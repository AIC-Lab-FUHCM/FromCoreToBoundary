"""
far_fp_loss.py - Far False-Positive (far-FP) Penalty Loss Module.

Mathematical Rationale:
  Standard segmentation losses (Dice, Cross-Entropy) and distillation losses (KIS, CPD)
  treat all false-positive predictions equally, irrespective of their spatial distance
  from the true anatomical structure.
  Consequently, high-capacity models (or models distilling from teachers with ghost activations)
  can develop spurious "ghost islands" far from the target organ (e.g., predicted spleen
  in the contralateral lower quadrant, distance > 20 voxels from the true spleen).

  L_far directly penalizes foreground class probability mass predicted in the "far zone"
  (3D/2D Euclidean distance > d_thresh, default 20 voxels) from the true ground truth.

Formulation:
  For each slice b and foreground class c in {1..num_classes-1}:
    1. Let G_{b, c} be the ground truth binary mask.
    2. Dilation with circular Euclidean kernel of radius R = floor(d_thresh):
         M_{near, c} = Dilation(G_{b, c}, radius=R)
         M_{far, c} = ~M_{near, c}  (distance strictly > d_thresh)
       If class c is absent on slice b:
         M_{far, c} is all foreground/tissue pixels (or full slice if penalty_absent=True).
    3. Far-FP Mass Penalty:
       - Mode 'fraction':
           L_{far, c} = sum(M_{far, c} * p_c) / (sum(p_c) + epsilon)
           Naturally bounded in [0, 1]: directly penalizes the proportion of predicted
           organ volume that constitutes far false positives.
       - Mode 'bce':
           L_{far, c} = mean(-log(1 - p_c + epsilon) on M_{far, c})
       - Mode 'focal':
           L_{far, c} = mean(p_c^2 on M_{far, c})
       - Mode 'combined':
           alpha * L_fraction + (1 - alpha) * L_bce
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_circular_disk_kernel(radius: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Create exact 2D circular Euclidean structuring element for dilation conv."""
    r = int(radius)
    y, x = torch.meshgrid(
        torch.arange(-r, r + 1, dtype=torch.float32, device=device),
        torch.arange(-r, r + 1, dtype=torch.float32, device=device),
        indexing="ij",
    )
    disk = (x ** 2 + y ** 2 <= r ** 2).float()
    return disk.unsqueeze(0).unsqueeze(0)  # (1, 1, 2r+1, 2r+1)


class FarFPLoss(nn.Module):
    """Far False-Positive Penalty Loss Module."""

    def __init__(
        self,
        num_classes: int = 9,
        distance_threshold: float = 20.0,
        mode: str = "fraction",
        lambda_far: float = 0.05,
        penalty_absent_slices: bool = True,
        ref_organ_size: float = 1000.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.distance_threshold = float(distance_threshold)
        self.radius = int(self.distance_threshold)
        self.mode = str(mode).lower()
        self.lambda_far = float(lambda_far)
        self.penalty_absent_slices = bool(penalty_absent_slices)
        self.ref_organ_size = float(ref_organ_size)
        self.eps = float(eps)

        if self.radius > 0:
            kernel = make_circular_disk_kernel(self.radius)
            self.register_buffer("kernel", kernel)
        else:
            self.kernel = None

    def compute_far_masks(self, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute spatial binary masks indicating regions > distance_threshold from GT.

        Args:
            label: (B, H, W) long tensor with integer class labels.

        Returns:
            far_masks: (B, num_classes, H, W) float tensor where 1.0 indicates far from GT of that class.
            has_organ: (B, num_classes) bool tensor indicating presence of each class per slice.
        """
        B, H, W = label.shape
        device = label.device
        far_masks = torch.zeros((B, self.num_classes, H, W), device=device, dtype=torch.float32)
        has_organ = torch.zeros((B, self.num_classes), device=device, dtype=torch.bool)

        if self.kernel is None:
            return far_masks, has_organ

        for c in range(1, self.num_classes):
            g_c = (label == c).unsqueeze(1).float()  # (B, 1, H, W)
            has_c = (g_c.sum(dim=(2, 3), keepdim=True) > 0)  # (B, 1, 1, 1)
            has_organ[:, c] = has_c.squeeze(-1).squeeze(-1).squeeze(1)

            # Conv dilation with circular disk kernel (skip convolution if absent across entire batch)
            if g_c.sum() > 0:
                dilated = (F.conv2d(g_c, self.kernel, padding=self.radius) > 0.0).float()
            else:
                dilated = torch.zeros_like(g_c)
            far_c = 1.0 - dilated  # 1 where dist > radius

            if self.penalty_absent_slices:
                # On slices where organ c is completely absent, all pixels are far
                far_c = torch.where(has_c, far_c, torch.ones_like(far_c))
            else:
                # If absent slices penalty is disabled, only penalize slices where organ is present
                far_c = torch.where(has_c, far_c, torch.zeros_like(far_c))

            far_masks[:, c:c+1] = far_c

        return far_masks, has_organ

    def forward(
        self,
        logits: torch.Tensor,
        label: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute Far-FP penalty loss.

        Args:
            logits: (B, C, H, W) float tensor of unnormalized student logits.
            label: (B, H, W) long tensor with GT class labels in 0..C-1.

        Returns:
            loss: scalar torch.Tensor.
            metrics: dict with loss breakdown.
        """
        device = logits.device
        B, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1)

        far_masks, has_organ = self.compute_far_masks(label)  # (B, C, H, W), (B, C)

        per_class_losses = []
        active_losses = []
        far_mass_total = 0.0
        pred_mass_total = 0.0

        for c in range(1, self.num_classes):
            p_c = probs[:, c]  # (B, H, W)
            m_far = far_masks[:, c]  # (B, H, W)
            has_c = has_organ[:, c]  # (B,)

            # Soft-threshold near-zero residual softmax background noise (< 1%)
            p_c_eff = F.relu(p_c - 0.01)

            far_mass = (p_c_eff * m_far).sum(dim=(1, 2))  # (B,)
            near_mass = (p_c_eff * (1.0 - m_far)).sum(dim=(1, 2))  # (B,)
            total_mass = p_c_eff.sum(dim=(1, 2))  # (B,)

            far_mass_total += float(far_mass.sum().item())
            pred_mass_total += float(total_mass.sum().item())

            # Normalization scale:
            # - When organ is present: normalize by detached near mass (clamped to ref_organ_size)
            #   so loss is scale-invariant without distorting true organ gradients.
            # - When organ is absent: normalize by ref_organ_size, giving steady, strictly non-zero
            #   gradients to actively penalize spurious hallucinations on absent slices.
            denom = torch.where(
                has_c,
                torch.clamp(near_mass.detach(), min=self.ref_organ_size),
                torch.full_like(near_mass, self.ref_organ_size),
            )

            active = (total_mass > 1.0).float()
            has_active = bool(active.sum().item() > 0)

            if self.mode == "fraction":
                ratio = far_mass / denom
                if has_active:
                    loss_c = (ratio * active).sum() / (active.sum() + self.eps)
                    active_losses.append(loss_c)
                else:
                    loss_c = torch.tensor(0.0, device=device)
            elif self.mode == "bce":
                # Negative log-likelihood of background on far pixels normalized by organ scale
                bce = -torch.log(torch.clamp(1.0 - p_c, min=self.eps))
                bce_far = (bce * m_far).sum(dim=(1, 2)) / denom
                loss_c = (bce_far * active).sum() / (active.sum() + self.eps) if has_active else torch.tensor(0.0, device=device)
                if has_active:
                    active_losses.append(loss_c)
            elif self.mode == "focal":
                # Quadratic penalty on far probabilities normalized by organ scale
                focal_far = (((p_c_eff ** 2) * m_far).sum(dim=(1, 2))) / denom
                loss_c = (focal_far * active).sum() / (active.sum() + self.eps) if has_active else torch.tensor(0.0, device=device)
                if has_active:
                    active_losses.append(loss_c)
            elif self.mode == "combined":
                ratio = far_mass / denom
                loss_frac = (ratio * active).sum() / (active.sum() + self.eps) if has_active else torch.tensor(0.0, device=device)
                bce = -torch.log(torch.clamp(1.0 - p_c, min=self.eps))
                bce_far = (bce * m_far).sum(dim=(1, 2)) / denom
                loss_bce = (bce_far * active).sum() / (active.sum() + self.eps) if has_active else torch.tensor(0.0, device=device)
                loss_c = 0.5 * loss_frac + 0.5 * loss_bce
                if has_active:
                    active_losses.append(loss_c)
            else:
                raise ValueError(f"Unknown FarFPLoss mode: {self.mode}")

            per_class_losses.append(loss_c)

        if active_losses:
            loss = torch.stack(active_losses).mean()
        elif per_class_losses:
            loss = torch.stack(per_class_losses).mean()
        else:
            loss = torch.tensor(0.0, device=device)

        far_fp_ratio = far_mass_total / (pred_mass_total + self.eps) if pred_mass_total > 0 else 0.0

        metrics = {
            "loss_far": float(loss.item()),
            "far_mass": far_mass_total,
            "pred_mass": pred_mass_total,
            "far_fp_ratio": far_fp_ratio,
        }

        return loss, metrics
