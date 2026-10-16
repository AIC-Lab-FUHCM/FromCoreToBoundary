"""
kis_loss.py - Knowledge Infusion Stage (KIS) Loss Module.

Paper Reference:
  From Infusion to Assimilation Distillation for Medical Image Segmentation (IAD, CVPR 2026, Hong et al.)
  Eq. 2 / Fig. 9: Class-weighted Prototype Distillation (CPD) with Gallbladder (2x) and Pancreas (4x).
  Eq. 7 & Section 4.2: Total Stage 1 Infusion Loss:
    L_total = 0.8 * L_SGS + 0.2 * L_KIS
  where:
    L_SGS = (1 - dice_weight) * L_CE + dice_weight * L_Dice  (default: 0.1 * L_CE + 0.9 * L_Dice)
    L_KIS = MSE(outputs_s, outputs_t) + CPD_Loss
    CPD_Loss = mean(MSE(proto_s, proto_t) * weights_cpd)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

SYNAPSE_GALLBLADDER = 4
SYNAPSE_PANCREAS = 8

CPD_CLASS_WEIGHTS = {
    "paper": {SYNAPSE_GALLBLADDER: 2.0, SYNAPSE_PANCREAS: 4.0},
    "legacy_ids": {4: 2.0, 8: 4.0},
    "flat": {},
}


def cpd_class_weights(n_cls: int, mode: str = "flat", device: Optional[torch.device] = None) -> torch.Tensor:
    """Return CPD class weight tensor of shape [1, n_cls, 1] in float32."""
    if mode not in CPD_CLASS_WEIGHTS:
        raise ValueError(f"unknown cpd_weight_mode {mode!r}; choose from {sorted(CPD_CLASS_WEIGHTS)}")
    weights = torch.ones(n_cls, device=device, dtype=torch.float32)
    for cls, w in CPD_CLASS_WEIGHTS[mode].items():
        if cls < n_cls:
            weights[cls] = float(w)
    return weights.view(1, n_cls, 1)


def comp_class_pro(label: torch.Tensor, feature: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute per-class prototypes from features.

    Args:
        label: (B, H, W) long/int tensor.
        feature: (B, C, H, W) float tensor (e.g. logits or intermediate representation).
        num_classes: int number of classes (typically 9 for Synapse: background + 8 organs).

    Returns:
        prototypes: (B, num_classes, C) tensor strictly in float32.
    """
    B, C, H, W = feature.shape
    feature_fp32 = feature.float()
    prototypes = torch.zeros((B, num_classes, C), device=feature.device, dtype=torch.float32)

    for b in range(B):
        lbl_b = label[b]
        for cls in range(num_classes):
            mask = (lbl_b == cls)
            if mask.sum() == 0:
                continue
            cls_feats = feature_fp32[b, :, mask]  # [C, N]
            prototypes[b, cls] = cls_feats.mean(dim=1).float()

    return prototypes


class DiceLoss(nn.Module):
    """Multi-class Dice Loss matching Synapse_Tool_KD/utils.py implementation."""

    def __init__(self, n_classes: int = 9):
        super().__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor: torch.Tensor) -> torch.Tensor:
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = (input_tensor == i)
            tensor_list.append(temp_prob.unsqueeze(1))
        return torch.cat(tensor_list, dim=1).float()

    def _dice_loss(self, score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2.0 * intersect + smooth) / (z_sum + y_sum + smooth)
        return 1.0 - loss

    def forward(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[list] = None,
        softmax: bool = True,
    ) -> torch.Tensor:
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1.0] * self.n_classes
        assert inputs.size() == target.size(), (
            f"predict {inputs.size()} & target {target.size()} shape do not match"
        )
        loss = 0.0
        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes


def structure_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Weighted BCE + Weighted IoU loss for binary polyp segmentation (Polyp_Tool/train_isic.py)."""
    pred_f = pred.float()
    mask_f = mask.float()
    if mask_f.ndim == 3:
        mask_f = mask_f.unsqueeze(1)
    weit = 1.0 + 5.0 * torch.abs(F.avg_pool2d(mask_f, kernel_size=31, stride=1, padding=15) - mask_f)
    wbce = F.binary_cross_entropy_with_logits(pred_f, mask_f, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3)).clamp_min(1e-6)

    pred_sig = torch.sigmoid(pred_f)
    inter = ((pred_sig * mask_f) * weit).sum(dim=(2, 3))
    union = ((pred_sig + mask_f) * weit).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1.0) / (union - inter + 1.0).clamp_min(1e-6)
    return (wbce + wiou).mean()


def lift_binary_logits_for_cba(logits_1ch: torch.Tensor) -> torch.Tensor:
    """Lift 1-channel binary logits s [B, 1, H, W] to 2-class logits h = [1 - s/2, 1 + s/2] [B, 2, H, W].

    Satisfies:
      1. h[:, 1] - h[:, 0] == s, so softmax(h) == [1 - sigmoid(s), sigmoid(s)],
         argmax(h) == (s > 0), and cross_entropy(h, y) == binary_cross_entropy_with_logits(s, y).
      2. Offsets the 1D logit line off the origin so L2-normalization z = h / ||h||_2 maps
         s in (-inf, +inf) smoothly and monotonically onto the unit circle S^1 with non-zero gradients.
    """
    return torch.cat([1.0 - 0.5 * logits_1ch, 1.0 + 0.5 * logits_1ch], dim=1)


class KISLoss(nn.Module):
    """Knowledge Infusion Stage (KIS) Loss Module.

    Calculates:
      L_SGS = 0.1 * L_CE + 0.9 * L_Dice (multi-class) or structure_loss (binary num_classes=1)
      L_KIS = MSE(outputs_s, outputs_t) + mean(MSE(proto_s, proto_t) * cpd_weights)
      L_base = 0.8 * L_SGS + 0.2 * L_KIS
    """

    def __init__(
        self,
        num_classes: int = 9,
        dice_weight: float = 0.9,
        cpd_weight_mode: str = "flat",
        sgs_weight: float = 0.8,
        kis_weight: float = 0.2,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.dice_weight = float(dice_weight)
        self.cpd_weight_mode = str(cpd_weight_mode)
        self.sgs_weight = float(sgs_weight)
        self.kis_weight = float(kis_weight)

        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(n_classes=max(2, self.num_classes))

    def forward(
        self,
        outputs_student: torch.Tensor,
        outputs_teacher: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculate KIS loss in float32.

        Args:
            outputs_student: [B, num_classes, H, W] student logits.
            outputs_teacher: [B, num_classes, H, W] teacher logits (or detach() externally).
            labels: [B, H, W] ground-truth organ/polyp labels.

        Returns:
            loss_base: scalar combined base loss (0.8 * L_sgs + 0.2 * L_kis).
            metrics: Dict of individual loss terms as scalar tensors.
        """
        labels_long = labels.long()
        if labels_long.ndim == 4 and labels_long.shape[1] == 1:
            labels_long = labels_long.squeeze(1)

        stu_out = outputs_student.float()
        tea_out = outputs_teacher.detach().float()

        # Match spatial resolution if needed
        if tea_out.shape[-2:] != stu_out.shape[-2:]:
            tea_out = F.interpolate(
                tea_out,
                size=stu_out.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if self.num_classes == 1:
            mask_f = labels_long.float().unsqueeze(1)
            loss_ce = F.binary_cross_entropy_with_logits(stu_out, mask_f)
            loss_sgs = structure_loss(stu_out, mask_f)
            loss_dice = (loss_sgs - loss_ce).detach()
            loss_l2 = F.mse_loss(stu_out, tea_out)
            tea_proto = comp_class_pro(labels_long, tea_out, num_classes=2)
            stu_proto = comp_class_pro(labels_long, stu_out, num_classes=2)
            proloss_l2 = F.mse_loss(stu_proto, tea_proto)
        else:
            # 1. SGS Supervised Loss: (1 - dice_weight) * CE + dice_weight * Dice (strictly float32)
            loss_ce = self.ce_loss(stu_out, labels_long)
            loss_dice = self.dice_loss(stu_out, labels_long, softmax=True)
            loss_sgs = (1.0 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice

            # 3. Logit distillation: MSE(outputs_student, tea_out) strictly float32
            loss_l2 = F.mse_loss(stu_out, tea_out)

            # 4. Class prototype distillation (CPD) strictly float32
            tea_proto = comp_class_pro(labels_long, tea_out, num_classes=self.num_classes)
            stu_proto = comp_class_pro(labels_long, stu_out, num_classes=self.num_classes)
            mse_proto = F.mse_loss(stu_proto, tea_proto, reduction="none")
            weights = cpd_class_weights(
                self.num_classes, self.cpd_weight_mode, outputs_student.device
            )
            proloss_l2 = (mse_proto * weights).mean()

        loss_kis = loss_l2 + proloss_l2

        # 5. Combined base loss (Paper Eq. 7 & Section 4.2)
        loss_base = self.sgs_weight * loss_sgs + self.kis_weight * loss_kis

        metrics = {
            "loss_ce": loss_ce,
            "loss_dice": loss_dice,
            "loss_sgs": loss_sgs,
            "loss_l2": loss_l2,
            "proloss_l2": proloss_l2,
            "loss_kis": loss_kis,
            "loss_base": loss_base,
        }
        return loss_base, metrics

