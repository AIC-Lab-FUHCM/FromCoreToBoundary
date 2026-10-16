"""Knowledge distillation baseline losses for the 300-epoch BTCV benchmark.

Implements:
  - IAD Stage 2 (KAS: Residual Feature Calibration + Contrastive Semantic Separation)
    with registered Conv2d parameters ``pos`` (1x1) and ``neg`` (3x3).
  - LSKD (CVPR 2024): Logit Standardization Knowledge Distillation.
  - SinKD: Differentiable Sinkhorn Optimal Transport Distillation (with teacher
    detached, student attached, L2-normalized features, and row-min shift stabilization).
  - DIST+ (TPAMI 2025): Inter-class and intra-class Pearson correlation distillation
    computed in float32 to prevent mixed-precision overflow over spatial pixels.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
for _p in (_REPO_ROOT, _REPO_ROOT / "models", _REPO_ROOT / "scripts", _REPO_ROOT / "scripts" / "tools"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.cba_v2.losses.kis_loss import DiceLoss, structure_loss
from Synapse_Tool_KD.criterion.lskd import logit_standardization_kd_loss, z_score_standardize_2d
from Synapse_Tool_KD.criterion.SinKD import sinkhorn_loss
from Synapse_Tool_KD.distillation_loss import DIST, inter_class_relation, intra_class_relation


class IADKasHead(nn.Module):
    """Registered KAS projection head for IAD Stage 2 (RFC + CSS, Paper Eq. 6 & Eq. 8).

    Fixes the original author's inline ``nn.Conv2d`` bug in ``Synapse_Tool_KD/trainer.py``
    by registering ``pos`` (256 -> C, 1x1) and ``neg`` (C -> C, 3x3) as persistent
    ``nn.Module`` parameters optimized alongside the student.
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 9):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.pos = nn.Conv2d(self.in_channels, self.num_classes, kernel_size=1)
        self.neg = nn.Conv2d(self.num_classes, self.num_classes, kernel_size=3, padding=1)

    def forward(
        self,
        stu_logits: torch.Tensor,
        stu_global: torch.Tensor,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        dev_type = stu_logits.device.type if stu_logits.device.type in ("cuda", "cpu") else "cpu"
        with torch.amp.autocast(device_type=dev_type, enabled=False):
            os_f = stu_logits.float()
            fl_f = stu_global.float()
            if fl_f.shape[1] != self.in_channels:
                repeats = math.ceil(self.in_channels / fl_f.shape[1])
                fl_f = fl_f.repeat(1, repeats, 1, 1)[:, : self.in_channels]

            f_plus = F.interpolate(
                self.pos(fl_f).float(),
                size=os_f.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            if self.num_classes == 1:
                p_os = torch.sigmoid(os_f)
                p_fp = torch.sigmoid(f_plus)
            else:
                p_os = F.softmax(os_f, dim=1)
                p_fp = F.softmax(f_plus, dim=1)
            f_neg = self.neg(((1.0 - p_os) + (1.0 - p_fp)).float()).float()

            # RFC loss: L2 + self-Dice between F+ and Os
            loss_l2 = F.mse_loss(f_plus, os_f)
            inter = (p_fp * p_os).sum(dim=(2, 3))
            union = (p_fp + p_os).sum(dim=(2, 3))
            loss_self_dice = 1.0 - ((2.0 * inter + 1e-6) / (union + 1e-6)).mean()
            loss_rfc = loss_l2 + loss_self_dice

            # CSS loss: global cosine InfoNCE in float32 via numerically stable softplus
            s_pos = F.cosine_similarity(os_f.flatten(1), f_plus.flatten(1), dim=1, eps=1e-8)
            s_neg = F.cosine_similarity(os_f.flatten(1), f_neg.flatten(1), dim=1, eps=1e-8)
            loss_css = F.softplus((s_neg - s_pos) / float(temperature)).mean()

            loss_kas = loss_rfc + loss_css
        return {
            "loss_kas": loss_kas,
            "loss_rfc": loss_rfc,
            "loss_css": loss_css,
            "f_plus": f_plus,
            "f_neg": f_neg,
        }


class LSKDLoss(nn.Module):
    """Logit Standardization Knowledge Distillation (Sun et al., CVPR 2024)."""

    def __init__(self, tau: float = 2.0, eps: float = 1e-3):
        super().__init__()
        self.tau = float(tau)
        self.eps = float(eps)

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        stu_f = student_logits.float()
        tea_f = teacher_logits.detach().float()
        if tea_f.shape[-2:] != stu_f.shape[-2:]:
            tea_f = F.interpolate(tea_f, size=stu_f.shape[-2:], mode="bilinear", align_corners=False)
        if stu_f.shape[1] == 1:
            # Binary segmentation (C=1): standardize logits per-sample across spatial pixels,
            # then compute pixel-averaged binary KL divergence over [1 - sigmoid(z), sigmoid(z)].
            z_s = (stu_f - stu_f.mean(dim=(2, 3), keepdim=True)) / (
                (stu_f.std(dim=(2, 3), keepdim=True) + self.eps) * self.tau
            )
            z_t = (tea_f - tea_f.mean(dim=(2, 3), keepdim=True)) / (
                (tea_f.std(dim=(2, 3), keepdim=True) + self.eps) * self.tau
            )
            q_t = torch.cat([1.0 - torch.sigmoid(z_t), torch.sigmoid(z_t)], dim=1).clamp(min=1e-8, max=1.0)
            q_s_log = F.log_softmax(torch.cat([-0.5 * z_s, 0.5 * z_s], dim=1), dim=1)
            return F.kl_div(q_s_log, q_t, reduction="none").sum(dim=1).mean() * (self.tau ** 2)
        z_s = z_score_standardize_2d(stu_f, tau=self.tau, eps=self.eps)
        z_t = z_score_standardize_2d(tea_f, tau=self.tau, eps=self.eps)
        q_t = F.softmax(z_t, dim=1).clamp(min=1e-8, max=1.0)
        q_s_log = F.log_softmax(z_s, dim=1)
        return F.kl_div(q_s_log, q_t, reduction="none").sum(dim=1).mean() * (self.tau ** 2)


class SinKDLoss(nn.Module):
    """Differentiable Sinkhorn Distance Knowledge Distillation."""

    def __init__(
        self,
        lam: float = 10.0,
        max_iter: int = 5,
        p: float = 2.0,
        eps: float = 1e-8,
        normalize: bool = True,
    ):
        super().__init__()
        self.lam = float(lam)
        self.max_iter = int(max_iter)
        self.p = float(p)
        self.eps = float(eps)
        self.normalize = bool(normalize)

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        return sinkhorn_loss(
            t=teacher_logits,
            s=student_logits,
            lam=self.lam,
            max_iter=self.max_iter,
            p=self.p,
            eps=self.eps,
            normalize=self.normalize,
        )


class DISTPlusLoss(nn.Module):
    """DIST+ inter-class and intra-class relation distillation in float32."""

    def __init__(self, beta: float = 1.0, gamma: float = 1.0, tau: float = 1.0):
        super().__init__()
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.dist = DIST(beta=beta, gamma=gamma, tau=tau)

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        if student_logits.ndim == 4 and student_logits.shape[1] == 1:
            stu_f = student_logits.float()
            tea_f = teacher_logits.detach().float()
            if tea_f.shape[-2:] != stu_f.shape[-2:]:
                tea_f = F.interpolate(tea_f, size=stu_f.shape[-2:], mode="bilinear", align_corners=False)
            p_s = torch.sigmoid(stu_f / self.tau).flatten(1)
            p_t = torch.sigmoid(tea_f / self.tau).flatten(1)
            y_s_2c = torch.stack([1.0 - p_s.reshape(-1), p_s.reshape(-1)], dim=1)
            y_t_2c = torch.stack([1.0 - p_t.reshape(-1), p_t.reshape(-1)], dim=1)
            inter_loss = inter_class_relation(p_s, p_t)
            intra_loss = intra_class_relation(y_s_2c, y_t_2c)
            return self.beta * inter_loss + self.gamma * intra_loss
        return self.dist(student_logits, teacher_logits)


class BaselineKDLoss(nn.Module):
    """Combined supervised segmentation loss (L_SGS) + KD baseline objective.

    Follows the benchmark weighting:
      L_SGS = (1 - dice_weight) * CE + dice_weight * Dice (or structure_loss for C=1 binary)
      L_total = sgs_weight * L_SGS + kd_weight * L_KD
    """

    def __init__(
        self,
        method: str,
        num_classes: int = 9,
        dice_weight: float = 0.9,
        sgs_weight: float = 0.8,
        kd_weight: float = 0.2,
        lskd_tau: float = 2.0,
        sinkd_lam: float = 10.0,
        sinkd_iters: int = 5,
        dist_beta: float = 1.0,
        dist_gamma: float = 1.0,
        dist_tau: float = 1.0,
    ):
        super().__init__()
        self.method = method.lower()
        self.num_classes = int(num_classes)
        self.dice_weight = float(dice_weight)
        self.sgs_weight = float(sgs_weight)
        self.kd_weight = float(kd_weight)

        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(n_classes=max(2, self.num_classes))

        if self.method == "lskd":
            self.kd_criterion: Optional[nn.Module] = LSKDLoss(tau=lskd_tau)
        elif self.method == "sinkd":
            self.kd_criterion = SinKDLoss(lam=sinkd_lam, max_iter=sinkd_iters)
        elif self.method in ("dist_plus", "dist+", "dist"):
            self.kd_criterion = DISTPlusLoss(beta=dist_beta, gamma=dist_gamma, tau=dist_tau)
        elif self.method == "sgs_only":
            self.kd_criterion = None
        else:
            raise ValueError(f"Unsupported BaselineKDLoss method: {method!r}")

    def forward_sgs(self, outputs_student: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        dev_type = outputs_student.device.type if outputs_student.device.type in ("cuda", "cpu") else "cpu"
        with torch.amp.autocast(device_type=dev_type, enabled=False):
            labels_long = labels.long()
            if labels_long.ndim == 4 and labels_long.shape[1] == 1:
                labels_long = labels_long.squeeze(1)
            stu_out = outputs_student.float()
            if self.num_classes == 1:
                mask_f = labels_long.float().unsqueeze(1)
                loss_ce = F.binary_cross_entropy_with_logits(stu_out, mask_f)
                loss_sgs = structure_loss(stu_out, mask_f)
                loss_dice = (loss_sgs - loss_ce).detach()
            else:
                loss_ce = self.ce_loss(stu_out, labels_long)
                loss_dice = self.dice_loss(stu_out, labels_long, softmax=True)
                loss_sgs = (1.0 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice
        return loss_sgs, {
            "loss_ce": loss_ce.detach(),
            "loss_dice": loss_dice.detach(),
            "loss_sgs": loss_sgs.detach(),
        }

    def forward(
        self,
        outputs_student: torch.Tensor,
        outputs_teacher: Optional[torch.Tensor],
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        dev_type = outputs_student.device.type if outputs_student.device.type in ("cuda", "cpu") else "cpu"
        with torch.amp.autocast(device_type=dev_type, enabled=False):
            loss_sgs, metrics = self.forward_sgs(outputs_student, labels)
            if self.kd_criterion is None or outputs_teacher is None:
                metrics["loss_kd"] = outputs_student.new_zeros((), dtype=torch.float32)
                metrics["loss_base"] = loss_sgs.detach()
                return loss_sgs, metrics

            loss_kd = self.kd_criterion(outputs_student.float(), outputs_teacher.detach().float())
            loss_base = self.sgs_weight * loss_sgs + self.kd_weight * loss_kd
        metrics["loss_kd"] = loss_kd.detach()
        metrics["loss_base"] = loss_base.detach()
        return loss_base, metrics


__all__ = [
    "IADKasHead",
    "LSKDLoss",
    "SinKDLoss",
    "DISTPlusLoss",
    "BaselineKDLoss",
    "logit_standardization_kd_loss",
    "sinkhorn_loss",
    "DIST",
]

