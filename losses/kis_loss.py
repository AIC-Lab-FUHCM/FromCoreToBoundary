from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def cpd_class_weights(n_cls: int, device: Optional[torch.device] = None) -> torch.Tensor:
    return torch.ones((1, n_cls, 1), device=device, dtype=torch.float32)


def comp_class_pro(label: torch.Tensor, feature: torch.Tensor, num_classes: int) -> torch.Tensor:
    B, C, _, _ = feature.shape
    feature_fp32 = feature.float()
    prototypes = torch.zeros((B, num_classes, C), device=feature.device, dtype=torch.float32)
    for b in range(B):
        lbl_b = label[b]
        for cls in range(num_classes):
            mask = (lbl_b == cls)
            if mask.sum() > 0:
                prototypes[b, cls] = feature_fp32[b, :, mask].mean(dim=1).float()
    return prototypes


class DiceLoss(nn.Module):
    def __init__(self, n_classes: int = 9):
        super().__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat([(input_tensor == i).unsqueeze(1) for i in range(self.n_classes)], dim=1).float()

    def _dice_loss(self, score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        return 1.0 - (2.0 * intersect + smooth) / (z_sum + y_sum + smooth)

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
        loss = 0.0
        for i in range(self.n_classes):
            loss += self._dice_loss(inputs[:, i], target[:, i]) * weight[i]
        return loss / self.n_classes


def structure_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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


def lift_binary_logits(logits_1ch: torch.Tensor) -> torch.Tensor:
    return torch.cat([1.0 - 0.5 * logits_1ch, 1.0 + 0.5 * logits_1ch], dim=1)


class KISLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 9,
        dice_weight: float = 0.9,
        sgs_weight: float = 0.8,
        kis_weight: float = 0.2,
        normalize_logits: bool = False,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.dice_weight = float(dice_weight)
        self.sgs_weight = float(sgs_weight)
        self.kis_weight = float(kis_weight)
        self.normalize_logits = bool(normalize_logits)
        self.eps = float(eps)
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(n_classes=max(2, self.num_classes))

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize_logits:
            return x
        if x.shape[1] == 1:
            return (x - x.mean(dim=(2, 3), keepdim=True)) / (x.std(dim=(2, 3), keepdim=True) + self.eps)
        return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + self.eps)

    def forward(
        self,
        outputs_student: torch.Tensor,
        outputs_teacher: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        labels_long = labels.long()
        if labels_long.ndim == 4 and labels_long.shape[1] == 1:
            labels_long = labels_long.squeeze(1)

        stu_out = outputs_student.float()
        tea_out = outputs_teacher.detach().float()
        if tea_out.shape[-2:] != stu_out.shape[-2:]:
            tea_out = F.interpolate(tea_out, size=stu_out.shape[-2:], mode="bilinear", align_corners=False)

        stu_kd = self._standardize(stu_out)
        tea_kd = self._standardize(tea_out)

        if self.num_classes == 1:
            mask_f = labels_long.float().unsqueeze(1)
            loss_ce = F.binary_cross_entropy_with_logits(stu_out, mask_f)
            loss_sgs = structure_loss(stu_out, mask_f)
            loss_dice = (loss_sgs - loss_ce).detach()
            loss_l2 = F.mse_loss(stu_kd, tea_kd)
            tea_proto = comp_class_pro(labels_long, tea_kd, num_classes=2)
            stu_proto = comp_class_pro(labels_long, stu_kd, num_classes=2)
            proloss_l2 = F.mse_loss(stu_proto, tea_proto)
        else:
            loss_ce = self.ce_loss(stu_out, labels_long)
            loss_dice = self.dice_loss(stu_out, labels_long, softmax=True)
            loss_sgs = (1.0 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice
            loss_l2 = F.mse_loss(stu_kd, tea_kd)
            tea_proto = comp_class_pro(labels_long, tea_kd, num_classes=self.num_classes)
            stu_proto = comp_class_pro(labels_long, stu_kd, num_classes=self.num_classes)
            mse_proto = F.mse_loss(stu_proto, tea_proto, reduction="none")
            proloss_l2 = (mse_proto * cpd_class_weights(self.num_classes, outputs_student.device)).mean()

        loss_kis = loss_l2 + proloss_l2
        loss_base = self.sgs_weight * loss_sgs + self.kis_weight * loss_kis
        return loss_base, {
            "loss_ce": loss_ce,
            "loss_dice": loss_dice,
            "loss_sgs": loss_sgs,
            "loss_l2": loss_l2,
            "proloss_l2": proloss_l2,
            "loss_kis": loss_kis,
            "loss_base": loss_base,
        }
