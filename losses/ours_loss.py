from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class OursLoss(nn.Module):
    def __init__(
        self,
        m: float = 0.2,
        tau: float = 0.1,
        r: int = 2,
        beta: float = 0.5,
        k_bg: int = 512,
        k_fg: int = 512,
        num_classes: int = 9,
        feat_dim: int = 9,
        cache_momentum: float = 0.9,
        feature_space: str = "centered",
        error_reduction: str = "class_balanced",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.m = float(m)
        self.tau = float(tau)
        self.r = int(r)
        self.beta = float(beta)
        self.k_bg = int(k_bg)
        self.k_fg = int(k_fg)
        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.cache_momentum = float(cache_momentum)
        self.feature_space = str(feature_space).strip().lower()
        self.error_reduction = str(error_reduction).strip().lower()
        self.eps = float(eps)

        self.register_buffer("prototype_buffer", torch.zeros((self.num_classes, self.feat_dim), dtype=torch.float32), persistent=True)
        self.register_buffer("valid_mask", torch.zeros(self.num_classes, dtype=torch.bool), persistent=True)
        self.register_buffer("update_count", torch.zeros(self.num_classes, dtype=torch.long), persistent=True)

    def _center(self, x: torch.Tensor) -> torch.Tensor:
        return x - x.mean(dim=1, keepdim=True) if self.feature_space == "centered" else x

    def _project_grad(self, g: torch.Tensor) -> torch.Tensor:
        return g - g.mean(dim=-1, keepdim=True) if self.feature_space == "centered" else g

    def forward(
        self,
        fplus: torch.Tensor,
        y_gt: torch.Tensor,
        fplus_ema: Optional[torch.Tensor] = None,
        g_ref: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = fplus.device
        B, C, H, W = fplus.shape
        calc_dtype = torch.float32 if fplus.dtype in (torch.float16, torch.bfloat16) else fplus.dtype
        fplus_work = fplus.to(dtype=calc_dtype)

        if self.prototype_buffer.shape != (C, C) and not self.valid_mask.any():
            self.num_classes = C
            self.feat_dim = C
            self.register_buffer("prototype_buffer", torch.zeros((C, C), device=device, dtype=torch.float32), persistent=True)
            self.register_buffer("valid_mask", torch.zeros(C, dtype=torch.bool, device=device), persistent=True)
            self.register_buffer("update_count", torch.zeros(C, dtype=torch.long, device=device), persistent=True)

        h_feat = self._center(fplus_work)
        z = F.normalize(h_feat, p=2, dim=1, eps=self.eps)
        z_for_proto = (
            F.normalize(self._center(fplus_ema.to(dtype=calc_dtype)), p=2, dim=1, eps=self.eps)
            if fplus_ema is not None
            else z
        )

        y_gt_one_hot = F.one_hot(y_gt.long(), num_classes=C).permute(0, 3, 1, 2).to(dtype=calc_dtype)
        k_size = self.r * 2 + 1
        core_mask = -F.max_pool2d(-y_gt_one_hot, kernel_size=k_size, stride=1, padding=self.r)
        dilated_mask = F.max_pool2d(y_gt_one_hot, kernel_size=k_size, stride=1, padding=self.r)
        boundary_mask = y_gt_one_hot * (1.0 - core_mask)

        batch_protos = torch.zeros((C, C), device=device, dtype=torch.float32)
        batch_valid = torch.zeros(C, dtype=torch.bool, device=device)
        core_sums = core_mask.sum(dim=(0, 2, 3))
        gt_sums = y_gt_one_hot.sum(dim=(0, 2, 3))

        for c in range(C):
            if core_sums[c] > 0:
                mu_c = (z_for_proto * core_mask[:, c : c + 1]).sum(dim=(0, 2, 3)) / core_sums[c]
                batch_protos[c] = F.normalize(mu_c, p=2, dim=0, eps=self.eps).detach()
                batch_valid[c] = True
            elif gt_sums[c] > 0:
                mu_c = (z_for_proto * y_gt_one_hot[:, c : c + 1]).sum(dim=(0, 2, 3)) / gt_sums[c]
                batch_protos[c] = F.normalize(mu_c, p=2, dim=0, eps=self.eps).detach()
                batch_valid[c] = True

        if self.training:
            for c in range(C):
                if batch_valid[c]:
                    if self.valid_mask[c]:
                        self.prototype_buffer[c] = (
                            self.cache_momentum * self.prototype_buffer[c]
                            + (1.0 - self.cache_momentum) * batch_protos[c]
                        )
                    else:
                        self.prototype_buffer[c] = batch_protos[c].clone()
                        self.valid_mask[c] = True
                    self.update_count[c] += 1

        error_valid = batch_valid | self.valid_mask
        p_prob = F.softmax(fplus_work, dim=1)
        pred_labels = torch.argmax(fplus_work, dim=1)
        g_gt_all = g_ref.detach().to(dtype=calc_dtype) if g_ref is not None else (p_prob - y_gt_one_hot)
        h_norm_all = torch.norm(h_feat, p=2, dim=1, keepdim=True) + self.eps

        z_flat = z.permute(0, 2, 3, 1).reshape(-1, C)
        g_gt_flat = g_gt_all.permute(0, 2, 3, 1).reshape(-1, C)
        h_norm_flat = h_norm_all.permute(0, 2, 3, 1).reshape(-1, 1)

        pair_loss_sum = torch.tensor(0.0, device=device, dtype=calc_dtype)
        boundary_pair_count = 0
        boundary_present = (boundary_mask.sum(dim=(0, 2, 3)) > 0).tolist()

        for c in range(C):
            if not boundary_present[c] or not batch_valid[c].item():
                continue
            overlap = (boundary_mask[:, c : c + 1] > 0) & (dilated_mask > 0)
            adj_counts = overlap.sum(dim=(0, 2, 3)).tolist()
            for d in range(C):
                if c == d or adj_counts[d] == 0 or not batch_valid[d].item():
                    continue
                flat_valid = torch.where(overlap[:, d].reshape(-1))[0]
                if len(flat_valid) == 0:
                    continue

                z_v = z_flat[flat_valid]
                g_gt_v = g_gt_flat[flat_valid]
                h_n_v = h_norm_flat[flat_valid]
                mu_c, mu_d = batch_protos[c], batch_protos[d]
                delta_mu = (mu_d - mu_c).unsqueeze(0)

                sim_c = (z_v * mu_c.unsqueeze(0)).sum(dim=1, keepdim=True)
                sim_d = (z_v * mu_d.unsqueeze(0)).sum(dim=1, keepdim=True)
                u = (self.m + sim_d - sim_c) / self.tau
                ell_b = F.softplus(u)
                sigma = torch.sigmoid(u)

                proj = (z_v * delta_mu).sum(dim=1, keepdim=True)
                g_a_v = self._project_grad((sigma / (self.tau * h_n_v)) * (delta_mu - z_v * proj))

                dot = (g_gt_v * g_a_v).sum(dim=1, keepdim=True)
                norm_gt = torch.norm(g_gt_v, p=2, dim=1, keepdim=True) + self.eps
                norm_ga = torch.norm(g_a_v, p=2, dim=1, keepdim=True) + self.eps
                a_gate = torch.clamp(dot / (norm_gt * norm_ga), min=0.0, max=1.0).detach()

                pair_loss_sum = pair_loss_sum + (a_gate * ell_b).mean()
                boundary_pair_count += 1

        loss_b = pair_loss_sum / float(boundary_pair_count) if boundary_pair_count > 0 else 0.0 * fplus_work.sum()

        active_error_losses: List[torch.Tensor] = []
        total_pixel_e_loss = torch.tensor(0.0, device=device, dtype=calc_dtype)
        total_sampled_error = 0
        err_flat_indices = torch.where((y_gt != pred_labels).view(-1))[0]
        if len(err_flat_indices) > 0:
            y_err_all = y_gt.view(-1)[err_flat_indices]
            preds_err_all = pred_labels.view(-1)[err_flat_indices]
            fplus_flat = fplus_work.permute(0, 2, 3, 1).reshape(-1, C)

            for c in range(C):
                c_mask = (y_err_all == c)
                if not c_mask.any() or not error_valid[c]:
                    continue
                idx_c = torch.where(c_mask)[0]
                valid_opp = error_valid[preds_err_all[idx_c]]
                if not valid_opp.any():
                    continue
                idx_val = idx_c[valid_opp]
                budget = self.k_bg if c == 0 else self.k_fg
                if len(idx_val) > budget:
                    idx_sel = idx_val[torch.randperm(len(idx_val), device=device)[:budget]]
                else:
                    idx_sel = idx_val

                flat_sel = err_flat_indices[idx_sel]
                logits_sel = fplus_flat[flat_sel]
                target_sel = torch.full((len(flat_sel),), c, dtype=torch.long, device=device)
                ell_c = F.cross_entropy(logits_sel, target_sel, reduction="none")
                active_error_losses.append(ell_c.mean())
                total_pixel_e_loss = total_pixel_e_loss + ell_c.sum()
                total_sampled_error += len(flat_sel)

        if active_error_losses:
            if self.error_reduction == "pixel_mean" and total_sampled_error > 0:
                loss_e = total_pixel_e_loss / (float(total_sampled_error) + self.eps)
            else:
                loss_e = torch.stack(active_error_losses).mean()
        else:
            loss_e = 0.0 * fplus_work.sum()
        loss_ours = loss_b + self.beta * loss_e

        return loss_ours, {
            "loss_b": float(loss_b.item()),
            "loss_e": float(loss_e.item()),
            "loss_ours": float(loss_ours.item()),
        }
