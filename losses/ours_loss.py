"""
ours_loss.py - Directionally Gated Core-to-Boundary Assimilation & Hard-Error Distillation (Ours / FromCoreToBoundary).

Extends the Core-to-Boundary Assimilation formulation to dual cooperative streams:
  1. Boundary Stream (L_B):
     - Focuses on adjacent inter-class interface bands (radius r=2).
     - Directionally gated against supervised CE gradients (a_i^B).
  2. Error-Guided Stream (L_E):
     - Focuses on hard misclassified pixels anywhere in the 2D CT slice:
       E_c = {i | y_i = c, argmax_k p_i(k) != c}
     - Dedicated sampling budget for background (K_bg) and foreground organs (K_fg).
     - Class-balanced reduction: average within class S_c, then average across active classes C_E.
     - Directionally gated against supervised CE gradients (a_i^E).

EMA Prototype Memory Cache:
  - Maintains `prototype_buffer` [num_classes, feat_dim] across batches.
  - Extracted from EMA student representations when available.
  - When a class is present, updates cache via EMA momentum.
  - When an opponent class is absent in the batch, falls back to valid cached prototype.
  - If a class has never been seen, gracefully skips the pair and records diagnostics.

Feature space (feature_space):
  - "raw":      z = normalize(h)                        (h = student logits)
  - "centered": z = normalize(h - mean_k h_k)           (removes the common-mode component that
                dominates raw logits; prototypes, EMA features and ||h|| use the same centered h)
  The closed-form gradients below are taken w.r.t. the *original* logits: centering is the linear
  projection P = I - 11^T/K, so they are projected with P (a no-op when z and the prototypes are
  already zero-mean, which holds for centered features).

Total Loss Formulation:
  L_CBA_v2 = L_B + beta * L_E
  (Raw losses returned without internal lambda_A multiplication, so the runner controls:
   L_total = 0.8 * L_SGS + 0.2 * L_KIS + lambda_A(e) * (L_B + beta * L_E))
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


GATING_TYPE_ALIASES = {
    "none": "none",
    "ungated": "none",
    "arm_a": "none",
    "a": "none",
    "arm_b": "none",
    "b": "none",
    "arm_c": "none",
    "c": "none",
    "soft": "soft",
    "soft_gated": "soft",
    "gated": "soft",
    "arm_d": "soft",
    "d": "soft",
    "arm_d_cba_ema_gated": "soft",
    "binary": "binary",
    "binary_gated": "binary",
    "arm_e": "binary",
    "e": "binary",
    "arm_e_cba_ema_binarygated": "binary",
    "parametric": "parametric",
    "parametric_gated": "parametric",
    "arm_g": "parametric",
    "g": "parametric",
    "arm_g_cba_ema_parametricgated": "parametric",
}


class CBAv2Loss(nn.Module):
    """Directionally Gated Core-to-Boundary Assimilation Version 2 (CBA v2) Loss Module."""

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
        lambda_A: float = 1.0,
        gating_type: str = "soft",
        error_gating_type: Optional[str] = None,
        rho: float = 0.5,
        boundary_reduction: str = "pair_mean",
        error_reduction: str = "class_balanced",
        eps: float = 1e-8,
        force_ungated: bool = False,
        reduction: Optional[str] = None,
        enable_boundary: bool = True,
        enable_error: bool = True,
        error_mode: str = "contrastive",
        use_cache_for_all: bool = False,
        boundary_use_cache: bool = False,
        boundary_fallback_cache: bool = False,
        boundary_proto_source: Optional[str] = None,
        error_use_cache: bool = True,
        error_proto_source: str = "fallback",
        align_ce_hard_samples: Optional[bool] = None,
        feature_space: str = "raw",
        fp_by_pred_class: bool = False,
        fp_exclude_boundary: bool = False,
        empty_slice_suppression: bool = False,
        empty_slice_topk: int = 64,
        empty_slice_prob_thresh: float = 0.05,
        empty_slice_focal_gamma: float = 2.0,
        empty_slice_weight: float = 1.0,
    ):
        super().__init__()
        self.feature_space = str(feature_space).strip().lower()
        if self.feature_space not in ("raw", "centered"):
            raise ValueError(f"Unknown feature_space '{feature_space}'. Valid: 'raw', 'centered'")
        self.m = float(m)
        self.tau = float(tau)
        self.r = int(r)
        self.beta = float(beta)
        self.k_bg = int(k_bg)
        self.k_fg = int(k_fg)
        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.cache_momentum = float(cache_momentum)
        self.lambda_A = float(lambda_A)
        self.eps = float(eps)
        self.rho = float(rho)
        self.enable_boundary = bool(enable_boundary)
        self.enable_error = bool(enable_error)
        self.use_cache_for_all = bool(use_cache_for_all)
        self.boundary_fallback_cache = bool(boundary_fallback_cache)
        self.fp_by_pred_class = bool(fp_by_pred_class)
        self.fp_exclude_boundary = bool(fp_exclude_boundary)
        self.empty_slice_suppression = bool(empty_slice_suppression)
        self.align_ce_hard_samples = (
            (not self.fp_by_pred_class) if align_ce_hard_samples is None else bool(align_ce_hard_samples)
        )

        # Resolve boundary prototype source:
        # - "batch": fresh batch prototype (exact v1 behavior)
        # - "fallback": batch prototype when present, cached anchor hat{mu}_c when missing
        # - "cache": cached anchor hat{mu}_c always
        if boundary_proto_source is not None:
            self.boundary_proto_source = str(boundary_proto_source).strip().lower()
        elif self.use_cache_for_all or boundary_use_cache:
            self.boundary_proto_source = "cache"
        elif self.boundary_fallback_cache:
            self.boundary_proto_source = "fallback"
        else:
            self.boundary_proto_source = "batch"

        if self.boundary_proto_source not in ("batch", "fallback", "cache"):
            raise ValueError(
                f"Unknown boundary_proto_source '{self.boundary_proto_source}'. "
                f"Valid: 'batch' (exact v1), 'fallback' (batch if present, cache if absent), 'cache' (cache anchor)"
            )
        self.boundary_use_cache = (self.boundary_proto_source == "cache")

        self.error_use_cache = bool(error_use_cache)
        self.error_proto_source = str(error_proto_source).strip().lower()
        if self.use_cache_for_all:
            self.error_proto_source = "cache"

        self.error_mode = str(error_mode).strip().lower()
        if self.error_mode not in ("contrastive", "ce_hard"):
            raise ValueError(f"Unknown error_mode '{error_mode}'. Valid: 'contrastive', 'ce_hard'")

        self.empty_slice_topk = int(empty_slice_topk)
        self.empty_slice_prob_thresh = float(empty_slice_prob_thresh)
        self.empty_slice_focal_gamma = float(empty_slice_focal_gamma)
        self.empty_slice_weight = float(empty_slice_weight)

        if self.empty_slice_topk <= 0:
            raise ValueError(f"empty_slice_topk must be > 0, got {empty_slice_topk}")
        if not (0.0 <= self.empty_slice_prob_thresh < 1.0):
            raise ValueError(f"empty_slice_prob_thresh must be in [0.0, 1.0), got {empty_slice_prob_thresh}")
        if self.empty_slice_focal_gamma < 0.0:
            raise ValueError(f"empty_slice_focal_gamma must be >= 0.0, got {empty_slice_focal_gamma}")
        if self.empty_slice_weight < 0.0:
            raise ValueError(f"empty_slice_weight must be >= 0.0, got {empty_slice_weight}")

        if not (0.0 <= self.rho <= 1.0):
            raise ValueError(f"rho must be in [0.0, 1.0], got {rho}")

        # Reductions
        if reduction is not None:
            boundary_reduction = reduction
        self.boundary_reduction = str(boundary_reduction).strip().lower()
        if self.boundary_reduction not in ("pair_mean", "pixel_mean"):
            raise ValueError(f"Unknown boundary_reduction '{boundary_reduction}'. Must be 'pair_mean' or 'pixel_mean'.")

        self.error_reduction = str(error_reduction).strip().lower()
        if self.error_reduction not in ("class_balanced", "pixel_mean"):
            raise ValueError(f"Unknown error_reduction '{error_reduction}'. Must be 'class_balanced' or 'pixel_mean'.")

        # Gating configuration
        raw_gate = str(gating_type).strip().lower()
        if force_ungated or raw_gate in ("none", "ungated"):
            self.gating_type = "none"
        else:
            if raw_gate not in GATING_TYPE_ALIASES:
                raise ValueError(
                    f"Unknown gating_type '{gating_type}'. Valid: 'soft', 'binary', 'parametric', 'none'"
                )
            self.gating_type = GATING_TYPE_ALIASES[raw_gate]

        if error_gating_type is not None:
            raw_err_gate = str(error_gating_type).strip().lower()
            if raw_err_gate not in GATING_TYPE_ALIASES:
                raise ValueError(
                    f"Unknown error_gating_type '{error_gating_type}'. Valid: 'soft', 'binary', 'parametric', 'none'"
                )
            self.error_gating_type = GATING_TYPE_ALIASES[raw_err_gate]
        else:
            self.error_gating_type = self.gating_type

        # The CE-hard error stream is always applied ungated. Refuse a gated configuration instead of
        # silently ignoring it (previously a_gate_e was hard-coded to 1 regardless of the setting).
        if self.error_mode == "ce_hard" and self.enable_error and self.error_gating_type != "none":
            raise ValueError(
                f"error_mode='ce_hard' cannot be gated (got error_gating_type='{self.error_gating_type}'). "
                "Pass error_gating_type='none'."
            )

        # EMA Prototype Memory Cache Buffers
        # prototype_buffer stores accumulator vectors v_c [num_classes, feat_dim]
        self.register_buffer("prototype_buffer", torch.zeros((self.num_classes, self.feat_dim), dtype=torch.float32), persistent=True)
        self.register_buffer("valid_mask", torch.zeros(self.num_classes, dtype=torch.bool), persistent=True)
        self.register_buffer("update_count", torch.zeros(self.num_classes, dtype=torch.long), persistent=True)

    def reset_cache(self) -> None:
        """Clear all cached prototypes."""
        self.prototype_buffer.zero_()
        self.valid_mask.zero_()
        self.update_count.zero_()

    def sync_cache_ddp(self) -> None:
        """Synchronize prototype cache across all DDP ranks using all_reduce.

        Averages accumulator vectors v_c across all ranks.
        """
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        world_size = torch.distributed.get_world_size()
        if world_size <= 1:
            return

        counts = self.update_count.clone().float()
        accum = self.prototype_buffer.clone()
        valid_f = self.valid_mask.clone().float()

        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(accum, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(valid_f, op=torch.distributed.ReduceOp.SUM)

        valid = (counts > 0)
        for c in range(self.num_classes):
            if valid[c] and valid_f[c] > 0:
                self.prototype_buffer[c] = accum[c] / valid_f[c]
            else:
                self.prototype_buffer[c].zero_()

        self.valid_mask.copy_(valid)
        self.update_count.copy_(counts.long())

    def set_prototype(self, c: int, prototype: torch.Tensor) -> None:
        """Manually populate the cache for class c with an initial accumulator vector v_c (unit norm)."""
        if not (0 <= c < self.num_classes):
            raise IndexError(f"Class index {c} out of range [0, {self.num_classes})")
        p = prototype.detach().to(device=self.prototype_buffer.device, dtype=torch.float32).view(-1)
        if p.shape[0] != self.prototype_buffer.shape[1]:
            raise ValueError(f"Expected prototype dim {self.prototype_buffer.shape[1]}, got {p.shape[0]}")
        p_norm = F.normalize(p, p=2, dim=0, eps=self.eps)
        self.prototype_buffer[c] = p_norm  # initial accumulator has norm 1.0
        self.valid_mask[c] = True
        self.update_count[c] = 1

    def get_cache_anchor(self, c: int) -> Optional[torch.Tensor]:
        """Return unit-normalized prototype anchor hat{mu}_c = v_c / (||v_c|| + eps)."""
        if not self.valid_mask[c]:
            return None
        return F.normalize(self.prototype_buffer[c], p=2, dim=0, eps=self.eps)

    def get_cache_anchors(self) -> torch.Tensor:
        """Return unit-normalized prototype anchors hat{mu} for all classes [C, feat_dim]."""
        return F.normalize(self.prototype_buffer, p=2, dim=1, eps=self.eps)

    def get_cache_coherence(self) -> Dict[int, float]:
        """Return temporal directional stability R_c = ||v_c||_2 before normalization."""
        return {
            c: float(torch.norm(self.prototype_buffer[c], p=2).item())
            for c in range(self.num_classes)
            if self.valid_mask[c].item()
        }

    def get_cached_prototypes(self) -> Dict[int, torch.Tensor]:
        """Return dictionary mapping valid class indices to unit-normalized anchors hat{mu}_c."""
        anchors = self.get_cache_anchors()
        return {
            c: anchors[c].clone()
            for c in range(self.num_classes)
            if self.valid_mask[c].item()
        }

    def _center(self, x: torch.Tensor) -> torch.Tensor:
        """Remove the per-pixel mean over the class dimension when feature_space == 'centered'."""
        if self.feature_space == "centered":
            return x - x.mean(dim=1, keepdim=True)
        return x

    def _project_grad(self, g: torch.Tensor) -> torch.Tensor:
        """Map a gradient w.r.t. the (centered) feature back to the original logits: g <- P g."""
        if self.feature_space == "centered":
            return g - g.mean(dim=-1, keepdim=True)
        return g

    def compute_gate(self, cos_sim: torch.Tensor, gtype: Optional[str] = None) -> torch.Tensor:
        """Compute directional gating factor a_i from cosine similarity c_i."""
        mode = gtype if gtype is not None else self.gating_type
        if mode == "none":
            return torch.ones_like(cos_sim)
        elif mode == "binary":
            return (cos_sim > 0.0).to(dtype=cos_sim.dtype).detach()
        elif mode == "parametric":
            pos_mask = (cos_sim > 0.0).to(dtype=cos_sim.dtype)
            clamped_c = torch.clamp(cos_sim, min=0.0, max=1.0)
            return (pos_mask * (self.rho + (1.0 - self.rho) * clamped_c)).detach()
        elif mode == "soft":
            return torch.clamp(cos_sim, min=0.0, max=1.0).detach()
        else:
            raise ValueError(f"Unknown gating mode '{mode}'")

    def forward(
        self,
        fplus: torch.Tensor,
        y_gt: torch.Tensor,
        fplus_ema: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
        return_stream_losses: bool = False,
        g_ref: Optional[torch.Tensor] = None,
        update_cache: Optional[bool] = None,
        compute_autograd_norms: bool = False,
        sample_seed: Optional[int] = None,
        apply_lambda_A: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass for CBA v2 Loss.

        Args:
            fplus: [B, C, H, W] student feature or logits tensor (C classes: 0 background + organs)
            y_gt: [B, H, W] ground-truth class labels (0..C-1)
            fplus_ema: [B, C, H, W] optional EMA student representations for prototype extraction
            return_diagnostics: if True, returns (total_loss, diagnostics_dict) or (loss_b, loss_e, diag)
            return_stream_losses: if True, returns raw (loss_b, loss_e) tensors without combining
            g_ref: [B, C, H, W] optional external reference gradient (e.g. g_base = grad(L_base, fplus))
            update_cache: bool flag whether to update prototype cache (default: self.training)
            compute_autograd_norms: bool flag whether to compute autograd gradient norms
            sample_seed: optional integer seed for reproducible error pixel sampling
            apply_lambda_A: bool flag whether to multiply total_cba_loss by lambda_A (default: False, runner controls)

        Returns:
            total_loss: scalar tensor with autograd connection to fplus (or (loss_b, loss_e))
            diagnostics: (optional) rich dict with stream losses, overlap, cache hits, conflict rates
        """
        device = fplus.device
        B, C, H, W = fplus.shape

        # Ensure all operations run in at least float32 (FP32)
        calc_dtype = torch.float32 if fplus.dtype in (torch.float16, torch.bfloat16) else fplus.dtype
        fplus_work = fplus.to(dtype=calc_dtype)

        # Dynamic dimension check for prototype buffer
        if self.prototype_buffer.shape[0] != C or self.prototype_buffer.shape[1] != C:
            if not self.valid_mask.any():
                self.num_classes = C
                self.feat_dim = C
                self.register_buffer("prototype_buffer", torch.zeros((C, C), device=device, dtype=torch.float32), persistent=True)
                self.register_buffer("valid_mask", torch.zeros(C, dtype=torch.bool, device=device), persistent=True)
                self.register_buffer("update_count", torch.zeros(C, dtype=torch.long, device=device), persistent=True)
            else:
                raise ValueError(
                    f"Class count mismatch: prototype_buffer has {self.prototype_buffer.shape[0]} classes, "
                    f"but input has C={C} classes. Call reset_cache() if changing dataset classes."
                )

        # 1. Feature normalization and prototype sources (feature space: raw or centered logits)
        h_feat = self._center(fplus_work)
        z = F.normalize(h_feat, p=2, dim=1, eps=self.eps)  # [B, C, H, W]
        if fplus_ema is not None:
            z_for_proto = F.normalize(self._center(fplus_ema.to(dtype=calc_dtype)), p=2, dim=1, eps=self.eps)
        else:
            z_for_proto = z

        # 2. Morphological bands: Core (erosion) and Boundary (dilation - core)
        y_gt_one_hot = F.one_hot(y_gt.long(), num_classes=C).permute(0, 3, 1, 2).to(dtype=calc_dtype)  # [B, C, H, W]
        kernel_size = self.r * 2 + 1
        core_mask = -F.max_pool2d(-y_gt_one_hot, kernel_size=kernel_size, stride=1, padding=self.r)
        dilated_mask = F.max_pool2d(y_gt_one_hot, kernel_size=kernel_size, stride=1, padding=self.r)
        boundary_mask = y_gt_one_hot * (1.0 - core_mask)  # [B, C, H, W]

        # 3. Class Prototypes mu_c from core regions in current batch
        batch_protos = torch.zeros((C, C), device=device, dtype=torch.float32)
        batch_valid = torch.zeros(C, dtype=torch.bool, device=device)
        proto_coherence: Dict[int, float] = {}

        # Fast reduction to detect class presence
        core_sums = core_mask.sum(dim=(0, 2, 3))
        gt_sums = y_gt_one_hot.sum(dim=(0, 2, 3))

        for c in range(C):
            c_sum = core_sums[c]
            if c_sum > 0:
                c_m = core_mask[:, c:c+1, :, :]
                mu_c_unnorm = (z_for_proto * c_m).sum(dim=(0, 2, 3)) / c_sum
                proto_coherence[c] = float(torch.norm(mu_c_unnorm, p=2).item())
                mu_c = F.normalize(mu_c_unnorm, p=2, dim=0, eps=self.eps)
                batch_protos[c] = mu_c.detach()
                batch_valid[c] = True
            elif gt_sums[c] > 0:
                # Fallback to whole organ mask for thin/small organs
                c_m = y_gt_one_hot[:, c:c+1, :, :]
                mu_c_unnorm = (z_for_proto * c_m).sum(dim=(0, 2, 3)) / gt_sums[c]
                proto_coherence[c] = float(torch.norm(mu_c_unnorm, p=2).item())
                mu_c = F.normalize(mu_c_unnorm, p=2, dim=0, eps=self.eps)
                batch_protos[c] = mu_c.detach()
                batch_valid[c] = True

        # Cache update strictly guarded: ONLY when in training mode AND update_cache is not explicitly False.
        # When evaluating or calculating diagnostics, eval mode strictly freezes cache.
        do_update_cache = self.training and (True if update_cache is None else bool(update_cache))

        for c in range(C):
            if batch_valid[c]:
                if do_update_cache:
                    if self.valid_mask[c]:
                        # Accumulator update: v_c <- 0.9 v_c + 0.1 mu_c^batch (unnormalized!)
                        self.prototype_buffer[c] = (
                            self.cache_momentum * self.prototype_buffer[c]
                            + (1.0 - self.cache_momentum) * batch_protos[c]
                        )
                    else:
                        # First-time class appearance: directly initialize v_c = mu_c^batch (norm = 1.0)
                        self.prototype_buffer[c] = batch_protos[c].clone()
                        self.valid_mask[c] = True
                    self.update_count[c] += 1

        # Unit-normalized cache anchors: hat{mu}_c = v_c / (||v_c|| + eps)
        cache_anchors = F.normalize(self.prototype_buffer, p=2, dim=1, eps=self.eps)

        # Cross-batch temporal directional stability R_c^temporal = ||v_c||_2 (measured BEFORE normalization)
        temporal_coherence: Dict[int, float] = {}
        for c in range(C):
            if self.valid_mask[c]:
                temporal_coherence[c] = float(torch.norm(self.prototype_buffer[c], p=2).item())

        # Boundary stream prototypes:
        # - "batch": fresh batch prototypes (exact CBA v1 behavior)
        # - "fallback": batch prototype if present, fallback to cached anchor hat{mu}_c if missing
        # - "cache": cached anchor hat{mu}_c always
        boundary_protos = torch.zeros((C, C), device=device, dtype=torch.float32)
        boundary_valid = torch.zeros(C, dtype=torch.bool, device=device)

        if self.boundary_proto_source == "cache":
            for c in range(C):
                if self.valid_mask[c]:
                    boundary_protos[c] = cache_anchors[c]
                    boundary_valid[c] = True
                elif batch_valid[c]:
                    boundary_protos[c] = batch_protos[c]
                    boundary_valid[c] = True
        elif self.boundary_proto_source == "fallback":
            for c in range(C):
                if batch_valid[c]:
                    boundary_protos[c] = batch_protos[c]
                    boundary_valid[c] = True
                elif self.valid_mask[c]:
                    boundary_protos[c] = cache_anchors[c]
                    boundary_valid[c] = True
        else:
            # Exact v1 behavior: strictly batch prototypes
            boundary_protos = batch_protos
            boundary_valid = batch_valid

        # Error stream prototypes:
        # If error_proto_source == "cache": use cached anchor hat{mu}_c
        # Else (default "fallback"): use batch_protos if present, fallback to cached anchor hat{mu}_c if absent
        error_protos = torch.zeros((C, C), device=device, dtype=torch.float32)
        error_valid = torch.zeros(C, dtype=torch.bool, device=device)
        proto_sources: Dict[int, str] = {}
        cache_hits = 0
        cache_misses = 0
        batch_hits = 0

        for c in range(C):
            if self.error_proto_source == "cache":
                if self.valid_mask[c]:
                    error_protos[c] = cache_anchors[c]
                    error_valid[c] = True
                    proto_sources[c] = "cache_anchor"
                    if not batch_valid[c]:
                        cache_hits += 1
                    else:
                        batch_hits += 1
                elif batch_valid[c]:
                    error_protos[c] = batch_protos[c]
                    error_valid[c] = True
                    proto_sources[c] = "batch"
                    batch_hits += 1
                else:
                    error_valid[c] = False
                    proto_sources[c] = "none"
                    cache_misses += 1
            else:
                # Default "fallback" mode
                if batch_valid[c]:
                    error_protos[c] = batch_protos[c]
                    error_valid[c] = True
                    proto_sources[c] = "batch"
                    batch_hits += 1
                elif self.valid_mask[c] and self.error_use_cache:
                    error_protos[c] = cache_anchors[c]
                    error_valid[c] = True
                    proto_sources[c] = "cache"
                    cache_hits += 1
                else:
                    error_valid[c] = False
                    proto_sources[c] = "none"
                    cache_misses += 1

        # 4. Supervised CE probabilities or custom reference gradient
        p_prob = F.softmax(fplus_work, dim=1)                 # [B, C, H, W]
        pred_labels = torch.argmax(fplus_work, dim=1)          # [B, H, W]
        if g_ref is not None:
            g_gt_all = g_ref.detach().to(dtype=calc_dtype)
        else:
            g_gt_all = p_prob - y_gt_one_hot
        h_norm_all = torch.norm(h_feat, p=2, dim=1, keepdim=True) + self.eps  # [B, 1, H, W]

        # Fast 1D flattened views for zero-overhead vector indexing
        total_pixels = B * H * W
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, C)
        g_gt_flat = g_gt_all.permute(0, 2, 3, 1).reshape(-1, C)
        h_norm_flat = h_norm_all.permute(0, 2, 3, 1).reshape(-1, 1)

        # Overlap diagnostics between Boundary and Error (only if requested)
        if return_diagnostics:
            is_boundary = (boundary_mask.sum(dim=1) > 0)          # [B, H, W]
            is_error = (y_gt != pred_labels)                      # [B, H, W]
            is_overlap = is_boundary & is_error                   # [B, H, W]

            n_boundary_pix = int(is_boundary.sum().item())
            n_error_pix = int(is_error.sum().item())
            n_overlap_pix = int(is_overlap.sum().item())

            overlap_ratio_b = float(n_overlap_pix / (n_boundary_pix + self.eps))
            overlap_ratio_e = float(n_overlap_pix / (n_error_pix + self.eps))

            # Error breakdown: Background vs Foreground
            bg_error_mask = (y_gt == 0) & (pred_labels != 0)
            fg_error_mask = (y_gt > 0) & (pred_labels != y_gt)
            fg_missed_as_bg_mask = (y_gt > 0) & (pred_labels == 0)
            fg_confused_organ_mask = (y_gt > 0) & (pred_labels > 0) & (pred_labels != y_gt)

            bg_error_count = int(bg_error_mask.sum().item())
            fg_error_count = int(fg_error_mask.sum().item())
            fg_missed_as_bg_count = int(fg_missed_as_bg_mask.sum().item())
            fg_confused_with_organ_count = int(fg_confused_organ_mask.sum().item())

        # =========================================================================
        # 5. STREAM 1: BOUNDARY STREAM (L_B)
        # =========================================================================
        pair_loss_gated_sum = torch.tensor(0.0, device=device, dtype=calc_dtype)
        pair_loss_ungated_sum = torch.tensor(0.0, device=device, dtype=calc_dtype)
        pixel_loss_gated_sum = torch.tensor(0.0, device=device, dtype=calc_dtype)
        pixel_loss_ungated_sum = torch.tensor(0.0, device=device, dtype=calc_dtype)
        boundary_pair_count = 0
        total_boundary_count = 0
        skipped_boundary_pairs = 0

        # Boundary diagnostics per organ (classes 1..C-1)
        neg_interf_per_class = {c: 0.0 for c in range(1, C)}
        pos_interf_per_class = {c: 0.0 for c in range(1, C)}
        conflict_count_per_class = {c: 0 for c in range(1, C)}
        total_count_per_class = {c: 0 for c in range(1, C)}
        cosine_sum_per_class = {c: 0.0 for c in range(1, C)}
        active_count_per_class = {c: 0 for c in range(1, C)}
        sum_grad_norm_sq_b = torch.tensor(0.0, device=device, dtype=calc_dtype)
        gate_sum_b = 0.0
        gate_count_b = 0

        if self.enable_boundary:
            # Detect active boundary classes in batch
            boundary_present = (boundary_mask.sum(dim=(0, 2, 3)) > 0).tolist()
            classes_with_boundary = [c for c in range(C) if boundary_present[c]]

            for c in classes_with_boundary:
                b_mask_c = boundary_mask[:, c:c+1, :, :]
                overlap = (b_mask_c > 0) & (dilated_mask > 0)
                adj_counts = overlap.sum(dim=(0, 2, 3)).tolist()

                for d in range(C):
                    if c == d or adj_counts[d] == 0:
                        continue

                    # If an interface exists but either class lacks a prototype, record skipped
                    if not (boundary_valid[c].item() and boundary_valid[d].item()):
                        skipped_boundary_pairs += 1
                        continue

                    flat_valid = torch.where(overlap[:, d].reshape(-1))[0]
                    n_valid = len(flat_valid)
                    if n_valid == 0:
                        continue

                    z_v = z_flat[flat_valid]           # [N, C]
                    g_gt_v = g_gt_flat[flat_valid]     # [N, C]
                    h_n_v = h_norm_flat[flat_valid]    # [N, 1]

                    mu_c = boundary_protos[c]
                    mu_d = boundary_protos[d]
                    delta_mu = (mu_d - mu_c).unsqueeze(0)  # [1, C]
                    sim_c = (z_v * mu_c.unsqueeze(0)).sum(dim=1, keepdim=True)
                    sim_d = (z_v * mu_d.unsqueeze(0)).sum(dim=1, keepdim=True)

                    u = (self.m + sim_d - sim_c) / self.tau
                    ell_cba = F.softplus(u)
                    sigma = torch.sigmoid(u)

                    # Closed-form CBA gradient
                    proj = (z_v * delta_mu).sum(dim=1, keepdim=True)
                    tangent = delta_mu - z_v * proj
                    g_a_v = self._project_grad((sigma / (self.tau * h_n_v)) * tangent)

                    # Directional compatibility
                    dot = (g_gt_v * g_a_v).sum(dim=1, keepdim=True)
                    norm_gt = torch.norm(g_gt_v, p=2, dim=1, keepdim=True) + self.eps
                    norm_ga = torch.norm(g_a_v, p=2, dim=1, keepdim=True) + self.eps
                    cos_sim = dot / (norm_gt * norm_ga)

                    # Directional Gate
                    a_gate = self.compute_gate(cos_sim, self.gating_type)

                    # Loss accumulation
                    pair_gated_loss = (a_gate * ell_cba).mean()
                    pair_ungated_loss = ell_cba.mean()
                    pair_loss_gated_sum = pair_loss_gated_sum + pair_gated_loss
                    pair_loss_ungated_sum = pair_loss_ungated_sum + pair_ungated_loss
                    pixel_loss_gated_sum = pixel_loss_gated_sum + (a_gate * ell_cba).sum()
                    pixel_loss_ungated_sum = pixel_loss_ungated_sum + ell_cba.sum()
                    boundary_pair_count += 1
                    total_boundary_count += n_valid

                    # Analytical gradient norm tracking
                    if return_diagnostics:
                        g_eff_b = a_gate * g_a_v
                        sum_grad_norm_sq_b = sum_grad_norm_sq_b + (g_eff_b ** 2).sum()
                        gate_sum_b += float(a_gate.sum().item())
                        gate_count_b += n_valid

                        # Per-organ diagnostics (for foreground organs 1..C-1)
                        if c > 0:
                            neg_dot = torch.clamp(-dot, min=0.0)
                            pos_dot = torch.clamp(dot, min=0.0)
                            neg_interf_per_class[c] += float(neg_dot.sum().item())
                            pos_interf_per_class[c] += float(pos_dot.sum().item())
                            conflict_count_per_class[c] += int((cos_sim < 0).sum().item())
                            active_count_per_class[c] += int((a_gate > 0).sum().item())
                            cosine_sum_per_class[c] += float(cos_sim.sum().item())
                            total_count_per_class[c] += n_valid

            if boundary_pair_count > 0:
                if self.boundary_reduction == "pair_mean":
                    loss_b = pair_loss_gated_sum / float(boundary_pair_count)
                    ungated_loss_b = pair_loss_ungated_sum / float(boundary_pair_count)
                else:
                    loss_b = pixel_loss_gated_sum / (float(total_boundary_count) + self.eps)
                    ungated_loss_b = pixel_loss_ungated_sum / (float(total_boundary_count) + self.eps)
            else:
                loss_b = 0.0 * fplus_work.sum()
                ungated_loss_b = torch.tensor(0.0, device=device)
        else:
            loss_b = 0.0 * fplus_work.sum()
            ungated_loss_b = torch.tensor(0.0, device=device)

        # =========================================================================
        # 6. STREAM 2: ERROR-GUIDED STREAM (L_E)
        # =========================================================================
        active_error_class_losses: List[torch.Tensor] = []
        active_error_classes: List[int] = []
        total_pixel_e_loss = torch.tensor(0.0, device=device, dtype=calc_dtype)
        total_sampled_error = 0
        skipped_error_pixels = 0
        sampled_in_boundary_total = 0
        sampled_bg_error_count = 0
        sampled_fg_missed_as_bg_count = 0
        sampled_fg_confused_with_organ_count = 0

        # Solution 1 & Solution 2 specific tracking
        loss_e_fn = 0.0 * fplus_work.sum()
        loss_e_fp = 0.0 * fplus_work.sum()
        loss_e_ce = 0.0 * fplus_work.sum()
        loss_e_empty = 0.0 * fplus_work.sum()
        active_fn_classes: List[int] = []
        active_fp_pred_classes: List[int] = []
        active_empty_classes: List[int] = []
        fp_pixels_per_pred_class: Dict[int, int] = {k: 0 for k in range(1, C)}
        sampled_fp_per_pred_class: Dict[int, int] = {k: 0 for k in range(1, C)}
        bg_fp_excluded_boundary_pixels = 0
        empty_slice_active_pairs = 0
        empty_slice_total_absent_pairs = 0
        empty_slice_active_pixels = 0
        sum_grad_norm_sq_empty = torch.tensor(0.0, device=device, dtype=calc_dtype)

        # Diagnostics per error class
        error_count_per_class: Dict[int, int] = {c: 0 for c in range(C)}
        sampled_count_per_class: Dict[int, int] = {c: 0 for c in range(C)}
        error_conflict_count_per_class: Dict[int, int] = {c: 0 for c in range(C)}
        error_active_count_per_class: Dict[int, int] = {c: 0 for c in range(C)}
        error_cosine_sum_per_class: Dict[int, float] = {c: 0.0 for c in range(C)}
        error_total_per_class: Dict[int, int] = {c: 0 for c in range(C)}
        sum_grad_norm_sq_e = torch.tensor(0.0, device=device, dtype=calc_dtype)
        gate_sum_e = 0.0
        gate_count_e = 0

        # Fast error pixel detection across the whole batch
        err_bool = (y_gt != pred_labels)
        err_flat_indices = torch.where(err_bool.view(-1))[0]
        n_total_errors = len(err_flat_indices)

        if self.enable_error and n_total_errors > 0:
            if self.error_mode == "ce_hard" and self.fp_by_pred_class:
                # -----------------------------------------------------------------
                # Solution 1: Balanced FN (grouped by GT organ y > 0) +
                # Background FP (filtered outside boundary band r > 2, grouped by
                # predicted organ class hat{y} = k in {1..C-1})
                # -----------------------------------------------------------------
                fplus_flat = fplus_work.permute(0, 2, 3, 1).reshape(-1, C)
                y_gt_flat = y_gt.reshape(-1)
                preds_flat = pred_labels.reshape(-1)

                fn_class_losses: List[torch.Tensor] = []
                for c in range(1, C):
                    c_mask_1d = (y_gt_flat == c) & (preds_flat != c)
                    c_err_total = int(c_mask_1d.sum().item())
                    error_count_per_class[c] = c_err_total

                    if c_err_total == 0:
                        continue

                    if self.align_ce_hard_samples and not error_valid[c]:
                        skipped_error_pixels += c_err_total
                        continue

                    flat_idx_c = torch.where(c_mask_1d)[0]
                    preds_c = preds_flat[flat_idx_c]

                    if self.align_ce_hard_samples:
                        valid_opp = error_valid[preds_c]
                        n_unavail = int((~valid_opp).sum().item())
                        skipped_error_pixels += n_unavail
                        if not valid_opp.any():
                            continue
                        flat_val = flat_idx_c[valid_opp]
                        preds_val = preds_c[valid_opp]
                    else:
                        flat_val = flat_idx_c
                        preds_val = preds_c

                    n_val = len(preds_val)
                    budget = self.k_fg
                    if n_val > budget:
                        if sample_seed is not None:
                            g = torch.Generator(device=device)
                            g.manual_seed(int(sample_seed) + c)
                            perm = torch.randperm(n_val, generator=g, device=device)[:budget]
                        else:
                            perm = torch.randperm(n_val, device=device)[:budget]
                        flat_sel = flat_val[perm]
                        preds_sel = preds_val[perm]
                    else:
                        flat_sel = flat_val
                        preds_sel = preds_val

                    n_sel = len(preds_sel)
                    sampled_count_per_class[c] = n_sel
                    total_sampled_error += n_sel
                    sampled_fg_missed_as_bg_count += int((preds_sel == 0).sum().item())
                    sampled_fg_confused_with_organ_count += int((preds_sel > 0).sum().item())

                    if return_diagnostics:
                        sampled_in_boundary_total += int(is_boundary.reshape(-1)[flat_sel].sum().item())

                    logits_sel = fplus_flat[flat_sel]
                    target_sel = torch.full((n_sel,), c, dtype=torch.long, device=device)
                    ell_fn = F.cross_entropy(logits_sel, target_sel, reduction="none")
                    loss_c_fn = ell_fn.mean()

                    fn_class_losses.append(loss_c_fn)
                    active_fn_classes.append(c)
                    active_error_class_losses.append(loss_c_fn)
                    active_error_classes.append(c)
                    total_pixel_e_loss = total_pixel_e_loss + ell_fn.sum()

                    if return_diagnostics:
                        p_sel = F.softmax(logits_sel, dim=1)
                        g_e_v = p_sel - F.one_hot(target_sel, num_classes=C).to(dtype=p_sel.dtype)
                        sum_grad_norm_sq_e = sum_grad_norm_sq_e + (g_e_v ** 2).sum()
                        gate_sum_e += float(n_sel)
                        gate_count_e += n_sel
                        error_conflict_count_per_class[c] = 0
                        error_active_count_per_class[c] = n_sel
                        error_cosine_sum_per_class[c] = float(n_sel)
                        error_total_per_class[c] = n_sel

                # Background FP group (y = 0, hat{y} = k in {1..C-1})
                fp_class_losses: List[torch.Tensor] = []
                bg_raw_mask = (y_gt_flat == 0) & (preds_flat > 0)
                error_count_per_class[0] = int(bg_raw_mask.sum().item())

                for k in range(1, C):
                    fp_raw_k = (y_gt_flat == 0) & (preds_flat == k)
                    if not fp_raw_k.any():
                        continue

                    if self.fp_exclude_boundary:
                        outside_r_k = (dilated_mask[:, k, :, :].reshape(-1) <= 0.0)
                        fp_mask_k = fp_raw_k & outside_r_k
                        bg_fp_excluded_boundary_pixels += int((fp_raw_k & ~outside_r_k).sum().item())
                    else:
                        fp_mask_k = fp_raw_k

                    k_fp_total = int(fp_mask_k.sum().item())
                    fp_pixels_per_pred_class[k] = k_fp_total
                    if k_fp_total == 0:
                        continue

                    if self.align_ce_hard_samples and not (error_valid[0] and error_valid[k]):
                        skipped_error_pixels += k_fp_total
                        continue

                    flat_idx_k = torch.where(fp_mask_k)[0]
                    n_val_k = len(flat_idx_k)
                    budget_fp = self.k_bg
                    if n_val_k > budget_fp:
                        if sample_seed is not None:
                            g = torch.Generator(device=device)
                            g.manual_seed(int(sample_seed) + C + k)
                            perm = torch.randperm(n_val_k, generator=g, device=device)[:budget_fp]
                        else:
                            perm = torch.randperm(n_val_k, device=device)[:budget_fp]
                        flat_sel_k = flat_idx_k[perm]
                    else:
                        flat_sel_k = flat_idx_k

                    n_sel_k = len(flat_sel_k)
                    sampled_fp_per_pred_class[k] = n_sel_k
                    sampled_count_per_class[0] += n_sel_k
                    sampled_bg_error_count += n_sel_k
                    total_sampled_error += n_sel_k

                    if return_diagnostics:
                        sampled_in_boundary_total += int(is_boundary.reshape(-1)[flat_sel_k].sum().item())

                    logits_sel_k = fplus_flat[flat_sel_k]
                    target_sel_k = torch.zeros((n_sel_k,), dtype=torch.long, device=device)
                    ell_fp_k = F.cross_entropy(logits_sel_k, target_sel_k, reduction="none")
                    loss_k_fp = ell_fp_k.mean()

                    fp_class_losses.append(loss_k_fp)
                    active_fp_pred_classes.append(k)
                    total_pixel_e_loss = total_pixel_e_loss + ell_fp_k.sum()

                    if return_diagnostics:
                        p_sel_k = F.softmax(logits_sel_k, dim=1)
                        g_e_v_k = p_sel_k - F.one_hot(target_sel_k, num_classes=C).to(dtype=p_sel_k.dtype)
                        sum_grad_norm_sq_e = sum_grad_norm_sq_e + (g_e_v_k ** 2).sum()
                        gate_sum_e += float(n_sel_k)
                        gate_count_e += n_sel_k
                        error_active_count_per_class[0] += n_sel_k
                        error_cosine_sum_per_class[0] += float(n_sel_k)
                        error_total_per_class[0] += n_sel_k

                if len(fp_class_losses) > 0:
                    loss_e_fp = torch.stack(fp_class_losses).mean()
                    active_error_classes = [0] + active_fn_classes
                    active_error_class_losses = [loss_e_fp] + fn_class_losses
                if len(fn_class_losses) > 0:
                    loss_e_fn = torch.stack(fn_class_losses).mean()

                if self.error_reduction == "class_balanced":
                    if len(fn_class_losses) > 0 and len(fp_class_losses) > 0:
                        loss_e_ce = 0.5 * loss_e_fn + 0.5 * loss_e_fp
                    elif len(fn_class_losses) > 0:
                        loss_e_ce = loss_e_fn
                    elif len(fp_class_losses) > 0:
                        loss_e_ce = loss_e_fp
                    else:
                        loss_e_ce = 0.0 * fplus_work.sum()
                else:
                    if total_sampled_error > 0:
                        loss_e_ce = total_pixel_e_loss / (float(total_sampled_error) + self.eps)
                    else:
                        loss_e_ce = 0.0 * fplus_work.sum()
            else:
                # -----------------------------------------------------------------
                # Contrastive error stream (or legacy grouped-by-GT error stream)
                # -----------------------------------------------------------------
                y_err_all = y_gt.view(-1)[err_flat_indices]
                preds_err_all = pred_labels.view(-1)[err_flat_indices]
                fn_class_losses_legacy: List[torch.Tensor] = []

                for c in range(C):
                    c_mask_1d = (y_err_all == c)
                    c_err_total = int(c_mask_1d.sum().item())
                    error_count_per_class[c] = c_err_total

                    if c_err_total == 0:
                        continue

                    if c == 0 and self.fp_exclude_boundary:
                        idx_raw_0 = torch.where(c_mask_1d)[0]
                        flat_raw_0 = err_flat_indices[idx_raw_0]
                        preds_raw_0 = preds_err_all[idx_raw_0]
                        dil_flat = dilated_mask.permute(0, 2, 3, 1).reshape(-1, C)
                        in_b_0 = dil_flat[flat_raw_0, preds_raw_0] > 0.0
                        bg_fp_excluded_boundary_pixels += int(in_b_0.sum().item())
                        c_mask_1d = c_mask_1d.clone()
                        c_mask_1d[idx_raw_0[in_b_0]] = False
                        if not c_mask_1d.any():
                            continue

                    # Check if positive class c has a valid prototype
                    if (self.error_mode == "contrastive" or self.align_ce_hard_samples) and not error_valid[c]:
                        skipped_error_pixels += int(c_mask_1d.sum().item())
                        continue

                    idx_c = torch.where(c_mask_1d)[0]
                    preds_c = preds_err_all[idx_c]

                    # Opponent class d_i = pred_labels. Must have valid prototype in contrastive mode
                    # (and in aligned ce_hard mode to ensure the exact same sampled pixel set S_c)
                    if self.error_mode == "contrastive" or self.align_ce_hard_samples:
                        valid_opp = error_valid[preds_c]
                        n_unavail = int((~valid_opp).sum().item())
                        skipped_error_pixels += n_unavail

                        if not valid_opp.any():
                            continue

                        idx_val = idx_c[valid_opp]
                        preds_val = preds_c[valid_opp]
                    else:
                        idx_val = idx_c
                        preds_val = preds_c

                    n_val = len(preds_val)

                    # Dedicated sampling budget: K_bg for background, K_fg for organs
                    budget = self.k_bg if c == 0 else self.k_fg
                    if n_val > budget:
                        if sample_seed is not None:
                            g = torch.Generator(device=device)
                            g.manual_seed(int(sample_seed) + c)
                            perm = torch.randperm(n_val, generator=g, device=device)[:budget]
                        else:
                            perm = torch.randperm(n_val, device=device)[:budget]
                        idx_sel = idx_val[perm]
                        preds_sel = preds_val[perm]
                    else:
                        idx_sel = idx_val
                        preds_sel = preds_val

                    n_sel = len(preds_sel)
                    sampled_count_per_class[c] = n_sel
                    total_sampled_error += n_sel

                    # Track sampled category counts
                    if c == 0:
                        sampled_bg_error_count += n_sel
                    else:
                        sampled_fg_missed_as_bg_count += int((preds_sel == 0).sum().item())
                        sampled_fg_confused_with_organ_count += int((preds_sel > 0).sum().item())

                    # Extract features for sampled error pixels via 1D gather
                    flat_sel = err_flat_indices[idx_sel]
                    z_v_e = z_flat[flat_sel]
                    g_gt_v_e = g_gt_flat[flat_sel]
                    h_n_v_e = h_norm_flat[flat_sel]

                    if return_diagnostics:
                        # Track how many sampled error pixels lie on boundary
                        sampled_in_b = int(is_boundary.view(-1)[flat_sel].sum().item())
                        sampled_in_boundary_total += sampled_in_b

                    if self.error_mode == "ce_hard":
                        # Ablation Arm 5: Supervised Cross-Entropy on sampled hard pixels
                        fplus_flat = fplus_work.permute(0, 2, 3, 1).reshape(-1, C)
                        logits_sel = fplus_flat[flat_sel]
                        target_sel = torch.full((n_sel,), c, dtype=torch.long, device=device)
                        ell_e = F.cross_entropy(logits_sel, target_sel, reduction="none").unsqueeze(1)
                        a_gate_e = torch.ones_like(ell_e)
                        cos_sim_e = torch.ones_like(ell_e)
                        # Analytical gradient of CE: p_i - one_hot(c)
                        p_sel = F.softmax(logits_sel, dim=1)
                        g_e_v = p_sel - F.one_hot(target_sel, num_classes=C).to(dtype=p_sel.dtype)
                    else:
                        mu_c = error_protos[c].unsqueeze(0)   # [1, C]
                        mu_d = error_protos[preds_sel]        # [n_sel, C]
                        delta_mu_e = mu_d - mu_c                  # [n_sel, C]

                        sim_c = (z_v_e * mu_c).sum(dim=1, keepdim=True)
                        sim_d = (z_v_e * mu_d).sum(dim=1, keepdim=True)

                        u_e = (self.m + sim_d - sim_c) / self.tau
                        ell_e = F.softplus(u_e)
                        sigma_e = torch.sigmoid(u_e)

                        # Closed-form Error contrastive gradient
                        proj_e = (z_v_e * delta_mu_e).sum(dim=1, keepdim=True)
                        tangent_e = delta_mu_e - z_v_e * proj_e
                        g_e_v = self._project_grad((sigma_e / (self.tau * h_n_v_e)) * tangent_e)

                        # Directional compatibility with CE gradient
                        dot_e = (g_gt_v_e * g_e_v).sum(dim=1, keepdim=True)
                        norm_gt_e = torch.norm(g_gt_v_e, p=2, dim=1, keepdim=True) + self.eps
                        norm_ge = torch.norm(g_e_v, p=2, dim=1, keepdim=True) + self.eps
                        cos_sim_e = dot_e / (norm_gt_e * norm_ge)

                        # Directional Gate for Error stream
                        a_gate_e = self.compute_gate(cos_sim_e, self.error_gating_type)

                    # Class-level loss
                    loss_c_e = (a_gate_e * ell_e).mean()
                    active_error_class_losses.append(loss_c_e)
                    active_error_classes.append(c)
                    total_pixel_e_loss = total_pixel_e_loss + (a_gate_e * ell_e).sum()
                    if c == 0:
                        loss_e_fp = loss_c_e
                    else:
                        fn_class_losses_legacy.append(loss_c_e)
                        active_fn_classes.append(c)

                    # Diagnostics tracking
                    if return_diagnostics:
                        g_eff_e = a_gate_e * g_e_v
                        sum_grad_norm_sq_e = sum_grad_norm_sq_e + (g_eff_e ** 2).sum()
                        gate_sum_e += float(a_gate_e.sum().item())
                        gate_count_e += n_sel
                        error_conflict_count_per_class[c] = int((cos_sim_e < 0).sum().item())
                        error_active_count_per_class[c] = int((a_gate_e > 0).sum().item())
                        error_cosine_sum_per_class[c] = float(cos_sim_e.sum().item())
                        error_total_per_class[c] = n_sel

                if len(fn_class_losses_legacy) > 0:
                    loss_e_fn = torch.stack(fn_class_losses_legacy).mean()

                # Error reduction: Class-Balanced reduction across active classes C_E
                if len(active_error_class_losses) > 0:
                    if self.error_reduction == "class_balanced":
                        loss_e_ce = torch.stack(active_error_class_losses).mean()
                    else:
                        loss_e_ce = total_pixel_e_loss / (float(total_sampled_error) + self.eps)
                else:
                    loss_e_ce = 0.0 * fplus_work.sum()

        # ---------------------------------------------------------------------
        # Solution 2: Empty-Slice Top-K Suppression with focal/threshold guard
        # ---------------------------------------------------------------------
        if self.enable_error and self.empty_slice_suppression and self.empty_slice_weight > 0.0 and C > 1:
            slice_gt_sums = y_gt_one_hot[:, 1:, :, :].sum(dim=(2, 3))  # [B, C-1]
            absent_mask = (slice_gt_sums == 0.0)                       # [B, C-1]
            empty_slice_total_absent_pairs = int(absent_mask.sum().item())

            if empty_slice_total_absent_pairs > 0:
                p_fg = p_prob[:, 1:, :, :].reshape(B, C - 1, H * W)    # [B, C-1, H*W]
                k_eff = min(self.empty_slice_topk, H * W)
                top_p, top_idx = torch.topk(p_fg, k=k_eff, dim=2)      # [B, C-1, k_eff]

                # Threshold guard: already-clean slices with top_p <= tau_p are skipped
                active_pix_mask = (top_p > self.empty_slice_prob_thresh).to(dtype=calc_dtype)
                active_pix_counts = active_pix_mask.sum(dim=2)         # [B, C-1]
                dirty_absent_mask = absent_mask & (active_pix_counts > 0.0)
                empty_slice_active_pairs = int(dirty_absent_mask.sum().item())

                if empty_slice_active_pairs > 0:
                    # Gather all C class logits at top_idx [B, C-1, k_eff] and evaluate focal BCE in
                    # exact log-odds space u = z_c - logsumexp_{m != c}(z_m), preventing float32
                    # (1 - p) underflow, clamp gradient collapse, and Jacobian norm explosion.
                    fplus_bhw = fplus_work.reshape(B, C, H * W)
                    idx_flat = top_idx.reshape(B, 1, (C - 1) * k_eff).expand(B, C, (C - 1) * k_eff)
                    logits_top = (
                        torch.gather(fplus_bhw, dim=2, index=idx_flat)
                        .reshape(B, C, C - 1, k_eff)
                        .permute(0, 2, 3, 1)
                    )  # [B, C-1, k_eff, C]
                    class_ids = torch.arange(1, C, device=device).view(1, C - 1, 1, 1).expand(B, C - 1, k_eff, 1)
                    z_c = torch.gather(logits_top, dim=3, index=class_ids).squeeze(3)
                    logits_not_c = logits_top.scatter(3, class_ids, float("-inf"))
                    s_not_c = torch.logsumexp(logits_not_c, dim=3)     # [B, C-1, k_eff]
                    u_top = z_c - s_not_c                              # [B, C-1, k_eff]
                    p_top = torch.sigmoid(u_top)                       # [B, C-1, k_eff]
                    focal_factor = p_top.pow(self.empty_slice_focal_gamma)
                    bce_pen = F.softplus(u_top)                        # exact -log(1 - p_{b,c})
                    pixel_pen = active_pix_mask * focal_factor * bce_pen
                    slice_topk_loss = pixel_pen.sum(dim=2) / torch.clamp(active_pix_counts, min=1.0)  # [B, C-1]

                    empty_class_losses: List[torch.Tensor] = []
                    for j in range(C - 1):
                        col_mask = dirty_absent_mask[:, j]
                        if col_mask.any():
                            empty_class_losses.append(slice_topk_loss[col_mask, j].mean())
                            active_empty_classes.append(j + 1)

                    loss_e_empty = torch.stack(empty_class_losses).mean()

                    if return_diagnostics:
                        active_3d = dirty_absent_mask.unsqueeze(2) & (top_p > self.empty_slice_prob_thresh)
                        empty_slice_active_pixels = int(active_3d.sum().item())
                        if empty_slice_active_pixels > 0:
                            u_act = u_top[active_3d]
                            p_act = p_top[active_3d]
                            one_minus_p_act = torch.sigmoid(-u_act)
                            bce_act = bce_pen[active_3d]
                            focal_act = focal_factor[active_3d]
                            d_ell_du = focal_act * (
                                p_act + self.empty_slice_focal_gamma * one_minus_p_act * bce_act
                            )
                            q_not_c = F.softmax(logits_not_c[active_3d], dim=-1)
                            jac_u_norm_sq = 1.0 + (q_not_c ** 2).sum(dim=-1)
                            g_empty_sq = (d_ell_du ** 2) * jac_u_norm_sq
                            sum_grad_norm_sq_empty = g_empty_sq.sum()

        if self.enable_error:
            loss_e = loss_e_ce + self.empty_slice_weight * loss_e_empty
        else:
            loss_e = 0.0 * fplus_work.sum()

        # =========================================================================
        # 7. TOTAL CBA v2 LOSS
        # =========================================================================
        # Runner manages curriculum multiplier lambda_A(e):
        # L = 0.8 * L_SGS + 0.2 * L_KIS + lambda_A(e) * (L_B + beta * L_E)
        # Raw loss is returned without internal lambda_A multiplication by default
        raw_cba_loss = (loss_b if self.enable_boundary else 0.0 * fplus_work.sum()) + self.beta * (loss_e if self.enable_error else 0.0 * fplus_work.sum())
        total_cba_loss = self.lambda_A * raw_cba_loss if apply_lambda_A else raw_cba_loss

        if not return_diagnostics:
            if return_stream_losses:
                return loss_b, loss_e
            return total_cba_loss

        # =========================================================================
        # 8. DIAGNOSTICS COMPILATION
        # =========================================================================
        # Boundary diagnostics compilation
        conflict_rate_per_class: Dict[int, float] = {}
        active_frac_per_class: Dict[int, float] = {}
        mean_cosine_per_class: Dict[int, float] = {}
        r_minus_per_class: Dict[int, float] = {}

        total_neg_all = sum(neg_interf_per_class.values())
        total_pos_all = sum(pos_interf_per_class.values())
        r_minus_overall = float(total_neg_all / (total_neg_all + total_pos_all + self.eps))

        for c in range(1, C):
            tot = total_count_per_class[c]
            i_neg = neg_interf_per_class[c]
            i_pos = pos_interf_per_class[c]
            r_minus_per_class[c] = float(i_neg / (i_neg + i_pos + self.eps))
            if tot > 0:
                conflict_rate_per_class[c] = float(conflict_count_per_class[c] / tot)
                active_frac_per_class[c] = float(active_count_per_class[c] / tot)
                mean_cosine_per_class[c] = float(cosine_sum_per_class[c] / tot)
            else:
                conflict_rate_per_class[c] = 0.0
                active_frac_per_class[c] = 0.0
                mean_cosine_per_class[c] = 0.0

        # Error diagnostics compilation
        error_conflict_rate_per_class: Dict[int, float] = {}
        error_active_frac_per_class: Dict[int, float] = {}
        error_mean_cosine_per_class: Dict[int, float] = {}
        error_conflict_tot = 0
        error_sampled_tot = 0

        for c in range(C):
            tot_e = error_total_per_class[c]
            error_sampled_tot += tot_e
            error_conflict_tot += error_conflict_count_per_class[c]
            if tot_e > 0:
                error_conflict_rate_per_class[c] = float(error_conflict_count_per_class[c] / tot_e)
                error_active_frac_per_class[c] = float(error_active_count_per_class[c] / tot_e)
                error_mean_cosine_per_class[c] = float(error_cosine_sum_per_class[c] / tot_e)
            else:
                error_conflict_rate_per_class[c] = 0.0
                error_active_frac_per_class[c] = 0.0
                error_mean_cosine_per_class[c] = 0.0

        error_macro_conflict_rate = (
            float(sum(error_conflict_rate_per_class[c] for c in active_error_classes) / len(active_error_classes))
            if active_error_classes else 0.0
        )
        error_micro_conflict_rate = float(error_conflict_tot / (error_sampled_tot + self.eps))

        # Analytical gradient norms
        grad_norm_b_analytical = math.sqrt(float(sum_grad_norm_sq_b.item()) / (total_boundary_count + self.eps))
        grad_norm_empty_analytical = (
            math.sqrt(float(sum_grad_norm_sq_empty.item()) / (empty_slice_active_pixels + self.eps))
            if empty_slice_active_pixels > 0 else 0.0
        )
        if total_sampled_error > 0:
            grad_norm_e_analytical = math.sqrt(
                float((sum_grad_norm_sq_e + (self.empty_slice_weight ** 2) * sum_grad_norm_sq_empty).item())
                / (total_sampled_error + empty_slice_active_pixels + self.eps)
            )
        elif empty_slice_active_pixels > 0:
            grad_norm_e_analytical = float(self.empty_slice_weight * grad_norm_empty_analytical)
        else:
            grad_norm_e_analytical = 0.0
        grad_norm_ratio_analytical = float(grad_norm_e_analytical / (grad_norm_b_analytical + self.eps))
        effective_grad_norm_e_analytical = float(self.beta * grad_norm_e_analytical)
        effective_grad_ratio_analytical = float(self.beta * grad_norm_ratio_analytical)

        # Cache stats
        cache_denom = cache_hits + cache_misses
        cache_hit_rate = float(cache_hits / (cache_denom + self.eps)) if cache_denom > 0 else 0.0
        classes_in_cache = [c for c in range(C) if self.valid_mask[c].item()]
        classes_in_batch = [c for c in range(C) if batch_valid[c].item()]

        # Skip rates
        skipped_error_rate = float(skipped_error_pixels / (n_total_errors + self.eps)) if n_total_errors > 0 else 0.0
        total_boundary_attempted = boundary_pair_count + skipped_boundary_pairs
        skipped_boundary_rate = float(skipped_boundary_pairs / (total_boundary_attempted + self.eps)) if total_boundary_attempted > 0 else 0.0

        # Prototype coherence / quality monitoring
        # 1. Within-batch spatial coherence: R_c^batch = ||mu_c^raw||_2
        bg_proto_coherence = proto_coherence.get(0, 0.0)
        fg_coherences = [v for k, v in proto_coherence.items() if k > 0]
        fg_proto_coherence_mean = float(sum(fg_coherences) / len(fg_coherences)) if fg_coherences else 0.0

        # 2. Cross-batch temporal directional stability: R_c^temporal = ||v_c||_2 (measured on v_c before normalization)
        bg_temporal_coherence = temporal_coherence.get(0, 0.0)
        fg_temporal_coherences = [v for k, v in temporal_coherence.items() if k > 0]
        fg_temporal_coherence_mean = float(sum(fg_temporal_coherences) / len(fg_temporal_coherences)) if fg_temporal_coherences else 0.0

        diagnostics: Dict[str, Union[dict, list, float, int]] = {
            # Losses
            "loss_b": float(loss_b.item()),
            "loss_e": float(loss_e.item()),
            "loss_e_ce": float(loss_e_ce.item()),
            "loss_e_fn": float(loss_e_fn.item()),
            "loss_e_fp": float(loss_e_fp.item()),
            "loss_e_empty": float(loss_e_empty.item()),
            "loss_cba": float(total_cba_loss.item()),
            "loss_cba_raw": float(raw_cba_loss.item()),
            "ungated_loss_b": float(ungated_loss_b.item()),
            "beta": self.beta,
            "lambda_A": self.lambda_A,
            "apply_lambda_A": apply_lambda_A,
            "gating_type": self.gating_type,
            "error_gating_type": self.error_gating_type,
            "error_mode": self.error_mode,
            "enable_boundary": self.enable_boundary,
            "enable_error": self.enable_error,
            "boundary_use_cache": self.boundary_use_cache,
            "boundary_fallback_cache": self.boundary_fallback_cache,
            "boundary_proto_source": self.boundary_proto_source,
            "use_cache_for_all": self.use_cache_for_all,
            "error_proto_source": self.error_proto_source,
            "align_ce_hard_samples": self.align_ce_hard_samples,
            "feature_space": self.feature_space,
            "fp_by_pred_class": self.fp_by_pred_class,
            "fp_exclude_boundary": self.fp_exclude_boundary,
            "empty_slice_suppression": self.empty_slice_suppression,
            "empty_slice_topk": self.empty_slice_topk,
            "empty_slice_prob_thresh": self.empty_slice_prob_thresh,
            "empty_slice_focal_gamma": self.empty_slice_focal_gamma,
            "empty_slice_weight": self.empty_slice_weight,

            # Gate values and feature scale (mean gate value, not only the fraction of open gates)
            "boundary_gate_mean": float(gate_sum_b / gate_count_b) if gate_count_b > 0 else 0.0,
            "error_gate_mean": (
                float(gate_sum_e / gate_count_e)
                if gate_count_e > 0
                else (1.0 if empty_slice_active_pixels > 0 else 0.0)
            ),
            "h_norm_mean": float(h_norm_all.mean().item()),

            # Overlap metrics
            "total_boundary_pixels": total_boundary_count,
            "unique_boundary_pixels": n_boundary_pix,
            "total_error_pixels": n_error_pix,
            "sampled_error_pixels": total_sampled_error,
            "overlap_boundary_error_pixels": n_overlap_pix,
            "overlap_ratio_boundary": overlap_ratio_b,
            "overlap_ratio_error": overlap_ratio_e,
            "sampled_error_in_boundary": sampled_in_boundary_total,

            # Error category breakdown
            "bg_error_count": bg_error_count,
            "fg_error_count": fg_error_count,
            "fg_missed_as_bg_count": fg_missed_as_bg_count,
            "fg_confused_with_organ_count": fg_confused_with_organ_count,
            "sampled_bg_error_count": sampled_bg_error_count,
            "sampled_fg_error_count": sampled_fg_missed_as_bg_count + sampled_fg_confused_with_organ_count,
            "sampled_fg_missed_as_bg_count": sampled_fg_missed_as_bg_count,
            "sampled_fg_confused_with_organ_count": sampled_fg_confused_with_organ_count,
            "error_pixels_per_class": error_count_per_class,
            "sampled_pixels_per_class": sampled_count_per_class,
            "active_error_classes": active_error_classes,
            "active_fn_classes": active_fn_classes,
            "active_fp_pred_classes": active_fp_pred_classes,
            "fp_pixels_per_pred_class": fp_pixels_per_pred_class,
            "sampled_fp_per_pred_class": sampled_fp_per_pred_class,
            "bg_fp_excluded_boundary_pixels": bg_fp_excluded_boundary_pixels,
            "active_empty_classes": active_empty_classes,
            "empty_slice_active_pairs": empty_slice_active_pairs,
            "empty_slice_total_absent_pairs": empty_slice_total_absent_pairs,
            "empty_slice_active_pixels": empty_slice_active_pixels,

            # Prototype Cache metrics
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "batch_hits": batch_hits,
            "cache_hit_rate": cache_hit_rate,
            "classes_in_cache": classes_in_cache,
            "classes_in_batch": classes_in_batch,
            "proto_sources": proto_sources,
            "skipped_error_pixels": skipped_error_pixels,
            "skipped_error_rate": skipped_error_rate,
            "skipped_boundary_pairs": skipped_boundary_pairs,
            "skipped_boundary_rate": skipped_boundary_rate,

            # Prototype quality / coherence monitoring
            "bg_proto_coherence": bg_proto_coherence,
            "fg_proto_coherence_mean": fg_proto_coherence_mean,
            "proto_coherence_per_class": proto_coherence,
            "batch_proto_coherence": proto_coherence,
            "temporal_coherence_per_class": temporal_coherence,
            "bg_temporal_coherence": bg_temporal_coherence,
            "fg_temporal_coherence_mean": fg_temporal_coherence_mean,

            # Boundary directional conflict metrics
            "I_plus": pos_interf_per_class,
            "I_minus": neg_interf_per_class,
            "r_minus": r_minus_per_class,
            "r_minus_overall": r_minus_overall,
            "boundary_conflict_rate": conflict_rate_per_class,
            "boundary_active_fraction": active_frac_per_class,
            "boundary_mean_cosine": mean_cosine_per_class,

            # Error directional conflict metrics
            "error_conflict_rate": error_conflict_rate_per_class,
            "error_active_fraction": error_active_frac_per_class,
            "error_mean_cosine": error_mean_cosine_per_class,
            "error_macro_conflict_rate": error_macro_conflict_rate,
            "error_micro_conflict_rate": error_micro_conflict_rate,

            # Analytical gradient norms
            "grad_norm_b_analytical": grad_norm_b_analytical,
            "grad_norm_e_analytical": grad_norm_e_analytical,
            "grad_norm_empty_analytical": grad_norm_empty_analytical,
            "grad_norm_ratio_analytical": grad_norm_ratio_analytical,
            "effective_grad_norm_e_analytical": effective_grad_norm_e_analytical,
            "effective_grad_ratio_analytical": effective_grad_ratio_analytical,
        }

        # Optional Autograd gradient norms
        if compute_autograd_norms and fplus.requires_grad:
            if loss_b.requires_grad and loss_b.item() > 0.0:
                gb = torch.autograd.grad(loss_b, fplus, retain_graph=True)[0]
                diagnostics["grad_norm_b_autograd"] = float(gb.norm().item())
            else:
                diagnostics["grad_norm_b_autograd"] = 0.0

            if loss_e.requires_grad and loss_e.item() > 0.0:
                ge = torch.autograd.grad(loss_e, fplus, retain_graph=True)[0]
                diagnostics["grad_norm_e_autograd"] = float(ge.norm().item())
            else:
                diagnostics["grad_norm_e_autograd"] = 0.0

            nb = diagnostics["grad_norm_b_autograd"]
            ne = diagnostics["grad_norm_e_autograd"]
            diagnostics["effective_grad_norm_e_autograd"] = float(self.beta * ne)
            diagnostics["effective_grad_ratio_autograd"] = float(self.beta * ne / (nb + self.eps))

        if return_stream_losses:
            if not return_diagnostics:
                return loss_b, loss_e
            return loss_b, loss_e, diagnostics

        return total_cba_loss, diagnostics


# Convenient alias
CBAGatedLossV2 = CBAv2Loss


def morphological_dilation_2d(mask: torch.Tensor, radius: int = 2) -> torch.Tensor:
    """Perform 2D morphological dilation with a square structuring element of radius r."""
    if radius <= 0:
        return mask > 0.5
    kernel_size = 2 * radius + 1
    padding = radius

    if mask.ndim == 3:
        x = mask.unsqueeze(1).float()
    elif mask.ndim == 4:
        x = mask.float()
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {mask.shape}")

    dilated = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding) > 0.5
    if mask.ndim == 3:
        dilated = dilated.squeeze(1)
    return dilated


def morphological_erosion_2d(mask: torch.Tensor, radius: int = 2) -> torch.Tensor:
    """Perform 2D morphological erosion with a square structuring element of radius r."""
    if radius <= 0:
        return mask > 0.5
    kernel_size = 2 * radius + 1
    padding = radius

    if mask.ndim == 3:
        x = mask.unsqueeze(1).float()
    elif mask.ndim == 4:
        x = mask.float()
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {mask.shape}")

    eroded = 1.0 - F.max_pool2d(1.0 - x, kernel_size=kernel_size, stride=1, padding=padding) > 0.5
    if mask.ndim == 3:
        eroded = eroded.squeeze(1)
    return eroded


def compute_boundary_bands(
    y_gt: torch.Tensor,
    num_classes: int = 9,
    radius: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, Dict[str, torch.Tensor]]]:
    """Compute inner, outer, and full morphological boundary bands per class and pooled."""
    device = y_gt.device
    b, h, w = y_gt.shape
    inner_pooled = torch.zeros((b, h, w), dtype=torch.bool, device=device)
    outer_pooled = torch.zeros((b, h, w), dtype=torch.bool, device=device)
    full_pooled = torch.zeros((b, h, w), dtype=torch.bool, device=device)

    per_class_bands: Dict[int, Dict[str, torch.Tensor]] = {}

    for k in range(1, num_classes):
        m_k = y_gt == k
        if not m_k.any():
            per_class_bands[k] = {
                "inner": torch.zeros((b, h, w), dtype=torch.bool, device=device),
                "outer": torch.zeros((b, h, w), dtype=torch.bool, device=device),
                "full": torch.zeros((b, h, w), dtype=torch.bool, device=device),
            }
            continue

        dil_k = morphological_dilation_2d(m_k, radius=radius)
        ero_k = morphological_erosion_2d(m_k, radius=radius)

        inner_k = m_k & (~ero_k)
        outer_k = dil_k & (~m_k)
        full_k = dil_k & (~ero_k)

        inner_pooled = inner_pooled | inner_k
        outer_pooled = outer_pooled | outer_k
        full_pooled = full_pooled | full_k

        per_class_bands[k] = {"inner": inner_k, "outer": outer_k, "full": full_k}

    return inner_pooled, outer_pooled, full_pooled, per_class_bands


OursLoss = CBAv2Loss

