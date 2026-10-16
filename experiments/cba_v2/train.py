#!/usr/bin/env python3
"""
run_cba_v2_pilot.py - 5-Arm Controlled Benchmark Runner for CBA v2.

The 5 Controlled Experimental Arms:
  Arm_V1_KIS:
    L = 0.8 * L_SGS + 0.2 * L_KIS (lambda_A = 0.0)
    Supervised ground truth + Teacher distillation baseline.
  Arm_V2_Boundary:
    L = 0.8 * L_SGS + 0.2 * L_KIS + lambda_A * L_B (Directionally gated boundary stream only)
  Arm_V3_Error:
    L = 0.8 * L_SGS + 0.2 * L_KIS + lambda_A * beta * L_E (Directionally gated error stream only)
  Arm_V4_Dual_Contrastive:
    L = 0.8 * L_SGS + 0.2 * L_KIS + lambda_A * (L_B + beta * L_E) (Full CBA v2 dual contrastive)
  Arm_V5_Dual_CEHard:
    L = 0.8 * L_SGS + 0.2 * L_KIS + lambda_A * (L_B + beta * L_E^{CE}) (Boundary + CE on sampled hard pixels)

Scientific Controls:
  - Supports microbatch 8 and gradient accumulation 2 (effective batch size 16).
  - Checkpoint save and resume: saves cba_criterion.state_dict() (prototype_buffer, valid_mask, update_count).
  - Validation/evaluation strictly freezes cache via cba_criterion.eval().
  - Rich diagnostics: missing prototype rate, gate open fraction, gradient norms, temporal coherence.
  - End-to-end 3D volume evaluation with MedPy HD95 (voxelspacing), Surface NSD (2 mm), and Clean Macro Dice.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Dataset

# Resolve repository root
REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.cba_v2.losses.kis_loss import KISLoss, lift_binary_logits_for_cba, structure_loss
from experiments.cba_v2.losses.cba_v2_loss import CBAv2Loss
from experiments.cba_v2.losses.kd_baselines import BaselineKDLoss, IADKasHead
from experiments.cba_v2.models import DATASET_MODEL_SPECS, normalize_dataset_name, verify_checkpoint_compatibility

try:
    from medpy import metric as medpy_metric
except ImportError:
    medpy_metric = None


CLASS_NAMES = {
    1: "spleen",
    2: "right_kidney",
    3: "left_kidney",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "aorta",
    8: "pancreas",
}
ALL_CLASS_NAMES = {0: "background", **CLASS_NAMES}


def apply_dataset_defaults(args: argparse.Namespace) -> str:
    """Resolve and validate dataset-specific defaults (image_size, num_classes, checkpoints, paths).

    Ensures Synapse (9 classes, 512x512), ACDC (4 classes, 256x256), and Polyp (1 binary channel, 256x256)
    cannot be accidentally mixed.
    """
    ds = normalize_dataset_name(getattr(args, "dataset", "synapse"))
    args.dataset = ds
    spec = DATASET_MODEL_SPECS[ds]
    if not getattr(args, "img_size", 0):
        args.img_size = int(spec["image_size"])

    if ds == "acdc":
        if args.num_classes == 9:
            args.num_classes = 4
        elif args.num_classes != 4:
            raise ValueError(f"[Dataset Guard] ACDC requires num_classes=4, got num_classes={args.num_classes}")
        if getattr(args, "teacher_ckpt", None) in (None, "teacher_weights/epoch_299.pth"):
            args.teacher_ckpt = spec["default_teacher_ckpt"]
        if getattr(args, "data_dir", None) == "data/btcv/train_npz":
            args.data_dir = "data/ACDC/train"
        if getattr(args, "list_path", None) == "data/lists_Synapse/train.txt":
            args.list_path = "data/ACDC/lists_ACDC/train.txt"
        if getattr(args, "test_dir", None) == "data/btcv/test_vol_h5":
            args.test_dir = "data/ACDC/test"
        if getattr(args, "test_list", None) == "data/lists_Synapse/test_vol.txt":
            args.test_list = "data/ACDC/lists_ACDC/test.txt"
    elif ds == "polyp":
        if args.num_classes == 9:
            args.num_classes = 1
        elif args.num_classes != 1:
            raise ValueError(f"[Dataset Guard] Polyp requires num_classes=1 (binary output), got num_classes={args.num_classes}")
        if getattr(args, "teacher_ckpt", None) in (None, "teacher_weights/epoch_299.pth"):
            args.teacher_ckpt = spec["default_teacher_ckpt"]
        if getattr(args, "data_dir", None) == "data/btcv/train_npz":
            args.data_dir = "data/dataset/TrainDataset"
        if getattr(args, "test_dir", None) == "data/btcv/test_vol_h5":
            args.test_dir = "data/dataset"
    return ds

PILOT_ARMS_V2 = [
    "Arm_V1_KIS",
    "Arm_V2_Boundary",
    "Arm_V3_Error",
    "Arm_V4_Dual_Contrastive",
    "Arm_V5_Dual_CEHard",
    "Arm_V5_FarFP",
    "Arm_V1_KIS_FarFP",
    "Arm_IAD_200_100",
    "Arm_LSKD",
    "Arm_SinKD",
    "Arm_DIST_Plus",
    "Arm_SGS_Only",
]

NON_CBA_ARMS = (
    "Arm_V1_KIS",
    "Arm_V1_KIS_FarFP",
    "Arm_IAD_200_100",
    "Arm_LSKD",
    "Arm_SinKD",
    "Arm_DIST_Plus",
    "Arm_SGS_Only",
)


def resolve_arm_name(requested_name: str) -> str:
    """Resolve arm alias to canonical name."""
    req = requested_name.strip().lower()
    mapping = {
        "v1": "Arm_V1_KIS",
        "arm_v1": "Arm_V1_KIS",
        "arm_v1_kis": "Arm_V1_KIS",
        "kis": "Arm_V1_KIS",
        "baseline": "Arm_V1_KIS",
        "v2": "Arm_V2_Boundary",
        "arm_v2": "Arm_V2_Boundary",
        "arm_v2_boundary": "Arm_V2_Boundary",
        "boundary": "Arm_V2_Boundary",
        "boundary_only": "Arm_V2_Boundary",
        "ours_v2": "Arm_V2_Boundary",
        "v3": "Arm_V3_Error",
        "arm_v3": "Arm_V3_Error",
        "arm_v3_error": "Arm_V3_Error",
        "error": "Arm_V3_Error",
        "error_only": "Arm_V3_Error",
        "ours_v3": "Arm_V3_Error",
        "v4": "Arm_V4_Dual_Contrastive",
        "arm_v4": "Arm_V4_Dual_Contrastive",
        "arm_v4_dual_contrastive": "Arm_V4_Dual_Contrastive",
        "dual": "Arm_V4_Dual_Contrastive",
        "cba_v2": "Arm_V4_Dual_Contrastive",
        "contrastive": "Arm_V4_Dual_Contrastive",
        "ours_v4": "Arm_V4_Dual_Contrastive",
        "v5": "Arm_V5_Dual_CEHard",
        "arm_v5": "Arm_V5_Dual_CEHard",
        "arm_v5_dual_cehard": "Arm_V5_Dual_CEHard",
        "ce_hard": "Arm_V5_Dual_CEHard",
        "cehard": "Arm_V5_Dual_CEHard",
        "ours": "Arm_V5_Dual_CEHard",
        "ours_v5": "Arm_V5_Dual_CEHard",
        "v5_far": "Arm_V5_FarFP",
        "arm_v5_far": "Arm_V5_FarFP",
        "arm_v5_farfp": "Arm_V5_FarFP",
        "ours_far": "Arm_V5_FarFP",
        "ours_farfp": "Arm_V5_FarFP",
        "far_fp": "Arm_V5_FarFP",
        "farfp": "Arm_V5_FarFP",
        "kis_far": "Arm_V1_KIS_FarFP",
        "kis_farfp": "Arm_V1_KIS_FarFP",
        "arm_v1_far": "Arm_V1_KIS_FarFP",
        "arm_v1_farfp": "Arm_V1_KIS_FarFP",
        "iad": "Arm_IAD_200_100",
        "iad_200_100": "Arm_IAD_200_100",
        "arm_iad": "Arm_IAD_200_100",
        "arm_iad_200_100": "Arm_IAD_200_100",
        "lskd": "Arm_LSKD",
        "arm_lskd": "Arm_LSKD",
        "sinkd": "Arm_SinKD",
        "arm_sinkd": "Arm_SinKD",
        "dist_plus": "Arm_DIST_Plus",
        "dist+": "Arm_DIST_Plus",
        "dist": "Arm_DIST_Plus",
        "arm_dist_plus": "Arm_DIST_Plus",
        "sgs_only": "Arm_SGS_Only",
        "sgs": "Arm_SGS_Only",
        "arm_sgs": "Arm_SGS_Only",
        "arm_sgs_only": "Arm_SGS_Only",
    }
    if req in mapping:
        return mapping[req]
    for full in PILOT_ARMS_V2:
        if req in full.lower():
            return full
    raise ValueError(
        f"Unknown arm name '{requested_name}'. Valid choices: {PILOT_ARMS_V2} or aliases "
        "(ours, v1..v5, iad, lskd, sinkd, dist_plus, sgs_only)"
    )



def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = int(torch.initial_seed()) % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def atomic_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def get_curriculum_lambda(
    epoch_1indexed: int,
    max_lambda: float = 0.02,
    warmup_epochs: int = 0,
    ramp_epochs: int = 10,
) -> float:
    """Curriculum schedule for lambda_A(e):
      - If warmup_epochs == 0 and ramp_epochs == 0: returns max_lambda immediately.
      - If epoch <= warmup_epochs: returns 0.0.
      - If warmup < epoch <= warmup + ramp: ramps linearly from 0.0 to max_lambda.
      - If epoch > warmup + ramp: returns max_lambda.
    """
    if warmup_epochs <= 0 and ramp_epochs <= 0:
        return float(max_lambda)
    if epoch_1indexed <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return float(max_lambda)
    ramp = (epoch_1indexed - warmup_epochs) / float(ramp_epochs)
    return float(max_lambda * min(1.0, max(0.0, ramp)))


# -----------------------------------------------------------------------------
# Dataset Loaders & Synthetic Smoke Data
# -----------------------------------------------------------------------------

class SyntheticSliceDataset(Dataset):
    """Synthetic slice dataset for local verification, smoke testing, and checkpoint validation."""

    def __init__(self, num_samples: int = 32, num_classes: int = 9, h: int = 512, w: int = 512, img_size: int = 0):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.h = int(img_size) if img_size and img_size > 0 else h
        self.w = int(img_size) if img_size and img_size > 0 else w

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        label = np.zeros((self.h, self.w), dtype=np.int64)
        n_fg = max(1, self.num_classes - 1) if self.num_classes > 1 else 1
        sy = max(1, self.h // 8)
        sx = max(1, self.w // 8)
        if self.num_classes == 1:
            label[sy * 2 : sy * 5, sx * 2 : sx * 5] = 1
        elif self.h == 512 and self.w == 512 and self.num_classes >= 5:
            # Spleen
            label[100:220, 100:220] = 1
            # Right Kidney
            label[140:260, 270:380] = 2
            # Left Kidney
            label[260:380, 120:240] = 3
            # Gallbladder
            label[300:400, 280:380] = 4
            # Add index-dependent variation
            c_alt = (idx % n_fg) + 1
            label[50:120, 320:420] = c_alt
        else:
            label[sy * 1 : sy * 3, sx * 1 : sx * 3] = 1
            if n_fg >= 2:
                label[sy * 2 : sy * 4, sx * 4 : sx * 6] = 2
            if n_fg >= 3:
                label[sy * 4 : sy * 6, sx * 2 : sx * 4] = 3
            c_alt = (idx % n_fg) + 1
            label[sy * 1 : sy * 2, sx * 5 : sx * 7] = c_alt

        # Normalized image representation (3 channels for ViT/CNN)
        image = (label > 0).astype(np.float32) * 0.5 + 0.1
        image = np.stack([image, image, image], axis=0)
        return {
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(label),
            "case_name": f"synthetic_slice_{idx:03d}",
        }


class SimpleConvStudent(nn.Module):
    """Lightweight 2D segmentation student for smoke tests and unit validation."""

    def __init__(self, in_channels: int = 3, num_classes: int = 9):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x, return_logits: bool = True):
        logits = self.backbone(x)
        return (logits,)


# -----------------------------------------------------------------------------
# CBA v2 Loss Factory
# -----------------------------------------------------------------------------

def build_cba_v2_loss(arm_name: str, args: argparse.Namespace) -> Optional[CBAv2Loss]:
    """Instantiate CBA v2 loss configured specifically for the target experimental arm."""
    if arm_name in NON_CBA_ARMS:
        return None

    b_fallback = getattr(args, "boundary_fallback_cache", False)
    b_source = getattr(args, "boundary_proto_source", None)
    # For 1-channel binary segmentation (Polyp), logits are lifted to 2-class h = [1 - s/2, 1 + s/2]
    # with feature_space="raw" so L2-normalization maps s smoothly onto S^1 and CE(h, y) == BCEWithLogits(s, y).
    cba_num_classes = 2 if args.num_classes == 1 else args.num_classes
    cba_feat_dim = 2 if args.num_classes == 1 else args.num_classes
    cba_feature_space = "raw" if args.num_classes == 1 else args.cba_feature

    if arm_name == "Arm_V2_Boundary":
        return CBAv2Loss(
            m=args.m,
            tau=args.tau,
            r=args.r,
            beta=0.0,
            k_bg=args.k_bg,
            k_fg=args.k_fg,
            num_classes=cba_num_classes,
            feat_dim=cba_feat_dim,
            cache_momentum=args.cache_momentum,
            lambda_A=args.lambda_a,
            gating_type=args.gating_type,
            boundary_reduction="pair_mean",
            enable_boundary=True,
            enable_error=False,
            boundary_use_cache=args.boundary_use_cache,
            boundary_fallback_cache=b_fallback,
            boundary_proto_source=b_source,
            use_cache_for_all=args.use_cache_for_all,
            feature_space=cba_feature_space,
        )

    if arm_name == "Arm_V3_Error":
        return CBAv2Loss(
            m=args.m,
            tau=args.tau,
            r=args.r,
            beta=args.beta,
            k_bg=args.k_bg,
            k_fg=args.k_fg,
            num_classes=cba_num_classes,
            feat_dim=cba_feat_dim,
            cache_momentum=args.cache_momentum,
            lambda_A=args.lambda_a,
            gating_type=args.gating_type,
            error_gating_type=args.error_gating_type,
            error_reduction="class_balanced",
            enable_boundary=False,
            enable_error=True,
            error_mode="contrastive",
            error_proto_source="fallback",
            use_cache_for_all=args.use_cache_for_all,
            feature_space=cba_feature_space,
        )

    if arm_name == "Arm_V4_Dual_Contrastive":
        return CBAv2Loss(
            m=args.m,
            tau=args.tau,
            r=args.r,
            beta=args.beta,
            k_bg=args.k_bg,
            k_fg=args.k_fg,
            num_classes=cba_num_classes,
            feat_dim=cba_feat_dim,
            cache_momentum=args.cache_momentum,
            lambda_A=args.lambda_a,
            gating_type=args.gating_type,
            error_gating_type=args.error_gating_type,
            boundary_reduction="pair_mean",
            error_reduction="class_balanced",
            enable_boundary=True,
            enable_error=True,
            error_mode="contrastive",
            boundary_use_cache=args.boundary_use_cache,
            boundary_fallback_cache=b_fallback,
            boundary_proto_source=b_source,
            use_cache_for_all=args.use_cache_for_all,
            feature_space=cba_feature_space,
            error_proto_source="fallback",
        )

    if arm_name in ("Arm_V5_Dual_CEHard", "Arm_V5_FarFP"):
        return CBAv2Loss(
            m=args.m,
            tau=args.tau,
            r=args.r,
            beta=args.beta,
            k_bg=args.k_bg,
            k_fg=args.k_fg,
            num_classes=cba_num_classes,
            feat_dim=cba_feat_dim,
            cache_momentum=args.cache_momentum,
            lambda_A=args.lambda_a,
            gating_type=args.gating_type,
            error_gating_type=args.error_gating_type,
            boundary_reduction="pair_mean",
            error_reduction="class_balanced",
            enable_boundary=True,
            enable_error=True,
            error_mode="ce_hard",
            boundary_use_cache=args.boundary_use_cache,
            boundary_fallback_cache=b_fallback,
            boundary_proto_source=b_source,
            use_cache_for_all=args.use_cache_for_all,
            feature_space=cba_feature_space,
            error_proto_source="fallback",
            fp_by_pred_class=getattr(args, "fp_by_pred_class", False),
            fp_exclude_boundary=getattr(args, "fp_exclude_boundary", False),
            empty_slice_suppression=getattr(args, "empty_slice_suppression", False),
            empty_slice_topk=getattr(args, "empty_slice_topk", 64),
            empty_slice_prob_thresh=getattr(args, "empty_slice_prob_thresh", 0.05),
            empty_slice_focal_gamma=getattr(args, "empty_slice_focal_gamma", 2.0),
            empty_slice_weight=getattr(args, "empty_slice_weight", 1.0),
        )

    raise ValueError(f"Unsupported arm name: {arm_name}")


# -----------------------------------------------------------------------------
# Run bookkeeping helpers: metrics, checkpoints, RNG, provenance
# -----------------------------------------------------------------------------

def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def resolve_file(path: Optional[str]) -> Optional[str]:
    """Return an existing file path for `path` (as given, or relative to the repo root)."""
    if not path or str(path).lower() in ("none", "null"):
        return None
    for cand in (path, os.path.join(REPO_ROOT, path)):
        if os.path.isfile(cand):
            return cand
    return None


def git_info() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=15
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30
            ).stdout.strip()
        )
        return {"commit": commit or None, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def atomic_torch_save(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Optional[dict]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def summarize_eval(eval_res: dict) -> dict:
    """Flatten the output of evaluate_volumes() into scalar fields for metrics.jsonl/csv.

    evaluate_volumes() returns {"overall", "per_class", "cases"}; the headline numbers live in
    eval_res["overall"]. Indexing is strict on purpose: a missing key must raise instead of
    silently logging 0.0.
    """
    ov = eval_res["overall"]
    spacings = [tuple(c.get("voxelspacing", (1.0, 1.0, 1.0))) for c in eval_res.get("cases", [])]
    unit = "mm" if any(s != (1.0, 1.0, 1.0) for s in spacings) else "vox"
    res = {
        "eval_n_units": int(ov["primary_n_units"]),
        "eval_dice_raw": float(ov["primary_dice_raw_macro"]),
        "eval_dice_lcc": float(ov["primary_dice_lcc_macro"]),
        "eval_nsd_raw": float(ov["primary_nsd_raw_macro"]),
        "eval_nsd_lcc": float(ov["primary_nsd_lcc_macro"]),
        "eval_hd95_raw": float(ov["primary_hd95_raw_valid_macro"]),
        "eval_hd95_lcc": float(ov["primary_hd95_lcc_valid_macro"]),
        "eval_recall": float(ov["primary_recall_macro"]),
        "eval_precision": float(ov["primary_precision_macro"]),
        "eval_vol_ratio": float(ov["primary_vol_ratio_macro"]),
        "distance_unit": unit,
    }
    if "primary_jc_raw_macro" in ov:
        res["eval_jc_raw"] = float(ov["primary_jc_raw_macro"])
        res["eval_jc_lcc"] = float(ov.get("primary_jc_lcc_macro", ov["primary_jc_raw_macro"]))
    if "polyp_5sets_dice_raw_macro" in ov:
        res["eval_dice_5sets_raw"] = float(ov["polyp_5sets_dice_raw_macro"])
        res["eval_dice_5sets_lcc"] = float(ov.get("polyp_5sets_dice_lcc_macro", ov["polyp_5sets_dice_raw_macro"]))
    for k, v in ov.items():
        if k.startswith(("dice_", "jc_", "hd95_")) and isinstance(v, (int, float)):
            res[f"eval_{k}"] = float(v)
    if "primary_far_fp_raw" in ov:
        res["eval_far_fp_raw"] = int(ov["primary_far_fp_raw"])
        res["eval_far_fp_lcc"] = int(ov["primary_far_fp_lcc"])
    return res



def build_lr_fn(total_steps: int, warmup_iters: int, mode: str):
    """LambdaLR multiplier: linear warmup, then cosine or polynomial (power 0.9) decay to 0 at total_steps."""
    def lr_schedule_fn(step: int) -> float:
        if warmup_iters > 0 and step < warmup_iters:
            return float(step + 1) / float(max(1, warmup_iters))
        decay_steps = max(1, total_steps - warmup_iters)
        prog = float(step + 1 - warmup_iters) / float(decay_steps)
        prog = min(max(prog, 0.0), 1.0)
        if mode == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * prog))
        return max(0.0, 1.0 - prog) ** 0.9

    return lr_schedule_fn


def capped_lambda(curr_lambda: float, ratio_cap: float, norm_base: float, norm_cba: float) -> float:
    """Effective lambda_A under an optional gradient-ratio cap.

    ratio = lambda * norm_cba / norm_base, the norms being taken w.r.t. the student logits. When ratio_cap > 0 the
    effective lambda is limited so that ratio <= ratio_cap; it never exceeds the curriculum value.
    """
    if ratio_cap <= 0.0 or norm_cba <= 0.0:
        return curr_lambda
    return min(curr_lambda, ratio_cap * norm_base / norm_cba)


def make_grad_log(epoch: int, b_idx: int, curr_lambda: float, beta: float, g_base, diag: dict) -> dict:
    """One gradient-diagnostic record (norms w.r.t. logits, effective ratios vs the base loss)."""
    gb = float(g_base.norm().item()) if g_base is not None else 0.0
    nb = float(diag.get("grad_norm_b_autograd", 0.0))
    ne = float(diag.get("grad_norm_e_autograd", 0.0))
    eps = 1e-12
    cb = diag.get("boundary_conflict_rate", {}) or {}
    ce = diag.get("error_conflict_rate", {}) or {}
    return {
        "epoch": epoch,
        "microbatch": b_idx,
        "norm_base": gb,
        "norm_b": nb,
        "norm_e": ne,
        "ratio_b": float(curr_lambda * nb / (gb + eps)),
        "ratio_e": float(curr_lambda * beta * ne / (gb + eps)),
        "gate_b_mean": float(diag.get("boundary_gate_mean", 0.0)),
        "gate_e_mean": float(diag.get("error_gate_mean", 0.0)),
        "h_norm_mean": float(diag.get("h_norm_mean", 0.0)),
        "conflict_b_macro": float(sum(cb.values()) / len(cb)) if cb else 0.0,
        "conflict_e_macro": float(sum(ce.values()) / len(ce)) if ce else 0.0,
        "conflict_b_per_class": {str(k): float(v) for k, v in cb.items()},
        "conflict_e_per_class": {str(k): float(v) for k, v in ce.items()},
    }


def summarize_grad_logs(logs: List[dict]) -> dict:
    """Scalar per-epoch means of the gradient-diagnostic records (kept in metrics.csv)."""
    def mean(key: str) -> float:
        return float(sum(r[key] for r in logs) / len(logs))
    return {
        "gl_ratio_b_mean": mean("ratio_b"),
        "gl_ratio_e_mean": mean("ratio_e"),
        "gl_gate_b_mean": mean("gate_b_mean"),
        "gl_gate_e_mean": mean("gate_e_mean"),
        "gl_h_norm_mean": mean("h_norm_mean"),
        "gl_conflict_b_macro": mean("conflict_b_macro"),
        "gl_conflict_e_macro": mean("conflict_e_macro"),
        "gl_n": len(logs),
    }


def write_metrics_files(arm_dir: str, history: List[dict]) -> None:
    """Rewrite metrics.jsonl and metrics.csv from the full history (atomic, idempotent on resume)."""
    jsonl_path = os.path.join(arm_dir, "metrics.jsonl")
    tmp = f"{jsonl_path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for rec in history:
            handle.write(json.dumps(rec, default=str) + "\n")
    os.replace(tmp, jsonl_path)

    columns: List[str] = []
    for rec in history:
        for key, val in rec.items():
            if not isinstance(val, (dict, list)) and key not in columns:
                columns.append(key)
    csv_path = os.path.join(arm_dir, "metrics.csv")
    tmp = f"{csv_path}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", restval="")
        writer.writeheader()
        for rec in history:
            writer.writerow(rec)
    os.replace(tmp, csv_path)

    grad_path = os.path.join(arm_dir, "grad_log.jsonl")
    if any(rec.get("grad_log") for rec in history):
        tmp = f"{grad_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            for rec in history:
                for item in rec.get("grad_log", []):
                    handle.write(json.dumps(item) + "\n")
        os.replace(tmp, grad_path)


def ema_update(student: nn.Module, ema_student: nn.Module, decay: float = 0.99, buffer_mode: str = "average") -> None:
    """EMA of parameters (p_ema = decay*p_ema + (1-decay)*p).

    buffer_mode "average": float buffers (BatchNorm running_mean/var) are averaged the same way
    (as in the v1 pilot); integer buffers (num_batches_tracked) are copied.
    buffer_mode "copy": every buffer is copied from the student (the v2-pilot behaviour).
    """
    with torch.no_grad():
        for p_s, p_ema in zip(student.parameters(), ema_student.parameters()):
            p_ema.data.mul_(decay).add_(p_s.data, alpha=1.0 - decay)
        for b_s, b_ema in zip(student.buffers(), ema_student.buffers()):
            if buffer_mode == "average" and b_s.is_floating_point():
                b_ema.data.mul_(decay).add_(b_s.data, alpha=1.0 - decay)
            else:
                b_ema.data.copy_(b_s.data)


def copy_atomic(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    tmp = f"{dst}.tmp"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def recipe_summary(args: argparse.Namespace, use_amp: bool, is_bf16: bool) -> dict:
    """Training recipe actually used, and the known differences from the original KIS-300 run
    (outputs/drive_results_300ep log: batch 16, fp32, AdamW wd 0.1, warmup 250, 139 it/epoch)."""
    precision = ("bf16" if is_bf16 else "fp16") + " autocast" if use_amp else "fp32"
    return {
        "used": {
            "weight_decay": args.weight_decay,
            "warmup_iters": args.warmup_iters,
            "lr_schedule": args.lr_schedule,
            "aug": args.aug,
            "cpd_weight_mode": args.cpd_weight_mode,
            "ema_buffer_mode": args.ema_buffer_mode,
            "student_init": args.student_ckpt or "from scratch (ImageNet backbone)",
            "teacher_lora": args.teacher_ckpt,
            "precision": precision,
            "microbatch": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
        },
        "known_differences_from_kis300_run": [
            f"precision: {precision} here vs fp32 in the original run",
            f"BatchNorm statistics are computed on microbatches of {args.batch_size} "
            f"(effective batch {args.batch_size * args.grad_accum_steps} via gradient accumulation) "
            "vs a single batch of 16 in the original run",
            "teacher: teacher_weights/epoch_299.pth (LoRA); the original run loaded best_probe.pth",
            f"cpd_weight_mode={args.cpd_weight_mode} (user decision: own reproduced baseline, not the author's recipe); "
            "the original trainer default is 'paper'",
            "evaluation protocol: evaluate_volumes() drops units without GT and NaN HD95; "
            "distances are in voxels (test h5 files and NIfTI headers carry no real spacing)",
        ],
    }


def write_run_config(arm_dir: str, args: argparse.Namespace, arm_name: str, device: torch.device,
                     resumed_from_epoch: int, recipe: Optional[dict] = None) -> dict:
    """Write/extend run_config.json: args, git commit, checkpoint hashes, environment."""
    path = os.path.join(arm_dir, "run_config.json")
    cfg: dict = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                cfg = json.load(handle)
        except Exception:
            cfg = {}
    if "checkpoint_sha256" not in cfg:
        hashes = {}
        for label, p in (
            ("teacher_lora", getattr(args, "teacher_ckpt", None)),
            ("teacher_base", getattr(args, "teacher_base_ckpt", None)),
            ("student_init", args.student_ckpt),
        ):
            resolved = resolve_file(p)
            hashes[label] = (
                {"path": resolved, "bytes": os.path.getsize(resolved), "sha256": sha256_file(resolved)}
                if resolved
                else None
            )
        cfg["checkpoint_sha256"] = hashes
    cfg.update(
        {
            "arm": arm_name,
            "args": vars(args),
            "git": git_info(),
            "recipe": recipe,
            "env": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                "host": socket.gethostname(),
            },
        }
    )
    cfg.setdefault("invocations", []).append({"started_utc": utc_now(), "resumed_from_epoch": resumed_from_epoch})
    atomic_json(path, cfg)
    return cfg


# -----------------------------------------------------------------------------
# Main Training & Pilot Execution Routine
# -----------------------------------------------------------------------------

def run_pilot(args: argparse.Namespace) -> dict:
    dataset = apply_dataset_defaults(args)
    set_seed(args.seed)
    arm_name = resolve_arm_name(args.arm)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("=" * 78)
    print(" CBA v2 EXPERIMENTAL BENCHMARK RUNNER")
    print("=" * 78)
    print(f" Dataset      : {dataset} (img_size={args.img_size}, num_classes={args.num_classes})")
    print(f" Arm Name     : {arm_name}")
    print(f" Device       : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Microbatch   : {args.batch_size} (Grad Accum: {args.grad_accum_steps}, Effective: {args.batch_size * args.grad_accum_steps})")
    print(f" Epochs       : {args.epochs}")
    print(f" Output Dir   : {args.output_dir}")
    print(f" CBA Coeff    : lambda_A = {args.lambda_a if arm_name != 'Arm_V1_KIS' else 0.0}, beta = {args.beta}")
    if getattr(args, "use_curriculum", False):
        print(f" Curriculum   : Warmup={args.curriculum_warmup} ep, Ramp={args.curriculum_ramp} ep")
    print("-" * 78)

    os.makedirs(args.output_dir, exist_ok=True)
    arm_dir = os.path.join(args.output_dir, arm_name)
    os.makedirs(arm_dir, exist_ok=True)

    # 1. Models. The teacher is built first and is mandatory outside --smoke_test:
    #    a missing teacher must abort the run, never silently degrade to CE-only training.
    teacher = None
    if args.smoke_test:
        student = SimpleConvStudent(in_channels=3, num_classes=args.num_classes).to(device)
        ema_student = SimpleConvStudent(in_channels=3, num_classes=args.num_classes).to(device)
        ema_student.load_state_dict(student.state_dict())
    else:
        from experiments.cba_v2.models import build_student, build_frozen_teacher

        if arm_name != "Arm_SGS_Only":
            teacher = build_frozen_teacher(
                iad_root=REPO_ROOT,
                teacher_ckpt=getattr(args, "teacher_ckpt", "teacher_weights/epoch_299.pth"),
                base_ckpt=getattr(args, "teacher_base_ckpt", "data/weights/sam_vit_l_0b3195.pth"),
                device=device,
                dataset=dataset,
            )
            if teacher is None:
                raise RuntimeError("build_frozen_teacher returned None; refusing to train without a teacher.")
        init_p = getattr(args, "init_ckpt", None)
        stu_ckpt = init_p if init_p else args.student_ckpt
        stu_ckpt = None if (not stu_ckpt or str(stu_ckpt).lower() in ("none", "null", "")) else stu_ckpt
        student = build_student(REPO_ROOT, stu_ckpt, device, dataset=dataset)
        ema_student = build_student(REPO_ROOT, stu_ckpt, device, dataset=dataset)
        ema_student.load_state_dict(student.state_dict())

    ema_student.eval()
    for p in ema_student.parameters():
        p.requires_grad = False

    # 2. Dataset & DataLoader
    if args.smoke_test:
        print(f"[Dataset] Instantiating Synthetic Slice Dataset ({dataset}, {args.img_size}x{args.img_size}, C={args.num_classes}) for Smoke Verification...", flush=True)
        train_dataset = SyntheticSliceDataset(
            num_samples=args.batch_size * args.grad_accum_steps * 2,
            num_classes=args.num_classes,
            img_size=args.img_size,
        )
    elif dataset == "acdc":
        from experiments.cba_v2.data import ACDCSliceDataset

        train_dataset = ACDCSliceDataset(
            data_dir=args.data_dir,
            list_path=args.list_path,
            iad_root=REPO_ROOT,
            split="train",
            augment=True,
            aug=args.aug,
            img_size=args.img_size,
        )
    elif dataset == "polyp":
        from experiments.cba_v2.data import PolypDataset

        train_dataset = PolypDataset(
            data_dir=args.data_dir,
            iad_root=REPO_ROOT,
            split="train",
            augment=True,
            aug=args.aug,
            img_size=args.img_size,
        )
    else:
        from experiments.cba_v2.data import SliceDataset

        train_dataset = SliceDataset(
            data_dir=args.data_dir,
            list_path=args.list_path,
            iad_root=REPO_ROOT,
            augment=True,
            aug=args.aug,
        )

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0 if os.name == "nt" or args.smoke_test else 2,
        pin_memory=(device.type == "cuda"),
        drop_last=args.drop_last,
        worker_init_fn=seed_worker,
    )

    # 3. Loss Modules
    cba_criterion = build_cba_v2_loss(arm_name, args)
    if cba_criterion is not None:
        cba_criterion = cba_criterion.to(device)

    # If warm initializing via --init_ckpt, restore EMA student and prototype cache if available
    init_p = resolve_file(getattr(args, "init_ckpt", None))
    if init_p and os.path.isfile(init_p):
        try:
            init_data = torch.load(init_p, map_location=device, weights_only=False)
            if isinstance(init_data, dict):
                if init_data.get("ema_student") is not None:
                    ema_student.load_state_dict(init_data["ema_student"])
                    print(f"[{arm_name}] Restored EMA student from init_ckpt: {init_p}", flush=True)
                if cba_criterion is not None and init_data.get("cba_criterion_state_dict") is not None:
                    cba_criterion.load_state_dict(init_data["cba_criterion_state_dict"])
                    cached_count = int(cba_criterion.valid_mask.sum().item())
                    print(f"[{arm_name}] Restored CBA prototype cache ({cached_count}/{cba_criterion.num_classes} classes) from init_ckpt: {init_p}", flush=True)
        except Exception as e:
            print(f"[{arm_name} Warning] Could not load extra states from init_ckpt: {e}", flush=True)

    # Far-FP Penalty Loss
    lambda_far = float(getattr(args, "lambda_far", 0.0))
    if arm_name in ("Arm_V5_FarFP", "Arm_V1_KIS_FarFP") and lambda_far <= 0.0:
        lambda_far = 0.05
    far_criterion = None
    if lambda_far > 0.0:
        from experiments.cba_v2.losses.far_fp_loss import FarFPLoss
        far_criterion = FarFPLoss(
            num_classes=args.num_classes,
            distance_threshold=getattr(args, "far_threshold", 20.0),
            mode=getattr(args, "far_mode", "fraction"),
            lambda_far=lambda_far,
            penalty_absent_slices=getattr(args, "far_penalty_absent", True),
            ref_organ_size=getattr(args, "far_ref_size", 1000.0),
        ).to(device)
        print(f"[{arm_name}] Initialized FarFPLoss (lambda_far={lambda_far:.4f}, mode={getattr(args, 'far_mode', 'fraction')}, threshold={getattr(args, 'far_threshold', 20.0)} vox)", flush=True)

    ce_criterion = nn.CrossEntropyLoss()
    cpd_mode = getattr(args, "cpd_weight_mode", "paper")
    sgs_w = getattr(args, "sgs_weight", 0.8)
    kis_w = getattr(args, "kis_weight", 0.2)

    kis_criterion = None
    if teacher is not None or arm_name == "Arm_IAD_200_100":
        if arm_name not in ("Arm_LSKD", "Arm_SinKD", "Arm_DIST_Plus", "Arm_SGS_Only"):
            kis_criterion = KISLoss(
                num_classes=args.num_classes,
                dice_weight=0.9,
                cpd_weight_mode=cpd_mode,
                sgs_weight=sgs_w,
                kis_weight=kis_w,
            ).to(device)
            print(f"[{arm_name}] Initialized KISLoss (SGS: {sgs_w}, KIS: {kis_w}, CPD: {cpd_mode})", flush=True)

    baseline_kd_criterion = None
    if arm_name in ("Arm_LSKD", "Arm_SinKD", "Arm_DIST_Plus", "Arm_SGS_Only"):
        kd_method_map = {
            "Arm_LSKD": "lskd",
            "Arm_SinKD": "sinkd",
            "Arm_DIST_Plus": "dist_plus",
            "Arm_SGS_Only": "sgs_only",
        }
        eff_sgs_w = 1.0 if arm_name == "Arm_SGS_Only" else sgs_w
        eff_kd_w = 0.0 if arm_name == "Arm_SGS_Only" else kis_w
        baseline_kd_criterion = BaselineKDLoss(
            method=kd_method_map[arm_name],
            num_classes=args.num_classes,
            dice_weight=0.9,
            sgs_weight=eff_sgs_w,
            kd_weight=eff_kd_w,
            lskd_tau=getattr(args, "lskd_tau", 2.0),
            sinkd_lam=getattr(args, "sinkd_lam", 10.0),
            sinkd_iters=getattr(args, "sinkd_iters", 5),
            dist_beta=getattr(args, "dist_beta", 1.0),
            dist_gamma=getattr(args, "dist_gamma", 1.0),
            dist_tau=getattr(args, "dist_tau", 1.0),
        ).to(device)
        print(f"[{arm_name}] Initialized BaselineKDLoss (method={kd_method_map[arm_name]}, SGS: {eff_sgs_w}, KD: {eff_kd_w})", flush=True)

    kas_head = None
    kas_sgs_criterion = None
    kas_optimizer = None
    kis_stage_epochs = int(getattr(args, "kis_stage_epochs", 200))
    kas_weight = float(getattr(args, "kas_weight", 0.1))
    kas_temp = float(getattr(args, "kas_temp", 1.0))
    if arm_name == "Arm_IAD_200_100":
        # Preserve student/dataloader RNG state so Stage 1 (KIS) matches Arm_V1_KIS bit-for-bit.
        rng_before_kas = capture_rng_state()
        kas_head = IADKasHead(in_channels=256, num_classes=args.num_classes).to(device)
        kas_sgs_criterion = BaselineKDLoss(
            method="sgs_only",
            num_classes=args.num_classes,
            dice_weight=0.9,
            sgs_weight=1.0,
            kd_weight=0.0,
        ).to(device)
        kas_optimizer = torch.optim.AdamW(kas_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        restore_rng_state(rng_before_kas)
        print(
            f"[{arm_name}] Initialized 2-stage IAD (Stage 1 KIS: ep 1..{kis_stage_epochs}, "
            f"Stage 2 KAS: ep {kis_stage_epochs + 1}..{args.epochs}, kas_weight={kas_weight})",
            flush=True,
        )

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    epoch_batches = len(loader)
    if args.max_train_batches:
        epoch_batches = min(epoch_batches, args.max_train_batches)
    steps_per_epoch = math.ceil(epoch_batches / args.grad_accum_steps)
    # --sched_epochs decouples the LR schedule length from the number of epochs actually run
    # (e.g. a 20-epoch shared warmup of a 60-epoch tuning schedule must not decay to zero at epoch 20).
    sched_epochs = args.sched_epochs if args.sched_epochs else args.epochs
    total_steps = sched_epochs * steps_per_epoch
    warmup_iters = min(getattr(args, "warmup_iters", 100), max(1, total_steps // 4)) if total_steps > 0 else 0

    lr_schedule_fn = build_lr_fn(total_steps, warmup_iters, getattr(args, "lr_schedule", "cosine"))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule_fn)

    use_amp = (device.type == "cuda" and not args.no_amp)
    is_bf16 = use_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if is_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and not is_bf16))

    start_epoch = 0
    history: List[dict] = []

    # 4. Checkpoint Resume ("auto" = this arm's latest local checkpoint, else the Drive copy)
    resume_path = args.resume
    if resume_path and str(resume_path).lower() == "auto":
        candidates = [os.path.join(arm_dir, "latest_checkpoint.pth")]
        if args.drive_sync_dir:
            candidates.append(os.path.join(args.drive_sync_dir, "latest_checkpoint.pth"))
        resume_path = next((c for c in candidates if os.path.isfile(c)), None)
        print(f"[Resume] auto -> {resume_path if resume_path else 'no checkpoint found, starting fresh'}", flush=True)
    if resume_path:
        resolved_resume = resolve_file(resume_path)
        if not resolved_resume:
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_path = resolved_resume
        print(f"[Resume] Loading checkpoint from: {resume_path}", flush=True)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)

        if isinstance(ckpt, dict):
            ckpt_args = ckpt.get("args", {})
            if isinstance(ckpt_args, dict) and ckpt_args.get("dataset"):
                ckpt_ds = normalize_dataset_name(ckpt_args["dataset"])
                if ckpt_ds != dataset:
                    raise ValueError(
                        f"[Checkpoint Guard] Refusing to resume checkpoint from dataset={ckpt_ds!r} "
                        f"into run with dataset={dataset!r}: {resume_path}"
                    )
            if not args.smoke_test and "student" in ckpt:
                verify_checkpoint_compatibility(
                    ckpt["student"],
                    dataset=dataset,
                    role="student",
                    source_label=resume_path,
                )

        student.load_state_dict(ckpt["student"])
        if ckpt.get("ema_student") is not None:
            ema_student.load_state_dict(ckpt["ema_student"])
        if ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") is not None and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
        if kas_head is not None and ckpt.get("kas_head") is not None:
            kas_head.load_state_dict(ckpt["kas_head"])
        if kas_optimizer is not None and ckpt.get("kas_optimizer") is not None:
            kas_optimizer.load_state_dict(ckpt["kas_optimizer"])

        if cba_criterion is not None and ckpt.get("cba_criterion_state_dict") is not None:
            cba_criterion.load_state_dict(ckpt["cba_criterion_state_dict"])
            cached_count = int(cba_criterion.valid_mask.sum().item())
            print(f"[Resume] Restored CBA v2 prototype cache: {cached_count}/{cba_criterion.num_classes} classes valid.", flush=True)
            print(f"[Resume] Coherence: {cba_criterion.get_cache_coherence()}", flush=True)

        restore_rng_state(ckpt.get("rng"))
        start_epoch = int(ckpt.get("epoch", 0))
        history = list(ckpt.get("history", []))
        print(f"[Resume] Resumed successfully at epoch {start_epoch}.", flush=True)

    run_cfg = write_run_config(
        arm_dir, args, arm_name, device, resumed_from_epoch=start_epoch,
        recipe=recipe_summary(args, use_amp, is_bf16),
    )

    eval_freq = args.eval_freq
    save_freq = args.save_freq
    save_epochs = {int(e) for e in str(args.save_epochs).split(",") if e.strip()}
    if arm_name == "Arm_IAD_200_100" and 0 < kis_stage_epochs <= args.epochs:
        save_epochs.add(kis_stage_epochs)
    final_preds_dir = os.path.join(arm_dir, "final_preds")
    avg_loss = float("nan")
    final_eval_summary: Optional[dict] = None

    def build_ckpt(epoch_done: int) -> dict:
        return {
            "epoch": epoch_done,
            "student": student.state_dict(),
            "ema_student": ema_student.state_dict() if ema_student is not None else None,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "cba_criterion_state_dict": cba_criterion.state_dict() if cba_criterion is not None else None,
            "kas_head": kas_head.state_dict() if kas_head is not None else None,
            "kas_optimizer": kas_optimizer.state_dict() if kas_optimizer is not None else None,
            "rng": capture_rng_state(),
            "history": history,
            "arm_name": arm_name,
            "args": vars(args),
        }

    # 5. Training Loop
    for epoch in range(start_epoch, args.epochs):
        student.train()
        if cba_criterion is not None:
            cba_criterion.train()
        in_kas_stage = (arm_name == "Arm_IAD_200_100") and ((epoch + 1) > kis_stage_epochs)
        if kas_head is not None and in_kas_stage:
            kas_head.train()
            kas_optimizer.zero_grad(set_to_none=True)

        # Compute dynamic curriculum lambda_A for this epoch
        if args.use_curriculum and arm_name not in NON_CBA_ARMS:
            curr_lambda = get_curriculum_lambda(
                epoch_1indexed=epoch + 1,
                max_lambda=args.lambda_a,
                warmup_epochs=args.curriculum_warmup,
                ramp_epochs=args.curriculum_ramp,
            )
        else:
            curr_lambda = args.lambda_a if arm_name not in NON_CBA_ARMS else 0.0

        # While lambda_A == 0 the CBA term is skipped entirely (no EMA/CBA forward), so every arm
        # consumes identical compute and RNG during warmup and can be forked from one checkpoint.
        cba_active = cba_criterion is not None and curr_lambda > 0.0

        running_loss = 0.0
        running_base = 0.0
        running_cba = 0.0
        running_far = 0.0
        running_far_ratio = 0.0
        running_b = 0.0
        running_e = 0.0
        running_e_fn = 0.0
        running_e_fp = 0.0
        running_e_empty = 0.0
        running_skip_e = 0.0
        running_err_active = 0.0
        running_grad_norm_e = 0.0
        running_eff_grad_norm_e = 0.0
        running_grad_norm_b = 0.0
        running_lam_eff = 0.0
        running_ratio_cba = 0.0
        running_kd = 0.0
        running_rfc = 0.0
        running_css = 0.0
        grad_logs: List[dict] = []
        batches = 0

        optimizer.zero_grad(set_to_none=True)
        t_ep_start = time.perf_counter()

        for b_idx, batch in enumerate(loader):
            if args.max_train_batches and b_idx >= args.max_train_batches:
                break
            img = batch["image"].to(device, non_blocking=True)
            lab = batch["label"].to(device, non_blocking=True)

            tea_logits = None
            if teacher is not None and not in_kas_stage:
                with torch.no_grad():
                    if use_amp:
                        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                            tea_out = teacher(img, multimask_output=8)
                    else:
                        tea_out = teacher(img, multimask_output=8)
                    tea_logits = tea_out[0] if isinstance(tea_out, (tuple, list)) else tea_out
            elif teacher is None and not in_kas_stage and (
                (baseline_kd_criterion is not None and arm_name != "Arm_SGS_Only") or arm_name == "Arm_IAD_200_100"
            ):
                with torch.no_grad():
                    ema_out_tea = ema_student(img, return_logits=True) if args.smoke_test else ema_student(img, True)
                    tea_logits = (ema_out_tea[0] if isinstance(ema_out_tea, (tuple, list)) else ema_out_tea).detach()

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out = student(img, return_logits=True) if args.smoke_test else student(img, True)
                logits = out[0] if isinstance(out, (tuple, list)) else out

                loss_kd_val = 0.0
                loss_rfc_val = 0.0
                loss_css_val = 0.0

                if in_kas_stage and kas_head is not None and kas_sgs_criterion is not None:
                    stu_global = out[-1] if isinstance(out, (tuple, list)) else logits
                    loss_sgs, _ = kas_sgs_criterion.forward_sgs(logits, lab)
                    kas_res = kas_head(logits, stu_global, temperature=kas_temp)
                    loss_kas = kas_res["loss_kas"]
                    loss_rfc_val = float(kas_res["loss_rfc"].detach().item())
                    loss_css_val = float(kas_res["loss_css"].detach().item())
                    loss_kd_val = float(loss_kas.detach().item())
                    loss_base = loss_sgs + kas_weight * loss_kas
                elif baseline_kd_criterion is not None and (tea_logits is not None or arm_name == "Arm_SGS_Only"):
                    loss_base, kd_metrics = baseline_kd_criterion(
                        outputs_student=logits,
                        outputs_teacher=tea_logits,
                        labels=lab,
                    )
                    loss_kd_val = float(kd_metrics["loss_kd"].item())
                elif kis_criterion is not None and tea_logits is not None:
                    loss_base, kis_metrics = kis_criterion(
                        outputs_student=logits,
                        outputs_teacher=tea_logits,
                        labels=lab,
                    )
                    loss_kd_val = float(kis_metrics.get("loss_kis", torch.tensor(0.0)).item())
                else:
                    if args.num_classes == 1:
                        loss_base = structure_loss(logits, lab)
                    else:
                        loss_base = ce_criterion(logits, lab.long())

                loss_cba = torch.tensor(0.0, device=device)
                lb_val, le_val = 0.0, 0.0
                diag = {}
                lam_eff = curr_lambda
                ratio_cba = 0.0

                if cba_active:
                    fplus_ema = None
                    with torch.no_grad():
                        ema_out = ema_student(img, return_logits=True) if args.smoke_test else ema_student(img, True)
                        fplus_ema = ema_out[0] if isinstance(ema_out, (tuple, list)) else ema_out

                    gates_on = (cba_criterion.gating_type != "none") or (
                        cba_criterion.enable_error and cba_criterion.error_mode == "contrastive"
                        and cba_criterion.error_gating_type != "none")
                    log_step = bool(args.grad_log_every) and (
                        b_idx % (args.grad_accum_steps * args.grad_log_every) == 0)
                    g_base = None
                    if (args.gate_ref == "base" and gates_on) or log_step or args.lambda_ratio_cap > 0.0:
                        g_base = torch.autograd.grad(loss_base, logits, retain_graph=True)[0].detach()

                    if args.num_classes == 1:
                        cba_fplus = lift_binary_logits_for_cba(logits)
                        cba_fplus_ema = lift_binary_logits_for_cba(fplus_ema) if fplus_ema is not None else None
                        cba_g_ref = (
                            torch.cat([-g_base, g_base], dim=1)
                            if (args.gate_ref == "base" and gates_on and g_base is not None)
                            else None
                        )
                    else:
                        cba_fplus = logits
                        cba_fplus_ema = fplus_ema
                        cba_g_ref = g_base if (args.gate_ref == "base" and gates_on) else None

                    loss_b_t, loss_e_t, diag = cba_criterion(
                        fplus=cba_fplus,
                        y_gt=lab,
                        fplus_ema=cba_fplus_ema,
                        return_diagnostics=True,
                        return_stream_losses=True,
                        g_ref=cba_g_ref,
                        compute_autograd_norms=log_step,
                    )
                    loss_cba = (loss_b_t if cba_criterion.enable_boundary else 0.0 * loss_b_t) + \
                        cba_criterion.beta * (loss_e_t if cba_criterion.enable_error else 0.0 * loss_e_t)
                    if args.lambda_ratio_cap > 0.0 and loss_cba.requires_grad:
                        g_cba = torch.autograd.grad(loss_cba, logits, retain_graph=True, allow_unused=True)[0]
                        if g_cba is not None:
                            n_base = float(g_base.float().norm().item())
                            n_cba = float(g_cba.detach().float().norm().item())
                            lam_eff = capped_lambda(curr_lambda, args.lambda_ratio_cap, n_base, n_cba)
                            ratio_cba = lam_eff * n_cba / (n_base + 1e-12)
                    if log_step:
                        grad_logs.append(make_grad_log(epoch + 1, b_idx, lam_eff, cba_criterion.beta, g_base, diag))
                    lb_val = diag.get("loss_b", 0.0)
                    le_val = diag.get("loss_e", 0.0)
                    running_b += lb_val
                    running_e += le_val
                    running_e_fn += diag.get("loss_e_fn", 0.0)
                    running_e_fp += diag.get("loss_e_fp", 0.0)
                    running_e_empty += diag.get("loss_e_empty", 0.0)
                    running_skip_e += diag.get("skipped_error_rate", 0.0)
                    err_act_dict = diag.get("error_active_fraction", {})
                    if err_act_dict:
                        running_err_active += sum(err_act_dict.values()) / max(1, len(err_act_dict))
                    running_grad_norm_e += diag.get("grad_norm_e_analytical", 0.0)
                    running_eff_grad_norm_e += diag.get("effective_grad_norm_e_analytical", 0.0)
                    running_grad_norm_b += diag.get("grad_norm_b_analytical", 0.0)

                loss_far = torch.tensor(0.0, device=device)
                far_diag = {}
                if far_criterion is not None and lambda_far > 0.0:
                    loss_far, far_diag = far_criterion(logits, lab)

                total_loss = loss_base + (lam_eff * loss_cba) + (lambda_far * loss_far)
                scaled_loss = total_loss / args.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if (b_idx + 1) % args.grad_accum_steps == 0 or (b_idx + 1) == epoch_batches:
                if in_kas_stage and kas_optimizer is not None:
                    for g in kas_optimizer.param_groups:
                        g["lr"] = optimizer.param_groups[0]["lr"]
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    if in_kas_stage and kas_optimizer is not None:
                        scaler.step(kas_optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                    if in_kas_stage and kas_optimizer is not None:
                        kas_optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if in_kas_stage and kas_optimizer is not None:
                    kas_optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                # Update EMA student
                if ema_student is not None:
                    ema_update(student, ema_student, decay=0.99, buffer_mode=args.ema_buffer_mode)

            running_loss += float(total_loss.item())
            running_base += float(loss_base.item())
            running_cba += float(loss_cba.item())
            running_far += float(loss_far.item())
            running_far_ratio += far_diag.get("far_fp_ratio", 0.0)
            running_lam_eff += lam_eff
            running_ratio_cba += ratio_cba
            running_kd += loss_kd_val
            running_rfc += loss_rfc_val
            running_css += loss_css_val
            batches += 1

        nb = max(1, batches)
        ep_sec = time.perf_counter() - t_ep_start

        avg_loss = running_loss / nb
        avg_base = running_base / nb
        avg_cba = running_cba / nb
        avg_far = running_far / nb
        avg_far_ratio = running_far_ratio / nb
        avg_b = running_b / nb
        avg_e = running_e / nb
        avg_e_fn = running_e_fn / nb
        avg_e_fp = running_e_fp / nb
        avg_e_empty = running_e_empty / nb
        avg_skip_e = running_skip_e / nb
        avg_err_act = running_err_active / nb
        avg_gne = running_grad_norm_e / nb
        avg_eff_gne = running_eff_grad_norm_e / nb
        avg_gnb = running_grad_norm_b / nb
        avg_lam_eff = running_lam_eff / nb
        avg_ratio_cba = running_ratio_cba / nb
        avg_kd = running_kd / nb
        avg_rfc = running_rfc / nb
        avg_css = running_css / nb
        cap_str = (f", lambda_eff={avg_lam_eff:.5f}, ratio={avg_ratio_cba:.3f}"
                   if args.lambda_ratio_cap > 0.0 else "")
        far_str = (f", L_far={avg_far:.4f} (ratio={avg_far_ratio*100:.2f}%)"
                   if far_criterion is not None else "")

        # Cache stats
        cache_valid_count = int(cba_criterion.valid_mask.sum().item()) if cba_criterion is not None else 0
        cache_total_classes = cba_criterion.num_classes if cba_criterion is not None else args.num_classes
        cache_coherence = cba_criterion.get_cache_coherence() if cba_criterion is not None else {}
        mean_temporal_coh = (sum(cache_coherence.values()) / len(cache_coherence)) if cache_coherence else 0.0

        print(
            f"[{arm_name}] Ep {epoch+1:02d}/{args.epochs:02d} | "
            f"Loss: {avg_loss:.4f} (Base: {avg_base:.4f}, CBA: {avg_cba:.4f} [L_B: {avg_b:.4f}, L_E: {avg_e:.4f} "
            f"(FN:{avg_e_fn:.4f}, FP:{avg_e_fp:.4f}, Empty:{avg_e_empty:.4f})], lambda={curr_lambda:.4f}{cap_str}{far_str}) | "
            f"ErrGateOpen: {avg_err_act*100:.1f}% | SkipErr: {avg_skip_e*100:.1f}% | "
            f"GradNorm (||g_E||: {avg_gne:.4f}, eff_||g_E||: {avg_eff_gne:.4f}, ||g_B||: {avg_gnb:.4f}) | "
            f"Cache: {cache_valid_count}/{cache_total_classes} (R_temp: {mean_temporal_coh:.3f}) | Time: {ep_sec:.2f}s",
            flush=True,
        )

        ep_record = {
            "epoch": epoch + 1,
            "loss_total": avg_loss,
            "loss_base": avg_base,
            "loss_cba": avg_cba,
            "loss_far": avg_far,
            "far_fp_ratio_mean": avg_far_ratio,
            "loss_b": avg_b,
            "loss_e": avg_e,
            "loss_e_fn": avg_e_fn,
            "loss_e_fp": avg_e_fp,
            "loss_e_empty": avg_e_empty,
            "loss_kd": avg_kd,
            "loss_rfc": avg_rfc,
            "loss_css": avg_css,
            "stage": "KAS" if in_kas_stage else ("KIS" if arm_name == "Arm_IAD_200_100" else arm_name),
            "lambda_a": curr_lambda,
            "lambda_eff": avg_lam_eff,
            "ratio_cba_mean": avg_ratio_cba,
            "err_gate_open_frac": avg_err_act,
            "skipped_error_rate": avg_skip_e,
            "grad_norm_e_analytical": avg_gne,
            "effective_grad_norm_e_analytical": avg_eff_gne,
            "grad_norm_b_analytical": avg_gnb,
            "cache_valid_classes": cache_valid_count,
            "cache_temporal_coherence_mean": mean_temporal_coh,
            "time_sec": ep_sec,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        if grad_logs:
            ep_record["grad_log"] = grad_logs
            ep_record.update(summarize_grad_logs(grad_logs))

        # 6. 3D Volume / Benchmark Evaluation. Monitoring only: no checkpoint is selected on test metrics;
        #    the reported model is the final-epoch student.
        is_final = (epoch + 1) == args.epochs
        do_eval = (not args.smoke_test) and (not getattr(args, "skip_eval", False)) and (
            (eval_freq > 0 and (epoch + 1) % eval_freq == 0) or is_final)
        if do_eval:
            print(f"\n[{arm_name}] Commencing Evaluation ({dataset}) at Epoch {epoch+1}...", flush=True)
            from experiments.cba_v2.evaluation import evaluate_volumes

            # STRICT FREEZE: ensures cache is never contaminated by validation/test slices
            if cba_criterion is not None:
                cba_criterion.eval()

            eval_res = evaluate_volumes(
                student_net=student,
                test_dir=args.test_dir,
                test_list=args.test_list,
                device=device,
                batch_size=args.batch_size,
                use_amp=use_amp,
                max_cases=args.eval_max_cases or None,
                iad_root=REPO_ROOT,
                save_preds_dir=final_preds_dir if (is_final and args.save_final_preds) else None,
                tag="final" if is_final else f"ep{epoch+1:03d}",
                dataset=dataset,
                img_size=args.img_size,
            )

            if cba_criterion is not None:
                cba_criterion.train()

            summary = summarize_eval(eval_res)
            ep_record.update(summary)
            atomic_json(os.path.join(arm_dir, f"eval_ep{epoch+1:03d}.json"), eval_res)
            if is_final:
                atomic_json(os.path.join(arm_dir, "eval_final.json"), eval_res)
                final_eval_summary = summary

            print(
                f"[{arm_name}] Eval Ep {epoch+1:02d} -> "
                f"Dice raw/LCC: {summary['eval_dice_raw']*100:.2f}/{summary['eval_dice_lcc']*100:.2f}% | "
                f"NSD raw: {summary['eval_nsd_raw']*100:.2f}% | "
                f"HD95 raw: {summary['eval_hd95_raw']:.2f} {summary['distance_unit']} | "
                f"Recall: {summary['eval_recall']*100:.2f}% | Precision: {summary['eval_precision']*100:.2f}%",
                flush=True,
            )

        history.append(ep_record)
        write_metrics_files(arm_dir, history)

        # 7. Checkpointing: latest every epoch (local, atomic); permanent copies at save_freq,
        #    at --save_epochs (e.g. the shared-warmup fork point) and at the final epoch.
        latest_ckpt_path = os.path.join(arm_dir, "latest_checkpoint.pth")
        atomic_torch_save(build_ckpt(epoch + 1), latest_ckpt_path)

        permanent = (
            (save_freq > 0 and (epoch + 1) % save_freq == 0)
            or is_final
            or (epoch + 1) in save_epochs
        )
        ckpt_path = os.path.join(arm_dir, f"checkpoint_epoch_{epoch+1:03d}.pth")
        if permanent:
            copy_atomic(latest_ckpt_path, ckpt_path)

        # Drive sync: latest every N epochs; permanent fork/final checkpoints and metrics when created.
        drive_dir = args.drive_sync_dir
        if drive_dir:
            try:
                if args.drive_sync_every > 0 and ((epoch + 1) % args.drive_sync_every == 0 or is_final or permanent):
                    copy_atomic(latest_ckpt_path, os.path.join(drive_dir, "latest_checkpoint.pth"))
                    for name in ("metrics.jsonl", "metrics.csv", "run_config.json"):
                        src = os.path.join(arm_dir, name)
                        if os.path.isfile(src):
                            copy_atomic(src, os.path.join(drive_dir, name))
                    if permanent and ((epoch + 1) in save_epochs or is_final):
                        copy_atomic(ckpt_path, os.path.join(drive_dir, os.path.basename(ckpt_path)))
                    if do_eval:
                        for name in (f"eval_ep{epoch+1:03d}.json", "eval_final.json"):
                            src = os.path.join(arm_dir, name)
                            if os.path.isfile(src):
                                copy_atomic(src, os.path.join(drive_dir, name))
                    print(f"[DriveSync] Synced epoch {epoch+1:03d} to: {drive_dir}", flush=True)
            except Exception as sync_err:
                print(f"[DriveSync Warning] {sync_err}", flush=True)

    # Save final student (the reported model: the weights after the last epoch)
    final_path = os.path.join(arm_dir, "final_student.pth")
    atomic_torch_save(
        {
            "student": student.state_dict(),
            "cba_criterion_state_dict": cba_criterion.state_dict() if cba_criterion is not None else None,
            "kas_head": kas_head.state_dict() if kas_head is not None else None,
            "arm_name": arm_name,
            "epoch": args.epochs,
            "args": vars(args),
        },
        final_path,
    )
    print(f"[{arm_name}] Final weights & cache saved to: {final_path}", flush=True)
    if args.drive_sync_dir and args.drive_sync_every > 0:
        try:
            copy_atomic(final_path, os.path.join(args.drive_sync_dir, "final_student.pth"))
            if os.path.isdir(final_preds_dir):
                drive_preds_dir = os.path.join(args.drive_sync_dir, "final_preds")
                os.makedirs(drive_preds_dir, exist_ok=True)
                for cur_root, _, files in os.walk(final_preds_dir):
                    rel_dir = os.path.relpath(cur_root, final_preds_dir)
                    dst_sub = drive_preds_dir if rel_dir == "." else os.path.join(drive_preds_dir, rel_dir)
                    os.makedirs(dst_sub, exist_ok=True)
                    for pred_file in files:
                        pred_src = os.path.join(cur_root, pred_file)
                        if os.path.isfile(pred_src):
                            copy_atomic(pred_src, os.path.join(dst_sub, pred_file))
        except Exception as sync_err:
            print(f"[DriveSync Warning] {sync_err}", flush=True)

    # Free memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "arm_name": arm_name,
        "dataset": dataset,
        "final_loss": avg_loss,
        "final_eval": final_eval_summary,
        "history": history,
        "run_config": run_cfg,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CBA v2 & KD Baselines Benchmark Runner")
    parser.add_argument("--dataset", type=str, default="synapse", choices=["synapse", "btcv", "acdc", "polyp"], help="Target dataset (synapse: 9 classes 512x512; acdc: 4 classes 256x256; polyp: 1 binary channel 256x256)")
    parser.add_argument("--img_size", type=int, default=0, help="Input image size (0 = dataset default: 512 for synapse, 256 for acdc/polyp)")
    parser.add_argument("--arm", type=str, default="Arm_V5_Dual_CEHard", help="Arm name (v1..v5, iad, lskd, sinkd, dist_plus)")
    parser.add_argument("--epochs", type=int, default=300, help="Total training epochs (default: 300 for freshstart)")
    parser.add_argument("--batch_size", type=int, default=8, help="Microbatch size")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_schedule", type=str, default="polynomial", choices=["cosine", "polynomial"], help="LR scheduler")
    parser.add_argument("--sched_epochs", type=int, default=0, help="Length of the LR schedule in epochs (0 = --epochs). Use for shared-warmup / proxy runs.")
    parser.add_argument("--warmup_iters", type=int, default=250, help="Linear warmup iterations (KIS-300 recipe: 250)")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay (KIS-300 recipe: 0.1)")
    parser.add_argument("--aug", type=str, default="full", choices=["full", "flip"], help="Training augmentation: original IAD RandomGenerator chain (full) or flip+rot90 only (flip); identical for every arm")
    parser.add_argument("--ema_buffer_mode", type=str, default="average", choices=["average", "copy"], help="EMA handling of float buffers (BN stats): average (v1) or copy (v2 pilot)")
    parser.add_argument("--lambda_a", type=float, default=0.02, help="CBA loss coefficient")
    parser.add_argument("--beta", type=float, default=0.5, help="Error stream weight beta")
    parser.add_argument("--lambda_ratio_cap", type=float, default=0.0,
                        help="If > 0, cap lambda_A per microbatch so that lambda_A * ||grad_logits(L_CBA)|| / "
                             "||grad_logits(L_base)|| <= this value (0 = off, the default)")
    parser.add_argument("--m", type=float, default=0.2, help="Contrastive margin")
    parser.add_argument("--tau", type=float, default=0.1, help="Temperature tau")
    parser.add_argument("--r", type=int, default=2, help="Boundary radius r")
    parser.add_argument("--k_bg", type=int, default=512, help="Background sampling budget")
    parser.add_argument("--k_fg", type=int, default=512, help="Organ sampling budget")
    parser.add_argument("--num_classes", type=int, default=9, help="Number of classes")
    parser.add_argument("--cache_momentum", type=float, default=0.9, help="EMA cache momentum")
    parser.add_argument("--gating_type", type=str, default="soft", help="Boundary gating type")
    parser.add_argument("--error_gating_type", type=str, default="none", help="Error-stream gating type (default none: D7; ce_hard requires none)")
    parser.add_argument("--cba_feature", type=str, default="centered", choices=["raw", "centered"], help="Feature space for CBA: logits minus per-pixel class mean (centered, D5) or raw logits")
    parser.add_argument("--gate_ref", type=str, default="base", choices=["ce", "base"], help="Gate reference gradient: CE-to-GT (ce) or grad of the full base loss w.r.t. logits (base, D6)")
    parser.add_argument("--grad_log_every", type=int, default=50, help="Log CBA gradient diagnostics every N optimizer steps (0 disables)")
    parser.add_argument("--boundary_use_cache", action="store_true", help="Boundary uses cache instead of batch proto")
    parser.add_argument("--boundary_fallback_cache", action="store_true", help="Boundary uses batch proto if present, fallback to cache if absent")
    parser.add_argument("--boundary_proto_source", type=str, default=None, help="Explicit boundary prototype source: 'batch', 'fallback', 'cache'")
    parser.add_argument("--use_cache_for_all", action="store_true", help="Both streams use cache anchors")
    parser.add_argument("--cpd_weight_mode", type=str, default="flat", choices=["paper", "flat"], help="CPD class weight mode (default: flat)")
    parser.add_argument("--sgs_weight", type=float, default=0.8, help="SGS base loss weight")
    parser.add_argument("--kis_weight", type=float, default=0.2, help="KIS / KD base loss weight")
    parser.add_argument("--kis_stage_epochs", type=int, default=200, help="Stage 1 KIS epochs before Stage 2 KAS in Arm_IAD_200_100 (default: 200)")
    parser.add_argument("--kas_weight", type=float, default=0.1, help="Stage 2 KAS loss weight (default: 0.1, Paper Eq. 8)")
    parser.add_argument("--kas_temp", type=float, default=1.0, help="Stage 2 KAS CSS temperature (default: 1.0)")
    parser.add_argument("--lskd_tau", type=float, default=2.0, help="LSKD temperature tau (default: 2.0)")
    parser.add_argument("--sinkd_lam", type=float, default=10.0, help="SinKD Sinkhorn regularization lambda (default: 10.0)")
    parser.add_argument("--sinkd_iters", type=int, default=5, help="SinKD Sinkhorn-Knopp iterations (default: 5)")
    parser.add_argument("--dist_beta", type=float, default=1.0, help="DIST+ inter-class weight beta (default: 1.0)")
    parser.add_argument("--dist_gamma", type=float, default=1.0, help="DIST+ intra-class weight gamma (default: 1.0)")
    parser.add_argument("--dist_tau", type=float, default=1.0, help="DIST+ softmax temperature tau (default: 1.0)")
    parser.add_argument("--use_curriculum", action=argparse.BooleanOptionalAction, default=True, help="Enable curriculum warmup/ramp for lambda_A (--no-use_curriculum to disable)")
    parser.add_argument("--no_curriculum", action="store_false", dest="use_curriculum", help="Alias of --no-use_curriculum")
    parser.add_argument("--curriculum_warmup", type=int, default=20, help="Curriculum warmup epochs (lambda_A = 0, default: 20)")
    parser.add_argument("--curriculum_ramp", type=int, default=30, help="Curriculum ramp epochs (0 -> lambda_A, default: 30)")
    parser.add_argument("--eval_freq", type=int, default=0, help="3D test volume evaluation frequency; 0 = final epoch only (default)")
    parser.add_argument("--skip_eval", action="store_true", help="Skip all evaluations, including the final one; intended for shared warmup jobs")
    parser.add_argument("--eval_max_cases", type=int, default=0, help="Limit test volumes per evaluation (0 = all)")
    parser.add_argument("--save_freq", type=int, default=25, help="Permanent checkpoint frequency (epochs)")
    parser.add_argument("--save_epochs", type=str, default="20", help="Extra comma-separated epochs that always get a permanent checkpoint (default: 20, the shared-warmup fork point)")
    parser.add_argument("--max_train_batches", type=int, default=0, help="Limit training microbatches per epoch (0 = all; for smoke runs)")
    parser.add_argument("--save_final_preds", action=argparse.BooleanOptionalAction, default=True, help="Save 3D predictions of the final-epoch evaluation")
    parser.add_argument("--output_dir", type=str, default="runs/cba_v2", help="Output directory")
    parser.add_argument("--drive_sync_dir", type=str, default=None, help="Google Drive directory for auto-syncing checkpoints")
    parser.add_argument("--drive_sync_every", type=int, default=5, help="Drive sync period in epochs (permanent checkpoints and the final epoch always sync)")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from, or 'auto' (latest local/Drive checkpoint of this arm, else fresh start)")
    parser.add_argument("--smoke_test", action="store_true", help="Run quick smoke test with synthetic data (no teacher, CE supervision)")
    parser.add_argument("--seed", type=int, default=2345, help="Random seed")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--drop_last", action=argparse.BooleanOptionalAction, default=True, help="Drop last incomplete batch in loader (--no-drop_last to keep it)")
    parser.add_argument("--student_ckpt", type=str, default=None, help="Initial student checkpoint (default: None for fresh start from scratch)")
    parser.add_argument("--init_ckpt", type=str, default=None, help="Warm initialization from checkpoint (e.g. ours_v5 epoch 300) without resuming epoch/history/optimizer")
    parser.add_argument("--lambda_far", type=float, default=0.0, help="Weight for Far-FP penalty loss L_far (default: 0.0, recommended: 0.05 for FarFP arms)")
    parser.add_argument("--far_threshold", type=float, default=20.0, help="Distance threshold in voxels for L_far (default: 20.0)")
    parser.add_argument("--far_mode", type=str, default="fraction", choices=["fraction", "bce", "focal", "combined"], help="Penalty mode for Far-FP loss")
    parser.add_argument("--far_ref_size", type=float, default=1000.0, help="Reference organ size in voxels for FarFPLoss normalization (default: 1000.0)")
    parser.add_argument("--far_penalty_absent", action=argparse.BooleanOptionalAction, default=True, help="Penalize predictions on slices where organ is absent")
    parser.add_argument("--fp_by_pred_class", action=argparse.BooleanOptionalAction, default=False, help="Group background FP errors by predicted organ class (Solution 1, default: False)")
    parser.add_argument("--fp_exclude_boundary", action=argparse.BooleanOptionalAction, default=False, help="Exclude r<=2 boundary transition band from background FP error set (Solution 1, default: False)")
    parser.add_argument("--empty_slice_suppression", action=argparse.BooleanOptionalAction, default=False, help="Enable Top-K empty-slice focal suppression for absent organs (Solution 2, default: False)")
    parser.add_argument("--empty_slice_topk", type=int, default=64, help="Top-K highest-probability pixels per absent (slice, organ) pair (Solution 2)")
    parser.add_argument("--empty_slice_prob_thresh", type=float, default=0.05, help="Probability threshold tau_p below which absent-slice pixels are ignored as clean (Solution 2)")
    parser.add_argument("--empty_slice_focal_gamma", type=float, default=2.0, help="Focal exponent gamma for empty-slice suppression (Solution 2)")
    parser.add_argument("--empty_slice_weight", type=float, default=1.0, help="Relative weight of L_EmptySlice inside L_E (Solution 2)")
    parser.add_argument("--teacher_ckpt", type=str, default="teacher_weights/epoch_299.pth", help="Teacher LoRA checkpoint")
    parser.add_argument("--teacher_base_ckpt", type=str, default="data/weights/sam_vit_l_0b3195.pth", help="Teacher SAM ViT-L base checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/btcv/train_npz", help="Path to train slices")
    parser.add_argument("--list_path", type=str, default="data/lists_Synapse/train.txt", help="Path to train slice list")
    parser.add_argument("--test_dir", type=str, default="data/btcv/test_vol_h5", help="Path to 3D test volumes")
    parser.add_argument("--test_list", type=str, default="data/lists_Synapse/test_vol.txt", help="Path to 3D test volume list")
    return parser



def main(argv: Optional[List[str]] = None):
    args = build_parser().parse_args(argv)
    run_pilot(args)


if __name__ == "__main__":
    main()

