#!/usr/bin/env python3
"""
main.py - Unified Top-Level CLI Entry Point for the IAD / CBA v2 Benchmark Repository.

Exposes all input parameters required to train, reproduce, evaluate, tune, or package
any Knowledge Distillation method or ablation arm across all supported medical segmentation datasets:

  Methods / Arms:
    - ours (v5)   : Arm_V5_Dual_CEHard      (Proposed: 0.8*L_SGS + 0.2*L_KIS + lambda_A*(L_B + beta*L_E^CE))
    - iad         : Arm_IAD_200_100         (CVPR 2026 IAD: Stage-1 KIS [ep 1..200] -> Stage-2 KAS [ep 201..300])
    - dist_plus   : Arm_DIST_Plus           (TPAMI 2025 DIST+: Inter- & Intra-class Pearson relation KD)
    - lskd        : Arm_LSKD                (CVPR 2024 LSKD: Logit Standardization KD)
    - sinkd       : Arm_SinKD               (SinKD: Sinkhorn Optimal Transport KD)
    - kis (v1)    : Arm_V1_KIS              (Stage-1 Knowledge Infusion baseline: 0.8*L_SGS + 0.2*L_KIS)
    - v2          : Arm_V2_Boundary         (Ablation: KIS + Directionally Gated Boundary Stream L_B)
    - v3          : Arm_V3_Error            (Ablation: KIS + Contrastive Error Stream L_E)
    - v4          : Arm_V4_Dual_Contrastive (Ablation: KIS + Dual Contrastive L_B + beta*L_E^contra)
    - sgs_only    : Arm_SGS_Only            (Supervised Ground-Truth Segmentation only, no Teacher KD)
    - v5_far      : Arm_V5_FarFP            (Ablation: Ours v5 + Far-FP spatial distance penalty)
    - kis_far     : Arm_V1_KIS_FarFP        (Ablation: KIS v1 + Far-FP spatial distance penalty)

  Datasets:
    - synapse     : Synapse / BTCV Multi-Organ CT (9 classes, 512x512, 18 train / 12 test 3D volumes)
    - acdc        : ACDC Cardiac MRI (4 classes: BG, RV, Myo, LV, 256x256, 1304 train slices / 40 test 3D volumes)
    - polyp       : Polyp Endoscopy (1 binary channel, 256x256, 1450 train / 5 test sets: Kvasir, ClinicDB, ColonDB, ETIS, CVC-300)

Quick-Start Examples:
  1. Inspect all parameters, methods, or datasets:
       python main.py --help
       python main.py --list_methods
       python main.py --list_datasets

  2. Train any method on any dataset (single seed or multi-seed):
       python main.py --mode train --method ours --dataset synapse --epochs 300 --seed 2345
       python main.py --mode train --method iad --dataset acdc --epochs 300 --seed 2345
       python main.py --mode train --method dist_plus --dataset polyp --epochs 300 --seeds 2345 42 1234

  3. Reproduce official 300-epoch benchmark runs (with shared 20-epoch warmup, Drive sync, and auto-resume):
       python main.py --mode reproduce --method ours --dataset synapse --seeds 2345 42 1234 --make_warmup
       python main.py --mode reproduce --method all --dataset acdc --seed 2345
       python main.py --mode reproduce --method all --dataset all --seeds 2345 42 1234 --dry_run

  4. Evaluate any trained Student or Teacher checkpoint on 3D volumes / Polyp subdatasets:
       python main.py --mode eval --dataset synapse --ckpt runs/main/ours_v5/Arm_V5_Dual_CEHard/final_student.pth
       python main.py --mode eval --dataset acdc --eval_role teacher --ckpt teacher_weights/acdc_tea.pth
       python main.py --eval_only --dataset polyp --ckpt teacher_weights/polyp_stu_iad.pth

  5. Package self-contained Colab code bundle:
       python main.py --mode package --bundle cba_v2
"""

from __future__ import annotations

import argparse
import atexit
import copy
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch

# Ensure repository root, models, scripts, and tools directories are importable
REPO_ROOT = Path(__file__).resolve().parent
for _p in (REPO_ROOT, REPO_ROOT / "models", REPO_ROOT / "scripts", REPO_ROOT / "scripts" / "tools"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

atexit.register(
    lambda: [
        shutil.rmtree(r / "__pycache__", ignore_errors=True)
        for r in (REPO_ROOT, REPO_ROOT.parent)
        if (r / "__pycache__").is_dir()
    ]
)

from experiments.cba_v2.evaluation import evaluate_volumes  # noqa: E402
from experiments.cba_v2.models import (  # noqa: E402
    DATASET_MODEL_SPECS,
    build_frozen_teacher,
    build_student,
    normalize_dataset_name,
)
from experiments.cba_v2.train import (  # noqa: E402
    PILOT_ARMS_V2,
    SimpleConvStudent,
    apply_dataset_defaults,
    atomic_json,
    resolve_arm_name,
    run_pilot,
    set_seed,
    summarize_eval,
)

SUPPORTED_DATASETS: Tuple[str, ...] = ("synapse", "btcv", "acdc", "polyp")
CANONICAL_DATASETS: Tuple[str, ...] = ("synapse", "acdc", "polyp")
DEFAULT_SEEDS: Tuple[int, ...] = (2345, 42, 1234)
MAIN_BENCHMARK_METHODS: Tuple[str, ...] = ("ours", "iad", "dist_plus", "lskd", "sinkd")

METHOD_SPECS: Dict[str, Dict[str, Optional[str]]] = {
    "ours": {
        "arm_name": "Arm_V5_Dual_CEHard",
        "launcher_kind": "ours",
        "launcher_arm": "v5",
        "category": "main_5methods",
        "formula": "0.8*L_SGS + 0.2*L_KIS + lambda_A*(L_B + beta*L_E^{CE})",
        "description": "Proposed CBA v2 (KIS + Gated Boundary Stream + Error-Driven Hard-Pixel CE Stream)",
    },
    "iad": {
        "arm_name": "Arm_IAD_200_100",
        "launcher_kind": "iad",
        "launcher_arm": None,
        "category": "main_5methods",
        "formula": "Stage 1 (ep 1..200): 0.8*L_SGS + 0.2*L_KIS | Stage 2 (ep 201..300): L_SGS + 0.1*(L_RFC + L_CSS)",
        "description": "CVPR 2026 Two-Stage Infusion-to-Assimilation Distillation (KIS -> KAS)",
    },
    "dist_plus": {
        "arm_name": "Arm_DIST_Plus",
        "launcher_kind": "dist_plus",
        "launcher_arm": None,
        "category": "main_5methods",
        "formula": "0.8*L_SGS + 0.2*L_DIST+ (beta*L_inter + gamma*L_intra)",
        "description": "TPAMI 2025 DIST+ Inter-class & Intra-class Pearson Correlation Distillation",
    },
    "lskd": {
        "arm_name": "Arm_LSKD",
        "launcher_kind": "lskd",
        "launcher_arm": None,
        "category": "main_5methods",
        "formula": "0.8*L_SGS + 0.2*L_LSKD (Z-score standardized KL, tau=2.0)",
        "description": "CVPR 2024 Logit Standardization Knowledge Distillation",
    },
    "sinkd": {
        "arm_name": "Arm_SinKD",
        "launcher_kind": "sinkd",
        "launcher_arm": None,
        "category": "main_5methods",
        "formula": "0.8*L_SGS + 0.2*L_SinKD (Sinkhorn OT distance, lambda=10.0, iters=5)",
        "description": "Sinkhorn Optimal Transport Knowledge Distillation",
    },
    "kis": {
        "arm_name": "Arm_V1_KIS",
        "launcher_kind": "kis",
        "launcher_arm": None,
        "category": "ablation",
        "formula": "0.8*L_SGS + 0.2*L_KIS (lambda_A = 0)",
        "description": "Arm V1: Stage-1 Knowledge Infusion Baseline (Soft-Label KL + Class-Weighted Prototype Alignment)",
    },
    "v2": {
        "arm_name": "Arm_V2_Boundary",
        "launcher_kind": "ours",
        "launcher_arm": "v2",
        "category": "ablation",
        "formula": "0.8*L_SGS + 0.2*L_KIS + lambda_A*L_B",
        "description": "Arm V2: KIS + Directionally Gated Boundary Stream only",
    },
    "v3": {
        "arm_name": "Arm_V3_Error",
        "launcher_kind": "ours",
        "launcher_arm": "v3",
        "category": "ablation",
        "formula": "0.8*L_SGS + 0.2*L_KIS + lambda_A*beta*L_E^{contra}",
        "description": "Arm V3: KIS + Contrastive Error Stream only",
    },
    "v4": {
        "arm_name": "Arm_V4_Dual_Contrastive",
        "launcher_kind": "ours",
        "launcher_arm": "v4",
        "category": "ablation",
        "formula": "0.8*L_SGS + 0.2*L_KIS + lambda_A*(L_B + beta*L_E^{contra})",
        "description": "Arm V4: KIS + Dual Contrastive (Boundary + Contrastive Error Stream)",
    },
    "sgs_only": {
        "arm_name": "Arm_SGS_Only",
        "launcher_kind": "sgs_only",
        "launcher_arm": None,
        "category": "ablation",
        "formula": "1.0*L_SGS (0.1*CE + 0.9*Dice or structure_loss, no Teacher KD)",
        "description": "Supervised Ground-Truth Segmentation Baseline (Student trained without Teacher)",
    },
    "v5_far": {
        "arm_name": "Arm_V5_FarFP",
        "launcher_kind": None,
        "launcher_arm": None,
        "category": "experimental",
        "formula": "L_{Ours_V5} + lambda_far*L_FarFP",
        "description": "Ours V5 with spatial Far-FP distance penalty",
    },
    "kis_far": {
        "arm_name": "Arm_V1_KIS_FarFP",
        "launcher_kind": None,
        "launcher_arm": None,
        "category": "experimental",
        "formula": "L_{KIS} + lambda_far*L_FarFP",
        "description": "KIS V1 with spatial Far-FP distance penalty",
    },
}


def print_methods_table() -> None:
    print("=" * 108)
    print(f"{'Method / Alias':<14} | {'Canonical Arm':<24} | {'Category':<14} | {'Objective / Description'}")
    print("-" * 108)
    for key, spec in METHOD_SPECS.items():
        print(f"{key:<14} | {str(spec['arm_name']):<24} | {str(spec['category']):<14} | {spec['description']}")
        print(f"{'':<14} | {'':<24} | {'Formula':<14} | {spec['formula']}")
        print("-" * 108)


def print_datasets_table() -> None:
    dataset_details = {
        "synapse": {
            "alias": "btcv",
            "train_path": "data/btcv/train_npz (list: data/lists_Synapse/train.txt)",
            "test_path": "data/btcv/test_vol_h5 (list: data/lists_Synapse/test_vol.txt)",
            "notes": "9 classes (BG + 8 abdominal organs), 512x512 slices, 18 train / 12 test 3D CT volumes",
        },
        "acdc": {
            "alias": "-",
            "train_path": "data/ACDC/train (list: data/ACDC/lists_ACDC/train.txt)",
            "test_path": "data/ACDC/test (list: data/ACDC/lists_ACDC/test.txt)",
            "notes": "4 classes (BG, RV, Myo, LV), 256x256 slices, 1304 train slices / 40 test 3D MRI volumes",
        },
        "polyp": {
            "alias": "-",
            "train_path": "data/dataset/TrainDataset",
            "test_path": "data/dataset/TestDataset/{Kvasir,CVC-ClinicDB,CVC-ColonDB,ETIS-LaribPolypDB,CVC-300}",
            "notes": "1 binary channel (sigmoid), 256x256 images, 1450 train / 5 benchmark test sets (798 images)",
        },
    }
    print("=" * 108)
    print(f"{'Dataset':<10} | {'Classes':<7} | {'Size':<9} | {'Default Teacher Checkpoint':<32} | {'Details'}")
    print("-" * 108)
    for ds_name in CANONICAL_DATASETS:
        spec = DATASET_MODEL_SPECS[ds_name]
        info = dataset_details[ds_name]
        sz = f"{spec['image_size']}x{spec['image_size']}"
        print(f"{ds_name:<10} | {spec['num_classes']:<7} | {sz:<9} | {spec['default_teacher_ckpt']:<32} | {info['notes']}")
        print(f"{'':<10} | {'Train':<7} | {info['train_path']}")
        print(f"{'':<10} | {'Test':<7} | {info['test_path']}")
        print("-" * 108)


def parse_seeds(seed: int, seeds_arg: Optional[Sequence[str]]) -> List[int]:
    """Parse --seed and optional --seeds into a deduplicated list of integer seeds."""
    if not seeds_arg:
        return [int(seed)]
    tokens: List[str] = []
    for item in seeds_arg:
        for sub in str(item).replace(",", " ").split():
            if sub.strip():
                tokens.append(sub.strip().lower())
    if not tokens:
        return [int(seed)]
    if any(t in ("all", "3seeds", "default") for t in tokens):
        return list(DEFAULT_SEEDS)
    parsed: List[int] = []
    for t in tokens:
        val = int(t)
        if val not in parsed:
            parsed.append(val)
    return parsed


def parse_datasets(dataset_arg: str) -> List[str]:
    """Parse --dataset (single name, comma-separated list, or 'all') into canonical dataset names."""
    raw = str(dataset_arg or "synapse").strip().lower()
    if raw in ("all", "3datasets"):
        return list(CANONICAL_DATASETS)
    out: List[str] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        ds = normalize_dataset_name(part)
        if ds not in out:
            out.append(ds)
    return out or ["synapse"]


def resolve_effective_arm(args: argparse.Namespace, raw_argv: Optional[Sequence[str]] = None) -> str:
    """Resolve --method and --arm into the canonical arm name for training."""
    raw_tokens = list(raw_argv) if raw_argv is not None else []
    arm_explicit = any(tok == "--arm" or tok.startswith("--arm=") for tok in raw_tokens)
    method_val = getattr(args, "method", None)
    if method_val:
        m_clean = str(method_val).strip().lower()
        if m_clean in ("all", "5methods"):
            raise ValueError("--method all is supported in '--mode reproduce'; for '--mode train' specify one method.")
        if m_clean == "ours" and arm_explicit:
            return resolve_arm_name(args.arm)
        return resolve_arm_name(m_clean)
    return resolve_arm_name(args.arm)


def parse_methods_for_reproduce(
    method_arg: Optional[str],
    arm_arg: str,
    raw_argv: Optional[Sequence[str]] = None,
) -> List[Tuple[str, Optional[str]]]:
    """Map --method / --arm to a list of (launcher_kind, launcher_arm) tuples for scripts/main_common.py."""
    raw_tokens = list(raw_argv) if raw_argv is not None else []
    arm_explicit = any(tok == "--arm" or tok.startswith("--arm=") for tok in raw_tokens)

    if method_arg:
        raw = str(method_arg).strip().lower()
    elif arm_explicit:
        raw = str(arm_arg).strip().lower()
    else:
        raw = "ours"

    if raw in ("all", "5methods"):
        return [
            ("ours", "v5"),
            ("iad", None),
            ("dist_plus", None),
            ("lskd", None),
            ("sinkd", None),
        ]

    jobs: List[Tuple[str, Optional[str]]] = []
    for token in raw.split(","):
        tok = token.strip().lower()
        if not tok:
            continue
        if tok in ("ours", "ours_v5", "v5", "arm_v5", "arm_v5_dual_cehard", "ce_hard", "cehard"):
            sub_arm = "v5"
            if tok == "ours" and arm_explicit:
                canon = resolve_arm_name(arm_arg)
                canon_to_short = {
                    "Arm_V2_Boundary": "v2",
                    "Arm_V3_Error": "v3",
                    "Arm_V4_Dual_Contrastive": "v4",
                    "Arm_V5_Dual_CEHard": "v5",
                }
                if canon not in canon_to_short:
                    raise ValueError(f"--method ours supports --arm in {{v2, v3, v4, v5}}, got {arm_arg!r}")
                sub_arm = canon_to_short[canon]
            jobs.append(("ours", sub_arm))
        elif tok in ("v2", "ours_v2", "arm_v2", "arm_v2_boundary", "boundary", "boundary_only"):
            jobs.append(("ours", "v2"))
        elif tok in ("v3", "ours_v3", "arm_v3", "arm_v3_error", "error", "error_only"):
            jobs.append(("ours", "v3"))
        elif tok in ("v4", "ours_v4", "arm_v4", "arm_v4_dual_contrastive", "dual", "cba_v2", "contrastive"):
            jobs.append(("ours", "v4"))
        elif tok in ("kis", "v1", "arm_v1", "arm_v1_kis", "baseline"):
            jobs.append(("kis", None))
        elif tok in ("iad", "iad_200_100", "arm_iad", "arm_iad_200_100"):
            jobs.append(("iad", None))
        elif tok in ("dist_plus", "dist+", "dist", "arm_dist_plus"):
            jobs.append(("dist_plus", None))
        elif tok in ("lskd", "arm_lskd"):
            jobs.append(("lskd", None))
        elif tok in ("sinkd", "arm_sinkd"):
            jobs.append(("sinkd", None))
        elif tok in ("sgs_only", "sgs", "arm_sgs", "arm_sgs_only"):
            jobs.append(("sgs_only", None))
        else:
            raise ValueError(
                f"Unsupported method/arm {tok!r} for reproduce mode. "
                "Valid choices: ours, iad, dist_plus, lskd, sinkd, kis, v1..v5, sgs_only, or all."
            )
    return jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -------------------------------------------------------------------------
    # [1] Execution Mode & Discovery
    # -------------------------------------------------------------------------
    g_mode = parser.add_argument_group("[1] Execution Mode & Discovery")
    g_mode.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "reproduce", "eval", "tune", "package"],
        help="Execution mode: 'train' (direct trainer), 'reproduce' (300-epoch benchmark launcher with shared warmup & resume), "
             "'eval' (standalone checkpoint evaluation), 'tune' (hyperparameter tuning), or 'package' (build Colab zip bundle)",
    )
    g_mode.add_argument(
        "--eval_only",
        action="store_true",
        help="Shortcut for '--mode eval' (run standalone evaluation on test volumes/datasets without training)",
    )
    g_mode.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the resolved configuration or launcher command(s) without executing training/evaluation",
    )
    g_mode.add_argument(
        "--list_methods",
        action="store_true",
        help="Print table of all supported KD methods and ablation arms, then exit",
    )
    g_mode.add_argument(
        "--list_datasets",
        action="store_true",
        help="Print table of all supported datasets, class counts, resolutions, and paths, then exit",
    )

    # -------------------------------------------------------------------------
    # [2] Method / Arm & Dataset Selection
    # -------------------------------------------------------------------------
    g_target = parser.add_argument_group("[2] Method / Arm & Dataset Selection")
    g_target.add_argument(
        "--method",
        type=str,
        default=None,
        help="Method selector: 'ours' (or 'v5'), 'iad', 'dist_plus', 'lskd', 'sinkd', 'kis' (or 'v1'), "
             "'v2', 'v3', 'v4', 'sgs_only', 'v5_far', 'kis_far', or 'all' (in reproduce mode, runs all 5 main methods)",
    )
    g_target.add_argument(
        "--arm",
        type=str,
        default="Arm_V5_Dual_CEHard",
        help="Arm name or alias (ours, v1..v5, iad, lskd, sinkd, dist_plus, sgs_only; default: Arm_V5_Dual_CEHard)",
    )
    g_target.add_argument(
        "--dataset",
        type=str,
        default="synapse",
        help="Target dataset: 'synapse' (or 'btcv', 9 classes, 512x512), 'acdc' (4 classes, 256x256), "
             "'polyp' (1 binary channel, 256x256), comma-separated list (e.g. 'acdc,polyp'), or 'all'",
    )
    g_target.add_argument(
        "--img_size",
        type=int,
        default=0,
        help="Input image size (0 = dataset default: 512 for synapse, 256 for acdc/polyp)",
    )
    g_target.add_argument(
        "--num_classes",
        type=int,
        default=9,
        help="Number of segmentation classes (auto-resolved per dataset: 9 for synapse, 4 for acdc, 1 for polyp)",
    )
    g_target.add_argument("--data_dir", type=str, default="data/btcv/train_npz", help="Path to training slices/images")
    g_target.add_argument("--list_path", type=str, default="data/lists_Synapse/train.txt", help="Path to training split list")
    g_target.add_argument("--test_dir", type=str, default="data/btcv/test_vol_h5", help="Path to test volumes/images")
    g_target.add_argument("--test_list", type=str, default="data/lists_Synapse/test_vol.txt", help="Path to test volume list")

    # -------------------------------------------------------------------------
    # [3] Reproducibility & Multi-Seed Control
    # -------------------------------------------------------------------------
    g_seed = parser.add_argument_group("[3] Reproducibility & Multi-Seed Control")
    g_seed.add_argument("--seed", type=int, default=2345, help="Primary random seed (default: 2345)")
    g_seed.add_argument(
        "--seeds",
        nargs="*",
        default=None,
        help="Multiple random seeds to run sequentially, e.g. '--seeds 2345 42 1234' or '--seeds all'",
    )

    # -------------------------------------------------------------------------
    # [4] Training Schedule, Optimization & Augmentation
    # -------------------------------------------------------------------------
    g_opt = parser.add_argument_group("[4] Training Schedule, Optimization & Augmentation")
    g_opt.add_argument("--epochs", type=int, default=300, help="Total training epochs (default: 300)")
    g_opt.add_argument("--batch_size", type=int, default=8, help="Microbatch size (default: 8)")
    g_opt.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps (default: 2 -> eff batch 16)")
    g_opt.add_argument("--lr", type=float, default=1e-4, help="Base learning rate (default: 1e-4)")
    g_opt.add_argument(
        "--lr_schedule",
        type=str,
        default="polynomial",
        choices=["cosine", "polynomial"],
        help="Learning rate decay schedule (default: polynomial)",
    )
    g_opt.add_argument(
        "--sched_epochs",
        type=int,
        default=0,
        help="Total length of LR schedule in epochs (0 = --epochs; used for shared warmup / proxy runs)",
    )
    g_opt.add_argument("--warmup_iters", type=int, default=250, help="Linear LR warmup iterations (default: 250)")
    g_opt.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay (default: 0.1)")
    g_opt.add_argument(
        "--aug",
        type=str,
        default="full",
        choices=["full", "flip"],
        help="Training augmentation: 'full' (IAD RandomGenerator chain) or 'flip' (flip+rot90 only)",
    )
    g_opt.add_argument(
        "--ema_buffer_mode",
        type=str,
        default="average",
        choices=["average", "copy"],
        help="EMA handling of float buffers (BN stats): 'average' or 'copy'",
    )
    g_opt.add_argument(
        "--drop_last",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop last incomplete batch in training DataLoader (--no-drop_last to keep it)",
    )
    g_opt.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision (AMP)")
    g_opt.add_argument("--cpu", action="store_true", help="Force CPU execution")
    g_opt.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run fast smoke test with synthetic data and lightweight student (no teacher checkpoint required)",
    )
    g_opt.add_argument(
        "--max_train_batches",
        type=int,
        default=0,
        help="Limit training microbatches per epoch (0 = all; useful for quick smoke verification)",
    )

    # -------------------------------------------------------------------------
    # [5] Base Supervision (SGS) & Stage-1 Distillation (KIS)
    # -------------------------------------------------------------------------
    g_base = parser.add_argument_group("[5] Base Supervision (SGS) & Stage-1 Distillation (KIS)")
    g_base.add_argument("--sgs_weight", type=float, default=0.8, help="Supervised ground-truth loss (L_SGS) weight (default: 0.8)")
    g_base.add_argument("--kis_weight", type=float, default=0.2, help="Knowledge distillation loss (L_KIS / L_KD) weight (default: 0.2)")
    g_base.add_argument(
        "--cpd_weight_mode",
        type=str,
        default="flat",
        choices=["paper", "flat"],
        help="Class-weighted prototype distillation (CPD) mode in KISLoss (default: flat)",
    )

    # -------------------------------------------------------------------------
    # [6] Ours (CBA v2) Boundary & Error-Stream Hyperparameters
    # -------------------------------------------------------------------------
    g_cba = parser.add_argument_group("[6] Ours (CBA v2) Boundary & Error-Stream Hyperparameters")
    g_cba.add_argument("--lambda_a", type=float, default=0.02, help="CBA assimilation loss coefficient lambda_A (default: 0.02)")
    g_cba.add_argument("--beta", type=float, default=0.5, help="Error-stream relative weight beta (default: 0.5)")
    g_cba.add_argument(
        "--lambda_ratio_cap",
        type=float,
        default=0.0,
        help="If > 0, cap lambda_A per microbatch so lambda_A * ||g_CBA|| / ||g_base|| <= cap (0 = off)",
    )
    g_cba.add_argument("--m", type=float, default=0.2, help="Boundary contrastive margin m (default: 0.2)")
    g_cba.add_argument("--tau", type=float, default=0.1, help="Contrastive temperature tau (default: 0.1)")
    g_cba.add_argument("--r", type=int, default=2, help="Morphological boundary band radius r in pixels (default: 2)")
    g_cba.add_argument("--k_bg", type=int, default=512, help="Background pixel sampling budget per batch (default: 512)")
    g_cba.add_argument("--k_fg", type=int, default=512, help="Foreground organ pixel sampling budget per class (default: 512)")
    g_cba.add_argument("--cache_momentum", type=float, default=0.9, help="Prototype EMA cache momentum (default: 0.9)")
    g_cba.add_argument("--gating_type", type=str, default="soft", help="Boundary directional gating type (default: soft)")
    g_cba.add_argument(
        "--error_gating_type",
        type=str,
        default="none",
        help="Error-stream gating type (default: none; ce_hard requires none)",
    )
    g_cba.add_argument(
        "--cba_feature",
        type=str,
        default="centered",
        choices=["raw", "centered"],
        help="Feature space for CBA: 'centered' (logits minus per-pixel class mean) or 'raw'",
    )
    g_cba.add_argument(
        "--gate_ref",
        type=str,
        default="base",
        choices=["ce", "base"],
        help="Directional gate reference gradient: 'base' (full base loss grad) or 'ce' (CE-to-GT grad)",
    )
    g_cba.add_argument("--grad_log_every", type=int, default=50, help="Log CBA gradient diagnostics every N steps (0 disables)")
    g_cba.add_argument("--boundary_use_cache", action="store_true", help="Boundary stream uses EMA cache prototypes")
    g_cba.add_argument(
        "--boundary_fallback_cache",
        action="store_true",
        help="Boundary stream uses batch prototypes when present and falls back to EMA cache when absent",
    )
    g_cba.add_argument(
        "--boundary_proto_source",
        type=str,
        default=None,
        help="Explicit boundary prototype source: 'batch', 'fallback', or 'cache'",
    )
    g_cba.add_argument("--use_cache_for_all", action="store_true", help="Both boundary and error streams use cache prototypes")
    g_cba.add_argument(
        "--use_curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable curriculum warmup/ramp for lambda_A (--no-use_curriculum to disable)",
    )
    g_cba.add_argument("--no_curriculum", action="store_false", dest="use_curriculum", help="Alias for --no-use_curriculum")
    g_cba.add_argument("--curriculum_warmup", type=int, default=20, help="Curriculum warmup epochs where lambda_A = 0 (default: 20)")
    g_cba.add_argument("--curriculum_ramp", type=int, default=30, help="Curriculum linear ramp epochs from 0 to lambda_A (default: 30)")
    g_cba.add_argument("--state_json", type=str, default=None, help="Path to tune_state.json for loading tuned Ours parameters")
    g_cba.add_argument(
        "--make_warmup",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="In reproduce mode, create the shared 20-epoch warmup checkpoint if absent (default: True for kis / non-synapse)",
    )
    g_cba.add_argument(
        "--wait_min",
        type=float,
        default=240.0,
        help="Minutes to wait for shared warmup checkpoint in reproduce mode when --no-make_warmup is set",
    )

    # -------------------------------------------------------------------------
    # [7] Baseline KD Hyperparameters (IAD, LSKD, SinKD, DIST+)
    # -------------------------------------------------------------------------
    g_kd = parser.add_argument_group("[7] Baseline KD Hyperparameters (IAD, LSKD, SinKD, DIST+)")
    g_kd.add_argument(
        "--kis_stage_epochs",
        type=int,
        default=200,
        help="IAD Stage-1 KIS epochs before transitioning to Stage-2 KAS (default: 200)",
    )
    g_kd.add_argument("--kas_weight", type=float, default=0.1, help="IAD Stage-2 KAS loss weight (default: 0.1, Paper Eq. 8)")
    g_kd.add_argument("--kas_temp", type=float, default=1.0, help="IAD Stage-2 KAS CSS temperature (default: 1.0)")
    g_kd.add_argument("--lskd_tau", type=float, default=2.0, help="LSKD standardization temperature tau (default: 2.0)")
    g_kd.add_argument("--sinkd_lam", type=float, default=10.0, help="SinKD Sinkhorn regularization lambda (default: 10.0)")
    g_kd.add_argument("--sinkd_iters", type=int, default=5, help="SinKD Sinkhorn-Knopp iterations (default: 5)")
    g_kd.add_argument("--dist_beta", type=float, default=1.0, help="DIST+ inter-class correlation weight beta (default: 1.0)")
    g_kd.add_argument("--dist_gamma", type=float, default=1.0, help="DIST+ intra-class correlation weight gamma (default: 1.0)")
    g_kd.add_argument("--dist_tau", type=float, default=1.0, help="DIST+ softmax temperature tau (default: 1.0)")

    # -------------------------------------------------------------------------
    # [8] Far-FP & Experimental Error-Set Options
    # -------------------------------------------------------------------------
    g_far = parser.add_argument_group("[8] Far-FP & Experimental Error-Set Options")
    g_far.add_argument("--lambda_far", type=float, default=0.0, help="Weight for Far-FP penalty loss L_far (default: 0.0)")
    g_far.add_argument("--far_threshold", type=float, default=20.0, help="Distance threshold in voxels for L_far (default: 20.0)")
    g_far.add_argument(
        "--far_mode",
        type=str,
        default="fraction",
        choices=["fraction", "bce", "focal", "combined"],
        help="Penalty mode for FarFPLoss (default: fraction)",
    )
    g_far.add_argument("--far_ref_size", type=float, default=1000.0, help="Reference organ size in voxels for FarFPLoss")
    g_far.add_argument(
        "--far_penalty_absent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Penalize predictions on slices where organ is absent in FarFPLoss",
    )
    g_far.add_argument(
        "--fp_by_pred_class",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Group background FP errors by predicted organ class (default: False)",
    )
    g_far.add_argument(
        "--fp_exclude_boundary",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exclude r<=2 boundary transition band from background FP error set (default: False)",
    )
    g_far.add_argument(
        "--empty_slice_suppression",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Top-K empty-slice focal suppression for absent organs (default: False)",
    )
    g_far.add_argument("--empty_slice_topk", type=int, default=64, help="Top-K pixels per absent (slice, organ) pair")
    g_far.add_argument("--empty_slice_prob_thresh", type=float, default=0.05, help="Probability threshold tau_p on absent slices")
    g_far.add_argument("--empty_slice_focal_gamma", type=float, default=2.0, help="Focal exponent gamma for empty-slice suppression")
    g_far.add_argument("--empty_slice_weight", type=float, default=1.0, help="Relative weight of L_EmptySlice inside L_E")

    # -------------------------------------------------------------------------
    # [9] Checkpoints, Output Directories, Drive Sync & Evaluation
    # -------------------------------------------------------------------------
    g_io = parser.add_argument_group("[9] Checkpoints, Output Directories, Drive Sync & Evaluation")
    g_io.add_argument("--teacher_ckpt", type=str, default="teacher_weights/epoch_299.pth", help="Teacher LoRA/model checkpoint")
    g_io.add_argument(
        "--teacher_base_ckpt",
        type=str,
        default="data/weights/sam_vit_l_0b3195.pth",
        help="Teacher SAM ViT-L base checkpoint",
    )
    g_io.add_argument(
        "--student_ckpt",
        type=str,
        default=None,
        help="Initial student checkpoint (default: None for training from scratch with ImageNet EfficientNet-B1)",
    )
    g_io.add_argument(
        "--init_ckpt",
        type=str,
        default=None,
        help="Warm initialization checkpoint (restores student, EMA student, and prototype cache without resuming epoch)",
    )
    g_io.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint path to evaluate in '--mode eval' (falls back to --student_ckpt or --teacher_ckpt)",
    )
    g_io.add_argument(
        "--eval_role",
        type=str,
        default="student",
        choices=["student", "teacher"],
        help="Model role to build and evaluate in '--mode eval' (default: student)",
    )
    g_io.add_argument(
        "--eval_output",
        type=str,
        default=None,
        help="Path to write evaluation JSON output in '--mode eval' (default: <output_dir>/eval_<role>_<dataset>.json)",
    )
    g_io.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path to resume training from, or 'auto' (latest local/Drive checkpoint, else fresh start)",
    )
    g_io.add_argument("--output_dir", type=str, default="runs/cba_v2", help="Local output directory for train/eval mode")
    g_io.add_argument("--runs_dir", type=str, default="runs/main", help="Local runs root directory for reproduce/tune mode")
    g_io.add_argument(
        "--drive_dir",
        type=str,
        default=None,
        help="Google Drive root folder for reproduce/tune mode, e.g. /content/drive/MyDrive/IAD/main",
    )
    g_io.add_argument(
        "--drive_sync_dir",
        type=str,
        default=None,
        help="Google Drive folder for per-run checkpoint syncing in train mode",
    )
    g_io.add_argument("--drive_sync_every", type=int, default=5, help="Drive sync period in epochs (default: 5)")
    g_io.add_argument("--save_freq", type=int, default=25, help="Permanent checkpoint save frequency in epochs (default: 25)")
    g_io.add_argument(
        "--save_epochs",
        type=str,
        default="20",
        help="Comma-separated epochs that always save a permanent checkpoint (default: '20' for warmup fork)",
    )
    g_io.add_argument(
        "--eval_freq",
        type=int,
        default=0,
        help="3D volume / benchmark evaluation period in epochs (0 = final epoch only, default: 0)",
    )
    g_io.add_argument("--skip_eval", action="store_true", help="Skip all evaluations including final epoch (used for warmup)")
    g_io.add_argument("--eval_max_cases", type=int, default=0, help="Limit test volumes/cases per evaluation (0 = all)")
    g_io.add_argument(
        "--save_final_preds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save 3D/2D prediction arrays from the final evaluation (--no-save_final_preds to disable)",
    )

    # -------------------------------------------------------------------------
    # [10] Hyperparameter Tuning & Code Packaging
    # -------------------------------------------------------------------------
    g_aux = parser.add_argument_group("[10] Hyperparameter Tuning & Code Packaging")
    g_aux.add_argument(
        "--tune_action",
        type=str,
        default="run",
        choices=["plan", "run", "score"],
        help="Action for '--mode tune': 'plan' (cost estimate), 'run' (execute stage), or 'score' (select params)",
    )
    g_aux.add_argument(
        "--tune_stage",
        type=str,
        default="A",
        choices=["warmup", "ref", "A", "B", "C"],
        help="Tuning stage for '--mode tune' (default: A)",
    )
    g_aux.add_argument("--tune_L", type=int, default=60, help="Schedule length L in epochs for '--mode tune' (default: 60)")
    g_aux.add_argument(
        "--bundle",
        type=str,
        default="cba_v2",
        choices=["cba_v2", "cba_v2_arm5", "ours_curriculum", "ours_300ep", "cba_pilot"],
        help="Bundle target for '--mode package' (default: cba_v2)",
    )
    g_aux.add_argument(
        "--regen_notebooks",
        action="store_true",
        help="Regenerate Colab notebooks before packaging in '--mode package'",
    )

    return parser


def _explicit_cli_dests(parser: argparse.ArgumentParser, raw_argv: Sequence[str]) -> set:
    """Return the set of parser destination names explicitly provided on the CLI."""
    opt_to_dest: Dict[str, str] = {}
    for action in parser._actions:
        for opt in action.option_strings:
            opt_to_dest[opt] = action.dest
    explicit: set = set()
    for tok in raw_argv:
        if not str(tok).startswith("-"):
            continue
        opt = str(tok).split("=", 1)[0]
        if opt in opt_to_dest:
            explicit.add(opt_to_dest[opt])
    return explicit


_REPRODUCE_CONTROL_DESTS = {
    "mode", "eval_only", "dry_run", "list_methods", "list_datasets",
    "method", "arm", "dataset", "seed", "seeds", "epochs", "sched_epochs",
    "runs_dir", "drive_dir", "drive_sync_dir", "output_dir", "resume",
    "eval_freq", "wait_min", "make_warmup", "kis_stage_epochs",
    "state_json", "lambda_ratio_cap",
    "lambda_a", "tau", "m", "r", "beta",
    "ckpt", "eval_role", "eval_output",
    "tune_action", "tune_stage", "tune_L", "bundle", "regen_notebooks",
}


def _build_extra_reproduce_train_args(args: argparse.Namespace, explicit_dests: set) -> List[str]:
    """Collect explicitly provided train.py flags not already managed by main_common.py."""
    from experiments.cba_v2.train import build_parser as build_train_parser

    train_parser = build_train_parser()
    extra: List[str] = []
    seen_dests: set = set()
    for action in train_parser._actions:
        dest = action.dest
        if dest not in explicit_dests or dest in _REPRODUCE_CONTROL_DESTS or dest in seen_dests:
            continue
        seen_dests.add(dest)
        val = getattr(args, dest, None)
        if isinstance(action, argparse.BooleanOptionalAction):
            extra.append(f"--{dest}" if val else f"--no-{dest}")
        elif action.nargs == 0:
            if getattr(action, "const", None) is True and val is True:
                extra.append(action.option_strings[0])
            elif getattr(action, "const", None) is False and val is False:
                extra.append(action.option_strings[0])
        elif val is not None:
            extra.extend([action.option_strings[0], str(val)])
    return extra


def _aggregate_seed_summaries(seed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean and std across seeds for all scalar evaluation metrics."""
    summaries = [r["final_eval"] for r in seed_results if isinstance(r.get("final_eval"), dict)]
    if not summaries:
        return {
            "n_seeds": len(seed_results),
            "seeds": [r.get("seed") for r in seed_results],
            "metrics_mean": {},
            "metrics_std": {},
        }
    scalar_keys = [
        k for k, v in summaries[0].items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    mean_dict: Dict[str, float] = {}
    std_dict: Dict[str, float] = {}
    for k in scalar_keys:
        vals = [float(s[k]) for s in summaries if isinstance(s.get(k), (int, float))]
        if vals:
            mean_dict[k] = float(np.mean(vals))
            std_dict[k] = float(np.std(vals, ddof=0))
    return {
        "n_seeds": len(summaries),
        "seeds": [r.get("seed") for r in seed_results],
        "metrics_mean": mean_dict,
        "metrics_std": std_dict,
    }


def run_train_mode(args: argparse.Namespace, raw_argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Execute direct training (single or multi-seed, single or multi-dataset) via experiments/cba_v2/train.py."""
    import run_tuning as rt

    raw_tokens = list(raw_argv) if raw_argv is not None else []
    arm_name = resolve_effective_arm(args, raw_tokens)
    datasets = parse_datasets(args.dataset)
    seeds = parse_seeds(args.seed, args.seeds)

    # Optionally load tuned state parameters if --state_json was provided and not explicitly overridden on CLI
    if args.state_json:
        state = rt.load_state(args.state_json)
        for key, flag in rt.STATE_TO_FLAG.items():
            if not any(t == flag or t.startswith(f"{flag}=") for t in raw_tokens):
                if key in state:
                    setattr(args, key, state[key])

    if args.drive_dir and not args.drive_sync_dir:
        args.drive_sync_dir = args.drive_dir

    multi_run = len(datasets) > 1 or len(seeds) > 1
    planned_runs: List[Dict[str, Any]] = []
    for ds in datasets:
        for seed in seeds:
            run_args = copy.deepcopy(args)
            run_args.dataset = ds
            run_args.arm = arm_name
            run_args.seed = seed
            if len(datasets) > 1:
                run_args.output_dir = os.path.join(args.output_dir, ds)
                if run_args.drive_sync_dir:
                    run_args.drive_sync_dir = os.path.join(run_args.drive_sync_dir, ds)
            if len(seeds) > 1:
                run_args.output_dir = os.path.join(run_args.output_dir, f"seed_{seed}")
                if run_args.drive_sync_dir:
                    run_args.drive_sync_dir = f"{run_args.drive_sync_dir}_s{seed}"
            apply_dataset_defaults(run_args)
            planned_runs.append(
                {
                    "dataset": ds,
                    "arm": arm_name,
                    "seed": seed,
                    "epochs": run_args.epochs,
                    "output_dir": run_args.output_dir,
                    "drive_sync_dir": run_args.drive_sync_dir,
                    "args": run_args,
                }
            )

    if args.dry_run:
        for item in planned_runs:
            print(
                f"[dry_run:train] dataset={item['dataset']} arm={item['arm']} "
                f"seed={item['seed']} epochs={item['epochs']} output_dir={item['output_dir']}",
                flush=True,
            )
        return {
            "mode": "train",
            "dry_run": True,
            "arm_name": arm_name,
            "datasets": datasets,
            "seeds": seeds,
            "runs": [
                {k: v for k, v in item.items() if k != "args"}
                for item in planned_runs
            ],
        }

    if not multi_run:
        return run_pilot(planned_runs[0]["args"])

    results: List[Dict[str, Any]] = []
    for item in planned_runs:
        res = run_pilot(item["args"])
        res["seed"] = item["seed"]
        res["dataset"] = item["dataset"]
        res["output_dir"] = item["output_dir"]
        results.append(res)

    by_dataset: Dict[str, Dict[str, Any]] = {}
    for ds in datasets:
        ds_results = [r for r in results if r.get("dataset") == ds]
        ds_agg = _aggregate_seed_summaries(ds_results)
        by_dataset[ds] = ds_agg
        if len(datasets) > 1 and len(seeds) > 1:
            ds_summary_path = os.path.join(args.output_dir, ds, "multiseed_summary.json")
            atomic_json(
                ds_summary_path,
                {
                    "arm_name": arm_name,
                    "datasets": [ds],
                    "seeds": seeds,
                    "aggregate": ds_agg,
                },
            )

    if len(datasets) == 1:
        agg = dict(by_dataset[datasets[0]])
        agg["by_dataset"] = by_dataset
    else:
        agg = {
            "n_datasets": len(datasets),
            "n_seeds": len(seeds),
            "seeds": seeds,
            "by_dataset": by_dataset,
        }

    summary_path = os.path.join(args.output_dir, "multiseed_summary.json")
    atomic_json(
        summary_path,
        {
            "arm_name": arm_name,
            "datasets": datasets,
            "seeds": seeds,
            "aggregate": agg,
            "by_dataset": by_dataset,
        },
    )
    print(f"[Multi-Seed Summary] Saved aggregated metrics across {len(results)} runs to: {summary_path}", flush=True)
    return {
        "mode": "train",
        "arm_name": arm_name,
        "datasets": datasets,
        "seeds": seeds,
        "runs": results,
        "aggregate": agg,
        "by_dataset": by_dataset,
        "summary_path": summary_path,
    }


def run_reproduce_mode(args: argparse.Namespace, raw_argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Execute official 300-epoch benchmark reproduction launcher via scripts/main_common.py."""
    import main_common as mc

    raw_tokens = list(raw_argv) if raw_argv is not None else []
    explicit_dests = _explicit_cli_dests(build_parser(), raw_tokens)
    extra_train_args = _build_extra_reproduce_train_args(args, explicit_dests)
    datasets = parse_datasets(args.dataset)
    seeds = parse_seeds(args.seed, args.seeds)
    method_jobs = parse_methods_for_reproduce(args.method, args.arm, raw_tokens)

    drive_root = args.drive_dir or args.drive_sync_dir
    commands: List[List[str]] = []
    job_records: List[Dict[str, Any]] = []

    for ds in datasets:
        ds_runs_dir = args.runs_dir
        if len(datasets) > 1 and args.runs_dir == "runs/main" and ds != "synapse":
            ds_runs_dir = f"runs/{ds}_main"
        ds_drive_dir = drive_root
        if len(datasets) > 1 and drive_root and ds != "synapse":
            ds_drive_dir = f"{drive_root.rstrip('/\\')}_{ds}"

        for kind, arm_short in method_jobs:
            for seed in seeds:
                mc_argv: List[str] = [
                    "--dataset", ds,
                    "--epochs", str(args.epochs),
                    "--runs_dir", str(ds_runs_dir),
                    "--seed", str(seed),
                    "--eval_freq", str(args.eval_freq),
                    "--wait_min", str(args.wait_min),
                ]
                if ds_drive_dir:
                    mc_argv += ["--drive_dir", str(ds_drive_dir)]
                if args.make_warmup is True:
                    mc_argv.append("--make_warmup")
                elif args.make_warmup is False:
                    mc_argv.append("--no-make_warmup")
                if args.dry_run:
                    mc_argv.append("--dry_run")
                if kind == "iad" and args.kis_stage_epochs != 200:
                    mc_argv += ["--kis_stage_epochs", str(args.kis_stage_epochs)]
                if kind == "ours":
                    eff_arm_short = arm_short or "v5"
                    mc_argv += ["--arm", eff_arm_short]
                    if args.state_json:
                        mc_argv += ["--state_json", str(args.state_json)]
                    if args.lambda_ratio_cap > 0.0:
                        mc_argv += ["--lambda_ratio_cap", str(args.lambda_ratio_cap)]
                    for p_key in mc.OURS_PARAMS.get(eff_arm_short, ()):
                        if p_key in explicit_dests:
                            mc_argv += [f"--{p_key}", str(getattr(args, p_key))]

                cmd = mc.run_main(kind, mc_argv, extra_train_args=extra_train_args or None)
                if cmd:
                    commands.append(cmd)
                job_records.append(
                    {
                        "dataset": ds,
                        "kind": kind,
                        "arm": arm_short or kind,
                        "seed": seed,
                        "command": cmd,
                    }
                )

    return {
        "mode": "reproduce",
        "dry_run": bool(args.dry_run),
        "datasets": datasets,
        "seeds": seeds,
        "methods": [f"{k}:{a}" if a else k for k, a in method_jobs],
        "commands": commands,
        "jobs": job_records,
    }


def run_eval_mode(args: argparse.Namespace, raw_argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Run standalone evaluation of a Student or Teacher checkpoint on Synapse, ACDC, or Polyp."""
    from experiments.cba_v2.train import resolve_file

    raw_tokens = list(raw_argv) if raw_argv is not None else []
    explicit_dests = _explicit_cli_dests(build_parser(), raw_tokens)
    datasets = parse_datasets(args.dataset)
    if len(datasets) > 1:
        raise ValueError("--mode eval requires a single dataset (--dataset synapse, acdc, or polyp).")

    eval_args = copy.deepcopy(args)
    eval_args.dataset = datasets[0]
    ds = apply_dataset_defaults(eval_args)
    set_seed(eval_args.seed)
    role = str(eval_args.eval_role or "student").strip().lower()
    device = torch.device("cuda" if torch.cuda.is_available() and not eval_args.cpu else "cpu")
    use_amp = (device.type == "cuda" and not eval_args.no_amp)

    ckpt_path = eval_args.ckpt
    if not ckpt_path:
        if role == "teacher":
            ckpt_path = eval_args.teacher_ckpt
        else:
            ckpt_path = eval_args.student_ckpt or eval_args.init_ckpt or eval_args.resume

    out_json = eval_args.eval_output or os.path.join(eval_args.output_dir, f"eval_{role}_{ds}.json")
    preds_dir = os.path.join(eval_args.output_dir, f"preds_{role}_{ds}") if eval_args.save_final_preds else None

    if eval_args.dry_run:
        print(
            f"[dry_run:eval] dataset={ds} role={role} ckpt={ckpt_path} "
            f"test_dir={eval_args.test_dir} eval_output={out_json}",
            flush=True,
        )
        return {
            "mode": "eval",
            "dry_run": True,
            "dataset": ds,
            "role": role,
            "ckpt": ckpt_path,
            "eval_output": out_json,
        }

    if eval_args.smoke_test:
        model = SimpleConvStudent(in_channels=3, num_classes=eval_args.num_classes).to(device)
        resolved_ckpt = resolve_file(ckpt_path) if ckpt_path else None
        if "ckpt" in explicit_dests and ckpt_path and not resolved_ckpt:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        if resolved_ckpt and os.path.isfile(resolved_ckpt):
            state = torch.load(resolved_ckpt, map_location=device, weights_only=False)
            if isinstance(state, dict) and "student" in state:
                state = state["student"]
            model.load_state_dict(state, strict=False)
        model.eval()

        tmp_eval_root = None
        test_dir = eval_args.test_dir
        test_list = eval_args.test_list
        if not os.path.isdir(os.path.join(REPO_ROOT, str(test_dir))) and not os.path.isdir(str(test_dir)):
            tmp_eval_root = tempfile.mkdtemp(prefix=f"main_eval_smoke_{ds}_")
            vol_dir = os.path.join(tmp_eval_root, "test_vol_h5")
            os.makedirs(vol_dir, exist_ok=True)
            sz = int(eval_args.img_size or 64)
            label = np.zeros((2, sz, sz), dtype=np.uint8)
            label[:, sz // 4 : sz // 2, sz // 4 : sz // 2] = 1
            with h5py.File(os.path.join(vol_dir, "case0001.npy.h5"), "w") as hf:
                hf.create_dataset("image", data=(label > 0).astype(np.float32))
                hf.create_dataset("label", data=label)
            test_list = os.path.join(tmp_eval_root, "test_vol.txt")
            with open(test_list, "w", encoding="utf-8") as fh:
                fh.write("case0001\n")
            test_dir = vol_dir
        try:
            eval_res = evaluate_volumes(
                student_net=model,
                test_dir=test_dir,
                test_list=test_list,
                device=device,
                batch_size=eval_args.batch_size,
                use_amp=False,
                max_cases=eval_args.eval_max_cases or 1,
                iad_root=str(REPO_ROOT),
                save_preds_dir=preds_dir,
                tag=f"eval_{role}",
                dataset="synapse" if tmp_eval_root else ds,
                img_size=eval_args.img_size,
            )
        finally:
            if tmp_eval_root:
                shutil.rmtree(tmp_eval_root, ignore_errors=True)
    else:
        if role == "teacher":
            model = build_frozen_teacher(
                iad_root=str(REPO_ROOT),
                teacher_ckpt=ckpt_path,
                base_ckpt=eval_args.teacher_base_ckpt,
                device=device,
                dataset=ds,
            )
        else:
            if not ckpt_path:
                raise ValueError(
                    "--mode eval with --eval_role student requires a checkpoint via --ckpt (or --student_ckpt)."
                )
            model = build_student(
                iad_root=str(REPO_ROOT),
                student_ckpt=ckpt_path,
                device=device,
                strict_load=True,
                dataset=ds,
            )
            model.eval()

        eval_res = evaluate_volumes(
            student_net=model,
            test_dir=eval_args.test_dir,
            test_list=eval_args.test_list,
            device=device,
            batch_size=eval_args.batch_size,
            use_amp=use_amp,
            max_cases=eval_args.eval_max_cases or None,
            iad_root=str(REPO_ROOT),
            save_preds_dir=preds_dir,
            tag=f"eval_{role}",
            dataset=ds,
            img_size=eval_args.img_size,
        )

    summary = summarize_eval(eval_res)
    atomic_json(out_json, eval_res)
    print(
        f"[Eval Summary ({ds}, role={role})] "
        f"Dice raw/LCC: {summary['eval_dice_raw']*100:.2f}/{summary['eval_dice_lcc']*100:.2f}% | "
        f"HD95 raw/LCC: {summary['eval_hd95_raw']:.2f}/{summary['eval_hd95_lcc']:.2f} {summary['distance_unit']} | "
        f"NSD raw: {summary['eval_nsd_raw']*100:.2f}% | Saved to: {out_json}",
        flush=True,
    )
    return {
        "mode": "eval",
        "dataset": ds,
        "role": role,
        "ckpt": ckpt_path,
        "eval_output": out_json,
        "summary": summary,
        "eval": eval_res,
    }


def run_tune_mode(args: argparse.Namespace) -> Dict[str, Any]:
    """Dispatch hyperparameter tuning commands to scripts/run_tuning.py."""
    import run_tuning as rt

    action = args.tune_action
    if action == "plan":
        cost = rt.plan_cost(epoch_sec=100.0, eval_sec=60.0, L=args.tune_L)
        print(json.dumps(cost, indent=2), flush=True)
        return {"mode": "tune", "action": "plan", "cost": cost}

    tune_argv: List[str] = [
        action,
        "--stage", str(args.tune_stage),
        "--L", str(args.tune_L),
        "--runs_dir", str(args.runs_dir),
    ]
    if args.drive_dir:
        tune_argv += ["--drive_dir", str(args.drive_dir)]
    if args.state_json:
        tune_argv += ["--state_json", str(args.state_json)]
    if action == "run" and args.dry_run:
        tune_argv.append("--dry_run")
    out = rt.main(tune_argv)
    return {"mode": "tune", "action": action, "stage": args.tune_stage, "argv": tune_argv, "out": out}


def run_package_mode(args: argparse.Namespace) -> Dict[str, Any]:
    """Package self-contained Colab bundle (and optionally regenerate notebooks)."""
    import zip_code

    if args.dry_run:
        print(f"[dry_run:package] bundle={args.bundle} regen_notebooks={args.regen_notebooks}", flush=True)
        return {"mode": "package", "dry_run": True, "bundle": args.bundle, "regen_notebooks": bool(args.regen_notebooks)}

    if args.regen_notebooks:
        import make_main_notebook
        import prepare_5methods_notebooks
        import prepare_acdc_polyp_3notebooks

        make_main_notebook.main()
        prepare_5methods_notebooks.main()
        prepare_acdc_polyp_3notebooks.main()

    bundle_map = {
        "cba_v2": ("iad_cba_v2_code.zip", zip_code.CBA_V2_TARGETS, "run_cba_v2_colab.ipynb"),
        "cba_v2_arm5": ("iad_cba_v2_arm5_code.zip", zip_code.CBA_V2_ARM5_TARGETS, "run_cba_v2_arm5_colab.ipynb"),
        "ours_curriculum": ("iad_ours_curriculum_code.zip", zip_code.OURS_CURRICULUM_TARGETS, "run_ours_curriculum_colab.ipynb"),
        "ours_300ep": ("iad_ours_300ep_code.zip", zip_code.OURS_300EP_TARGETS, "run_ours_300ep_colab.ipynb"),
        "cba_pilot": ("iad_cba_pilot_code.zip", zip_code.CBA_PILOT_TARGETS, "run_cba_pilot_colab.ipynb"),
    }
    zip_name, targets, nb_name = bundle_map[args.bundle]
    out_zip = zip_code.build_bundle(zip_name=zip_name, targets=targets, notebook_name=nb_name)
    return {"mode": "package", "bundle": args.bundle, "output_zip": str(out_zip)}


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(raw_argv)

    if args.list_methods:
        print_methods_table()
        return {"mode": "list_methods", "methods": list(METHOD_SPECS.keys())}

    if args.list_datasets:
        print_datasets_table()
        return {"mode": "list_datasets", "datasets": list(CANONICAL_DATASETS)}

    mode = "eval" if args.eval_only else args.mode
    if mode == "train":
        return run_train_mode(args, raw_argv)
    if mode == "reproduce":
        return run_reproduce_mode(args, raw_argv)
    if mode == "eval":
        return run_eval_mode(args, raw_argv)
    if mode == "tune":
        return run_tune_mode(args)
    if mode == "package":
        return run_package_mode(args)
    raise ValueError(f"Unsupported mode: {mode!r}")


if __name__ == "__main__":
    main()

