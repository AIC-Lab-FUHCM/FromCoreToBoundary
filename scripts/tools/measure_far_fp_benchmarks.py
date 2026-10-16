#!/usr/bin/env python3
"""
tools/measure_far_fp_benchmarks.py

Direct Far False-Positive (far-FP) Benchmark Evaluation across BTCV test cases.

Definition:
    far-FP voxels: Number of predicted voxels of class c with 3D Euclidean distance > 20 voxels
    to the ground truth G_c (or all predicted voxels if G_c is absent in the patient).

Computes:
    - Raw far-FP voxels and post-LCC far-FP voxels across 12 BTCV test cases x 8 organs (96 units).
    - Near-FP voxels (0 < dist <= 20 voxels) and True Positive (TP) voxels.
    - Clean Macro Dice (raw and LCC), HD95 (raw and LCC), and Surface NSD (2mm).
    - Cross-checkpoint comparative benchmarks across canonical models:
        * kis300:      I:/My Drive/IAD/main/kis300/checkpoint_epoch_300.pth
        * ours_v5:     I:/My Drive/IAD/main/ours_v5/checkpoint_epoch_300.pth
        * ours_v5cap:  I:/My Drive/IAD/main/ours_v5cap/checkpoint_epoch_300.pth
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn
from scipy.ndimage import zoom

# Resolve repository paths
_HERE = Path(__file__).resolve().parent
_REPO_PATH = _HERE.parent.parent if _HERE.parent.name == "scripts" else _HERE.parent
REPO_ROOT = str(_REPO_PATH)
for _p in (_REPO_PATH, _REPO_PATH / "models", _REPO_PATH / "scripts", _REPO_PATH / "scripts" / "tools"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.cba_v2.data import find_test_vol_dir, resolve_list_file, is_valid_case_list
from experiments.cba_v2.evaluation import (
    CLASS_NAMES,
    ALL_CLASS_NAMES,
    apply_3d_lcc_per_class,
    compute_volume_metrics_per_class,
    get_case_voxelspacing,
)
from experiments.cba_v2.models import build_student

# Default canonical benchmark checkpoints
CANONICAL_CHECKPOINTS = {
    "kis300": r"I:\My Drive\IAD\main\kis300\checkpoint_epoch_300.pth",
    "ours_v5": r"I:\My Drive\IAD\main\ours_v5\checkpoint_epoch_300.pth",
    "ours_v5cap": r"I:\My Drive\IAD\main\ours_v5cap\checkpoint_epoch_300.pth",
}


def resolve_checkpoint_path(name: str, requested: Optional[str] = None) -> Optional[str]:
    """Resolve checkpoint path across Windows Drive I:, Colab Drive, or local storage."""
    cands = []
    if requested:
        cands.append(requested)
    win_p = CANONICAL_CHECKPOINTS.get(name, "")
    if win_p:
        cands.append(win_p)
    colab_p1 = f"/content/drive/MyDrive/IAD/main/{name}/checkpoint_epoch_300.pth"
    colab_p2 = f"/content/drive/My Drive/IAD/main/{name}/checkpoint_epoch_300.pth"
    colab_p3 = f"/content/drive/MyDrive/IAD/{name}/checkpoint_epoch_300.pth"
    local_p1 = os.path.join(REPO_ROOT, "runs", "main", name, "checkpoint_epoch_300.pth")
    local_p2 = os.path.join(str(Path(REPO_ROOT).parent), "runs", "main", name, "checkpoint_epoch_300.pth")
    cands.extend([colab_p1, colab_p2, colab_p3, local_p1, local_p2])
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return requested or win_p


def compute_far_fp_voxels(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    distance_threshold: float = 20.0,
) -> Dict[str, int]:
    """Compute exact far-FP voxels (dist > distance_threshold voxels to GT).

    Uses an exact ROI bounding-box Euclidean Distance Transform optimization:
    Any voxel outside bbox(GT, pad=int(distance_threshold)+1) is guaranteed to have
    Euclidean distance > distance_threshold. Inside bbox, exact EDT is computed.
    Mathematically 100% equivalent to global EDT on the entire volume.
    """
    p_bin = (pred_mask > 0)
    g_bin = (gt_mask > 0)

    p_sum = int(p_bin.sum())
    g_sum = int(g_bin.sum())

    if p_sum == 0:
        return {
            "far_fp": 0,
            "near_fp": 0,
            "total_fp": 0,
            "tp": 0,
            "p_sum": 0,
            "g_sum": g_sum,
        }

    if g_sum == 0:
        # GT absent: every predicted voxel has infinite distance to GT
        return {
            "far_fp": p_sum,
            "near_fp": 0,
            "total_fp": p_sum,
            "tp": 0,
            "p_sum": p_sum,
            "g_sum": 0,
        }

    tp = int(np.logical_and(p_bin, g_bin).sum())
    total_fp = p_sum - tp

    if total_fp == 0:
        return {
            "far_fp": 0,
            "near_fp": 0,
            "total_fp": 0,
            "tp": tp,
            "p_sum": p_sum,
            "g_sum": g_sum,
        }

    # Bounding box of GT padded by distance_threshold + 1
    where_g = np.where(g_bin)
    pad = int(np.ceil(distance_threshold)) + 1
    slices = tuple(
        slice(max(0, int(w.min()) - pad), min(g_bin.shape[ax], int(w.max()) + pad + 1))
        for ax, w in enumerate(where_g)
    )

    sub_g = g_bin[slices]
    sub_p = p_bin[slices]

    # EDT inside cropped ROI
    dt_sub = ndi.distance_transform_edt(~sub_g)
    far_inside = int((sub_p & (dt_sub > distance_threshold)).sum())

    # Voxels outside ROI bounding box are strictly > distance_threshold from GT
    p_inside = int(sub_p.sum())
    far_outside = p_sum - p_inside
    far_fp = far_inside + far_outside
    near_fp = total_fp - far_fp

    return {
        "far_fp": far_fp,
        "near_fp": near_fp,
        "total_fp": total_fp,
        "tp": tp,
        "p_sum": p_sum,
        "g_sum": g_sum,
    }


@torch.no_grad()
def infer_case_volume(
    student_net: nn.Module,
    image: np.ndarray,
    device: torch.device,
    batch_size: int = 8,
    use_amp: bool = True,
) -> np.ndarray:
    """Run batched slice inference on 3D CT volume."""
    student_net.eval()
    num_slices = image.shape[0]
    preds = []
    curr_bs = batch_size
    slice_idx = 0

    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    while slice_idx < num_slices:
        end_idx = min(slice_idx + curr_bs, num_slices)
        chunk = image[slice_idx:end_idx]
        h, w = chunk.shape[1], chunk.shape[2]
        if h != 512 or w != 512:
            zoomed = [zoom(chunk[s], (512 / h, 512 / w), order=3) for s in range(chunk.shape[0])]
            chunk = np.stack(zoomed, axis=0)

        x_in = torch.from_numpy(chunk).unsqueeze(1).repeat(1, 3, 1, 1).float().to(device)
        try:
            if use_amp and ("cuda" in str(device)):
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    out = student_net(x_in, True)
            else:
                out = student_net(x_in, True)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            pred_chunk = torch.argmax(logits, dim=1).to(torch.uint8).cpu().numpy()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as oom_err:
            if "out of memory" in str(oom_err).lower() and curr_bs > 1:
                curr_bs = max(1, curr_bs // 2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise oom_err
        finally:
            del x_in, out, logits

        if h != 512 or w != 512:
            zoomed_back = [zoom(pred_chunk[s], (h / 512, w / 512), order=0) for s in range(pred_chunk.shape[0])]
            pred_chunk = np.stack(zoomed_back, axis=0)

        preds.append(pred_chunk)
        slice_idx = end_idx

    pred_vol = np.concatenate(preds, axis=0)
    del preds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred_vol


def evaluate_checkpoint_far_fp(
    ckpt_path: str,
    model_name: str,
    test_dir: str,
    cases: List[str],
    device: torch.device,
    cache_dir: Optional[str] = None,
    distance_threshold: float = 20.0,
    force_infer: bool = False,
    batch_size: int = 8,
) -> Dict[str, Union[dict, list, float, int]]:
    """Evaluate a single checkpoint for far-FP and standard metrics across all test cases."""
    print(f"\n{'='*78}\nEvaluating {model_name} from: {ckpt_path}\n{'='*78}", flush=True)

    if cache_dir:
        os.makedirs(os.path.join(cache_dir, model_name), exist_ok=True)

    student_net = None

    unit_records = []
    case_records = []

    for ci, case in enumerate(cases):
        t0 = time.time()
        cand_paths = [
            os.path.join(test_dir, f"{case}.npy.h5"),
            os.path.join(test_dir, f"{case}.h5"),
        ]
        case_path = next((p for p in cand_paths if os.path.isfile(p)), None)
        if not case_path:
            print(f"Warning: case file not found for {case}, skipping.")
            continue

        raw_cache_file = os.path.join(cache_dir, model_name, f"{case}_pred.npy") if cache_dir else None
        lcc_cache_file = os.path.join(cache_dir, model_name, f"{case}_pred_lcc.npy") if cache_dir else None

        has_cache = (
            raw_cache_file and os.path.isfile(raw_cache_file)
            and lcc_cache_file and os.path.isfile(lcc_cache_file)
            and not force_infer
        )

        with h5py.File(case_path, "r") as f:
            image = f["image"][:]
            label = f["label"][:]
        while image.ndim > 3:
            image = image.squeeze(0)
        while label.ndim > 3:
            label = label.squeeze(0)

        if has_cache:
            pred_vol = np.load(raw_cache_file)
            pred_vol_lcc = np.load(lcc_cache_file)
        else:
            if student_net is None:
                student_net = build_student(REPO_ROOT, ckpt_path, device, strict_load=True)
            pred_vol = infer_case_volume(student_net, image, device, batch_size=batch_size)
            pred_vol_lcc, _ = apply_3d_lcc_per_class(pred_vol)

            if raw_cache_file and lcc_cache_file:
                np.save(raw_cache_file, pred_vol)
                np.save(lcc_cache_file, pred_vol_lcc)

        voxelspacing = get_case_voxelspacing(test_dir, case_path)

        case_dices_raw = []
        case_dices_lcc = []
        case_hds_raw = []
        case_hds_lcc = []
        case_far_fp_raw = 0
        case_far_fp_lcc = 0

        for c in range(1, 9):
            c_name = CLASS_NAMES[c]
            gt_c = (label == c)
            p_c_raw = (pred_vol == c)
            p_c_lcc = (pred_vol_lcc == c)

            # Standard metrics (Dice, HD95, NSD)
            res_raw = compute_volume_metrics_per_class(p_c_raw, gt_c, voxelspacing=voxelspacing)
            res_lcc = compute_volume_metrics_per_class(p_c_lcc, gt_c, voxelspacing=voxelspacing)

            # Far-FP metrics
            far_res_raw = compute_far_fp_voxels(p_c_raw, gt_c, distance_threshold=distance_threshold)
            far_res_lcc = compute_far_fp_voxels(p_c_lcc, gt_c, distance_threshold=distance_threshold)

            case_far_fp_raw += far_res_raw["far_fp"]
            case_far_fp_lcc += far_res_lcc["far_fp"]

            unit_info = {
                "case": case,
                "organ": c_name,
                "class_id": c,
                "gt_present": (not res_raw["gt_empty"]),
                "gt_voxels": far_res_raw["g_sum"],
                "pred_voxels_raw": far_res_raw["p_sum"],
                "pred_voxels_lcc": far_res_lcc["p_sum"],
                "tp_raw": far_res_raw["tp"],
                "tp_lcc": far_res_lcc["tp"],
                "far_fp_raw": far_res_raw["far_fp"],
                "far_fp_lcc": far_res_lcc["far_fp"],
                "near_fp_raw": far_res_raw["near_fp"],
                "near_fp_lcc": far_res_lcc["near_fp"],
                "total_fp_raw": far_res_raw["total_fp"],
                "total_fp_lcc": far_res_lcc["total_fp"],
                "dice_raw": res_raw["dice_clean"],
                "dice_lcc": res_lcc["dice_clean"],
                "hd95_raw": res_raw["hd95"],
                "hd95_lcc": res_lcc["hd95"],
                "nsd_raw": res_raw["surface_dice"],
                "nsd_lcc": res_lcc["surface_dice"],
            }
            unit_records.append(unit_info)

            if unit_info["gt_present"]:
                case_dices_raw.append(res_raw["dice_clean"])
                case_dices_lcc.append(res_lcc["dice_clean"])
                if not np.isnan(res_raw["hd95"]):
                    case_hds_raw.append(res_raw["hd95"])
                if not np.isnan(res_lcc["hd95"]):
                    case_hds_lcc.append(res_lcc["hd95"])

        elapsed = round(time.time() - t0, 2)
        mean_d_raw = np.mean(case_dices_raw) * 100 if case_dices_raw else 0.0
        mean_d_lcc = np.mean(case_dices_lcc) * 100 if case_dices_lcc else 0.0
        mean_hd_raw = np.mean(case_hds_raw) if case_hds_raw else float("nan")
        mean_hd_lcc = np.mean(case_hds_lcc) if case_hds_lcc else float("nan")

        print(
            f"[{ci+1:02d}/{len(cases):02d}] {case:8s} | "
            f"Dice Raw/LCC: {mean_d_raw:5.2f}% / {mean_d_lcc:5.2f}% | "
            f"HD95 Raw/LCC: {mean_hd_raw:5.2f} / {mean_hd_lcc:5.2f} mm | "
            f"Far-FP Raw/LCC: {case_far_fp_raw:7,d} / {case_far_fp_lcc:7,d} vox ({elapsed}s)",
            flush=True,
        )

        case_records.append({
            "case": case,
            "dice_raw_mean": mean_d_raw,
            "dice_lcc_mean": mean_d_lcc,
            "hd95_raw_mean": mean_hd_raw,
            "hd95_lcc_mean": mean_hd_lcc,
            "far_fp_raw": case_far_fp_raw,
            "far_fp_lcc": case_far_fp_lcc,
            "time_sec": elapsed,
        })

        del pred_vol, pred_vol_lcc, image, label
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if student_net is not None:
        del student_net
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate Metrics
    real_units = [u for u in unit_records if u["gt_present"]]

    # Macro organ metrics (mean of 8 organ averages across cases)
    per_organ_summary = {}
    macro_dices_raw = []
    macro_dices_lcc = []
    macro_hds_raw = []
    macro_hds_lcc = []
    macro_nsds_raw = []
    macro_nsds_lcc = []

    total_far_fp_raw = sum(u["far_fp_raw"] for u in unit_records)
    total_far_fp_lcc = sum(u["far_fp_lcc"] for u in unit_records)
    total_near_fp_raw = sum(u["near_fp_raw"] for u in unit_records)
    total_fp_raw = sum(u["total_fp_raw"] for u in unit_records)

    for c in range(1, 9):
        c_name = CLASS_NAMES[c]
        c_units_real = [u for u in real_units if u["class_id"] == c]
        c_units_all = [u for u in unit_records if u["class_id"] == c]

        c_d_raw = float(np.mean([u["dice_raw"] for u in c_units_real])) if c_units_real else float("nan")
        c_d_lcc = float(np.mean([u["dice_lcc"] for u in c_units_real])) if c_units_real else float("nan")
        c_n_raw = float(np.mean([u["nsd_raw"] for u in c_units_real])) if c_units_real else float("nan")
        c_n_lcc = float(np.mean([u["nsd_lcc"] for u in c_units_real])) if c_units_real else float("nan")

        v_hd_raw = [u["hd95_raw"] for u in c_units_real if not np.isnan(u["hd95_raw"])]
        v_hd_lcc = [u["hd95_lcc"] for u in c_units_real if not np.isnan(u["hd95_lcc"])]
        c_h_raw = float(np.mean(v_hd_raw)) if v_hd_raw else float("nan")
        c_h_lcc = float(np.mean(v_hd_lcc)) if v_hd_lcc else float("nan")

        c_far_raw = sum(u["far_fp_raw"] for u in c_units_all)
        c_far_lcc = sum(u["far_fp_lcc"] for u in c_units_all)
        c_near_raw = sum(u["near_fp_raw"] for u in c_units_all)
        c_tot_fp = sum(u["total_fp_raw"] for u in c_units_all)

        per_organ_summary[c_name] = {
            "dice_raw": c_d_raw,
            "dice_lcc": c_d_lcc,
            "hd95_raw": c_h_raw,
            "hd95_lcc": c_h_lcc,
            "nsd_raw": c_n_raw,
            "nsd_lcc": c_n_lcc,
            "far_fp_raw": c_far_raw,
            "far_fp_lcc": c_far_lcc,
            "near_fp_raw": c_near_raw,
            "total_fp_raw": c_tot_fp,
        }

        macro_dices_raw.append(c_d_raw)
        macro_dices_lcc.append(c_d_lcc)
        macro_nsds_raw.append(c_n_raw)
        macro_nsds_lcc.append(c_n_lcc)
        if not np.isnan(c_h_raw):
            macro_hds_raw.append(c_h_raw)
        if not np.isnan(c_h_lcc):
            macro_hds_lcc.append(c_h_lcc)

    overall = {
        "model_name": model_name,
        "checkpoint_path": ckpt_path,
        "total_units": len(unit_records),
        "primary_real_gt_units": len(real_units),
        "macro_dice_raw": float(np.mean(macro_dices_raw)) if macro_dices_raw else float("nan"),
        "macro_dice_lcc": float(np.mean(macro_dices_lcc)) if macro_dices_lcc else float("nan"),
        "macro_hd95_raw": float(np.mean(macro_hds_raw)) if macro_hds_raw else float("nan"),
        "macro_hd95_lcc": float(np.mean(macro_hds_lcc)) if macro_hds_lcc else float("nan"),
        "macro_nsd_raw": float(np.mean(macro_nsds_raw)) if macro_nsds_raw else float("nan"),
        "macro_nsd_lcc": float(np.mean(macro_nsds_lcc)) if macro_nsds_lcc else float("nan"),
        "total_far_fp_raw": total_far_fp_raw,
        "total_far_fp_lcc": total_far_fp_lcc,
        "total_near_fp_raw": total_near_fp_raw,
        "total_fp_raw": total_fp_raw,
        "mean_far_fp_per_case_raw": float(total_far_fp_raw / len(cases)) if cases else 0.0,
        "mean_far_fp_per_case_lcc": float(total_far_fp_lcc / len(cases)) if cases else 0.0,
        "far_fp_ratio_in_fp": float(total_far_fp_raw / total_fp_raw) if total_fp_raw > 0 else 0.0,
    }

    print(
        f"\n[{model_name} SUMMARY]\n"
        f"  Total far-FP Voxels (Raw): {total_far_fp_raw:10,d} vox ({overall['mean_far_fp_per_case_raw']:,.1f} vox/case)\n"
        f"  Total far-FP Voxels (LCC): {total_far_fp_lcc:10,d} vox ({overall['mean_far_fp_per_case_lcc']:,.1f} vox/case)\n"
        f"  Macro Dice Raw/LCC       : {overall['macro_dice_raw']*100:.2f}% / {overall['macro_dice_lcc']*100:.2f}%\n"
        f"  Macro HD95 Raw/LCC       : {overall['macro_hd95_raw']:.2f} mm / {overall['macro_hd95_lcc']:.2f} mm\n"
        f"  Macro NSD Raw (2mm)      : {overall['macro_nsd_raw']*100:.2f}%",
        flush=True,
    )

    return {
        "overall": overall,
        "per_organ": per_organ_summary,
        "cases": case_records,
        "units": unit_records,
    }


def format_markdown_comparison_table(results: Dict[str, dict]) -> str:
    """Generate clean comparison markdown tables across models."""
    lines = [
        "# Far False-Positive (far-FP) Benchmark Comparison Across 12 BTCV Test Cases x 8 Organs",
        "",
        "> **Metric Definition**: `far-FP voxels` = number of predicted voxels of class $c$ with 3D Euclidean distance $> 20$ voxels to true GT $G_c$ (isotropic voxel grid EDT).",
        "",
        "## 1. Overall Comparative Benchmark Matrix",
        "",
        "| Model | Total far-FP (Raw) | far-FP / Case (Raw) | Total far-FP (LCC) | Dice Raw (%) | Dice LCC (%) | HD95 Raw (mm) | HD95 LCC (mm) | NSD (2mm) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name, res in results.items():
        ov = res["overall"]
        lines.append(
            f"| **{name}** | {ov['total_far_fp_raw']:,d} | {ov['mean_far_fp_per_case_raw']:,.1f} | "
            f"{ov['total_far_fp_lcc']:,d} | {ov['macro_dice_raw']*100:.2f}% | {ov['macro_dice_lcc']*100:.2f}% | "
            f"{ov['macro_hd95_raw']:.2f} | {ov['macro_hd95_lcc']:.2f} | {ov['macro_nsd_raw']*100:.2f}% |"
        )

    lines.extend([
        "",
        "## 2. Organ-by-Organ Breakdown of far-FP Voxels (Raw)",
        "",
    ])

    header = "| Organ | " + " | ".join(f"{name} (far-FP)" for name in results.keys()) + " | " + " | ".join(f"{name} Dice_LCC" for name in results.keys()) + " |"
    sep = "|---| " + " | ".join("---:" for _ in results.keys()) + " | " + " | ".join("---:" for _ in results.keys()) + " |"
    lines.append(header)
    lines.append(sep)

    for c in range(1, 9):
        c_name = CLASS_NAMES[c]
        row_vals = [f"**{c_name}**"]
        for name, res in results.items():
            c_far = res["per_organ"][c_name]["far_fp_raw"]
            row_vals.append(f"{c_far:,d}")
        for name, res in results.items():
            c_d = res["per_organ"][c_name]["dice_lcc"] * 100
            row_vals.append(f"{c_d:.2f}%")
        lines.append("| " + " | ".join(row_vals) + " |")

    lines.extend([
        "",
        "## 3. Case-by-Case Breakdown of far-FP Voxels (Raw)",
        "",
    ])

    first_res = next(iter(results.values()))
    cases = [c["case"] for c in first_res["cases"]]
    c_header = "| Case | " + " | ".join(f"{name} far-FP" for name in results.keys()) + " | " + " | ".join(f"{name} Dice_LCC" for name in results.keys()) + " |"
    c_sep = "|---| " + " | ".join("---:" for _ in results.keys()) + " | " + " | ".join("---:" for _ in results.keys()) + " |"
    lines.append(c_header)
    lines.append(c_sep)

    for case in cases:
        row_vals = [f"`{case}`"]
        for name, res in results.items():
            c_rec = next((x for x in res["cases"] if x["case"] == case), None)
            far = c_rec["far_fp_raw"] if c_rec else 0
            row_vals.append(f"{far:,d}")
        for name, res in results.items():
            c_rec = next((x for x in res["cases"] if x["case"] == case), None)
            d = c_rec["dice_lcc_mean"] if c_rec else 0.0
            row_vals.append(f"{d:.2f}%")
        lines.append("| " + " | ".join(row_vals) + " |")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure Far False-Positive (far-FP) Benchmarks")
    parser.add_argument("--ckpts", nargs="+", default=None,
                        help="List of model_name=ckpt_path pairs, or pass without name for auto-detect.")
    parser.add_argument("--test_dir", type=str, default="data/btcv/test_vol_h5",
                        help="Path to 3D test volumes")
    parser.add_argument("--test_list", type=str, default="data/lists_Synapse/test_vol.txt",
                        help="Path to test cases list file")
    parser.add_argument("--output_dir", type=str, default="outputs/far_fp_benchmarks",
                        help="Output directory for reports and metrics")
    parser.add_argument("--cache_dir", type=str, default="outputs/far_fp_benchmarks/cached_preds",
                        help="Directory to cache prediction volumes")
    parser.add_argument("--distance_threshold", type=float, default=20.0,
                        help="Euclidean distance threshold in voxels (default: 20.0 voxels)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Microbatch size for volume inference")
    parser.add_argument("--force_infer", action="store_true",
                        help="Force re-inference even if cached prediction exists")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU evaluation")
    parser.add_argument("--canonical_only", action="store_true",
                        help="Run only canonical checkpoints (kis300, ours_v5, ours_v5cap)")
    return parser


def main(argv: Optional[List[str]] = None):
    args = build_parser().parse_args(argv)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Resolve checkpoints
    checkpoints_to_run: Dict[str, str] = {}
    if args.ckpts:
        for item in args.ckpts:
            if "=" in item:
                k, v = item.split("=", 1)
                checkpoints_to_run[k.strip()] = v.strip()
            else:
                k = Path(item).stem
                checkpoints_to_run[k] = item.strip()
    else:
        checkpoints_to_run = dict(CANONICAL_CHECKPOINTS)

    # Validate checkpoint paths
    valid_ckpts = {}
    for name, p in checkpoints_to_run.items():
        resolved = resolve_checkpoint_path(name, p)
        if resolved and os.path.isfile(resolved):
            valid_ckpts[name] = resolved
        else:
            print(f"Warning: Checkpoint '{name}' at '{p}' not found, skipping.")

    if not valid_ckpts:
        raise FileNotFoundError(f"No valid checkpoints found from: {checkpoints_to_run}")

    # Resolve test directory and case list
    test_dir = find_test_vol_dir(args.test_dir, REPO_ROOT)
    resolved_list = resolve_list_file(
        requested=args.test_list,
        list_name="test_vol.txt",
        iad_root=REPO_ROOT,
        data_dir=test_dir,
    )
    cases = []
    if resolved_list and os.path.isfile(resolved_list):
        with open(resolved_list, "r", encoding="utf-8") as f:
            cases = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if not cases and os.path.isdir(test_dir):
        cases = sorted([
            os.path.splitext(f)[0].replace(".npy", "")
            for f in os.listdir(test_dir) if f.endswith((".h5", ".npz"))
        ])

    print(f"Loaded {len(cases)} test cases from: {test_dir}")
    print(f"Checkpoints to evaluate ({len(valid_ckpts)}): {list(valid_ckpts.keys())}")

    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(args.output_dir, "cached_preds")

    all_results = {}
    for model_name, ckpt_path in valid_ckpts.items():
        t0 = time.time()
        res = evaluate_checkpoint_far_fp(
            ckpt_path=ckpt_path,
            model_name=model_name,
            test_dir=test_dir,
            cases=cases,
            device=device,
            cache_dir=cache_dir,
            distance_threshold=args.distance_threshold,
            force_infer=args.force_infer,
            batch_size=args.batch_size,
        )
        res["overall"]["total_eval_time_sec"] = round(time.time() - t0, 2)
        all_results[model_name] = res

    # Generate output reports
    md_content = format_markdown_comparison_table(all_results)
    md_path = os.path.join(args.output_dir, "far_fp_benchmark_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    json_path = os.path.join(args.output_dir, "far_fp_benchmark_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print(md_content)
    print("=" * 80)
    print(f"\nArtifacts saved successfully to:\n  - Markdown: {md_path}\n  - JSON:     {json_path}")
    return all_results


if __name__ == "__main__":
    main()
