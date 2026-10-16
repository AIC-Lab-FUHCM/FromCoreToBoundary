"""3D Connected Component Diagnostic Tool for Knowledge Distillation Evaluation.

Compares:
  - Condition A: Vanilla_TGBD (Raw)
  - Condition B: Vanilla_TGBD + 3D Largest Connected Component (LCC)
  - Condition C: Vanilla_TGBD_CBA_EMA (Raw)
  - Condition D: Vanilla_TGBD_CBA_EMA + 3D Largest Connected Component (LCC)

Performs:
  1. Volume inference and caching for Vanilla, CBA_EMA, and Teacher SAM ViT-L.
  2. 3D 26-connectivity LCC post-processing per organ class (handling LK and RK separately, no GT leakage).
  3. Metric evaluation (Clean Dice, Buggy Dice, BRD r=2, NSD tau=2mm, HD95) for all 4 conditions.
  4. False-Positive Island quantification:
     - Discarded island count, volume, surface area
     - Euclidean distance from discarded islands to true GT organ (min, mean, max)
     - Overlap with Teacher predictions to test if islands were inherited from Teacher.
  5. Paired difference analysis and tail distribution breakdown.
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import zoom
import torch
import torch.nn as nn
from medpy import metric as medpy_metric

_HERE = Path(__file__).resolve().parent
_REPO_PATH = _HERE.parent.parent if _HERE.parent.name == "scripts" else _HERE.parent
REPO_ROOT = str(_REPO_PATH)
for _p in (_REPO_PATH, _REPO_PATH / "models", _REPO_PATH / "scripts", _REPO_PATH / "scripts" / "tools"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.cba_v2.models import build_student, build_frozen_teacher
from experiments.cba_v2.evaluation import get_bounding_box_3d, get_surface_3d

CLASS_NAMES = [
    "background",
    "spleen",
    "right_kidney",
    "left_kidney",
    "gallbladder",
    "liver",
    "stomach",
    "aorta",
    "pancreas",
]


def compute_volume_metrics_per_class(
    pred_vol: np.ndarray,
    gt_vol: np.ndarray,
    radius: int = 2,
    tau_mm: float = 2.0,
    voxelspacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, Union[float, bool, int]]:
    """Compute Clean Dice, Buggy Dice, HD95, BRD (radius r), and Surface Dice (tau_mm) for one organ class.
    
    Fixed Case 0035 Bug & Physical Spacing:
      - If pred=0 and gt=0: dice_clean = 1.0 (true negative), dice_buggy = 0.0, hd95 = 0.0.
      - If pred>0 and gt=0: dice_clean = 0.0 (false positive), dice_buggy = 1.0 (author bug), hd95 = NaN.
      - If pred=0 and gt>0: dice_clean = 0.0 (false negative, defined failure), dice_buggy = 0.0, hd95 = NaN.
      - Evaluates all non-empty prediction masks with unconstrained HD95 (no dice > 0.02 guardrail).
      - Applies exact voxelspacing to HD95 and distance transforms for true physical millimeter units.
    """
    p_bin = pred_vol > 0
    g_bin = gt_vol > 0
    p_sum = int(p_bin.sum())
    g_sum = int(g_bin.sum())

    if p_sum == 0 and g_sum == 0:
        return {
            "dice_clean": 1.0,
            "dice_buggy": 0.0,
            "hd95": 0.0,
            "brd": 1.0,
            "surface_dice": 1.0,
            "empty_pred": False,
            "gt_empty": True,
            "p_sum": 0,
            "g_sum": 0,
            "tp": 0,
        }
    if p_sum > 0 and g_sum == 0:
        return {
            "dice_clean": 0.0,
            "dice_buggy": 1.0,
            "hd95": float("nan"),
            "brd": 0.0,
            "surface_dice": 0.0,
            "empty_pred": False,
            "gt_empty": True,
            "p_sum": p_sum,
            "g_sum": 0,
            "tp": 0,
        }
    if p_sum == 0 and g_sum > 0:
        return {
            "dice_clean": 0.0,
            "dice_buggy": 0.0,
            "hd95": float("nan"),
            "brd": 0.0,
            "surface_dice": 0.0,
            "empty_pred": True,
            "gt_empty": False,
            "p_sum": 0,
            "g_sum": g_sum,
            "tp": 0,
        }

    inter = int(np.logical_and(p_bin, g_bin).sum())
    dice = float(2.0 * inter / (p_sum + g_sum))

    bb = get_bounding_box_3d(p_bin | g_bin, pad=max(radius + 4, int(np.ceil(tau_mm)) + 4))
    if bb is not None:
        p_c = p_bin[bb]
        g_c = g_bin[bb]
    else:
        p_c = p_bin
        g_c = g_bin

    hd = float("nan")
    if medpy_metric is not None:
        try:
            hd = float(medpy_metric.binary.hd95(p_c, g_c, voxelspacing=voxelspacing))
        except Exception:
            hd = float("nan")

    dil = ndi.binary_dilation(g_c, iterations=radius)
    ero = ndi.binary_erosion(g_c, iterations=radius)
    band = dil ^ ero
    p_b = p_c & band
    g_b = g_c & band
    inter_b = (p_b & g_b).sum()
    denom_b = p_b.sum() + g_b.sum()
    brd = float(2.0 * inter_b / denom_b) if denom_b > 0 else 0.0

    s_pred = get_surface_3d(p_c)
    s_gt = get_surface_3d(g_c)
    if s_pred.sum() == 0 or s_gt.sum() == 0:
        sd = 0.0
    else:
        dt_gt = ndi.distance_transform_edt(~s_gt, sampling=voxelspacing)
        dt_pred = ndi.distance_transform_edt(~s_pred, sampling=voxelspacing)
        pred_close = (dt_gt[s_pred] <= tau_mm).sum()
        gt_close = (dt_pred[s_gt] <= tau_mm).sum()
        sd = float((pred_close + gt_close) / (s_pred.sum() + s_gt.sum()))

    return {
        "dice_clean": dice,
        "dice_buggy": dice,
        "hd95": hd,
        "brd": brd,
        "surface_dice": sd,
        "empty_pred": False,
        "gt_empty": False,
        "p_sum": p_sum,
        "g_sum": g_sum,
        "tp": inter,
    }


def apply_3d_lcc_per_class(
    pred_vol: np.ndarray,
    connectivity: int = 3,
) -> Tuple[np.ndarray, Dict[int, Dict[str, Union[int, float, np.ndarray]]]]:
    """Apply 3D Largest Connected Component (LCC) post-processing per organ class.
    
    Treats each organ class (1..8) independently.
    Uses 26-connectivity (connectivity=3).
    Returns:
      filtered_vol: 3D numpy array with non-largest components removed (set to 0)
      island_info: Dict mapping class_id -> {
          'num_components': int,
          'num_discarded': int,
          'discarded_voxels': int,
          'discarded_mask': bool np.ndarray (or None if discarded_voxels == 0)
      }
    """
    struct = ndi.generate_binary_structure(3, connectivity)
    filtered_vol = np.zeros_like(pred_vol)
    island_info = {}

    for c in range(1, 9):
        mask_c = (pred_vol == c)
        vox_count = int(mask_c.sum())
        if vox_count == 0:
            island_info[c] = {
                "num_components": 0,
                "num_discarded": 0,
                "discarded_voxels": 0,
                "discarded_mask": None,
            }
            continue

        labeled, num_features = ndi.label(mask_c, structure=struct)
        if num_features <= 1:
            filtered_vol[mask_c] = c
            island_info[c] = {
                "num_components": num_features,
                "num_discarded": 0,
                "discarded_voxels": 0,
                "discarded_mask": None,
            }
        else:
            sizes = np.bincount(labeled.flat)[1:]
            max_label = int(np.argmax(sizes)) + 1
            lcc_mask = (labeled == max_label)
            filtered_vol[lcc_mask] = c

            discarded_mask = mask_c & (~lcc_mask)
            discarded_voxels = int(discarded_mask.sum())

            island_info[c] = {
                "num_components": num_features,
                "num_discarded": num_features - 1,
                "discarded_voxels": discarded_voxels,
                "discarded_mask": discarded_mask,
            }

    return filtered_vol, island_info


@torch.no_grad()
def infer_volume(
    model: nn.Module,
    image_3d: np.ndarray,
    device: torch.device,
    batch_size: int = 8,
    is_teacher: bool = False,
) -> np.ndarray:
    """Run slice-by-slice inference on 3D volume, returning predicted class labels (D, H, W)."""
    model.eval()
    num_slices = image_3d.shape[0]
    preds = []
    curr_bs = batch_size
    slice_idx = 0

    amp_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    while slice_idx < num_slices:
        end_idx = min(slice_idx + curr_bs, num_slices)
        chunk = image_3d[slice_idx:end_idx]
        h, w = chunk.shape[1], chunk.shape[2]
        if h != 512 or w != 512:
            zoomed = [zoom(chunk[s], (512 / h, 512 / w), order=3) for s in range(chunk.shape[0])]
            chunk = np.stack(zoomed, axis=0)

        x_in = torch.from_numpy(chunk).unsqueeze(1).repeat(1, 3, 1, 1).float().to(device)
        try:
            if "cuda" in str(device):
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    if is_teacher:
                        out = model(x_in, multimask_output=8)
                    else:
                        out = model(x_in, True)
            else:
                out = model(x_in, multimask_output=8) if is_teacher else model(x_in, True)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            pred_chunk = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy().astype(np.uint8)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as oom_err:
            if "out of memory" in str(oom_err).lower() and curr_bs > 1:
                curr_bs = max(1, curr_bs // 2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise oom_err

        if h != 512 or w != 512:
            zoomed_back = [zoom(pred_chunk[s], (h / 512, w / 512), order=0) for s in range(pred_chunk.shape[0])]
            pred_chunk = np.stack(zoomed_back, axis=0)

        preds.append(pred_chunk)
        slice_idx = end_idx

    pred_vol = np.concatenate(preds, axis=0)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred_vol


def get_case_voxelspacing(test_dir: str, case: str) -> Tuple[float, float, float]:
    """Return spacing in array order (z, y, x)."""
    h5_path = os.path.join(test_dir, f"{case}.npy.h5")
    if os.path.exists(h5_path):
        try:
            with h5py.File(h5_path, "r") as f:
                if "voxelspacing" in f.attrs:
                    sp = f.attrs["voxelspacing"]
                    return tuple(float(x) for x in sp)
                if "spacing" in f.attrs:
                    sp = f.attrs["spacing"]
                    return tuple(float(x) for x in sp)
        except Exception:
            pass
    nii_cand = os.path.join(REPO_ROOT, "data", "test_nii", f"{case}_img.nii.gz")
    if os.path.exists(nii_cand):
        try:
            import SimpleITK as sitk
            img = sitk.ReadImage(nii_cand)
            sp_itk = img.GetSpacing()  # (x, y, z)
            # In numpy array order (D, H, W) -> (z, y, x):
            return (float(sp_itk[2]), float(sp_itk[1]), float(sp_itk[0]))
        except Exception:
            pass
    return (1.0, 1.0, 1.0)


def analyze_discarded_islands(
    case_id: str,
    class_id: int,
    discarded_mask: np.ndarray,
    gt_vol: np.ndarray,
    pred_teacher: Optional[np.ndarray],
    voxelspacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, Union[int, float, str]]:
    """Compute detailed geometric and spatial metrics for discarded false-positive islands in physical mm."""
    if discarded_mask is None or discarded_mask.sum() == 0:
        return {
            "discarded_voxels": 0,
            "surface_voxels": 0,
            "min_dist_to_gt": 0.0,
            "mean_dist_to_gt": 0.0,
            "max_dist_to_gt": 0.0,
            "teacher_overlap_voxels": 0,
            "teacher_overlap_frac": 0.0,
            "dist_category": "none",
        }

    gt_c = (gt_vol == class_id)
    discarded_voxels = int(discarded_mask.sum())
    s_discarded = get_surface_3d(discarded_mask)
    surface_voxels = int(s_discarded.sum())

    if gt_c.sum() > 0:
        bb = get_bounding_box_3d(gt_c | discarded_mask, pad=4)
        if bb is not None:
            dt_gt = ndi.distance_transform_edt(~gt_c[bb], sampling=voxelspacing)
            dists = dt_gt[discarded_mask[bb]]
        else:
            dt_gt = ndi.distance_transform_edt(~gt_c, sampling=voxelspacing)
            dists = dt_gt[discarded_mask]
        min_dist = float(dists.min())
        mean_dist = float(dists.mean())
        max_dist = float(dists.max())
    else:
        # Organ absent in GT
        min_dist = float("nan")
        mean_dist = float("nan")
        max_dist = float("nan")

    teacher_overlap = 0
    teacher_frac = 0.0
    if pred_teacher is not None:
        t_c = (pred_teacher == class_id)
        teacher_overlap = int((t_c & discarded_mask).sum())
        teacher_frac = float(teacher_overlap / discarded_voxels)

    if np.isnan(min_dist):
        dist_cat = "gt_absent"
    elif min_dist < 10.0:
        dist_cat = "near_boundary (<10mm)"
    elif min_dist < 50.0:
        dist_cat = "intermediate (10-50mm)"
    else:
        dist_cat = "distant_ghost (>50mm)"

    return {
        "discarded_voxels": discarded_voxels,
        "surface_voxels": surface_voxels,
        "min_dist_to_gt": min_dist,
        "mean_dist_to_gt": mean_dist,
        "max_dist_to_gt": max_dist,
        "teacher_overlap_voxels": teacher_overlap,
        "teacher_overlap_frac": teacher_frac,
        "dist_category": dist_cat,
    }


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test_dir", default=os.path.join(REPO_ROOT, "data", "btcv", "test_vol_h5"))
    parser.add_argument("--test_list", default=os.path.join(REPO_ROOT, "data", "lists_Synapse", "test_vol.txt"))
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "runs", "lcc_diagnostic"))
    parser.add_argument("--cache_dir", default=os.path.join(REPO_ROOT, "runs", "lcc_cached_preds"))
    args = parser.parse_args(argv)

    print("=" * 80)
    print("3D CONNECTED COMPONENT DIAGNOSTIC SUITE (CODEX AUDIT)")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    test_dir = args.test_dir
    test_list_path = args.test_list
    out_dir = args.out_dir
    cache_dir = args.cache_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    with open(test_list_path, "r", encoding="utf-8") as f:
        cases = [ln.strip() for ln in f if ln.strip()]
    print(f"Loaded {len(cases)} test cases: {cases}")

    # Check if all cache files already exist
    all_cached = all(
        os.path.exists(os.path.join(cache_dir, f"{c}_pred_vanilla.npy"))
        and os.path.exists(os.path.join(cache_dir, f"{c}_pred_ema.npy"))
        and os.path.exists(os.path.join(cache_dir, f"{c}_pred_teacher.npy"))
        for c in cases
    )

    student_v, student_e, teacher_net = None, None, None
    if not all_cached:
        print("\n[Loading Models for Inference]...")
        v_ckpt = os.environ.get(
            "IAD_VANILLA_CKPT",
            os.path.join(REPO_ROOT, "runs", "main", "kis300", "checkpoint_epoch_300.pth"),
        )
        e_ckpt = os.environ.get(
            "IAD_EMA_CKPT",
            os.path.join(REPO_ROOT, "runs", "main", "ours_v5", "checkpoint_epoch_300.pth"),
        )
        tea_lora = os.environ.get(
            "IAD_TEA_LORA",
            os.path.join(REPO_ROOT, "teacher_weights", "btcv_samed_l_epoch_299.pth"),
        )
        tea_base = os.environ.get(
            "IAD_TEA_BASE",
            os.path.join(REPO_ROOT, "pretrained_ckpt", "sam_vit_l_0b3195.pth"),
        )
        student_v = build_student(REPO_ROOT, v_ckpt, device)
        student_e = build_student(REPO_ROOT, e_ckpt, device)
        teacher_net = build_frozen_teacher(REPO_ROOT, tea_lora, tea_base, device)
        print("All 3 models loaded successfully.")
    else:
        print("\n[All 36 prediction volumes cached on disk. Skipping GPU model loading.]")

    # 2. Volume inference & LCC processing
    cond_names = ["A_Vanilla_Raw", "B_Vanilla_LCC", "C_EMA_Raw", "D_EMA_LCC"]
    per_case_results = []
    island_audit_records = []

    # Container for all paired evaluations
    all_pairs = []

    print("\n[Evaluating Volumes & Connected Components]...")
    for ci, case in enumerate(cases):
        t0 = time.time()
        h5_path = os.path.join(test_dir, f"{case}.npy.h5")
        with h5py.File(h5_path, "r") as f:
            image = f["image"][:]
            label = f["label"][:]
        while image.ndim > 3:
            image = image.squeeze(0)
        while label.ndim > 3:
            label = label.squeeze(0)

        # Check cached predictions
        v_cache_file = os.path.join(cache_dir, f"{case}_pred_vanilla.npy")
        e_cache_file = os.path.join(cache_dir, f"{case}_pred_ema.npy")
        t_cache_file = os.path.join(cache_dir, f"{case}_pred_teacher.npy")

        if os.path.exists(v_cache_file):
            pred_v_raw = np.load(v_cache_file)
        else:
            pred_v_raw = infer_volume(student_v, image, device, batch_size=8, is_teacher=False)
            np.save(v_cache_file, pred_v_raw)

        if os.path.exists(e_cache_file):
            pred_e_raw = np.load(e_cache_file)
        else:
            pred_e_raw = infer_volume(student_e, image, device, batch_size=8, is_teacher=False)
            np.save(e_cache_file, pred_e_raw)

        if os.path.exists(t_cache_file):
            pred_t_raw = np.load(t_cache_file)
        else:
            pred_t_raw = infer_volume(teacher_net, image, device, batch_size=4, is_teacher=True)
            np.save(t_cache_file, pred_t_raw)

        # Apply 3D LCC
        pred_v_lcc, islands_v = apply_3d_lcc_per_class(pred_v_raw, connectivity=3)
        pred_e_lcc, islands_e = apply_3d_lcc_per_class(pred_e_raw, connectivity=3)

        preds_dict = {
            "A_Vanilla_Raw": pred_v_raw,
            "B_Vanilla_LCC": pred_v_lcc,
            "C_EMA_Raw": pred_e_raw,
            "D_EMA_LCC": pred_e_lcc,
        }

        voxelspacing = get_case_voxelspacing(test_dir, case)
        case_rec = {"case": case, "time_sec": round(time.time() - t0, 2), "voxelspacing": voxelspacing}

        for c in range(1, 9):
            c_name = CLASS_NAMES[c]
            gt_c = (label == c)
            g_sum = int(gt_c.sum())
            is_gt_present = (g_sum > 0)

            # Metrics for all 4 conditions
            m_res = {}
            for cond in cond_names:
                p_c = (preds_dict[cond] == c)
                res = compute_volume_metrics_per_class(p_c, gt_c, radius=2, tau_mm=2.0, voxelspacing=voxelspacing)
                m_res[cond] = res
                case_rec[f"{cond}_{c_name}_dice_clean"] = res["dice_clean"]
                case_rec[f"{cond}_{c_name}_brd"] = res["brd"]
                case_rec[f"{cond}_{c_name}_nsd"] = res["surface_dice"]
                case_rec[f"{cond}_{c_name}_hd95"] = res["hd95"]

            # Analyze discarded islands with true physical mm voxelspacing
            isl_v = analyze_discarded_islands(case, c, islands_v[c]["discarded_mask"], label, pred_t_raw, voxelspacing=voxelspacing)
            isl_e = analyze_discarded_islands(case, c, islands_e[c]["discarded_mask"], label, pred_t_raw, voxelspacing=voxelspacing)

            # Recall and Retention decomposition
            tp_raw_v = int(m_res["A_Vanilla_Raw"]["tp"])
            tp_lcc_v = int(m_res["B_Vanilla_LCC"]["tp"])
            tp_removed_v = tp_raw_v - tp_lcc_v
            recall_raw_v = float(tp_raw_v / g_sum) if is_gt_present else float("nan")
            retention_v = float(tp_lcc_v / tp_raw_v) if tp_raw_v > 0 else (1.0 if tp_raw_v == 0 else float("nan"))
            tissue_lost_pct_v = float(tp_removed_v / g_sum * 100.0) if is_gt_present else float("nan")

            tp_raw_e = int(m_res["C_EMA_Raw"]["tp"])
            tp_lcc_e = int(m_res["D_EMA_LCC"]["tp"])
            tp_removed_e = tp_raw_e - tp_lcc_e
            recall_raw_e = float(tp_raw_e / g_sum) if is_gt_present else float("nan")
            retention_e = float(tp_lcc_e / tp_raw_e) if tp_raw_e > 0 else (1.0 if tp_raw_e == 0 else float("nan"))
            tissue_lost_pct_e = float(tp_removed_e / g_sum * 100.0) if is_gt_present else float("nan")

            island_audit_records.append({
                "case": case,
                "organ": c_name,
                "class_id": c,
                "is_gt_present": is_gt_present,
                "gt_voxels": g_sum,
                "vanilla_components": islands_v[c]["num_components"],
                "vanilla_discarded_islands": islands_v[c]["num_discarded"],
                "vanilla_discarded_voxels": isl_v["discarded_voxels"],
                "vanilla_min_dist_to_gt": isl_v["min_dist_to_gt"],
                "vanilla_mean_dist_to_gt": isl_v["mean_dist_to_gt"],
                "vanilla_teacher_overlap_voxels": isl_v["teacher_overlap_voxels"],
                "vanilla_teacher_overlap_frac": isl_v["teacher_overlap_frac"],
                "vanilla_dist_category": isl_v["dist_category"],
                "vanilla_tp_raw": tp_raw_v,
                "vanilla_tp_lcc": tp_lcc_v,
                "vanilla_tp_removed": tp_removed_v,
                "vanilla_recall_raw": recall_raw_v,
                "vanilla_retention": retention_v,
                "vanilla_tissue_lost_pct": tissue_lost_pct_v,
                "ema_components": islands_e[c]["num_components"],
                "ema_discarded_islands": islands_e[c]["num_discarded"],
                "ema_discarded_voxels": isl_e["discarded_voxels"],
                "ema_min_dist_to_gt": isl_e["min_dist_to_gt"],
                "ema_mean_dist_to_gt": isl_e["mean_dist_to_gt"],
                "ema_teacher_overlap_voxels": isl_e["teacher_overlap_voxels"],
                "ema_teacher_overlap_frac": isl_e["teacher_overlap_frac"],
                "ema_dist_category": isl_e["dist_category"],
                "ema_tp_raw": tp_raw_e,
                "ema_tp_lcc": tp_lcc_e,
                "ema_tp_removed": tp_removed_e,
                "ema_recall_raw": recall_raw_e,
                "ema_retention": retention_e,
                "ema_tissue_lost_pct": tissue_lost_pct_e,
            })

            # Check if this pair is valid for paired comparison (both have valid HD95)
            hd_a = m_res["A_Vanilla_Raw"]["hd95"]
            hd_b = m_res["B_Vanilla_LCC"]["hd95"]
            hd_c = m_res["C_EMA_Raw"]["hd95"]
            hd_d = m_res["D_EMA_LCC"]["hd95"]

            all_pairs.append({
                "case": case,
                "organ": c_name,
                "class_id": c,
                "is_gt_present": is_gt_present,
                "gt_voxels": g_sum,
                "voxelspacing": voxelspacing,
                "dice_A": m_res["A_Vanilla_Raw"]["dice_clean"],
                "dice_B": m_res["B_Vanilla_LCC"]["dice_clean"],
                "dice_C": m_res["C_EMA_Raw"]["dice_clean"],
                "dice_D": m_res["D_EMA_LCC"]["dice_clean"],
                "brd_A": m_res["A_Vanilla_Raw"]["brd"],
                "brd_B": m_res["B_Vanilla_LCC"]["brd"],
                "brd_C": m_res["C_EMA_Raw"]["brd"],
                "brd_D": m_res["D_EMA_LCC"]["brd"],
                "nsd_A": m_res["A_Vanilla_Raw"]["surface_dice"],
                "nsd_B": m_res["B_Vanilla_LCC"]["surface_dice"],
                "nsd_C": m_res["C_EMA_Raw"]["surface_dice"],
                "nsd_D": m_res["D_EMA_LCC"]["surface_dice"],
                "hd95_A": hd_a,
                "hd95_B": hd_b,
                "hd95_C": hd_c,
                "hd95_D": hd_d,
                "tp_raw_v": tp_raw_v,
                "tp_lcc_v": tp_lcc_v,
                "tp_removed_v": tp_removed_v,
                "recall_raw_v": recall_raw_v,
                "retention_v": retention_v,
                "tissue_lost_pct_v": tissue_lost_pct_v,
                "tp_raw_e": tp_raw_e,
                "tp_lcc_e": tp_lcc_e,
                "tp_removed_e": tp_removed_e,
                "recall_raw_e": recall_raw_e,
                "retention_e": retention_e,
                "tissue_lost_pct_e": tissue_lost_pct_e,
                "valid_hd_AC": (not np.isnan(hd_a)) and (not np.isnan(hd_c)),
                "valid_hd_BD": (not np.isnan(hd_b)) and (not np.isnan(hd_d)),
                "valid_hd_all4": (not np.isnan(hd_a)) and (not np.isnan(hd_b)) and (not np.isnan(hd_c)) and (not np.isnan(hd_d)),
            })

        per_case_results.append(case_rec)
        print(f"[{ci+1:02d}/{len(cases):02d}] {case} evaluated ({case_rec['time_sec']}s)", flush=True)

    # 3. Compute Summary Metrics
    print("\n" + "=" * 80)
    print("METRIC AGGREGATION & COMPARATIVE BENCHMARK MATRIX (PRIMARY: 95 REAL GT UNITS)")
    print("=" * 80)

    # Primary evaluation partition: All 95 units with real GT
    primary_pairs = [p for p in all_pairs if p["is_gt_present"]]
    empty_gt_pairs = [p for p in all_pairs if not p["is_gt_present"]]
    valid_primary_all4 = [p for p in primary_pairs if p["valid_hd_all4"]]
    common_92 = [p for p in primary_pairs if p["valid_hd_all4"]]  # 92 common units

    print(f"Total patient-organ units evaluated: {len(all_pairs)} (12 cases x 8 organs)")
    print(f"Primary benchmark set (Real GT present): {len(primary_pairs)} units")
    print(f"Empty GT units (case0035 gallbladder): {len(empty_gt_pairs)} unit(s)")
    print(f"Units with valid HD95 across all 4 conditions: {len(valid_primary_all4)}")

    # Overall Summary Table (Both Macro and Pooled on specified pairs)
    def calc_stats(cond_suffix, pairs_subset):
        dices = [p[f"dice_{cond_suffix}"] for p in pairs_subset]
        brds = [p[f"brd_{cond_suffix}"] for p in pairs_subset]
        nsds = [p[f"nsd_{cond_suffix}"] for p in pairs_subset]
        hds_pooled = [p[f"hd95_{cond_suffix}"] for p in pairs_subset if not np.isnan(p[f"hd95_{cond_suffix}"])]

        # Macro average (mean of 8 organ means)
        org_dices = []
        org_brds = []
        org_nsds = []
        org_hds = []
        for c_name in CLASS_NAMES[1:]:
            c_pairs = [p for p in pairs_subset if p["organ"] == c_name]
            if c_pairs:
                org_dices.append(np.mean([p[f"dice_{cond_suffix}"] for p in c_pairs]))
                org_brds.append(np.mean([p[f"brd_{cond_suffix}"] for p in c_pairs]))
                org_nsds.append(np.mean([p[f"nsd_{cond_suffix}"] for p in c_pairs]))
                valid_c_hds = [p[f"hd95_{cond_suffix}"] for p in c_pairs if not np.isnan(p[f"hd95_{cond_suffix}"])]
                if valid_c_hds:
                    org_hds.append(np.mean(valid_c_hds))

        return {
            "n_units": len(pairs_subset),
            "n_valid_hd": len(hds_pooled),
            "dice_pooled": np.mean(dices) if dices else float("nan"),
            "dice_macro": np.mean(org_dices) if org_dices else float("nan"),
            "brd_pooled": np.mean(brds) if brds else float("nan"),
            "brd_macro": np.mean(org_brds) if org_brds else float("nan"),
            "nsd_pooled": np.mean(nsds) if nsds else float("nan"),
            "nsd_macro": np.mean(org_nsds) if org_nsds else float("nan"),
            "hd95_pooled_mean": np.mean(hds_pooled) if hds_pooled else float("nan"),
            "hd95_pooled_median": np.median(hds_pooled) if hds_pooled else float("nan"),
            "hd95_macro_mean": np.mean(org_hds) if org_hds else float("nan"),
            "hd95_p75": np.percentile(hds_pooled, 75) if hds_pooled else float("nan"),
            "hd95_p90": np.percentile(hds_pooled, 90) if hds_pooled else float("nan"),
            "hd95_p95": np.percentile(hds_pooled, 95) if hds_pooled else float("nan"),
            "hd95_max": np.max(hds_pooled) if hds_pooled else float("nan"),
        }

    # Primary Statistics (All 95 Real GT units)
    stats_A = calc_stats("A", primary_pairs)
    stats_B = calc_stats("B", primary_pairs)
    stats_C = calc_stats("C", primary_pairs)
    stats_D = calc_stats("D", primary_pairs)

    # Sensitivity Statistics (92 common units)
    stats_A_92 = calc_stats("A", valid_primary_all4)
    stats_B_92 = calc_stats("B", valid_primary_all4)
    stats_C_92 = calc_stats("C", valid_primary_all4)
    stats_D_92 = calc_stats("D", valid_primary_all4)

    print("\n--- 1. OVERALL 4-WAY COMPARISON TABLE (POOLED & MACRO) ---")
    header = f"{'Condition':<20} | {'Clean Dice (M/P)':<18} | {'BRD (r=2)':<12} | {'NSD (tau=2)':<12} | {'HD95 Macro':<12} | {'HD95 Pooled':<12} | {'HD95 Med':<8}"
    print(header)
    print("-" * len(header))
    for name, s in [
        ("A: Vanilla (Raw)", stats_A),
        ("B: Vanilla + LCC", stats_B),
        ("C: EMA (Raw)", stats_C),
        ("D: EMA + LCC", stats_D),
    ]:
        print(f"{name:<20} | {s['dice_macro']*100:.2f}% / {s['dice_pooled']*100:.2f}% | {s['brd_macro']*100:.2f}%      | {s['nsd_macro']*100:.2f}%      | {s['hd95_macro_mean']:>6.2f} mm    | {s['hd95_pooled_mean']:>6.2f} mm    | {s['hd95_pooled_median']:>4.2f} mm")

    print("\n--- 2. TAIL DISTRIBUTION OF HD95 (POOLED) ---")
    print(f"{'Condition':<20} | {'P50 (Med)':<10} | {'P75':<10} | {'P90':<10} | {'P95':<10} | {'Max':<10}")
    print("-" * 75)
    for name, s in [
        ("A: Vanilla (Raw)", stats_A),
        ("B: Vanilla + LCC", stats_B),
        ("C: EMA (Raw)", stats_C),
        ("D: EMA + LCC", stats_D),
    ]:
        print(f"{name:<20} | {s['hd95_pooled_median']:>6.2f} mm | {s['hd95_p75']:>6.2f} mm | {s['hd95_p90']:>6.2f} mm | {s['hd95_p95']:>6.2f} mm | {s['hd95_max']:>6.2f} mm")

    # 4. Paired Comparisons (Primary 95 Real GT units and Sensitivity 92 units)
    print("\n--- 3. PAIRED DELTA ANALYSIS (PRIMARY: 95 REAL GT UNITS) ---")
    deltas_AB_hd = [p["hd95_B"] - p["hd95_A"] for p in primary_pairs if p["valid_hd_all4"]]
    deltas_CD_hd = [p["hd95_D"] - p["hd95_C"] for p in primary_pairs if p["valid_hd_all4"]]
    deltas_BD_hd = [p["hd95_D"] - p["hd95_B"] for p in primary_pairs if p["valid_hd_all4"]]
    deltas_AC_hd = [p["hd95_C"] - p["hd95_A"] for p in primary_pairs if p["valid_hd_all4"]]

    def report_deltas(name, deltas):
        arr = np.array(deltas)
        n_imp = int(np.sum(arr < -0.1))
        n_unc = int(np.sum(np.abs(arr) <= 0.1))
        n_wor = int(np.sum(arr > 0.1))
        print(f"[{name}] (n={len(arr)})")
        print(f"  * Mean Delta    : {arr.mean():+.4f} mm")
        print(f"  * Median Delta  : {np.median(arr):+.4f} mm")
        print(f"  * Improved      : {n_imp}/{len(arr)} ({n_imp/len(arr)*100:.1f}%)")
        print(f"  * Unchanged     : {n_unc}/{len(arr)} ({n_unc/len(arr)*100:.1f}%)")
        print(f"  * Worse         : {n_wor}/{len(arr)} ({n_wor/len(arr)*100:.1f}%)")
        print(f"  * Sum of Deltas : {arr.sum():+.2f} mm")

    report_deltas("1. Vanilla LCC Effect: Delta(B - A) [Vanilla+LCC vs Vanilla Raw]", deltas_AB_hd)
    report_deltas("2. EMA LCC Effect: Delta(D - C) [EMA+LCC vs EMA Raw]", deltas_CD_hd)
    report_deltas("3. Head-to-Head Post-LCC: Delta(D - B) [EMA+LCC vs Vanilla+LCC]", deltas_BD_hd)
    report_deltas("4. Original Raw Head-to-Head: Delta(C - A) [EMA Raw vs Vanilla Raw]", deltas_AC_hd)

    # Codex Advantage Narrowing Metric
    raw_adv = stats_A["hd95_pooled_mean"] - stats_C["hd95_pooled_mean"]
    lcc_adv = stats_B["hd95_pooled_mean"] - stats_D["hd95_pooled_mean"]
    if abs(raw_adv) > 1e-8:
        narrowing = (1.0 - lcc_adv / raw_adv) * 100.0
        print(f"\n[Codex LCC Advantage Narrowing on All 95 Real GT Units]:")
        print(f"  * Raw EMA advantage   : {raw_adv:+.2f} mm (Vanilla {stats_A['hd95_pooled_mean']:.2f} -> EMA {stats_C['hd95_pooled_mean']:.2f})")
        print(f"  * Post-LCC advantage  : {lcc_adv:+.2f} mm (Vanilla+LCC {stats_B['hd95_pooled_mean']:.2f} -> EMA+LCC {stats_D['hd95_pooled_mean']:.2f})")
        print(f"  * Narrowing of Margin : {narrowing:.1f}%")

    # 4b. Sensitivity Analysis on 92 common units
    print("\n--- 3b. SENSITIVITY SUBSET: 92 COMMON UNITS (DIRECT RECONCILIATION) ---")
    header_92 = f"{'Condition':<20} | {'Clean Dice (M/P)':<18} | {'NSD (tau=2)':<12} | {'HD95 Macro':<12} | {'HD95 Pooled':<12} | {'HD95 Med':<8}"
    print(header_92)
    print("-" * len(header_92))
    for name, s in [
        ("A: Vanilla (Raw)", stats_A_92),
        ("B: Vanilla + LCC", stats_B_92),
        ("C: EMA (Raw)", stats_C_92),
        ("D: EMA + LCC", stats_D_92),
    ]:
        print(f"{name:<20} | {s['dice_macro']*100:.2f}% / {s['dice_pooled']*100:.2f}% | {s['nsd_macro']*100:.2f}%      | {s['hd95_macro_mean']:>6.2f} mm    | {s['hd95_pooled_mean']:>6.2f} mm    | {s['hd95_pooled_median']:>4.2f} mm")

    # 4c. Separate False-Positive Statistics for Absent Organ (Case 0035 Gallbladder)
    print("\n--- 3c. ABSENT ORGAN FALSE-POSITIVE AUDIT (CASE 0035 GALLBLADDER) ---")
    c35_gall = [p for p in empty_gt_pairs if p["case"] == "case0035" and p["organ"] == "gallbladder"]
    c35_isl = [i for i in island_audit_records if i["case"] == "case0035" and i["organ"] == "gallbladder"]
    if c35_gall and c35_isl:
        g_pair = c35_gall[0]
        g_isl = c35_isl[0]
        print(f"Patient case0035 (Absent Gallbladder in GT):")
        print(f"  * Vanilla Raw  : Dice Clean = {g_pair['dice_A']:.2f}, Buggy Dice = 1.0 (author phantom artifact), FP voxels = {g_isl['vanilla_discarded_voxels'] + (g_pair['tp_lcc_v'])}")
        print(f"  * Vanilla + LCC: Post-LCC Discarded FP voxels = {g_isl['vanilla_discarded_voxels']}")
        print(f"  * EMA Raw      : Dice Clean = {g_pair['dice_C']:.2f}, Buggy Dice = 1.0 (author phantom artifact), FP voxels = {g_isl['ema_discarded_voxels'] + (g_pair['tp_lcc_e'])}")
        print(f"  * EMA + LCC    : Post-LCC Discarded FP voxels = {g_isl['ema_discarded_voxels']}")

    # 5. Organ-by-organ breakdown table across all 4 conditions
    print("\n--- 4. PER-ORGAN BREAKDOWN TABLE WITH RECALL AND RETENTION ---")
    org_table_header = f"{'Organ':<14} | {'Rec(V)':<8} | {'Ret(V)':<8} | {'Rec(EMA)':<8} | {'Ret(EMA)':<8} | {'Dice A':<8} | {'Dice B':<8} | {'Dice C':<8} | {'Dice D':<8} | {'HD95 A':<8} | {'HD95 B':<8} | {'HD95 C':<8} | {'HD95 D':<8} | {'Delta(D-B)':<10}"
    print(org_table_header)
    print("-" * len(org_table_header))

    per_organ_summary = {}
    for c in range(1, 9):
        c_name = CLASS_NAMES[c]
        c_pairs = [p for p in primary_pairs if p["organ"] == c_name]
        d_A = np.mean([p["dice_A"] for p in c_pairs])
        d_B = np.mean([p["dice_B"] for p in c_pairs])
        d_C = np.mean([p["dice_C"] for p in c_pairs])
        d_D = np.mean([p["dice_D"] for p in c_pairs])

        rec_v = np.nanmean([p["recall_raw_v"] for p in c_pairs])
        ret_v = np.nanmean([p["retention_v"] for p in c_pairs])
        rec_e = np.nanmean([p["recall_raw_e"] for p in c_pairs])
        ret_e = np.nanmean([p["retention_e"] for p in c_pairs])

        loss_pct_v = np.nanmean([p["tissue_lost_pct_v"] for p in c_pairs])
        loss_pct_e = np.nanmean([p["tissue_lost_pct_e"] for p in c_pairs])

        hds_A = [p["hd95_A"] for p in c_pairs if not np.isnan(p["hd95_A"])]
        hds_B = [p["hd95_B"] for p in c_pairs if not np.isnan(p["hd95_B"])]
        hds_C = [p["hd95_C"] for p in c_pairs if not np.isnan(p["hd95_C"])]
        hds_D = [p["hd95_D"] for p in c_pairs if not np.isnan(p["hd95_D"])]

        hd_A = np.mean(hds_A) if hds_A else float("nan")
        hd_B = np.mean(hds_B) if hds_B else float("nan")
        hd_C = np.mean(hds_C) if hds_C else float("nan")
        hd_D = np.mean(hds_D) if hds_D else float("nan")

        delta_DB = hd_D - hd_B

        per_organ_summary[c_name] = {
            "dice_A": d_A, "dice_B": d_B, "dice_C": d_C, "dice_D": d_D,
            "recall_raw_v": rec_v, "retention_v": ret_v, "tissue_lost_pct_v": loss_pct_v,
            "recall_raw_e": rec_e, "retention_e": ret_e, "tissue_lost_pct_e": loss_pct_e,
            "hd95_A": hd_A, "hd95_B": hd_B, "hd95_C": hd_C, "hd95_D": hd_D,
            "delta_DB": delta_DB,
        }

        print(f"{c_name:<14} | {rec_v*100:>5.1f}%  | {ret_v*100:>5.1f}%  | {rec_e*100:>5.1f}%   | {ret_e*100:>5.1f}%   | {d_A*100:>5.1f}%  | {d_B*100:>5.1f}%  | {d_C*100:>5.1f}%  | {d_D*100:>5.1f}%  | {hd_A:>6.2f} | {hd_B:>6.2f} | {hd_C:>6.2f} | {hd_D:>6.2f} | {delta_DB:>+6.2f} mm")

    # Explicit Aorta Clarification
    aorta_pairs = [p for p in primary_pairs if p["organ"] == "aorta"]
    total_aorta_gt = sum(p["gt_voxels"] for p in aorta_pairs)
    total_aorta_tp_rem_v = sum(p["tp_removed_v"] for p in aorta_pairs)
    total_aorta_tp_rem_e = sum(p["tp_removed_e"] for p in aorta_pairs)
    aorta_islands = [i for i in island_audit_records if i["organ"] == "aorta"]
    total_aorta_discarded_v = sum(i["vanilla_discarded_voxels"] for i in aorta_islands)
    print("\n[Aorta Clarification]:")
    print(f"  * Total Aorta GT volume |G| = {total_aorta_gt:,} voxels")
    print(f"  * Vanilla: TP removed = {total_aorta_tp_rem_v:,} voxels out of {total_aorta_gt:,} ({total_aorta_tp_rem_v/total_aorta_gt*100:.2f}% true tissue lost)")
    if total_aorta_discarded_v > 0:
        print(f"  * Vanilla: TP removed represents {total_aorta_tp_rem_v/total_aorta_discarded_v*100:.1f}% of discarded voxels ({total_aorta_tp_rem_v:,}/{total_aorta_discarded_v:,})")
    print(f"  * EMA    : TP removed = {total_aorta_tp_rem_e:,} voxels out of {total_aorta_gt:,} ({total_aorta_tp_rem_e/total_aorta_gt*100:.2f}% true tissue lost)")

    # 6. Deep Dive into Critical Cases
    print("\n--- 5. DEEP DIVE: CRITICAL CASES (OUTLIERS & PRIMARY TARGETS) ---")
    critical_pairs = [
        ("case0029", "spleen"),
        ("case0029", "left_kidney"),
        ("case0029", "right_kidney"),
        ("case0022", "right_kidney"),
        ("case0035", "spleen"),
        ("case0001", "gallbladder"),
        ("case0001", "liver"),
        ("case0003", "liver"),
        ("case0003", "gallbladder"),
        ("case0003", "spleen"),
        ("case0032", "right_kidney"),
    ]

    print(f"{'Case':<10} | {'Organ':<14} | {'HD95 A(V)':<10} | {'HD95 B(V+LCC)':<14} | {'HD95 C(EMA)':<12} | {'HD95 D(E+LCC)':<14} | {'V Islands':<10} | {'E Islands':<10} | {'V Isld Dist':<12}")
    print("-" * 115)
    for c_id, org in critical_pairs:
        pair = next((p for p in all_pairs if p["case"] == c_id and p["organ"] == org), None)
        isl = next((i for i in island_audit_records if i["case"] == c_id and i["organ"] == org), None)
        if pair is not None and isl is not None:
            v_dist_str = f"{isl['vanilla_min_dist_to_gt']:.1f}mm" if not np.isnan(isl['vanilla_min_dist_to_gt']) else "N/A"
            print(f"{c_id:<10} | {org:<14} | {pair['hd95_A']:>6.2f} mm | {pair['hd95_B']:>6.2f} mm     | {pair['hd95_C']:>6.2f} mm   | {pair['hd95_D']:>6.2f} mm     | {isl['vanilla_discarded_islands']:>3d} ({isl['vanilla_discarded_voxels']:>5d}v) | {isl['ema_discarded_islands']:>3d} ({isl['ema_discarded_voxels']:>5d}v) | {v_dist_str}")

    # 7. False-Positive Island Aggregate Statistics & Teacher Overlap
    print("\n--- 6. DISCARDED FALSE-POSITIVE ISLAND QUANTIFICATION ---")
    total_v_islands = sum(i["vanilla_discarded_islands"] for i in island_audit_records)
    total_e_islands = sum(i["ema_discarded_islands"] for i in island_audit_records)
    total_v_voxels = sum(i["vanilla_discarded_voxels"] for i in island_audit_records)
    total_e_voxels = sum(i["ema_discarded_voxels"] for i in island_audit_records)
    total_v_tea_overlap = sum(i["vanilla_teacher_overlap_voxels"] for i in island_audit_records)
    total_e_tea_overlap = sum(i["ema_teacher_overlap_voxels"] for i in island_audit_records)

    print(f"Total Discarded Islands : Vanilla = {total_v_islands} | EMA = {total_e_islands} ({total_e_islands - total_v_islands:+d})")
    print(f"Total Discarded Voxels  : Vanilla = {total_v_voxels} | EMA = {total_e_voxels} ({total_e_voxels - total_v_voxels:+d})")
    print(f"Teacher Overlap Voxels  : Vanilla = {total_v_tea_overlap} ({total_v_tea_overlap/total_v_voxels*100:.1f}%) | EMA = {total_e_tea_overlap} ({total_e_tea_overlap/total_e_voxels*100:.1f}%)" if total_v_voxels > 0 and total_e_voxels > 0 else "")

    # 8. Complete 3-Tier Distance Stratification of Discarded False Positives
    print("\n--- 7. THREE-TIER DISTANCE STRATIFICATION OF DISCARDED FALSE POSITIVES ---", flush=True)
    tier_header = f"{'Condition':<20} | {'Near (<10mm)':<16} | {'Intermediate (10-50mm)':<22} | {'Distant Ghost (>=50mm)':<24} | {'GT Absent':<12}"
    print(tier_header, flush=True)
    print("-" * len(tier_header), flush=True)

    tier_summary = {}
    for model_name, prefix in [("Vanilla", "vanilla"), ("EMA", "ema")]:
        near = sum(i[f"{prefix}_discarded_voxels"] for i in island_audit_records if i["is_gt_present"] and 0 <= i[f"{prefix}_min_dist_to_gt"] < 10.0)
        mid = sum(i[f"{prefix}_discarded_voxels"] for i in island_audit_records if i["is_gt_present"] and 10.0 <= i[f"{prefix}_min_dist_to_gt"] < 50.0)
        far = sum(i[f"{prefix}_discarded_voxels"] for i in island_audit_records if i["is_gt_present"] and i[f"{prefix}_min_dist_to_gt"] >= 50.0)
        absent = sum(i[f"{prefix}_discarded_voxels"] for i in island_audit_records if not i["is_gt_present"])
        tier_summary[model_name] = {
            "near_lt_10mm": near,
            "intermediate_10_50mm": mid,
            "distant_gte_50mm": far,
            "gt_absent_fp": absent,
        }
        print(f"{model_name:<20} | {near:>10,d} vox   | {mid:>16,d} vox        | {far:>18,d} vox          | {absent:>6,d} vox", flush=True)

    # Distribution of island distances (physically calibrated in mm)
    v_distant = [i for i in island_audit_records if i["vanilla_discarded_voxels"] > 0 and i["vanilla_min_dist_to_gt"] >= 50.0]
    e_distant = [i for i in island_audit_records if i["ema_discarded_voxels"] > 0 and i["ema_min_dist_to_gt"] >= 50.0]
    print(f"\nDistant Ghost Islands (>= 50mm from true organ):", flush=True)
    print(f"  * Vanilla: {len(v_distant)} pairs had ghost islands (Total {sum(i['vanilla_discarded_voxels'] for i in v_distant)} voxels)", flush=True)
    for i in v_distant:
        print(f"    - {i['case']} [{i['organ']}]: {i['vanilla_discarded_voxels']} voxels, min dist = {i['vanilla_min_dist_to_gt']:.1f} mm, teacher overlap = {i['vanilla_teacher_overlap_frac']*100:.1f}%", flush=True)
    print(f"  * EMA: {len(e_distant)} pairs had ghost islands (Total {sum(i['ema_discarded_voxels'] for i in e_distant)} voxels)", flush=True)
    for i in e_distant:
        print(f"    - {i['case']} [{i['organ']}]: {i['ema_discarded_voxels']} voxels, min dist = {i['ema_min_dist_to_gt']:.1f} mm, teacher overlap = {i['ema_teacher_overlap_frac']*100:.1f}%", flush=True)

    # Save output artifacts
    output_payload = {
        "overall_summary_primary_95": {
            "Condition_A_Vanilla_Raw": stats_A,
            "Condition_B_Vanilla_LCC": stats_B,
            "Condition_C_EMA_Raw": stats_C,
            "Condition_D_EMA_LCC": stats_D,
        },
        "overall_summary_sensitivity_92": {
            "Condition_A_Vanilla_Raw": stats_A_92,
            "Condition_B_Vanilla_LCC": stats_B_92,
            "Condition_C_EMA_Raw": stats_C_92,
            "Condition_D_EMA_LCC": stats_D_92,
        },
        "per_organ_summary": per_organ_summary,
        "tier_summary_distance_stratification": tier_summary,
        "paired_deltas": {
            "deltas_AB_hd": deltas_AB_hd,
            "deltas_CD_hd": deltas_CD_hd,
            "deltas_BD_hd": deltas_BD_hd,
            "deltas_AC_hd": deltas_AC_hd,
            "raw_advantage_ema": raw_adv,
            "lcc_advantage_ema": lcc_adv,
            "narrowing_pct": narrowing if abs(raw_adv) > 1e-8 else 0.0,
        },
        "all_pairs": all_pairs,
        "island_audit_records": island_audit_records,
    }

    json_path = os.path.join(out_dir, "lcc_diagnostic_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nFull results saved to: {json_path}", flush=True)


if __name__ == "__main__":
    main()
