"""3D volume evaluation (Dice/HD95/NSD, raw and post-LCC). Bodies copied verbatim from
experiments/kas_oa_h4/run_kis_cba_pilot.py (lines 64-81 and 748-1210)."""
from __future__ import annotations

import gc
import os
import time
from typing import Dict, Optional, Tuple, Union

import h5py
import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn
from scipy.ndimage import zoom

from experiments.cba_v2.data import (
    ACDC_CLASS_NAMES,
    POLYP_PRIMARY_4_SETS,
    POLYP_TEST_SETS,
    PolypDataset,
    find_acdc_split_dir,
    find_polyp_root,
    find_test_vol_dir,
    is_valid_case_list,
    resolve_acdc_list_file,
    resolve_list_file,
)

# Optional MedPy import for HD95
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


def get_surface_3d(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return mask & (~ndi.binary_erosion(mask))


def get_bounding_box_3d(mask: np.ndarray, pad: int = 8) -> Optional[Tuple[slice, ...]]:
    where = np.where(mask)
    if len(where[0]) == 0:
        return None
    slices = []
    for dim_idx, axis_coords in enumerate(where):
        min_c = max(0, int(axis_coords.min()) - pad)
        max_c = min(mask.shape[dim_idx], int(axis_coords.max()) + pad + 1)
        slices.append(slice(min_c, max_c))
    return tuple(slices)


def get_case_voxelspacing(test_dir: str, case_path: str) -> Tuple[float, float, float]:
    try:
        if case_path.endswith((".h5", ".npy.h5")):
            with h5py.File(case_path, "r") as f:
                for attr_key in ["voxelspacing", "spacing", "pixel_spacing"]:
                    if attr_key in f.attrs:
                        val = f.attrs[attr_key]
                        return (float(val[0]), float(val[1]), float(val[2]))
    except Exception:
        pass
    return (1.0, 1.0, 1.0)


def apply_3d_lcc_per_class(
    pred_vol: np.ndarray,
    connectivity: int = 3,
    num_classes: int = 9,
) -> Tuple[np.ndarray, Dict[int, Dict[str, Union[int, float]]]]:
    ndim = pred_vol.ndim
    struct = ndi.generate_binary_structure(ndim, min(connectivity, ndim))
    filtered_vol = np.zeros_like(pred_vol)
    island_info = {}

    for c in range(1, num_classes):
        mask_c = (pred_vol == c)
        vox_count = int(mask_c.sum())
        if vox_count == 0:
            island_info[c] = {"num_components": 0, "num_discarded": 0, "discarded_voxels": 0}
            continue

        labeled, num_features = ndi.label(mask_c, structure=struct)
        if num_features <= 1:
            filtered_vol[mask_c] = c
            island_info[c] = {"num_components": num_features, "num_discarded": 0, "discarded_voxels": 0}
        else:
            sizes = np.bincount(labeled.flat)[1:]
            max_label = int(np.argmax(sizes)) + 1
            lcc_mask = (labeled == max_label)
            filtered_vol[lcc_mask] = c
            discarded_voxels = int((mask_c & (~lcc_mask)).sum())
            island_info[c] = {
                "num_components": num_features,
                "num_discarded": num_features - 1,
                "discarded_voxels": discarded_voxels,
            }

    return filtered_vol, island_info


def compute_volume_metrics_per_class(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    radius: int = 2,
    tau_mm: float = 2.0,
    voxelspacing: Optional[Tuple[float, ...]] = (1.0, 1.0, 1.0),
) -> Dict[str, Union[float, bool, int]]:
    p_bin = (pred_mask > 0)
    g_bin = (gt_mask > 0)
    p_sum = int(p_bin.sum())
    g_sum = int(g_bin.sum())

    if p_sum == 0 and g_sum == 0:
        return {
            "dice_clean": 1.0, "dice_buggy": 1.0, "jc": 1.0, "hd95": float("nan"),
            "brd": 1.0, "surface_dice": 1.0, "empty_pred": True, "gt_empty": True,
            "tp": 0, "p_sum": 0, "g_sum": 0, "far_fp": 0, "near_fp": 0,
        }
    if p_sum > 0 and g_sum == 0:
        return {
            "dice_clean": 0.0, "dice_buggy": 1.0, "jc": 0.0, "hd95": float("nan"),
            "brd": 0.0, "surface_dice": 0.0, "empty_pred": False, "gt_empty": True,
            "tp": 0, "p_sum": p_sum, "g_sum": 0, "far_fp": p_sum, "near_fp": 0,
        }
    if p_sum == 0 and g_sum > 0:
        return {
            "dice_clean": 0.0, "dice_buggy": 0.0, "jc": 0.0, "hd95": float("nan"),
            "brd": 0.0, "surface_dice": 0.0, "empty_pred": True, "gt_empty": False,
            "tp": 0, "p_sum": 0, "g_sum": g_sum, "far_fp": 0, "near_fp": 0,
        }

    inter = int(np.logical_and(p_bin, g_bin).sum())
    union = int(np.logical_or(p_bin, g_bin).sum())
    dice = float(2.0 * inter / (p_sum + g_sum))
    jc = float(inter / union) if union > 0 else 1.0

    bb = get_bounding_box_3d(p_bin | g_bin, pad=max(radius + 4, int(np.ceil(tau_mm)) + 4))
    if bb is not None:
        p_c = p_bin[bb]
        g_c = g_bin[bb]
    else:
        p_c = p_bin
        g_c = g_bin

    spacing_use = voxelspacing[: p_c.ndim] if voxelspacing is not None else None
    hd = float("nan")
    if medpy_metric is not None:
        try:
            hd = float(medpy_metric.binary.hd95(p_c, g_c, voxelspacing=spacing_use))
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
        s_gt_inv = np.ascontiguousarray(~s_gt, dtype=bool)
        s_pred_inv = np.ascontiguousarray(~s_pred, dtype=bool)
        dt_gt = ndi.distance_transform_edt(s_gt_inv, sampling=spacing_use)
        dt_pred = ndi.distance_transform_edt(s_pred_inv, sampling=spacing_use)
        pred_close = int((dt_gt[s_pred] <= tau_mm).sum())
        gt_close = int((dt_pred[s_gt] <= tau_mm).sum())
        sd = float((pred_close + gt_close) / (s_pred.sum() + s_gt.sum()))
        del dt_gt, dt_pred, s_gt_inv, s_pred_inv

    # Far-FP computation (distance > 20 voxels to GT)
    where_g = np.where(g_bin)
    pad_far = 21
    slices_far = tuple(
        slice(max(0, int(w.min()) - pad_far), min(g_bin.shape[ax], int(w.max()) + pad_far + 1))
        for ax, w in enumerate(where_g)
    )
    sub_g_far = g_bin[slices_far]
    sub_p_far = p_bin[slices_far]
    dt_sub_far = ndi.distance_transform_edt(~sub_g_far)
    far_inside = int((sub_p_far & (dt_sub_far > 20.0)).sum())
    far_outside = p_sum - int(sub_p_far.sum())
    far_fp = far_inside + far_outside
    near_fp = (p_sum - inter) - far_fp

    return {
        "dice_clean": dice,
        "dice_buggy": dice,
        "jc": jc,
        "hd95": hd,
        "brd": brd,
        "surface_dice": sd,
        "empty_pred": False,
        "gt_empty": False,
        "tp": inter,
        "p_sum": p_sum,
        "g_sum": g_sum,
        "far_fp": far_fp,
        "near_fp": near_fp,
    }


@torch.no_grad()
def evaluate_volumes(
    student_net: nn.Module,
    test_dir: str,
    test_list: str,
    device: torch.device,
    batch_size: int = 8,
    use_amp: bool = True,
    max_cases: Optional[int] = None,
    brd_radius: int = 2,
    nsd_tau: float = 2.0,
    list_dir: Optional[str] = None,
    iad_root: Optional[str] = None,
    save_preds_dir: Optional[str] = None,
    tag: str = "",
    dataset: str = "synapse",
    img_size: Optional[int] = None,
    class_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Union[dict, list, float]]:
    student_net.eval()
    ds = str(dataset or "synapse").strip().lower()
    if ds == "polyp":
        return evaluate_polyp(
            student_net=student_net,
            test_dir=test_dir,
            device=device,
            batch_size=batch_size,
            use_amp=use_amp,
            max_cases=max_cases,
            brd_radius=brd_radius,
            nsd_tau=nsd_tau,
            iad_root=iad_root,
            save_preds_dir=save_preds_dir,
            tag=tag,
            img_size=img_size or 256,
        )

    if ds == "acdc":
        eval_img_size = int(img_size or 256)
        eval_class_names = class_names or ACDC_CLASS_NAMES
        test_dir = find_acdc_split_dir(split="test", requested=test_dir, iad_root=iad_root)
        resolved_test_list = resolve_acdc_list_file(
            requested=test_list,
            split="test",
            iad_root=iad_root,
            data_dir=test_dir,
        )
    else:
        eval_img_size = int(img_size or 512)
        eval_class_names = class_names or CLASS_NAMES
        test_dir = find_test_vol_dir(test_dir, iad_root)
        resolved_test_list = resolve_list_file(
            requested=test_list,
            list_name="test_vol.txt",
            iad_root=iad_root,
            list_dir=list_dir,
            data_dir=test_dir,
        )

    num_classes_total = max(eval_class_names.keys()) + 1
    valid_list = resolved_test_list and os.path.isfile(resolved_test_list) and is_valid_case_list(resolved_test_list, test_dir)

    cases = []
    if valid_list:
        with open(resolved_test_list, "r", encoding="utf-8") as f:
            raw_cases = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        for c in raw_cases:
            c_clean = c[:-4] if c.endswith(".npz") else c.replace(".npy.h5", "").replace(".h5", "")
            if any(os.path.isfile(os.path.join(test_dir, f"{c_clean}{ext}")) for ext in [".npy.h5", ".h5", ".npz"]):
                cases.append(c_clean)

    if not cases and os.path.isdir(test_dir):
        cases = sorted([
            os.path.splitext(f)[0].replace(".npy", "")
            for f in os.listdir(test_dir)
            if not f.startswith("._") and f.endswith((".h5", ".npz"))
        ])

    if not cases:
        raise FileNotFoundError(
            f"[evaluate_volumes:{ds}] No valid volume files (.npy.h5, .h5, or .npz) found in '{test_dir}' using list '{test_list}'"
        )

    if max_cases:
        cases = cases[:max_cases]

    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    per_case_records = []
    all_real_gt_units = []
    empty_gt_units = []
    all_evaluated_units = []

    if save_preds_dir:
        os.makedirs(save_preds_dir, exist_ok=True)

    print(f"\n[{tag}] Commencing {ds.upper()} evaluation across {len(cases)} test volume(s) at {eval_img_size}x{eval_img_size}...", flush=True)

    for ci, case in enumerate(cases):
        t0 = time.time()
        cand_paths = [
            os.path.join(test_dir, f"{case}.npy.h5"),
            os.path.join(test_dir, f"{case}.h5"),
            os.path.join(test_dir, f"{case}.npz"),
        ]
        case_path = next((p for p in cand_paths if os.path.isfile(p)), None)
        if not case_path:
            continue

        if case_path.endswith(".npz"):
            with np.load(case_path) as raw:
                image = (raw["img"] if "img" in raw else raw["image"])[:]
                label = raw["label"][:]
        else:
            with h5py.File(case_path, "r") as f:
                image = f["image"][:]
                label = f["label"][:]
        while image.ndim > 3:
            image = image.squeeze(0)
        while label.ndim > 3:
            label = label.squeeze(0)

        num_slices = image.shape[0]
        preds = []
        curr_bs = batch_size
        slice_idx = 0

        while slice_idx < num_slices:
            end_idx = min(slice_idx + curr_bs, num_slices)
            chunk = image[slice_idx:end_idx]
            h, w = chunk.shape[1], chunk.shape[2]
            if h != eval_img_size or w != eval_img_size:
                zoomed = [zoom(chunk[s], (eval_img_size / h, eval_img_size / w), order=3) for s in range(chunk.shape[0])]
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

            if h != eval_img_size or w != eval_img_size:
                zoomed_back = [zoom(pred_chunk[s], (h / eval_img_size, w / eval_img_size), order=0) for s in range(pred_chunk.shape[0])]
                pred_chunk = np.stack(zoomed_back, axis=0)

            preds.append(pred_chunk)
            slice_idx = end_idx

        pred_vol = np.concatenate(preds, axis=0)
        del preds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if save_preds_dir:
            np.save(os.path.join(save_preds_dir, f"{case}_pred_{tag}.npy"), pred_vol)

        pred_vol_lcc, island_info = apply_3d_lcc_per_class(pred_vol, num_classes=num_classes_total)
        if save_preds_dir:
            np.save(os.path.join(save_preds_dir, f"{case}_pred_{tag}_lcc.npy"), pred_vol_lcc)

        voxelspacing = get_case_voxelspacing(test_dir, case_path)
        case_rec = {"case": case, "voxelspacing": voxelspacing}

        case_dices_raw = []
        case_dices_lcc = []
        case_hds = []

        for c, c_name in sorted(eval_class_names.items()):
            gt_c = (label == c)
            res_raw = compute_volume_metrics_per_class(
                pred_vol == c, gt_c, radius=brd_radius, tau_mm=nsd_tau, voxelspacing=voxelspacing
            )
            res_lcc = compute_volume_metrics_per_class(
                pred_vol_lcc == c, gt_c, radius=brd_radius, tau_mm=nsd_tau, voxelspacing=voxelspacing
            )

            g_sum = res_raw["g_sum"]
            p_sum_raw = res_raw["p_sum"]
            p_sum_lcc = res_lcc["p_sum"]
            tp_raw = res_raw["tp"]
            tp_lcc = res_lcc["tp"]

            recall_raw = float(tp_raw / g_sum) if g_sum > 0 else (1.0 if tp_raw == 0 else 0.0)
            recall_lcc = float(tp_lcc / g_sum) if g_sum > 0 else (1.0 if tp_lcc == 0 else 0.0)
            precision_raw = float(tp_raw / p_sum_raw) if p_sum_raw > 0 else (1.0 if g_sum == 0 else 0.0)
            precision_lcc = float(tp_lcc / p_sum_lcc) if p_sum_lcc > 0 else (1.0 if g_sum == 0 else 0.0)
            vol_ratio_raw = float(p_sum_raw / g_sum) if g_sum > 0 else (1.0 if p_sum_raw == 0 else float("inf"))
            vol_ratio_lcc = float(p_sum_lcc / g_sum) if g_sum > 0 else (1.0 if p_sum_lcc == 0 else float("inf"))
            retention = float(tp_lcc / tp_raw) if tp_raw > 0 else 1.0

            unit_info = {
                "case": case,
                "organ": c_name,
                "class_id": c,
                "gt_present": not res_raw["gt_empty"],
                "empty_pred_raw": res_raw.get("empty_pred", False),
                "empty_pred_lcc": res_lcc.get("empty_pred", False),
                "clean_dice_raw": res_raw["dice_clean"],
                "clean_dice_lcc": res_lcc["dice_clean"],
                "jc_raw": res_raw["jc"],
                "jc_lcc": res_lcc["jc"],
                "nsd_raw": res_raw["surface_dice"],
                "nsd_lcc": res_lcc["surface_dice"],
                "hd95_raw": res_raw["hd95"],
                "hd95_lcc": res_lcc["hd95"],
                "tp_raw": tp_raw,
                "tp_lcc": tp_lcc,
                "p_sum_raw": p_sum_raw,
                "p_sum_lcc": p_sum_lcc,
                "g_sum": g_sum,
                "recall_raw": recall_raw,
                "recall_lcc": recall_lcc,
                "precision_raw": precision_raw,
                "precision_lcc": precision_lcc,
                "vol_ratio_raw": vol_ratio_raw,
                "vol_ratio_lcc": vol_ratio_lcc,
                "retention": retention,
                "far_fp_raw": res_raw.get("far_fp", 0),
                "far_fp_lcc": res_lcc.get("far_fp", 0),
                "near_fp_raw": res_raw.get("near_fp", 0),
                "near_fp_lcc": res_lcc.get("near_fp", 0),
            }
            all_evaluated_units.append(unit_info)

            if unit_info["gt_present"]:
                all_real_gt_units.append(unit_info)
                case_dices_raw.append(res_raw["dice_clean"])
                case_dices_lcc.append(res_lcc["dice_clean"])
                if not np.isnan(res_raw["hd95"]):
                    case_hds.append(res_raw["hd95"])
            else:
                empty_gt_units.append(unit_info)

            case_rec[f"dice_clean_{c_name}"] = res_raw["dice_clean"]
            case_rec[f"dice_lcc_{c_name}"] = res_lcc["dice_clean"]
            if ds != "synapse":
                case_rec[f"jc_{c_name}"] = res_raw["jc"]
            case_rec[f"recall_{c_name}"] = recall_raw
            case_rec[f"precision_{c_name}"] = precision_raw
            case_rec[f"vol_ratio_{c_name}"] = vol_ratio_raw

        case_rec["time_sec"] = round(time.time() - t0, 2)
        per_case_records.append(case_rec)

        mean_case_dice = np.mean(case_dices_raw) * 100 if case_dices_raw else 0.0
        mean_case_hd = np.mean(case_hds) if case_hds else float("nan")
        print(f"[{ci+1:02d}/{len(cases)}] {case:8s}: Raw Dice = {mean_case_dice:5.2f}% | HD95 = {mean_case_hd:5.2f} mm ({case_rec['time_sec']}s)", flush=True)

        del pred_vol, pred_vol_lcc, image, label
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    real_dices_raw = [u["clean_dice_raw"] for u in all_real_gt_units]
    real_dices_lcc = [u["clean_dice_lcc"] for u in all_real_gt_units]
    real_jcs_raw = [u["jc_raw"] for u in all_real_gt_units]
    real_jcs_lcc = [u["jc_lcc"] for u in all_real_gt_units]
    real_nsds_raw = [u["nsd_raw"] for u in all_real_gt_units]
    real_nsds_lcc = [u["nsd_lcc"] for u in all_real_gt_units]
    real_recalls_raw = [u["recall_raw"] for u in all_real_gt_units]
    real_precisions_raw = [u["precision_raw"] for u in all_real_gt_units]
    real_vols_raw = [u["vol_ratio_raw"] for u in all_real_gt_units]

    real_hds_raw_valid = [u["hd95_raw"] for u in all_real_gt_units if not np.isnan(u["hd95_raw"])]
    real_hds_lcc_valid = [u["hd95_lcc"] for u in all_real_gt_units if not np.isnan(u["hd95_lcc"])]

    # Per-class summary
    per_class_summary = {}
    macro_dices_raw, macro_dices_lcc = [], []
    macro_jcs_raw, macro_jcs_lcc = [], []
    macro_nsds_raw, macro_nsds_lcc = [], []
    macro_hds_raw_valid, macro_hds_lcc_valid = [], []
    macro_recalls, macro_precisions, macro_vols = [], [], []

    for c, c_name in sorted(eval_class_names.items()):
        c_units = [u for u in all_real_gt_units if u["class_id"] == c]
        c_units_all = [u for u in all_evaluated_units if u["class_id"] == c]
        if c_units or c_units_all:
            c_d_raw = float(np.mean([u["clean_dice_raw"] for u in c_units])) if c_units else float("nan")
            c_d_lcc = float(np.mean([u["clean_dice_lcc"] for u in c_units])) if c_units else float("nan")
            c_j_raw = float(np.mean([u["jc_raw"] for u in c_units])) if c_units else float("nan")
            c_j_lcc = float(np.mean([u["jc_lcc"] for u in c_units])) if c_units else float("nan")
            c_n_raw = float(np.mean([u["nsd_raw"] for u in c_units])) if c_units else float("nan")
            c_n_lcc = float(np.mean([u["nsd_lcc"] for u in c_units])) if c_units else float("nan")
            v_hd_raw = [u["hd95_raw"] for u in c_units if not np.isnan(u["hd95_raw"])]
            v_hd_lcc = [u["hd95_lcc"] for u in c_units if not np.isnan(u["hd95_lcc"])]
            c_h_raw_valid = float(np.mean(v_hd_raw)) if v_hd_raw else float("nan")
            c_h_lcc_valid = float(np.mean(v_hd_lcc)) if v_hd_lcc else float("nan")
            c_rec = float(np.mean([u["recall_raw"] for u in c_units])) if c_units else float("nan")
            c_prec = float(np.mean([u["precision_raw"] for u in c_units])) if c_units else float("nan")
            c_vol = float(np.mean([u["vol_ratio_raw"] for u in c_units])) if c_units else float("nan")

            c_rec_summary = {
                "dice_clean_raw": c_d_raw,
                "dice_clean_lcc": c_d_lcc,
                "nsd_raw": c_n_raw,
                "nsd_lcc": c_n_lcc,
                "hd95_raw_valid": c_h_raw_valid,
                "hd95_lcc_valid": c_h_lcc_valid,
                "far_fp_raw": int(sum(u["far_fp_raw"] for u in c_units_all)),
                "far_fp_lcc": int(sum(u["far_fp_lcc"] for u in c_units_all)),
                "recall_raw": c_rec,
                "precision_raw": c_prec,
                "vol_ratio_raw": c_vol,
                "n_cases": len(c_units),
            }
            if ds != "synapse":
                c_rec_summary["jc_raw"] = c_j_raw
                c_rec_summary["jc_lcc"] = c_j_lcc
            per_class_summary[c_name] = c_rec_summary
            if c_units:
                macro_dices_raw.append(c_d_raw)
                macro_dices_lcc.append(c_d_lcc)
                macro_jcs_raw.append(c_j_raw)
                macro_jcs_lcc.append(c_j_lcc)
                macro_nsds_raw.append(c_n_raw)
                macro_nsds_lcc.append(c_n_lcc)
                if not np.isnan(c_h_raw_valid):
                    macro_hds_raw_valid.append(c_h_raw_valid)
                if not np.isnan(c_h_lcc_valid):
                    macro_hds_lcc_valid.append(c_h_lcc_valid)
                macro_recalls.append(c_rec)
                macro_precisions.append(c_prec)
                macro_vols.append(c_vol)

    overall = {
        "primary_n_units": len(all_real_gt_units),
        "primary_dice_raw_macro": float(np.mean(macro_dices_raw)) if macro_dices_raw else float("nan"),
        "primary_dice_lcc_macro": float(np.mean(macro_dices_lcc)) if macro_dices_lcc else float("nan"),
        "primary_dice_raw_pooled": float(np.mean(real_dices_raw)) if real_dices_raw else float("nan"),
        "primary_dice_lcc_pooled": float(np.mean(real_dices_lcc)) if real_dices_lcc else float("nan"),
        "primary_nsd_raw_macro": float(np.mean(macro_nsds_raw)) if macro_nsds_raw else float("nan"),
        "primary_nsd_lcc_macro": float(np.mean(macro_nsds_lcc)) if macro_nsds_lcc else float("nan"),
        "primary_hd95_raw_valid_macro": float(np.mean(macro_hds_raw_valid)) if macro_hds_raw_valid else float("nan"),
        "primary_hd95_lcc_valid_macro": float(np.mean(macro_hds_lcc_valid)) if macro_hds_lcc_valid else float("nan"),
        "primary_hd95_raw_valid_pooled": float(np.mean(real_hds_raw_valid)) if real_hds_raw_valid else float("nan"),
        "primary_recall_macro": float(np.mean(macro_recalls)) if macro_recalls else float("nan"),
        "primary_precision_macro": float(np.mean(macro_precisions)) if macro_precisions else float("nan"),
        "primary_vol_ratio_macro": float(np.mean(macro_vols)) if macro_vols else float("nan"),
        "primary_far_fp_raw": int(sum(u["far_fp_raw"] for u in all_evaluated_units)),
        "primary_far_fp_lcc": int(sum(u["far_fp_lcc"] for u in all_evaluated_units)),
    }
    if ds != "synapse":
        overall["dataset"] = ds
        overall["primary_jc_raw_macro"] = float(np.mean(macro_jcs_raw)) if macro_jcs_raw else float("nan")
        overall["primary_jc_lcc_macro"] = float(np.mean(macro_jcs_lcc)) if macro_jcs_lcc else float("nan")

    print(
        f"\n[{tag}] SUMMARY ({ds.upper()}): Raw Macro Dice: {overall['primary_dice_raw_macro']*100:.2f}% | "
        f"LCC Macro Dice: {overall['primary_dice_lcc_macro']*100:.2f}% | "
        f"Raw NSD (2mm): {overall['primary_nsd_raw_macro']*100:.2f}% | "
        f"HD95: {overall['primary_hd95_raw_valid_macro']:.2f} mm | "
        f"Far-FP raw/LCC: {overall['primary_far_fp_raw']:,d} / {overall['primary_far_fp_lcc']:,d} vox | "
        f"Recall: {overall['primary_recall_macro']*100:.2f}% | "
        f"Precision: {overall['primary_precision_macro']*100:.2f}% | "
        f"Vol Ratio: {overall['primary_vol_ratio_macro']:.4f}",
        flush=True,
    )

    def _py(v):
        if isinstance(v, (np.bool_, bool)):
            return bool(v)
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, (np.floating, float)):
            return float(v)
        return v

    units = [{k: _py(v) for k, v in u.items()} for u in (all_real_gt_units + empty_gt_units)]

    return {
        "overall": overall,
        "per_class": per_class_summary,
        "cases": per_case_records,
        "units": units,
    }


@torch.no_grad()
def evaluate_polyp(
    student_net: nn.Module,
    test_dir: Optional[str],
    device: torch.device,
    batch_size: int = 8,
    use_amp: bool = True,
    max_cases: Optional[int] = None,
    brd_radius: int = 2,
    nsd_tau: float = 2.0,
    iad_root: Optional[str] = None,
    save_preds_dir: Optional[str] = None,
    tag: str = "",
    img_size: int = 256,
    subdatasets: Tuple[str, ...] = POLYP_TEST_SETS,
) -> Dict[str, Union[dict, list, float]]:
    """Evaluate a binary Polyp segmentation model across Kvasir, CVC-ClinicDB, CVC-ColonDB, ETIS-LaribPolypDB, and CVC-300."""
    student_net.eval()
    polyp_root = find_polyp_root(test_dir, iad_root)

    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    per_dataset_summary: Dict[str, dict] = {}
    per_case_records: List[dict] = []
    all_units: List[dict] = []

    if save_preds_dir:
        os.makedirs(save_preds_dir, exist_ok=True)

    for sub in subdatasets:
        try:
            ds_sub = PolypDataset(
                data_dir=polyp_root,
                split="test",
                subdataset=sub,
                iad_root=iad_root,
                augment=False,
                img_size=img_size,
            )
        except FileNotFoundError:
            continue

        n_eval = min(len(ds_sub), max_cases) if max_cases else len(ds_sub)
        sub_units: List[dict] = []
        t0 = time.time()

        for idx in range(n_eval):
            sample = ds_sub[idx]
            x_in = sample["image"].unsqueeze(0).to(device)
            gt_mask = sample["label"].numpy().astype(np.uint8)
            case_name = f"{sub}/{sample['case_name']}"

            if use_amp and ("cuda" in str(device)):
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    out = student_net(x_in, False)
            else:
                out = student_net(x_in, False)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            if logits.shape[1] == 1:
                pred_raw = (torch.sigmoid(logits.float()) >= 0.5).squeeze().cpu().numpy().astype(np.uint8)
            else:
                pred_raw = torch.argmax(logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)

            pred_lcc, _ = apply_3d_lcc_per_class(pred_raw, connectivity=2, num_classes=2)

            if save_preds_dir:
                sub_pred_dir = os.path.join(save_preds_dir, sub)
                os.makedirs(sub_pred_dir, exist_ok=True)
                np.save(os.path.join(sub_pred_dir, f"{sample['case_name']}_pred_{tag}.npy"), pred_raw)

            res_raw = compute_volume_metrics_per_class(
                pred_raw == 1, gt_mask == 1, radius=brd_radius, tau_mm=nsd_tau, voxelspacing=(1.0, 1.0)
            )
            res_lcc = compute_volume_metrics_per_class(
                pred_lcc == 1, gt_mask == 1, radius=brd_radius, tau_mm=nsd_tau, voxelspacing=(1.0, 1.0)
            )

            g_sum = res_raw["g_sum"]
            p_sum_raw = res_raw["p_sum"]
            p_sum_lcc = res_lcc["p_sum"]
            tp_raw = res_raw["tp"]
            tp_lcc = res_lcc["tp"]

            recall_raw = float(tp_raw / g_sum) if g_sum > 0 else (1.0 if tp_raw == 0 else 0.0)
            recall_lcc = float(tp_lcc / g_sum) if g_sum > 0 else (1.0 if tp_lcc == 0 else 0.0)
            precision_raw = float(tp_raw / p_sum_raw) if p_sum_raw > 0 else (1.0 if g_sum == 0 else 0.0)
            precision_lcc = float(tp_lcc / p_sum_lcc) if p_sum_lcc > 0 else (1.0 if g_sum == 0 else 0.0)
            vol_ratio_raw = float(p_sum_raw / g_sum) if g_sum > 0 else (1.0 if p_sum_raw == 0 else float("inf"))
            vol_ratio_lcc = float(p_sum_lcc / g_sum) if g_sum > 0 else (1.0 if p_sum_lcc == 0 else float("inf"))

            unit_info = {
                "case": case_name,
                "subdataset": sub,
                "organ": "polyp",
                "class_id": 1,
                "gt_present": not res_raw["gt_empty"],
                "empty_pred_raw": res_raw.get("empty_pred", False),
                "empty_pred_lcc": res_lcc.get("empty_pred", False),
                "clean_dice_raw": res_raw["dice_clean"],
                "clean_dice_lcc": res_lcc["dice_clean"],
                "jc_raw": res_raw["jc"],
                "jc_lcc": res_lcc["jc"],
                "nsd_raw": res_raw["surface_dice"],
                "nsd_lcc": res_lcc["surface_dice"],
                "hd95_raw": res_raw["hd95"],
                "hd95_lcc": res_lcc["hd95"],
                "tp_raw": tp_raw,
                "tp_lcc": tp_lcc,
                "p_sum_raw": p_sum_raw,
                "p_sum_lcc": p_sum_lcc,
                "g_sum": g_sum,
                "recall_raw": recall_raw,
                "recall_lcc": recall_lcc,
                "precision_raw": precision_raw,
                "precision_lcc": precision_lcc,
                "vol_ratio_raw": vol_ratio_raw,
                "vol_ratio_lcc": vol_ratio_lcc,
                "far_fp_raw": res_raw.get("far_fp", 0),
                "far_fp_lcc": res_lcc.get("far_fp", 0),
                "near_fp_raw": res_raw.get("near_fp", 0),
                "near_fp_lcc": res_lcc.get("near_fp", 0),
            }
            sub_units.append(unit_info)
            all_units.append(unit_info)
            per_case_records.append({
                "case": case_name,
                "subdataset": sub,
                "voxelspacing": (1.0, 1.0, 1.0),
                "dice_clean_polyp": res_raw["dice_clean"],
                "dice_lcc_polyp": res_lcc["dice_clean"],
                "jc_polyp": res_raw["jc"],
            })

        v_hd_raw = [u["hd95_raw"] for u in sub_units if not np.isnan(u["hd95_raw"])]
        v_hd_lcc = [u["hd95_lcc"] for u in sub_units if not np.isnan(u["hd95_lcc"])]
        # In Polyp_Tool/test_isic.py, empty-pred HD95 is 0.0 when computing mean HD95 across all test images
        hd_paper_raw = [0.0 if np.isnan(u["hd95_raw"]) else u["hd95_raw"] for u in sub_units]
        sub_rec = {
            "n_cases": len(sub_units),
            "dice_clean_raw": float(np.mean([u["clean_dice_raw"] for u in sub_units])),
            "dice_clean_lcc": float(np.mean([u["clean_dice_lcc"] for u in sub_units])),
            "jc_raw": float(np.mean([u["jc_raw"] for u in sub_units])),
            "jc_lcc": float(np.mean([u["jc_lcc"] for u in sub_units])),
            "nsd_raw": float(np.mean([u["nsd_raw"] for u in sub_units])),
            "nsd_lcc": float(np.mean([u["nsd_lcc"] for u in sub_units])),
            "hd95_raw_valid": float(np.mean(v_hd_raw)) if v_hd_raw else float("nan"),
            "hd95_lcc_valid": float(np.mean(v_hd_lcc)) if v_hd_lcc else float("nan"),
            "hd95_paper": float(np.mean(hd_paper_raw)) if hd_paper_raw else float("nan"),
            "recall_raw": float(np.mean([u["recall_raw"] for u in sub_units])),
            "precision_raw": float(np.mean([u["precision_raw"] for u in sub_units])),
            "vol_ratio_raw": float(np.mean([u["vol_ratio_raw"] for u in sub_units if np.isfinite(u["vol_ratio_raw"])])) if sub_units else float("nan"),
            "far_fp_raw": int(sum(u["far_fp_raw"] for u in sub_units)),
            "far_fp_lcc": int(sum(u["far_fp_lcc"] for u in sub_units)),
            "time_sec": round(time.time() - t0, 2),
        }
        per_dataset_summary[sub] = sub_rec
        print(
            f"[{tag}] Polyp {sub:18s} (n={len(sub_units):3d}): "
            f"DSC={sub_rec['dice_clean_raw']*100:5.2f}% (LCC {sub_rec['dice_clean_lcc']*100:5.2f}%) | "
            f"JC={sub_rec['jc_raw']*100:5.2f}% | HD95={sub_rec['hd95_raw_valid']:5.2f} px ({sub_rec['time_sec']}s)",
            flush=True,
        )

    if not per_dataset_summary:
        raise FileNotFoundError(f"[evaluate_polyp] No Polyp test datasets found under '{polyp_root}'")

    primary_sets = [s for s in POLYP_PRIMARY_4_SETS if s in per_dataset_summary]
    if not primary_sets:
        primary_sets = list(per_dataset_summary.keys())
    all_sets = list(per_dataset_summary.keys())

    def _macro(sets_list, key):
        vals = [per_dataset_summary[s][key] for s in sets_list if not np.isnan(per_dataset_summary[s][key])]
        return float(np.mean(vals)) if vals else float("nan")

    overall: Dict[str, Union[int, float, str]] = {
        "dataset": "polyp",
        "primary_n_units": int(sum(per_dataset_summary[s]["n_cases"] for s in primary_sets)),
        "total_n_units": len(all_units),
        "primary_dice_raw_macro": _macro(primary_sets, "dice_clean_raw"),
        "primary_dice_lcc_macro": _macro(primary_sets, "dice_clean_lcc"),
        "primary_jc_raw_macro": _macro(primary_sets, "jc_raw"),
        "primary_jc_lcc_macro": _macro(primary_sets, "jc_lcc"),
        "polyp_5sets_dice_raw_macro": _macro(all_sets, "dice_clean_raw"),
        "polyp_5sets_dice_lcc_macro": _macro(all_sets, "dice_clean_lcc"),
        "polyp_5sets_jc_raw_macro": _macro(all_sets, "jc_raw"),
        "primary_dice_raw_pooled": float(np.mean([u["clean_dice_raw"] for u in all_units])),
        "primary_dice_lcc_pooled": float(np.mean([u["clean_dice_lcc"] for u in all_units])),
        "primary_nsd_raw_macro": _macro(primary_sets, "nsd_raw"),
        "primary_nsd_lcc_macro": _macro(primary_sets, "nsd_lcc"),
        "primary_hd95_raw_valid_macro": _macro(primary_sets, "hd95_raw_valid"),
        "primary_hd95_lcc_valid_macro": _macro(primary_sets, "hd95_lcc_valid"),
        "primary_recall_macro": _macro(primary_sets, "recall_raw"),
        "primary_precision_macro": _macro(primary_sets, "precision_raw"),
        "primary_vol_ratio_macro": _macro(primary_sets, "vol_ratio_raw"),
        "primary_far_fp_raw": int(sum(u["far_fp_raw"] for u in all_units)),
        "primary_far_fp_lcc": int(sum(u["far_fp_lcc"] for u in all_units)),
    }
    for s, s_rec in per_dataset_summary.items():
        overall[f"dice_{s}"] = s_rec["dice_clean_raw"]
        overall[f"dice_lcc_{s}"] = s_rec["dice_clean_lcc"]
        overall[f"jc_{s}"] = s_rec["jc_raw"]
        overall[f"hd95_{s}"] = s_rec["hd95_raw_valid"]

    print(
        f"\n[{tag}] SUMMARY (POLYP): 4-Set Macro DSC: {overall['primary_dice_raw_macro']*100:.2f}% "
        f"(5-Set: {overall['polyp_5sets_dice_raw_macro']*100:.2f}%) | "
        f"4-Set Macro JC: {overall['primary_jc_raw_macro']*100:.2f}% | "
        f"HD95: {overall['primary_hd95_raw_valid_macro']:.2f} px",
        flush=True,
    )

    return {
        "overall": overall,
        "per_class": per_dataset_summary,
        "per_dataset": per_dataset_summary,
        "cases": per_case_records,
        "units": all_units,
    }

