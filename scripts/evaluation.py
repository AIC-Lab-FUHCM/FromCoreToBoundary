from __future__ import annotations

import gc
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn
from scipy.ndimage import zoom

from datasets.data import (
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


def get_surface_3d(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return mask & (~ndi.binary_erosion(mask))


def get_bounding_box_3d(mask: np.ndarray, pad: int = 8) -> Optional[Tuple[slice, ...]]:
    where = np.where(mask)
    if len(where[0]) == 0:
        return None
    return tuple(
        slice(max(0, int(c.min()) - pad), min(mask.shape[i], int(c.max()) + pad + 1))
        for i, c in enumerate(where)
    )


def get_case_voxelspacing(case_path: str) -> Tuple[float, float, float]:
    try:
        if case_path.endswith((".h5", ".npy.h5")):
            with h5py.File(case_path, "r") as f:
                for k in ("voxelspacing", "spacing", "pixel_spacing"):
                    if k in f.attrs:
                        val = f.attrs[k]
                        return (float(val[0]), float(val[1]), float(val[2]))
    except Exception:
        pass
    return (1.0, 1.0, 1.0)


def apply_3d_lcc_per_class(pred_vol: np.ndarray, connectivity: int = 3, num_classes: int = 9) -> np.ndarray:
    ndim = pred_vol.ndim
    struct = ndi.generate_binary_structure(ndim, min(connectivity, ndim))
    filtered_vol = np.zeros_like(pred_vol)
    for c in range(1, num_classes):
        mask_c = (pred_vol == c)
        if not mask_c.any():
            continue
        labeled, num_features = ndi.label(mask_c, structure=struct)
        if num_features <= 1:
            filtered_vol[mask_c] = c
        else:
            sizes = np.bincount(labeled.flat)[1:]
            filtered_vol[labeled == (int(np.argmax(sizes)) + 1)] = c
    return filtered_vol


def compute_volume_metrics_per_class(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    tau_mm: float = 2.0,
    voxelspacing: Optional[Tuple[float, ...]] = (1.0, 1.0, 1.0),
) -> Dict[str, Union[float, bool, int]]:
    p_bin = (pred_mask > 0)
    g_bin = (gt_mask > 0)
    p_sum = int(p_bin.sum())
    g_sum = int(g_bin.sum())

    if p_sum == 0 and g_sum == 0:
        return {"dice_clean": 1.0, "jc": 1.0, "hd95": float("nan"), "surface_dice": 1.0, "gt_empty": True, "tp": 0, "p_sum": 0, "g_sum": 0}
    if p_sum > 0 and g_sum == 0:
        return {"dice_clean": 0.0, "jc": 0.0, "hd95": float("nan"), "surface_dice": 0.0, "gt_empty": True, "tp": 0, "p_sum": p_sum, "g_sum": 0}
    if p_sum == 0 and g_sum > 0:
        return {"dice_clean": 0.0, "jc": 0.0, "hd95": float("nan"), "surface_dice": 0.0, "gt_empty": False, "tp": 0, "p_sum": 0, "g_sum": g_sum}

    inter = int(np.logical_and(p_bin, g_bin).sum())
    union = int(np.logical_or(p_bin, g_bin).sum())
    dice = float(2.0 * inter / (p_sum + g_sum))
    jc = float(inter / union) if union > 0 else 1.0

    bb = get_bounding_box_3d(p_bin | g_bin, pad=int(np.ceil(tau_mm)) + 4)
    p_c = p_bin[bb] if bb is not None else p_bin
    g_c = g_bin[bb] if bb is not None else g_bin

    spacing_use = voxelspacing[: p_c.ndim] if voxelspacing is not None else None
    hd = float("nan")
    if medpy_metric is not None:
        try:
            hd = float(medpy_metric.binary.hd95(p_c, g_c, voxelspacing=spacing_use))
        except Exception:
            hd = float("nan")

    s_pred = get_surface_3d(p_c)
    s_gt = get_surface_3d(g_c)
    if s_pred.sum() == 0 or s_gt.sum() == 0:
        sd = 0.0
    else:
        dt_gt = ndi.distance_transform_edt(np.ascontiguousarray(~s_gt, dtype=bool), sampling=spacing_use)
        dt_pred = ndi.distance_transform_edt(np.ascontiguousarray(~s_pred, dtype=bool), sampling=spacing_use)
        pred_close = int((dt_gt[s_pred] <= tau_mm).sum())
        gt_close = int((dt_pred[s_gt] <= tau_mm).sum())
        sd = float((pred_close + gt_close) / (s_pred.sum() + s_gt.sum()))

    return {
        "dice_clean": dice,
        "jc": jc,
        "hd95": hd,
        "surface_dice": sd,
        "gt_empty": False,
        "tp": inter,
        "p_sum": p_sum,
        "g_sum": g_sum,
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
            use_amp=use_amp,
            max_cases=max_cases,
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
        resolved_test_list = resolve_acdc_list_file(requested=test_list, split="test", iad_root=iad_root, data_dir=test_dir)
    else:
        eval_img_size = int(img_size or 512)
        eval_class_names = class_names or CLASS_NAMES
        test_dir = find_test_vol_dir(test_dir, iad_root)
        resolved_test_list = resolve_list_file(
            requested=test_list, list_name="test_vol.txt", iad_root=iad_root, list_dir=list_dir, data_dir=test_dir
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
        raise FileNotFoundError(f"No valid test volumes found in '{test_dir}'")
    if max_cases:
        cases = cases[:max_cases]

    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    per_case_records = []
    all_real_gt_units = []

    if save_preds_dir:
        os.makedirs(save_preds_dir, exist_ok=True)

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
                chunk = np.stack([zoom(chunk[s], (eval_img_size / h, eval_img_size / w), order=3) for s in range(chunk.shape[0])], axis=0)

            x_in = torch.from_numpy(chunk).unsqueeze(1).repeat(1, 3, 1, 1).float().to(device)
            if use_amp and ("cuda" in str(device)):
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    out = student_net(x_in, True)
            else:
                out = student_net(x_in, True)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            pred_chunk = torch.argmax(logits, dim=1).to(torch.uint8).cpu().numpy()

            if h != eval_img_size or w != eval_img_size:
                pred_chunk = np.stack([zoom(pred_chunk[s], (h / eval_img_size, w / eval_img_size), order=0) for s in range(pred_chunk.shape[0])], axis=0)
            preds.append(pred_chunk)
            slice_idx = end_idx

        pred_vol = np.concatenate(preds, axis=0)
        pred_vol_lcc = apply_3d_lcc_per_class(pred_vol, num_classes=num_classes_total)
        if save_preds_dir:
            np.save(os.path.join(save_preds_dir, f"{case}_pred_{tag}.npy"), pred_vol)
            np.save(os.path.join(save_preds_dir, f"{case}_pred_{tag}_lcc.npy"), pred_vol_lcc)

        voxelspacing = get_case_voxelspacing(case_path)
        case_rec = {"case": case, "voxelspacing": voxelspacing}
        case_dices_raw, case_hds = [], []

        for c, c_name in sorted(eval_class_names.items()):
            gt_c = (label == c)
            res_raw = compute_volume_metrics_per_class(pred_vol == c, gt_c, tau_mm=nsd_tau, voxelspacing=voxelspacing)
            res_lcc = compute_volume_metrics_per_class(pred_vol_lcc == c, gt_c, tau_mm=nsd_tau, voxelspacing=voxelspacing)
            g_sum = res_raw["g_sum"]
            p_sum_raw = res_raw["p_sum"]
            tp_raw = res_raw["tp"]
            recall_raw = float(tp_raw / g_sum) if g_sum > 0 else (1.0 if tp_raw == 0 else 0.0)
            precision_raw = float(tp_raw / p_sum_raw) if p_sum_raw > 0 else (1.0 if g_sum == 0 else 0.0)
            vol_ratio_raw = float(p_sum_raw / g_sum) if g_sum > 0 else (1.0 if p_sum_raw == 0 else float("inf"))

            unit_info = {
                "case": case,
                "organ": c_name,
                "class_id": c,
                "gt_present": not res_raw["gt_empty"],
                "clean_dice_raw": res_raw["dice_clean"],
                "clean_dice_lcc": res_lcc["dice_clean"],
                "jc_raw": res_raw["jc"],
                "jc_lcc": res_lcc["jc"],
                "nsd_raw": res_raw["surface_dice"],
                "nsd_lcc": res_lcc["surface_dice"],
                "hd95_raw": res_raw["hd95"],
                "hd95_lcc": res_lcc["hd95"],
                "recall_raw": recall_raw,
                "precision_raw": precision_raw,
                "vol_ratio_raw": vol_ratio_raw,
            }
            if unit_info["gt_present"]:
                all_real_gt_units.append(unit_info)
                case_dices_raw.append(res_raw["dice_clean"])
                if not np.isnan(res_raw["hd95"]):
                    case_hds.append(res_raw["hd95"])

            case_rec[f"dice_clean_{c_name}"] = res_raw["dice_clean"]
            case_rec[f"dice_lcc_{c_name}"] = res_lcc["dice_clean"]

        case_rec["time_sec"] = round(time.time() - t0, 2)
        per_case_records.append(case_rec)
        mean_case_dice = np.mean(case_dices_raw) * 100 if case_dices_raw else 0.0
        mean_case_hd = np.mean(case_hds) if case_hds else float("nan")
        print(f"[{ci+1:02d}/{len(cases)}] {case:8s}: Raw Dice = {mean_case_dice:5.2f}% | HD95 = {mean_case_hd:5.2f} ({case_rec['time_sec']}s)", flush=True)

        del pred_vol, pred_vol_lcc, image, label
        gc.collect()

    per_class_summary = {}
    macro_dices_raw, macro_dices_lcc = [], []
    macro_jcs_raw, macro_jcs_lcc = [], []
    macro_nsds_raw, macro_nsds_lcc = [], []
    macro_hds_raw_valid, macro_hds_lcc_valid = [], []
    macro_recalls, macro_precisions, macro_vols = [], [], []

    for c, c_name in sorted(eval_class_names.items()):
        c_units = [u for u in all_real_gt_units if u["class_id"] == c]
        if not c_units:
            continue
        c_d_raw = float(np.mean([u["clean_dice_raw"] for u in c_units]))
        c_d_lcc = float(np.mean([u["clean_dice_lcc"] for u in c_units]))
        c_j_raw = float(np.mean([u["jc_raw"] for u in c_units]))
        c_j_lcc = float(np.mean([u["jc_lcc"] for u in c_units]))
        c_n_raw = float(np.mean([u["nsd_raw"] for u in c_units]))
        c_n_lcc = float(np.mean([u["nsd_lcc"] for u in c_units]))
        v_hd_raw = [u["hd95_raw"] for u in c_units if not np.isnan(u["hd95_raw"])]
        v_hd_lcc = [u["hd95_lcc"] for u in c_units if not np.isnan(u["hd95_lcc"])]
        c_h_raw = float(np.mean(v_hd_raw)) if v_hd_raw else float("nan")
        c_h_lcc = float(np.mean(v_hd_lcc)) if v_hd_lcc else float("nan")
        c_rec = float(np.mean([u["recall_raw"] for u in c_units]))
        c_prec = float(np.mean([u["precision_raw"] for u in c_units]))
        c_vol = float(np.mean([u["vol_ratio_raw"] for u in c_units]))

        per_class_summary[c_name] = {
            "dice_clean_raw": c_d_raw,
            "dice_clean_lcc": c_d_lcc,
            "nsd_raw": c_n_raw,
            "nsd_lcc": c_n_lcc,
            "hd95_raw_valid": c_h_raw,
            "hd95_lcc_valid": c_h_lcc,
            "recall_raw": c_rec,
            "precision_raw": c_prec,
            "vol_ratio_raw": c_vol,
            "n_cases": len(c_units),
        }
        macro_dices_raw.append(c_d_raw)
        macro_dices_lcc.append(c_d_lcc)
        macro_jcs_raw.append(c_j_raw)
        macro_jcs_lcc.append(c_j_lcc)
        macro_nsds_raw.append(c_n_raw)
        macro_nsds_lcc.append(c_n_lcc)
        if not np.isnan(c_h_raw):
            macro_hds_raw_valid.append(c_h_raw)
        if not np.isnan(c_h_lcc):
            macro_hds_lcc_valid.append(c_h_lcc)
        macro_recalls.append(c_rec)
        macro_precisions.append(c_prec)
        macro_vols.append(c_vol)

    overall = {
        "primary_n_units": len(all_real_gt_units),
        "primary_dice_raw_macro": float(np.mean(macro_dices_raw)) if macro_dices_raw else float("nan"),
        "primary_dice_lcc_macro": float(np.mean(macro_dices_lcc)) if macro_dices_lcc else float("nan"),
        "primary_nsd_raw_macro": float(np.mean(macro_nsds_raw)) if macro_nsds_raw else float("nan"),
        "primary_nsd_lcc_macro": float(np.mean(macro_nsds_lcc)) if macro_nsds_lcc else float("nan"),
        "primary_hd95_raw_valid_macro": float(np.mean(macro_hds_raw_valid)) if macro_hds_raw_valid else float("nan"),
        "primary_hd95_lcc_valid_macro": float(np.mean(macro_hds_lcc_valid)) if macro_hds_lcc_valid else float("nan"),
        "primary_recall_macro": float(np.mean(macro_recalls)) if macro_recalls else float("nan"),
        "primary_precision_macro": float(np.mean(macro_precisions)) if macro_precisions else float("nan"),
        "primary_vol_ratio_macro": float(np.mean(macro_vols)) if macro_vols else float("nan"),
    }
    if ds != "synapse":
        overall["dataset"] = ds
        overall["primary_jc_raw_macro"] = float(np.mean(macro_jcs_raw)) if macro_jcs_raw else float("nan")
        overall["primary_jc_lcc_macro"] = float(np.mean(macro_jcs_lcc)) if macro_jcs_lcc else float("nan")

    return {
        "overall": overall,
        "per_class": per_class_summary,
        "cases": per_case_records,
    }


@torch.no_grad()
def evaluate_polyp(
    student_net: nn.Module,
    test_dir: Optional[str],
    device: torch.device,
    use_amp: bool = True,
    max_cases: Optional[int] = None,
    nsd_tau: float = 2.0,
    iad_root: Optional[str] = None,
    save_preds_dir: Optional[str] = None,
    tag: str = "",
    img_size: int = 256,
    subdatasets: Tuple[str, ...] = POLYP_TEST_SETS,
) -> Dict[str, Union[dict, list, float]]:
    student_net.eval()
    polyp_root = find_polyp_root(test_dir, iad_root)
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    per_dataset_summary: Dict[str, dict] = {}
    per_case_records: List[dict] = []
    all_units: List[dict] = []

    if save_preds_dir:
        os.makedirs(save_preds_dir, exist_ok=True)

    for sub in subdatasets:
        try:
            ds_sub = PolypDataset(data_dir=polyp_root, split="test", subdataset=sub, iad_root=iad_root, augment=False, img_size=img_size)
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

            pred_lcc = apply_3d_lcc_per_class(pred_raw, connectivity=2, num_classes=2)
            res_raw = compute_volume_metrics_per_class(pred_raw == 1, gt_mask == 1, tau_mm=nsd_tau, voxelspacing=(1.0, 1.0))
            res_lcc = compute_volume_metrics_per_class(pred_lcc == 1, gt_mask == 1, tau_mm=nsd_tau, voxelspacing=(1.0, 1.0))

            g_sum = res_raw["g_sum"]
            p_sum_raw = res_raw["p_sum"]
            tp_raw = res_raw["tp"]
            recall_raw = float(tp_raw / g_sum) if g_sum > 0 else (1.0 if tp_raw == 0 else 0.0)
            precision_raw = float(tp_raw / p_sum_raw) if p_sum_raw > 0 else (1.0 if g_sum == 0 else 0.0)
            vol_ratio_raw = float(p_sum_raw / g_sum) if g_sum > 0 else (1.0 if p_sum_raw == 0 else float("inf"))

            unit_info = {
                "case": case_name,
                "subdataset": sub,
                "clean_dice_raw": res_raw["dice_clean"],
                "clean_dice_lcc": res_lcc["dice_clean"],
                "jc_raw": res_raw["jc"],
                "jc_lcc": res_lcc["jc"],
                "nsd_raw": res_raw["surface_dice"],
                "nsd_lcc": res_lcc["surface_dice"],
                "hd95_raw": res_raw["hd95"],
                "hd95_lcc": res_lcc["hd95"],
                "recall_raw": recall_raw,
                "precision_raw": precision_raw,
                "vol_ratio_raw": vol_ratio_raw,
            }
            sub_units.append(unit_info)
            all_units.append(unit_info)
            per_case_records.append({"case": case_name, "subdataset": sub, "voxelspacing": (1.0, 1.0, 1.0)})

        v_hd_raw = [u["hd95_raw"] for u in sub_units if not np.isnan(u["hd95_raw"])]
        v_hd_lcc = [u["hd95_lcc"] for u in sub_units if not np.isnan(u["hd95_lcc"])]
        per_dataset_summary[sub] = {
            "n_cases": len(sub_units),
            "dice_clean_raw": float(np.mean([u["clean_dice_raw"] for u in sub_units])),
            "dice_clean_lcc": float(np.mean([u["clean_dice_lcc"] for u in sub_units])),
            "jc_raw": float(np.mean([u["jc_raw"] for u in sub_units])),
            "jc_lcc": float(np.mean([u["jc_lcc"] for u in sub_units])),
            "nsd_raw": float(np.mean([u["nsd_raw"] for u in sub_units])),
            "nsd_lcc": float(np.mean([u["nsd_lcc"] for u in sub_units])),
            "hd95_raw_valid": float(np.mean(v_hd_raw)) if v_hd_raw else float("nan"),
            "hd95_lcc_valid": float(np.mean(v_hd_lcc)) if v_hd_lcc else float("nan"),
            "recall_raw": float(np.mean([u["recall_raw"] for u in sub_units])),
            "precision_raw": float(np.mean([u["precision_raw"] for u in sub_units])),
            "vol_ratio_raw": float(np.mean([u["vol_ratio_raw"] for u in sub_units if np.isfinite(u["vol_ratio_raw"])])) if sub_units else float("nan"),
            "time_sec": round(time.time() - t0, 2),
        }

    if not per_dataset_summary:
        raise FileNotFoundError(f"No Polyp test datasets found under '{polyp_root}'")

    primary_sets = [s for s in POLYP_PRIMARY_4_SETS if s in per_dataset_summary] or list(per_dataset_summary.keys())
    all_sets = list(per_dataset_summary.keys())

    def _macro(sets_list, key):
        vals = [per_dataset_summary[s][key] for s in sets_list if not np.isnan(per_dataset_summary[s][key])]
        return float(np.mean(vals)) if vals else float("nan")

    overall: Dict[str, Union[int, float, str]] = {
        "dataset": "polyp",
        "primary_n_units": int(sum(per_dataset_summary[s]["n_cases"] for s in primary_sets)),
        "primary_dice_raw_macro": _macro(primary_sets, "dice_clean_raw"),
        "primary_dice_lcc_macro": _macro(primary_sets, "dice_clean_lcc"),
        "primary_jc_raw_macro": _macro(primary_sets, "jc_raw"),
        "primary_jc_lcc_macro": _macro(primary_sets, "jc_lcc"),
        "polyp_5sets_dice_raw_macro": _macro(all_sets, "dice_clean_raw"),
        "polyp_5sets_dice_lcc_macro": _macro(all_sets, "dice_clean_lcc"),
        "primary_nsd_raw_macro": _macro(primary_sets, "nsd_raw"),
        "primary_nsd_lcc_macro": _macro(primary_sets, "nsd_lcc"),
        "primary_hd95_raw_valid_macro": _macro(primary_sets, "hd95_raw_valid"),
        "primary_hd95_lcc_valid_macro": _macro(primary_sets, "hd95_lcc_valid"),
        "primary_recall_macro": _macro(primary_sets, "recall_raw"),
        "primary_precision_macro": _macro(primary_sets, "precision_raw"),
        "primary_vol_ratio_macro": _macro(primary_sets, "vol_ratio_raw"),
    }
    for s, s_rec in per_dataset_summary.items():
        overall[f"dice_{s}"] = s_rec["dice_clean_raw"]
        overall[f"jc_{s}"] = s_rec["jc_raw"]
        overall[f"hd95_{s}"] = s_rec["hd95_raw_valid"]

    return {
        "overall": overall,
        "per_class": per_dataset_summary,
        "cases": per_case_records,
    }
