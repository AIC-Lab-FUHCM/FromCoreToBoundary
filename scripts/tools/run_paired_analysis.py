import json
import os
import numpy as np

path_v = "outputs/colab_pilot_results/Vanilla_TGBD/eval_results.json"
path_e = "outputs/colab_pilot_results/Vanilla_TGBD_CBA_EMA/eval_results.json"

with open(path_v, "r") as f:
    res_v = json.load(f)
with open(path_e, "r") as f:
    res_e = json.load(f)

CLASS_NAMES = ["spleen", "right_kidney", "left_kidney", "gallbladder", "liver", "stomach", "aorta", "pancreas"]

cases_v = {r["case"]: r for r in res_v["per_case"]}
cases_e = {r["case"]: r for r in res_e["per_case"]}
common_cases = sorted(list(set(cases_v.keys()) & set(cases_e.keys())))

paired_deltas_hd = []
paired_details = []

for c_id in common_cases:
    rv, re_ = cases_v[c_id], cases_e[c_id]
    for c_name in CLASS_NAMES:
        hd_v = rv.get(f"hd95_{c_name}")
        hd_e = re_.get(f"hd95_{c_name}")
        dc_v = rv.get(f"dice_clean_{c_name}", 0.0)
        dc_e = re_.get(f"dice_clean_{c_name}", 0.0)
        
        valid_v = hd_v is not None and not np.isnan(hd_v)
        valid_e = hd_e is not None and not np.isnan(hd_e)
        
        if valid_v and valid_e:
            delta = float(hd_e - hd_v)
            paired_deltas_hd.append(delta)
            paired_details.append({
                "case": c_id, "organ": c_name,
                "hd_vanilla": float(hd_v), "hd_ema": float(hd_e), "delta_hd": delta,
                "dice_v": float(dc_v), "dice_e": float(dc_e),
                "delta_dice": float(dc_e - dc_v)
            })

paired_deltas_hd = np.array(paired_deltas_hd)
n_total = len(paired_deltas_hd)
n_improved = int(np.sum(paired_deltas_hd < -0.1))
n_unchanged = int(np.sum(np.abs(paired_deltas_hd) <= 0.1))
n_worse = int(np.sum(paired_deltas_hd > 0.1))

print("=== 1. PAIRED STATS (DELTA_i = EMA - Vanilla) ===")
print(f"Total valid patient-organ pairs: {n_total}")
print(f"Improved pairs (HD95 decreased >0.1mm): {n_improved} ({n_improved/n_total*100:.2f}%)")
print(f"Unchanged pairs (|delta| <= 0.1mm):   {n_unchanged} ({n_unchanged/n_total*100:.2f}%)")
print(f"Worse pairs (HD95 increased >0.1mm):     {n_worse} ({n_worse/n_total*100:.2f}%)")
print(f"Median of paired delta_i: {np.median(paired_deltas_hd):.4f} mm")
print(f"Mean of paired delta_i:   {np.mean(paired_deltas_hd):.4f} mm")

print("\n=== 2. TAIL DISTRIBUTION & OUTLIERS ===")
hd_v_all = np.array([d["hd_vanilla"] for d in paired_details])
hd_e_all = np.array([d["hd_ema"] for d in paired_details])
print(f"P50 (Median): Vanilla = {np.percentile(hd_v_all, 50):.2f} mm | EMA = {np.percentile(hd_e_all, 50):.2f} mm")
print(f"P75:         Vanilla = {np.percentile(hd_v_all, 75):.2f} mm | EMA = {np.percentile(hd_e_all, 75):.2f} mm")
print(f"P90:         Vanilla = {np.percentile(hd_v_all, 90):.2f} mm | EMA = {np.percentile(hd_e_all, 90):.2f} mm")
print(f"P95:         Vanilla = {np.percentile(hd_v_all, 95):.2f} mm | EMA = {np.percentile(hd_e_all, 95):.2f} mm")
print(f"Max HD95:    Vanilla = {np.max(hd_v_all):.2f} mm | EMA = {np.max(hd_e_all):.2f} mm")

zero_dice_v = sum(1 for d in paired_details if d["dice_v"] == 0.0)
zero_dice_e = sum(1 for d in paired_details if d["dice_e"] == 0.0)
print(f"Completely lost organs (Dice == 0): Vanilla = {zero_dice_v} | EMA = {zero_dice_e}")

print("\n=== 3. TOP IMPROVED PAIRS (Explaining the reduction) ===")
paired_details.sort(key=lambda x: x["delta_hd"])
for i, d in enumerate(paired_details[:8]):
    print(f"{i+1}. {d['case']} [{d['organ']}]: Vanilla={d['hd_vanilla']:.2f}mm -> EMA={d['hd_ema']:.2f}mm (Delta = {d['delta_hd']:.2f}mm, Dice: {d['dice_v']*100:.1f}% -> {d['dice_e']*100:.1f}%)")

print("\n=== 4. TOP REGRESSED PAIRS ===")
for i, d in enumerate(paired_details[-5:]):
    print(f"* {d['case']} [{d['organ']}]: Vanilla={d['hd_vanilla']:.2f}mm -> EMA={d['hd_ema']:.2f}mm (Delta = +{d['delta_hd']:.2f}mm, Dice: {d['dice_v']*100:.1f}% -> {d['dice_e']*100:.1f}%)")

print("\n=== 5. PER-ORGAN SUMMARY TABLE ===")
print(f"{'Organ':<15} | {'Vanilla Dice':<12} | {'EMA Dice':<12} | {'Vanilla HD95':<12} | {'EMA HD95':<12} | {'Delta HD95':<10}")
print("-" * 80)
for c in CLASS_NAMES:
    dv = res_v["per_class"][c]["mean_dice_clean"] * 100
    de = res_e["per_class"][c]["mean_dice_clean"] * 100
    hv = res_v["per_class"][c]["mean_hd95"]
    he = res_e["per_class"][c]["mean_hd95"]
    dh = he - hv
    print(f"{c:<15} | {dv:>11.2f}% | {de:>11.2f}% | {hv:>10.2f} mm | {he:>10.2f} mm | {dh:>9.2f} mm")
