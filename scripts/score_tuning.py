"""Score tuning runs against the KIS reference on the validation volumes (paired, case-clustered bootstrap).

Inputs are the per-evaluation JSON files written by experiments/cba_v2/train.py (eval_epXXX.json), which
hold one record per (case, organ) unit ("units"). Selection rule (registered before running, tuning_plan.md):

  1. For every candidate, average each unit over the last --last_k evaluations (reduces epoch-to-epoch noise).
  2. Feasible = macro Dice of the candidate is at most --dice_tol (default 0.003) below the KIS reference.
  3. Among feasible candidates choose the largest macro-HD95 reduction versus KIS (tie: higher Dice).
  4. The choice replaces the default configuration only if its HD95 improvement over the default is
     outside the bootstrap noise (95% CI of the paired difference lies below 0); otherwise the default is kept.
     The point-estimate winner is always reported as well.

Bootstrap: resample validation CASES (an organ's units within one patient are dependent), --n_boot times.

Example:
  python scripts/score_tuning.py --ref KIS=runs/tune/kis/Arm_V1_KIS --default lam0.02=runs/tune/a_lam002/Arm_V2_Boundary \
      --cand lam0.05=runs/tune/a_lam005/Arm_V2_Boundary --cand lam0.1=runs/tune/a_lam01/Arm_V2_Boundary --out tune_a.json
"""
import argparse
import glob
import json
import math
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Unit = Dict[str, float]
Key = Tuple[str, str]  # (case, organ)


def last_eval_files(run_dir: str, last_k: int) -> List[str]:
    files = sorted(glob.glob(os.path.join(run_dir, "eval_ep*.json")),
                   key=lambda p: int(re.search(r"eval_ep(\d+)\.json$", p).group(1)))
    if len(files) < last_k:
        raise FileNotFoundError(f"{run_dir}: need {last_k} eval_ep*.json files, found {len(files)}")
    return files[-last_k:]


def load_units(eval_json: dict, variant: str = "raw") -> Dict[Key, Dict[str, float]]:
    """GT-present units only: dice, nsd, hd95 (NaN allowed) for the chosen variant."""
    if "units" not in eval_json:
        raise KeyError("eval JSON has no 'units' (written by the current evaluation.py); re-run the evaluation")
    suffix = "raw" if variant == "raw" else "lcc"
    out: Dict[Key, Dict[str, float]] = {}
    for u in eval_json["units"]:
        if not u.get("gt_present"):
            continue
        out[(u["case"], u["organ"])] = {
            "dice": float(u[f"clean_dice_{suffix}"]),
            "nsd": float(u[f"nsd_{suffix}"]),
            "hd95": float(u[f"hd95_{suffix}"]) if u[f"hd95_{suffix}"] is not None else float("nan"),
        }
    return out


def average_units(per_eval: Sequence[Dict[Key, Dict[str, float]]]) -> Dict[Key, Dict[str, float]]:
    keys = set(per_eval[0])
    for d in per_eval[1:]:
        if set(d) != keys:
            raise ValueError("evaluations cover different (case, organ) units")
    out = {}
    for k in keys:
        rec = {}
        for m in ("dice", "nsd", "hd95"):
            vals = np.array([d[k][m] for d in per_eval], dtype=np.float64)
            rec[m] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
        out[k] = rec
    return out


def macro(units: Dict[Key, Dict[str, float]], metric: str, cases: Optional[Sequence[str]] = None,
          counts: Optional[Dict[str, int]] = None) -> float:
    """Mean over organs of the (valid) mean over cases. `counts` gives bootstrap multiplicities per case."""
    by_organ: Dict[str, List[Tuple[float, int]]] = {}
    for (case, organ), rec in units.items():
        if cases is not None and case not in cases:
            continue
        v = rec[metric]
        if not math.isfinite(v):
            continue
        by_organ.setdefault(organ, []).append((v, 1 if counts is None else counts.get(case, 0)))
    means = []
    for vals in by_organ.values():
        w = sum(c for _, c in vals)
        if w > 0:
            means.append(sum(v * c for v, c in vals) / w)
    return float(np.mean(means)) if means else float("nan")


def paired_common(a: Dict[Key, Dict[str, float]], b: Dict[Key, Dict[str, float]], metric: str):
    """Restrict to units valid (finite) in both runs for `metric`; returns two dicts."""
    keys = [k for k in a if k in b and math.isfinite(a[k][metric]) and math.isfinite(b[k][metric])]
    return {k: a[k] for k in keys}, {k: b[k] for k in keys}


def bootstrap_diff(a: Dict[Key, Dict[str, float]], b: Dict[Key, Dict[str, float]], metric: str,
                   n_boot: int = 1000, seed: int = 0) -> dict:
    """Paired macro difference a - b with a case-clustered bootstrap CI."""
    ua, ub = paired_common(a, b, metric)
    cases = sorted({c for c, _ in ua})
    point = macro(ua, metric) - macro(ub, metric)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        draw = rng.integers(0, len(cases), size=len(cases))
        counts: Dict[str, int] = {}
        for i in draw:
            counts[cases[i]] = counts.get(cases[i], 0) + 1
        d = macro(ua, metric, counts=counts) - macro(ub, metric, counts=counts)
        if math.isfinite(d):
            diffs.append(d)
    lo, hi = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))) if diffs else (float("nan"),) * 2
    return {"diff": float(point), "ci95": [lo, hi], "n_units": len(ua), "n_cases": len(cases)}


def load_run(run_dir: str, last_k: int, variant: str) -> Dict[Key, Dict[str, float]]:
    files = last_eval_files(run_dir, last_k)
    return average_units([load_units(json.load(open(f, encoding="utf-8")), variant) for f in files])


def select(ref: Dict[Key, dict], default_name: str, cands: Dict[str, Dict[Key, dict]], dice_tol: float,
           n_boot: int, seed: int) -> dict:
    rows = {}
    for name, units in cands.items():
        rows[name] = {
            "dice": macro(units, "dice"), "nsd": macro(units, "nsd"), "hd95": macro(units, "hd95"),
            "vs_ref_dice": bootstrap_diff(units, ref, "dice", n_boot, seed),
            "vs_ref_hd95": bootstrap_diff(units, ref, "hd95", n_boot, seed),
            "vs_ref_nsd": bootstrap_diff(units, ref, "nsd", n_boot, seed),
        }
        rows[name]["feasible"] = rows[name]["vs_ref_dice"]["diff"] >= -dice_tol
    feasible = [n for n, r in rows.items() if r["feasible"]]
    best = None
    if feasible:
        best = sorted(feasible, key=lambda n: (rows[n]["vs_ref_hd95"]["diff"], -rows[n]["dice"]))[0]
    decision, reason = default_name, "no feasible candidate; keep default"
    if best is not None and best != default_name:
        vs_def = bootstrap_diff(cands[best], cands[default_name], "hd95", n_boot, seed)
        rows[best]["vs_default_hd95"] = vs_def
        if vs_def["ci95"][1] < 0:
            decision, reason = best, "HD95 improvement over default is outside bootstrap noise"
        else:
            reason = "point-estimate winner is within bootstrap noise of the default; keep default"
    elif best == default_name:
        reason = "default is the point-estimate winner"
    return {"rows": rows, "best_by_point_estimate": best, "decision": decision, "reason": reason,
            "default": default_name, "dice_tol": dice_tol}


def parse_named(items: Sequence[str]) -> Dict[str, str]:
    out = {}
    for it in items:
        name, _, path = it.partition("=")
        if not path:
            raise ValueError(f"expected NAME=PATH, got {it!r}")
        out[name] = path
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="KIS reference as NAME=run_dir")
    ap.add_argument("--default", required=True, help="Default configuration as NAME=run_dir")
    ap.add_argument("--cand", action="append", default=[], help="Candidate NAME=run_dir (repeatable)")
    ap.add_argument("--variant", choices=["raw", "lcc"], default="raw")
    ap.add_argument("--last_k", type=int, default=2)
    ap.add_argument("--dice_tol", type=float, default=0.003)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    ref_name, ref_dir = list(parse_named([a.ref]).items())[0]
    def_name, def_dir = list(parse_named([a.default]).items())[0]
    cands = {def_name: load_run(def_dir, a.last_k, a.variant)}
    for n, d in parse_named(a.cand).items():
        cands[n] = load_run(d, a.last_k, a.variant)
    ref = load_run(ref_dir, a.last_k, a.variant)

    res = select(ref, def_name, cands, a.dice_tol, a.n_boot, a.seed)
    res["ref"] = {"name": ref_name, "dice": macro(ref, "dice"), "nsd": macro(ref, "nsd"), "hd95": macro(ref, "hd95")}
    res["variant"], res["last_k"] = a.variant, a.last_k
    print(f"{ref_name} (ref): Dice {res['ref']['dice']*100:.2f}  NSD {res['ref']['nsd']*100:.2f}  HD95 {res['ref']['hd95']:.2f}")
    for n, r in res["rows"].items():
        print(f"{n:>14}: Dice {r['dice']*100:.2f} ({r['vs_ref_dice']['diff']*100:+.2f})  "
              f"HD95 {r['hd95']:.2f} ({r['vs_ref_hd95']['diff']:+.2f}, CI {r['vs_ref_hd95']['ci95'][0]:+.2f}..{r['vs_ref_hd95']['ci95'][1]:+.2f})  "
              f"{'feasible' if r['feasible'] else 'INFEASIBLE'}")
    print(f"best by point estimate: {res['best_by_point_estimate']} | decision: {res['decision']} ({res['reason']})")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
    return res


if __name__ == "__main__":
    main()
