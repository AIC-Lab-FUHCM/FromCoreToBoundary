"""Driver for the hyperparameter-tuning runs (see tuning_plan.md).

All runs use the 14-case tuning-train split, evaluate on the 4 validation cases (never the test set) and
share the first 20 epochs (the "warmup" job, which is the KIS arm itself). Every other job forks from the
epoch-20 checkpoint, so jobs are comparable and can be split over several Colab sessions.

Subcommands
  plan   estimate GPU hours from a measured epoch time
  run    execute the jobs of a stage (skips finished jobs, auto-resumes, waits for the shared warmup)
  score  apply the registered selection rule (scripts/score_tuning.py) and update tune_state.json

Stages
  warmup  KIS for 20 epochs on the tuning split (LR schedule length = L)
  ref     KIS continued to L epochs (the reference every candidate is compared with)
  A       Arm 2 (Boundary): lambda_A grid
  B       Arm 2: tau in {0.07, 0.2}, m in {0.1, 0.3}, r in {1, 3}, one at a time at the selected lambda_A
  C       Arm 5 (Boundary + Error-CE): beta in {0.25, 0.5, 1.0} with the selected boundary parameters
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "models", REPO_ROOT / "scripts", REPO_ROOT / "scripts" / "tools"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

ARM_DIRS = {
    "v1": "Arm_V1_KIS",
    "v2": "Arm_V2_Boundary",
    "v3": "Arm_V3_Error",
    "v4": "Arm_V4_Dual_Contrastive",
    "v5": "Arm_V5_Dual_CEHard",
    "iad": "Arm_IAD_200_100",
    "lskd": "Arm_LSKD",
    "sinkd": "Arm_SinKD",
    "dist_plus": "Arm_DIST_Plus",
    "sgs_only": "Arm_SGS_Only",
}
WARMUP_EPOCHS = 20
DEFAULT_STATE = {"lambda_a": 0.02, "tau": 0.1, "m": 0.2, "r": 2, "beta": 0.5, "decisions": {}}

STATE_TO_FLAG = {"lambda_a": "--lambda_a", "tau": "--tau", "m": "--m", "r": "--r", "beta": "--beta"}
B_GRID = {"tau": [0.07, 0.2], "m": [0.1, 0.3], "r": [1, 3]}
C_BETA_GRID = [0.25, 0.5, 1.0]
SPLIT_ARGS = [
    "--list_path", "data/lists_Synapse/train_tune.txt",
    "--test_dir", "data/btcv/val_vol_h5",
    "--test_list", "data/lists_Synapse/val_tune_vol.txt",
]


def fmt(x) -> str:
    return f"{x:g}"


# ----------------------------------------------------------------------------- state

def _runs_dir_variants(runs_dir: str) -> List[str]:
    out = [runs_dir]
    if runs_dir and not os.path.isabs(runs_dir) and Path.cwd().resolve() != REPO_ROOT.resolve():
        out.append(os.path.join(str(REPO_ROOT), runs_dir))
    return out


def load_state(path: str) -> dict:
    for cand in _runs_dir_variants(path) if path else []:
        if os.path.isfile(cand):
            with open(cand, "r", encoding="utf-8") as fh:
                st = json.load(fh)
            merged = dict(DEFAULT_STATE)
            merged.update(st)
            return merged
    return json.loads(json.dumps(DEFAULT_STATE))


def save_state(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- jobs

def lambda_grid_from_calibration(calib_json: str, always_include: float = 0.02) -> List[float]:
    with open(calib_json, "r", encoding="utf-8") as fh:
        calib = json.load(fh)
    vals = [v for v in calib["lambda_for_target_ratio_b"].values() if v]
    grid = sorted({float(f"{v:.2g}") for v in vals} | {always_include})
    return grid


def check_length(L: int) -> None:
    if L < 60 or L % 10:
        raise ValueError("L must be a multiple of 10 and at least 60 (the lambda ramp ends at epoch 50 and the "
                         "score uses the last two evaluations)")


def build_jobs(stage: str, L: int, state: dict, lambda_grid: Optional[Sequence[float]] = None) -> List[dict]:
    """Job = {name, arm, epochs, params (dict of state keys), needs_warmup}."""
    check_length(L)
    base = {k: state[k] for k in STATE_TO_FLAG}
    if stage == "warmup":
        return [dict(name="warmup", arm="v1", epochs=WARMUP_EPOCHS, params={}, needs_warmup=False)]
    if stage == "ref":
        return [dict(name="ref_kis", arm="v1", epochs=L, params={}, needs_warmup=True)]
    if stage == "A":
        grid = sorted(set(lambda_grid or [0.01, 0.02, 0.05]) | {DEFAULT_STATE["lambda_a"]})
        return [dict(name=f"a_lam{fmt(v)}", arm="v2", epochs=L,
                     params={**{k: DEFAULT_STATE[k] for k in ("tau", "m", "r")}, "lambda_a": v},
                     needs_warmup=True) for v in grid]
    if stage == "B":
        jobs = []
        for key, vals in B_GRID.items():
            for v in vals:
                params = {**{k: DEFAULT_STATE[k] for k in ("tau", "m", "r")}, "lambda_a": base["lambda_a"], key: v}
                jobs.append(dict(name=f"b_{key}{fmt(v)}", arm="v2", epochs=L, params=params, needs_warmup=True))
        return jobs
    if stage == "C":
        return [dict(name=f"c_beta{fmt(b)}", arm="v5", epochs=L,
                     params={"lambda_a": base["lambda_a"], "tau": base["tau"], "m": base["m"], "r": base["r"],
                             "beta": b}, needs_warmup=True) for b in C_BETA_GRID]
    raise ValueError(f"unknown stage {stage!r}")


def shard_jobs(jobs: Sequence[dict], shard: str) -> List[dict]:
    i, n = (int(x) for x in shard.split("/"))
    if not (1 <= i <= n):
        raise ValueError(f"bad shard {shard!r}")
    return [j for k, j in enumerate(jobs) if k % n == i - 1]


def arm_dir(runs_dir: str, job: dict) -> str:
    return os.path.join(runs_dir, job["name"], ARM_DIRS[job["arm"]])


def is_done(job: dict, runs_dir: str, drive_dir: Optional[str]) -> bool:
    for r_dir in _runs_dir_variants(runs_dir):
        if os.path.isfile(os.path.join(arm_dir(r_dir, job), "eval_final.json")):
            return True
    return bool(drive_dir) and os.path.isfile(os.path.join(drive_dir, job["name"], "eval_final.json"))


def warmup_ckpt_candidates(runs_dir: str, drive_dir: Optional[str]) -> List[str]:
    name = f"checkpoint_epoch_{WARMUP_EPOCHS:03d}.pth"
    out = [os.path.join(r_dir, "warmup", ARM_DIRS["v1"], name) for r_dir in _runs_dir_variants(runs_dir)]
    if drive_dir:
        out.append(os.path.join(drive_dir, "warmup", name))
    return out


def find_warmup_ckpt(runs_dir: str, drive_dir: Optional[str]) -> Optional[str]:
    return next((p for p in warmup_ckpt_candidates(runs_dir, drive_dir) if os.path.isfile(p)), None)


def has_own_checkpoint(job: dict, runs_dir: str, drive_dir: Optional[str]) -> bool:
    for r_dir in _runs_dir_variants(runs_dir):
        if os.path.isfile(os.path.join(arm_dir(r_dir, job), "latest_checkpoint.pth")):
            return True
    return bool(drive_dir) and os.path.isfile(os.path.join(drive_dir, job["name"], "latest_checkpoint.pth"))


def job_command(job: dict, L: int, runs_dir: str, drive_dir: Optional[str], warmup_ckpt: Optional[str],
                python: str = sys.executable, seed: Optional[int] = None) -> List[str]:
    cmd = [python, "run_cba_v2.py", "--arm", job["arm"], "--epochs", str(job["epochs"]), "--sched_epochs", str(L)]
    cmd += SPLIT_ARGS
    cmd += ["--output_dir", os.path.join(runs_dir, job["name"]),
            "--eval_freq", "0" if job["name"] == "warmup" else "10",
            "--no-save_final_preds", "--drive_sync_every", "5"]
    if drive_dir:
        cmd += ["--drive_sync_dir", os.path.join(drive_dir, job["name"])]
    for key, val in job["params"].items():
        cmd += [STATE_TO_FLAG[key], fmt(val)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if job["needs_warmup"]:
        cmd += ["--resume", "auto" if has_own_checkpoint(job, runs_dir, drive_dir) else str(warmup_ckpt)]
    else:
        cmd += ["--resume", "auto"]
    return cmd


def wait_for_warmup(runs_dir: str, drive_dir: Optional[str], wait_min: float, poll_s: float = 60.0) -> Optional[str]:
    deadline = time.time() + wait_min * 60
    while True:
        ckpt = find_warmup_ckpt(runs_dir, drive_dir)
        if ckpt or time.time() >= deadline:
            return ckpt
        print(f"[wait] shared warmup checkpoint not found yet; retrying in {poll_s:.0f}s", flush=True)
        time.sleep(poll_s)


# ----------------------------------------------------------------------------- plan / cost

def plan_cost(epoch_sec: float, eval_sec: float, L: int, n_a: int = 3, n_b: int = 6, n_c: int = 3,
              kis_epoch_sec: Optional[float] = None) -> dict:
    """GPU-hours. epoch_sec: measured seconds per epoch with CBA active (all fork epochs, since lambda > 0 from
    epoch 21) on the 14-case tuning split; kis_epoch_sec: seconds per KIS-only epoch (warmup and the KIS
    reference), defaulting to epoch_sec."""
    check_length(L)
    kis_sec = epoch_sec if kis_epoch_sec is None else kis_epoch_sec
    per_fork_epochs = L - WARMUP_EPOCHS
    n_evals = L // 10 - WARMUP_EPOCHS // 10  # evaluations at 30, 40, ... L
    cba_fork_h = (per_fork_epochs * epoch_sec + n_evals * eval_sec) / 3600.0
    ref_h = (per_fork_epochs * kis_sec + n_evals * eval_sec) / 3600.0
    warmup_h = (WARMUP_EPOCHS * kis_sec + eval_sec) / 3600.0
    stages = {"ref_kis": ref_h, "A": n_a * cba_fork_h, "B": n_b * cba_fork_h, "C": n_c * cba_fork_h}
    return {"warmup_h": warmup_h, "fork_h_each": cba_fork_h, "stages": stages,
            "total_h": warmup_h + sum(stages.values()), "n_fork_runs": 1 + n_a + n_b + n_c}


# ----------------------------------------------------------------------------- scoring

def resolve_run_dir(job: dict, runs_dir: str, drive_dir: Optional[str]) -> str:
    local = arm_dir(runs_dir, job)
    if glob.glob(os.path.join(local, "eval_ep*.json")):
        return local
    if drive_dir and glob.glob(os.path.join(drive_dir, job["name"], "eval_ep*.json")):
        return os.path.join(drive_dir, job["name"])
    raise FileNotFoundError(f"no eval_ep*.json for job {job['name']} (looked in {local} and the Drive copy)")


def score_stage(stage: str, L: int, runs_dir: str, drive_dir: Optional[str], state_path: str,
                lambda_grid: Optional[Sequence[float]] = None, last_k: int = 2, dice_tol: float = 0.003,
                n_boot: int = 1000, seed: int = 0, variant: str = "raw") -> dict:
    import score_tuning as st

    state = load_state(state_path)
    ref_job = build_jobs("ref", L, state)[0]
    ref = st.load_run(resolve_run_dir(ref_job, runs_dir, drive_dir), last_k, variant)

    def load(job):
        return st.load_run(resolve_run_dir(job, runs_dir, drive_dir), last_k, variant)

    report: Dict[str, dict] = {}
    if stage == "A":
        jobs = build_jobs("A", L, state, lambda_grid)
        default_name = f"a_lam{fmt(DEFAULT_STATE['lambda_a'])}"
        res = st.select(ref, default_name, {j["name"]: load(j) for j in jobs}, dice_tol, n_boot, seed)
        chosen = next(j for j in jobs if j["name"] == res["decision"])
        state["lambda_a"] = chosen["params"]["lambda_a"]
        state["decisions"]["A"] = {"lambda_a": state["lambda_a"], "reason": res["reason"]}
        report["A"] = res
    elif stage == "B":
        default_job = dict(name=f"a_lam{fmt(state['lambda_a'])}", arm="v2")
        default_units = load(default_job)
        jobs = build_jobs("B", L, state)
        for key in B_GRID:
            cands = {f"default_{key}{fmt(DEFAULT_STATE[key])}": default_units}
            for j in jobs:
                if j["name"].startswith(f"b_{key}"):
                    cands[j["name"]] = load(j)
            default_name = f"default_{key}{fmt(DEFAULT_STATE[key])}"
            res = st.select(ref, default_name, cands, dice_tol, n_boot, seed)
            if res["decision"] != default_name:
                state[key] = next(j["params"][key] for j in jobs if j["name"] == res["decision"])
            else:
                state[key] = DEFAULT_STATE[key]
            state["decisions"][f"B_{key}"] = {key: state[key], "reason": res["reason"]}
            report[f"B_{key}"] = res
    elif stage == "C":
        jobs = build_jobs("C", L, state)
        default_name = f"c_beta{fmt(DEFAULT_STATE['beta'])}"
        res = st.select(ref, default_name, {j["name"]: load(j) for j in jobs}, dice_tol, n_boot, seed)
        chosen = next(j for j in jobs if j["name"] == res["decision"])
        state["beta"] = chosen["params"]["beta"]
        state["decisions"]["C"] = {"beta": state["beta"], "reason": res["reason"]}
        report["C"] = res
    else:
        raise ValueError(f"cannot score stage {stage!r}")
    save_state(state_path, state)
    return {"state": state, "report": report}


# ----------------------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--L", type=int, default=60, help="Proxy run length in epochs (multiple of 10, >= 60)")
        p.add_argument("--runs_dir", default="runs/tune")
        p.add_argument("--drive_dir", default=None, help="Drive folder for checkpoints, evals and tune_state.json")
        p.add_argument("--state_json", default=None, help="Path to tune_state.json (overrides default under drive_dir/runs_dir)")
        p.add_argument("--lambda_grid", default=None, help="Comma-separated lambda_A grid for stage A")
        p.add_argument("--calib_json", default=None, help="Build the stage-A grid from calibrate_lambda.py output")

    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--L", type=int, default=60)
    p_plan.add_argument("--epoch_sec", type=float, required=True, help="Measured seconds per epoch with CBA active (tuning split)")
    p_plan.add_argument("--kis_epoch_sec", type=float, default=None, help="Seconds per KIS-only epoch (default: --epoch_sec)")
    p_plan.add_argument("--eval_sec", type=float, default=90.0, help="Seconds per validation evaluation")
    p_plan.add_argument("--budget_h", type=float, default=200.0)
    p_plan.add_argument("--main_h_per_arm", type=float, default=None, help="Estimated hours of one 300-epoch main arm")

    p_run = sub.add_parser("run")
    common(p_run)
    p_run.add_argument("--stage", required=True, choices=["warmup", "ref", "A", "B", "C"])
    p_run.add_argument("--shard", default="1/1")
    p_run.add_argument("--wait_min", type=float, default=120.0)
    p_run.add_argument("--dry_run", action="store_true")
    p_run.add_argument("--seed", type=int, default=None)

    p_sc = sub.add_parser("score")
    common(p_sc)
    p_sc.add_argument("--stage", required=True, choices=["A", "B", "C"])
    p_sc.add_argument("--n_boot", type=int, default=1000)
    p_sc.add_argument("--dice_tol", type=float, default=0.003)

    a = ap.parse_args(argv)
    if a.cmd == "plan":
        c = plan_cost(a.epoch_sec, a.eval_sec, a.L, kis_epoch_sec=a.kis_epoch_sec)
        print(f"warmup {c['warmup_h']:.2f} h; one fork run {c['fork_h_each']:.2f} h; {c['n_fork_runs']} fork runs")
        for k, v in c["stages"].items():
            print(f"  stage {k:>7}: {v:6.2f} h")
        print(f"tuning total: {c['total_h']:.2f} h of {a.budget_h:.0f} h")
        if a.main_h_per_arm is not None:
            left = a.budget_h - c["total_h"]
            print(f"left for the 3 main arms: {left:.1f} h (needed ~{3 * a.main_h_per_arm:.1f} h)")
        return c

    grid = None
    if a.calib_json:
        grid = lambda_grid_from_calibration(a.calib_json)
    elif a.lambda_grid:
        grid = [float(x) for x in a.lambda_grid.split(",") if x.strip()]
    state_path = a.state_json or os.path.join(a.drive_dir or a.runs_dir, "tune_state.json")

    if a.cmd == "score":
        out = score_stage(a.stage, a.L, a.runs_dir, a.drive_dir, state_path, grid, n_boot=a.n_boot,
                          dice_tol=a.dice_tol)
        for name, res in out["report"].items():
            print(f"[{name}] decision={res['decision']} ({res['reason']}); best by point estimate: "
                  f"{res['best_by_point_estimate']}")
            for cn, r in res["rows"].items():
                print(f"   {cn:>16}: Dice {r['dice']*100:.2f} ({r['vs_ref_dice']['diff']*100:+.2f} vs KIS)  "
                      f"HD95 {r['hd95']:.2f} ({r['vs_ref_hd95']['diff']:+.2f} vs KIS)  "
                      f"{'feasible' if r['feasible'] else 'INFEASIBLE'}")
        print(json.dumps(out["state"], indent=2))
        os.makedirs(a.drive_dir or a.runs_dir, exist_ok=True)
        with open(os.path.join(a.drive_dir or a.runs_dir, f"score_{a.stage}.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        return out

    state = load_state(state_path)
    jobs = shard_jobs(build_jobs(a.stage, a.L, state, grid), a.shard)
    for job in jobs:
        if is_done(job, a.runs_dir, a.drive_dir):
            print(f"[skip] {job['name']} already finished")
            continue
        wck = None
        if job["needs_warmup"] and not has_own_checkpoint(job, a.runs_dir, a.drive_dir):
            wck = wait_for_warmup(a.runs_dir, a.drive_dir, 0 if a.dry_run else a.wait_min)
            if wck is None:
                if a.dry_run:
                    wck = "<warmup checkpoint>"
                else:
                    raise RuntimeError("shared warmup checkpoint not available; run stage 'warmup' first")
        cmd = job_command(job, a.L, a.runs_dir, a.drive_dir, wck)
        print("[run] " + " ".join(cmd), flush=True)
        if not a.dry_run:
            subprocess.run(_resolve_exec_cmd(cmd), cwd=str(REPO_ROOT), check=True)
    return jobs


def _resolve_exec_cmd(cmd: List[str]) -> List[str]:
    if len(cmd) >= 2 and cmd[1] == "run_cba_v2.py" and not (REPO_ROOT / "run_cba_v2.py").exists():
        cand = REPO_ROOT / "scripts" / "run_cba_v2.py"
        if cand.exists():
            return [cmd[0], str(cand)] + list(cmd[2:])
    return cmd


if __name__ == "__main__":
    main()
