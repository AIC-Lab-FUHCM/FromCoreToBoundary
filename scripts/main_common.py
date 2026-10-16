"""Shared launcher for the 300-epoch main runs (see run_main_kis.py and run_main_ours.py).

Both runs train on all 18 training cases and evaluate on the 12 official test volumes only at epoch 300.
The reported model is the epoch-300 model; there is no best-on-test checkpoint.

The first 20 epochs are identical for every arm (lambda_A = 0 during the curriculum warmup), so they are run once
as a KIS "warmup" job (LR schedule length = --epochs) and every arm forks from its epoch-20 checkpoint:

    <runs_dir>/warmup/Arm_V1_KIS/checkpoint_epoch_020.pth   (+ the same file in <drive_dir>/warmup/)

Those legacy names are retained for seed 2345. Additional seeds use warmup_s<seed>,
kis300_s<seed>, and ours_<arm>_s<seed> so warmups, resumes, and completion checks cannot cross seeds.

The first script that needs the warmup creates it when --make_warmup is set; otherwise it waits for another
session to create it. Both arms must be launched with the same --epochs. Finished runs are skipped and
interrupted runs resume from their latest checkpoint (local or Drive), so re-running the same command is safe.

Method parameters of "ours" come from the tuning state file (default: <drive_dir>/../tune/tune_state.json),
falling back to the defaults lambda_A=0.02, tau=0.1, m=0.2, r=2, beta=0.5.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "models", REPO_ROOT / "scripts", REPO_ROOT / "scripts" / "tools"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_tuning as rt  # noqa: E402

KIND_DIRS = {
    "kis": "kis300",
    "iad": "iad_200_100",
    "lskd": "lskd300",
    "sinkd": "sinkd300",
    "dist_plus": "dist_plus300",
    "sgs_only": "sgs300",
}
OURS_PARAMS = {
    "v2": ("lambda_a", "tau", "m", "r"),
    "v3": ("lambda_a", "tau", "m", "r", "beta"),
    "v4": ("lambda_a", "tau", "m", "r", "beta"),
    "v5": ("lambda_a", "tau", "m", "r", "beta"),
}


def seed_name(name: str, seed: int) -> str:
    """Keep the existing seed-2345 paths; isolate every additional seed."""
    return name if seed == 2345 else f"{name}_s{seed}"


def job_for(kind: str, arm: Optional[str], state: dict, epochs: int, ratio_cap: float = 0.0,
            seed: int = 2345, kis_stage_epochs: int = 200) -> dict:
    use_warmup = bool(epochs > rt.WARMUP_EPOCHS)
    if kind == "kis":
        return dict(name=seed_name(KIND_DIRS["kis"], seed), arm="v1", epochs=epochs, params={}, needs_warmup=use_warmup)
    if kind == "iad":
        extra = ["--kis_stage_epochs", str(kis_stage_epochs)] if kis_stage_epochs != 200 else []
        return dict(name=seed_name(KIND_DIRS["iad"], seed), arm="iad", epochs=epochs, params={},
                    extra=extra, needs_warmup=False)
    if kind in ("lskd", "sinkd", "dist_plus", "sgs_only"):
        return dict(name=seed_name(KIND_DIRS[kind], seed), arm=kind, epochs=epochs, params={}, needs_warmup=False)
    if arm not in OURS_PARAMS:
        raise ValueError(f"ours supports arms {sorted(OURS_PARAMS)}, got {arm!r}")
    job = dict(name=f"ours_{arm}", arm=arm, epochs=epochs,
               params={k: state[k] for k in OURS_PARAMS[arm]}, needs_warmup=use_warmup)
    if ratio_cap > 0.0:  # separate folder so it never mixes with the uncapped run
        job.update(name=f"ours_{arm}cap", extra=["--lambda_ratio_cap", rt.fmt(ratio_cap)])
    job["name"] = seed_name(job["name"], seed)
    return job


def _is_sol12_run_dir(folder: Optional[str]) -> bool:
    if not folder or not os.path.isdir(folder):
        return False
    cfg_candidates = [
        os.path.join(folder, "run_config.json"),
        os.path.join(folder, rt.ARM_DIRS.get("v5", "Arm_V5_Dual_CEHard"), "run_config.json"),
    ]
    for cfg_path in cfg_candidates:
        if not os.path.isfile(cfg_path):
            continue
        try:
            with open(cfg_path, encoding="utf-8") as fh:
                cfg = json.load(fh)
            args = cfg.get("args", cfg) if isinstance(cfg, dict) else {}
            if any(
                bool(args.get(k, False))
                for k in ("fp_by_pred_class", "fp_exclude_boundary", "empty_slice_suppression")
            ):
                return True
        except Exception:
            continue
    return False


def archive_sol12_run_if_needed(job: dict, runs_dir: str, drive_dir: Optional[str], dry_run: bool = False) -> List[str]:
    """Archive any stale Solution 1+2 v5 run folder so original ours_v5 can resume cleanly from warmup."""
    if job.get("arm") != "v5":
        return []
    archived: List[str] = []
    targets = [os.path.join(runs_dir, job["name"])]
    if drive_dir:
        targets.append(os.path.join(drive_dir, job["name"]))
    for src in targets:
        if not _is_sol12_run_dir(src):
            continue
        dst = f"{src}_sol12_archive"
        if os.path.exists(dst):
            dst = f"{src}_sol12_archive_{int(time.time())}"
        print(f"[archive] moving Solution 1+2 run {src} -> {dst}", flush=True)
        if not dry_run:
            shutil.move(src, dst)
        archived.append(dst)
    return archived


def find_warmup_ckpt(runs_dir: str, drive_dir: Optional[str], seed: int) -> Optional[str]:
    name = seed_name("warmup", seed)
    checkpoint = f"checkpoint_epoch_{rt.WARMUP_EPOCHS:03d}.pth"
    candidates = [
        os.path.join(r_dir, name, rt.ARM_DIRS["v1"], checkpoint)
        for r_dir in rt._runs_dir_variants(runs_dir)
    ]
    if drive_dir:
        candidates.append(os.path.join(drive_dir, name, checkpoint))
        candidates.append(os.path.join(drive_dir, name, rt.ARM_DIRS["v1"], checkpoint))
    return next((path for path in candidates if os.path.isfile(path)), None)


def wait_for_warmup(runs_dir: str, drive_dir: Optional[str], seed: int, wait_min: float) -> Optional[str]:
    deadline = time.monotonic() + wait_min * 60
    while True:
        checkpoint = find_warmup_ckpt(runs_dir, drive_dir, seed)
        remaining = deadline - time.monotonic()
        if checkpoint or remaining <= 0:
            return checkpoint
        print(f"[wait] {seed_name('warmup', seed)} checkpoint not found; retrying in up to 60s", flush=True)
        time.sleep(min(60.0, remaining))


def normalize_dataset(dataset: str) -> str:
    ds = str(dataset or "synapse").strip().lower()
    if ds in ("btcv", "synapse"):
        return "synapse"
    if ds in ("acdc", "polyp"):
        return ds
    raise ValueError(f"Unsupported dataset {dataset!r}. Expected 'synapse', 'acdc', or 'polyp'.")


def warmup_command(epochs: int, runs_dir: str, drive_dir: Optional[str], seed: int,
                   python: str = sys.executable, dataset: str = "synapse",
                   extra_train_args: Optional[List[str]] = None) -> List[str]:
    ds = normalize_dataset(dataset)
    name = seed_name("warmup", seed)
    cmd = [python, "run_cba_v2.py"]
    if ds != "synapse":
        cmd += ["--dataset", ds]
    cmd += ["--arm", "v1", "--epochs", str(rt.WARMUP_EPOCHS), "--sched_epochs", str(epochs),
            "--output_dir", os.path.join(runs_dir, name), "--eval_freq", "0", "--skip_eval", "--no-save_final_preds",
            "--drive_sync_every", "5", "--seed", str(seed), "--resume", "auto"]
    if drive_dir:
        cmd += ["--drive_sync_dir", os.path.join(drive_dir, name)]
    if extra_train_args:
        cmd += list(extra_train_args)
    return cmd


def main_command(job: dict, runs_dir: str, drive_dir: Optional[str], warmup_ckpt: Optional[str], seed: int,
                 eval_freq: int, python: str = sys.executable, dataset: str = "synapse",
                 extra_train_args: Optional[List[str]] = None) -> List[str]:
    ds = normalize_dataset(dataset)
    cmd = [python, "run_cba_v2.py"]
    if ds != "synapse":
        cmd += ["--dataset", ds]
    cmd += ["--arm", job["arm"], "--epochs", str(job["epochs"]),
            "--sched_epochs", str(job["epochs"]), "--output_dir", os.path.join(runs_dir, job["name"]),
            "--eval_freq", str(eval_freq), "--drive_sync_every", "5", "--seed", str(seed)]
    if drive_dir:
        cmd += ["--drive_sync_dir", os.path.join(drive_dir, job["name"])]
    for key, val in job["params"].items():
        cmd += [rt.STATE_TO_FLAG[key], rt.fmt(val)]
    cmd += job.get("extra", [])
    if extra_train_args:
        cmd += list(extra_train_args)
    own = rt.has_own_checkpoint(job, runs_dir, drive_dir)
    if job.get("needs_warmup", True):
        cmd += ["--resume", "auto" if own else str(warmup_ckpt)]
    else:
        cmd += ["--resume", "auto"]
    return cmd


def default_state_path(drive_dir: Optional[str]) -> Optional[str]:
    if not drive_dir:
        return None
    return os.path.join(os.path.dirname(os.path.normpath(drive_dir)), "tune", "tune_state.json")


def build_parser(kind: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=str, default="synapse", choices=["synapse", "btcv", "acdc", "polyp"],
                    help="Target dataset: synapse (default), acdc, or polyp")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--runs_dir", default="runs/main", help="Local output root")
    ap.add_argument("--drive_dir", default=None, help="Drive folder for checkpoints and results, e.g. /content/drive/MyDrive/IAD/main")
    ap.add_argument("--seed", type=int, default=2345)
    ap.add_argument("--eval_freq", type=int, default=0, help="Test-volume evaluation period; 0 = final epoch only (default)")
    ap.add_argument("--wait_min", type=float, default=240.0, help="Minutes to wait for the shared warmup when not creating it")
    ap.add_argument("--make_warmup", action=argparse.BooleanOptionalAction, default=(kind == "kis"),
                    help="Create the shared epoch-20 warmup checkpoint when it does not exist (default: %(default)s)")
    ap.add_argument("--dry_run", action="store_true")
    if kind == "iad":
        ap.add_argument("--kis_stage_epochs", type=int, default=200,
                        help="Stage 1 KIS epochs before Stage 2 KAS (default: 200)")
    if kind == "ours":
        ap.add_argument("--arm", choices=sorted(OURS_PARAMS), default="v5",
                        help="v5 = KIS + Boundary + Error-CE-hard (default), v2 = KIS + Boundary")
        ap.add_argument("--state_json", default=None, help="Tuning state file with the method parameters")
        ap.add_argument("--lambda_ratio_cap", type=float, default=0.0,
                        help="If > 0, cap lambda_A * ||g_CBA|| / ||g_base|| (logit space) at this value; the run is "
                             "written to ours_<arm>cap (default: off)")
        ap.add_argument("--lambda_a", type=float, default=None, help="Explicit CLI override for lambda_A")
        ap.add_argument("--tau", type=float, default=None, help="Explicit CLI override for tau")
        ap.add_argument("--m", type=float, default=None, help="Explicit CLI override for m")
        ap.add_argument("--r", type=int, default=None, help="Explicit CLI override for r")
        ap.add_argument("--beta", type=float, default=None, help="Explicit CLI override for beta")
    return ap


def run_main(kind: str, argv=None, extra_train_args: Optional[List[str]] = None) -> List[str]:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    a = build_parser(kind).parse_args(argv)
    ds = normalize_dataset(a.dataset)
    if ds != "synapse" and a.runs_dir == "runs/main":
        a.runs_dir = f"runs/{ds}_main"

    state = dict(rt.DEFAULT_STATE)
    if kind == "ours":
        path = a.state_json or default_state_path(a.drive_dir)
        state = rt.load_state(path) if path else rt.load_state("")
        for k in OURS_PARAMS[a.arm]:
            cli_override = getattr(a, k, None)
            if cli_override is not None:
                state[k] = cli_override
        print(f"[params] {path or 'defaults'}: " + ", ".join(f"{k}={state[k]}" for k in OURS_PARAMS[a.arm]), flush=True)
    job = job_for(
        kind,
        getattr(a, "arm", None),
        state,
        a.epochs,
        getattr(a, "lambda_ratio_cap", 0.0),
        a.seed,
        getattr(a, "kis_stage_epochs", 200),
    )

    archive_sol12_run_if_needed(job, a.runs_dir, a.drive_dir)

    if rt.is_done(job, a.runs_dir, a.drive_dir):
        print(f"[skip] {job['name']} ({ds}) already finished")
        return []

    wck = None
    effective_make_warmup = a.make_warmup or (ds != "synapse" and "--no-make_warmup" not in raw_argv)
    if job.get("needs_warmup", True) and not rt.has_own_checkpoint(job, a.runs_dir, a.drive_dir):
        wck = find_warmup_ckpt(a.runs_dir, a.drive_dir, a.seed)
        if wck is None and effective_make_warmup:
            cmd = warmup_command(
                a.epochs, a.runs_dir, a.drive_dir, a.seed,
                dataset=ds, extra_train_args=extra_train_args,
            )
            print("[warmup] " + " ".join(cmd), flush=True)
            if not a.dry_run:
                subprocess.run(rt._resolve_exec_cmd(cmd), cwd=str(REPO_ROOT), check=True)
                wck = find_warmup_ckpt(a.runs_dir, a.drive_dir, a.seed)
        elif wck is None:
            wck = wait_for_warmup(a.runs_dir, a.drive_dir, a.seed, 0 if a.dry_run else a.wait_min)
        if wck is None:
            if not a.dry_run:
                raise RuntimeError("shared warmup checkpoint not available; run with --make_warmup or start the other run first")
            wck = "<warmup checkpoint>"

    cmd = main_command(
        job, a.runs_dir, a.drive_dir, wck, a.seed, a.eval_freq,
        dataset=ds, extra_train_args=extra_train_args,
    )
    print("[run] " + " ".join(cmd), flush=True)
    if not a.dry_run:
        subprocess.run(rt._resolve_exec_cmd(cmd), cwd=str(REPO_ROOT), check=True)
    return cmd


