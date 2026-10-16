"""Generate the 5 method-specific 3-seed Colab notebooks and update colab_upload/MULTISEED_README.md.

The 5 notebooks (each configured with SEEDS = [2345, 42, 1234] and EPOCHS = 300):
  1. run_cba_v2_main_colab_ours_3seeds.ipynb      (RUN = "ours", Arm_V5_Dual_CEHard: 20 KIS warmup + 280 Ours v5)
  2. run_cba_v2_main_colab_iad_3seeds.ipynb       (RUN = "iad", Arm_IAD_200_100: 200 KIS + 100 KAS)
  3. run_cba_v2_main_colab_lskd_3seeds.ipynb      (RUN = "lskd", Arm_LSKD: 300 epochs LSKD)
  4. run_cba_v2_main_colab_sinkd_3seeds.ipynb     (RUN = "sinkd", Arm_SinKD: 300 epochs differentiable SinKD)
  5. run_cba_v2_main_colab_dist_plus_3seeds.ipynb (RUN = "dist_plus", Arm_DIST_Plus: 300 epochs fp32 DIST+)
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPLOAD = ROOT.parent / "colab_upload"
for _p in (ROOT, ROOT / "models", ROOT / "scripts", ROOT / "scripts" / "tools"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import zip_code  # noqa: E402


METHOD_SPECS = {
    "ours": {
        "notebook": "run_cba_v2_main_colab_ours_3seeds.ipynb",
        "title": "Ours v5 (KIS + Dual Boundary CBA + Error-Guided Hard-Pixel CE) — 3 Seeds (2345, 42, 1234)",
        "base_dir": "ours_v5",
        "arm_dir": "Arm_V5_Dual_CEHard",
        "summary": (
            "Trains the original Ours v5 (`Arm_V5_Dual_CEHard`, `lambda_a=0.02, beta=0.5, tau=0.1, m=0.2, r=2`, uncapped, `flat` CPD weights, "
            "`fp_by_pred_class=False, fp_exclude_boundary=False, empty_slice_suppression=False, align_ce_hard_samples=True`) "
            "for 300 total epochs per seed (20-epoch KIS warmup + 280-epoch joint KIS + Boundary CBA + class-balanced Error-Guided Hard-Pixel CE). "
            "If a seed's 20-epoch warmup checkpoint is already present on local disk or Drive (e.g. `warmup_s42/checkpoint_epoch_020.pth`), "
            "training resumes directly from epoch 20 (`Ep 21/300`); otherwise `--make_warmup` creates it automatically with `--skip_eval`."
        ),
    },
    "iad": {
        "notebook": "run_cba_v2_main_colab_iad_3seeds.ipynb",
        "title": "IAD (200 Epochs KIS + 100 Epochs KAS) — 3 Seeds (2345, 42, 1234)",
        "base_dir": "iad_200_100",
        "arm_dir": "Arm_IAD_200_100",
        "summary": (
            "Trains two-stage IAD (`Arm_IAD_200_100`) under a matched 300-epoch total budget per seed: "
            "Stage 1 (epochs 1..200) runs KIS (`0.8 * L_SGS + 0.2 * L_KIS`, `flat` CPD weights), and "
            "Stage 2 (epochs 201..300) runs KAS (`L_SGS + 0.1 * (L_RFC + L_CSS)`) with registered `IADKasHead` "
            "convolutions (`pos` 1x1 and `neg` 3x3). Permanent checkpoints are saved at epoch 200 and epoch 300."
        ),
    },
    "lskd": {
        "notebook": "run_cba_v2_main_colab_lskd_3seeds.ipynb",
        "title": "LSKD (CVPR 2024 Logit Standardization KD) — 3 Seeds (2345, 42, 1234)",
        "base_dir": "lskd300",
        "arm_dir": "Arm_LSKD",
        "summary": (
            "Trains LSKD (`Arm_LSKD`, `0.8 * L_SGS + 0.2 * L_LSKD`, `tau=2.0`) for 300 epochs per seed "
            "with float32 Z-score logit standardization and pixel-averaged KL divergence."
        ),
    },
    "sinkd": {
        "notebook": "run_cba_v2_main_colab_sinkd_3seeds.ipynb",
        "title": "SinKD (Sinkhorn Optimal Transport KD) — 3 Seeds (2345, 42, 1234)",
        "base_dir": "sinkd300",
        "arm_dir": "Arm_SinKD",
        "summary": (
            "Trains SinKD (`Arm_SinKD`, `0.8 * L_SGS + 0.2 * L_SinKD`, `lam=10.0, max_iter=5`) for 300 epochs per seed "
            "using the fixed differentiable Sinkhorn loss (teacher detached, student attached, L2-normalized features, "
            "and row-minimum shift stabilization to prevent exponential underflow)."
        ),
    },
    "dist_plus": {
        "notebook": "run_cba_v2_main_colab_dist_plus_3seeds.ipynb",
        "title": "DIST+ (TPAMI 2025 Inter/Intra-Class Relation KD) — 3 Seeds (2345, 42, 1234)",
        "base_dir": "dist_plus300",
        "arm_dir": "Arm_DIST_Plus",
        "summary": (
            "Trains DIST+ (`Arm_DIST_Plus`, `0.8 * L_SGS + 0.2 * L_DIST_Plus`, `beta=1.0, gamma=1.0, tau=1.0`) "
            "for 300 epochs per seed with float32-stabilized inter-class and intra-class Pearson correlation distillation."
        ),
    },
}


def cell(kind: str, text: str) -> dict:
    result = {"cell_type": kind, "metadata": {}, "source": text.strip().splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


SETUP = r'''
# 2. SETUP: use uploaded files in /content, or switch INPUT_MODE to drive.
import json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

if INPUT_MODE == "drive":
    from google.colab import drive
    drive.mount("/content/drive")
    INPUT = Path(DRIVE_ROOT)
    MAIN_DRIVE = str(INPUT / "main")
elif INPUT_MODE == "upload":
    INPUT = Path("/content")
    MAIN_DRIVE = str(Path(DRIVE_ROOT) / "main") if os.path.isdir(DRIVE_ROOT) else None
else:
    raise ValueError("INPUT_MODE must be 'upload' or 'drive'")

work = Path(WORK)
work.mkdir(parents=True, exist_ok=True)
for name in ("iad_cba_v2_code.zip", "btcv_data.zip"):
    if name == "btcv_data.zip":
        npz_dir = work / "data" / "btcv" / "train_npz"
        h5_dir = work / "data" / "btcv" / "test_vol_h5"
        if npz_dir.is_dir() and h5_dir.is_dir() and len(list(npz_dir.glob("*.npz"))) >= 2211 and len(list(h5_dir.glob("*.h5"))) >= 12:
            print("BTCV data already extracted in", work)
            continue
    archive = INPUT / name
    if not archive.is_file() or not zipfile.is_zipfile(archive):
        raise RuntimeError(f"Missing or incomplete ZIP: {archive}. Finish uploading the file before running setup.")
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"Corrupt ZIP entry: {bad} in {archive}")
        z.extractall(work)

for _mod in list(sys.modules):
    if _mod in ("main_common", "run_tuning") or _mod.startswith(("experiments.", "datasets.", "Synapse_Tool_KD.")):
        sys.modules.pop(_mod, None)

tea_src = INPUT / "epoch_299.pth"
if not tea_src.is_file():
    tea_src = INPUT / "teacher_weights" / "epoch_299.pth"
if not tea_src.is_file():
    raise FileNotFoundError(f"Upload epoch_299.pth into {INPUT}")
tea_dst = work / "teacher_weights" / "epoch_299.pth"
tea_dst.parent.mkdir(parents=True, exist_ok=True)
if not tea_dst.is_file() or tea_dst.stat().st_size != tea_src.stat().st_size:
    shutil.copy2(tea_src, tea_dst)
sam_dst = work / "data" / "weights" / "sam_vit_l_0b3195.pth"
sam_dst.parent.mkdir(parents=True, exist_ok=True)
sam_src = INPUT / "data" / "weights" / sam_dst.name
if not sam_dst.is_file() or sam_dst.stat().st_size < 1_200_000_000:
    if sam_src.is_file():
        shutil.copy2(sam_src, sam_dst)
    else:
        subprocess.run(["curl", "-fL", "--retry", "3", "-o", str(sam_dst),
                        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"], check=True)
assert sam_dst.stat().st_size >= 1_200_000_000, "Incomplete SAM base weights"
os.chdir(WORK)
if WORK not in sys.path:
    sys.path.insert(0, WORK)
if f"{WORK}/scripts" not in sys.path:
    sys.path.insert(0, f"{WORK}/scripts")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements_colab.txt"], check=True)

# Freeze the canonical v5 parameters for all seeds; do not read a changing tuning state.
STATE = str(work / "multiseed_v5_state.json")
Path(STATE).write_text(json.dumps({"lambda_a": 0.02, "tau": 0.1, "m": 0.2, "r": 2, "beta": 0.5}))

def run_command(command):
    print("Running:", " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=WORK, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    if code:
        raise RuntimeError(f"Training command failed with exit code {code}")

def command_for(seed, dry_run=False):
    cmd = [sys.executable, f"scripts/run_main_{RUN}.py", "--runs_dir", RUNS,
           "--epochs", str(EPOCHS), "--seed", str(seed), "--eval_freq", "0"]
    if MAIN_DRIVE:
        cmd += ["--drive_dir", MAIN_DRIVE]
    if RUN == "ours":
        cmd += ["--arm", "v5", "--state_json", STATE, "--make_warmup"]
    if dry_run:
        cmd.append("--dry_run")
    return cmd
print("Input mode:", INPUT_MODE, "| method:", RUN, "| seeds:", SEEDS, "| checkpoint backup:", MAIN_DRIVE or "local /content only")
'''

VERIFY = r'''
# 3. VERIFY: environment, bundled launcher, data, and seed-isolated dry runs across all 3 seeds.
import importlib, torch
assert torch.cuda.is_available(), "Select a GPU runtime in Colab first"
print("Torch:", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
for module in ("icecream", "cv2", "einops", "medpy", "h5py", "timm", "scipy"):
    importlib.import_module(module)
import datasets.dataset_synapse as ds
assert Path(ds.__file__).resolve().is_relative_to(Path(WORK).resolve()), ds.__file__
assert (work / "data/lists_Synapse/train.txt").is_file()
assert (work / "data/btcv/test_vol_h5").is_dir()
import main_common as mc
assert mc.seed_name("warmup", 42) == "warmup_s42", "Old code ZIP: upload the rebuilt bundle"
assert {"kis", "iad", "lskd", "sinkd", "dist_plus"} <= set(mc.KIND_DIRS), "Old code ZIP: missing 5-method support"
from experiments.cba_v2.losses.cba_v2_loss import CBAv2Loss
_v5_check = CBAv2Loss(error_mode="ce_hard", error_gating_type="none")
assert (
    (not _v5_check.fp_by_pred_class)
    and (not _v5_check.fp_exclude_boundary)
    and (not _v5_check.empty_slice_suppression)
    and _v5_check.align_ce_hard_samples
), (
    "Code ZIP still has Solution 1+2 enabled by default in CBAv2Loss; upload/extract the reverted iad_cba_v2_code.zip"
)
del _v5_check
for seed in SEEDS:
    if RUN == "ours":
        _job = mc.job_for("ours", "v5", mc.rt.DEFAULT_STATE, EPOCHS, seed=seed)
        mc.archive_sol12_run_if_needed(_job, RUNS, MAIN_DRIVE)
        if MAIN_DRIVE and not mc.rt.is_done(_job, RUNS, MAIN_DRIVE) and not mc.rt.has_own_checkpoint(_job, RUNS, MAIN_DRIVE):
            local_wck = Path(RUNS) / mc.seed_name("warmup", seed) / "Arm_V1_KIS" / "checkpoint_epoch_020.pth"
            drive_wck = mc.find_warmup_ckpt(RUNS, MAIN_DRIVE, seed)
            if drive_wck and not local_wck.is_file():
                local_wck.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(drive_wck, local_wck)
                print(f"Staged existing Drive warmup for seed {seed}: {local_wck}", flush=True)
    run_command(command_for(seed, dry_run=True))
'''

RUN_CELL = r'''
# 4. RUN: train all 3 seeds (2345, 42, 1234) for 300 total epochs per seed.
# Final 12-volume 3D BTCV evaluation runs only at epoch 300 (--eval_freq 0).
# Re-running after a disconnect skips completed seeds and resumes any interrupted seed from its latest checkpoint.
for seed in SEEDS:
    if RUN == "ours":
        _job = mc.job_for("ours", "v5", mc.rt.DEFAULT_STATE, EPOCHS, seed=seed)
        mc.archive_sol12_run_if_needed(_job, RUNS, MAIN_DRIVE)
        uploaded = Path(f"/content/warmup_s{seed}.pth")
        if not uploaded.is_file() and seed == 2345:
            uploaded = Path("/content/warmup.pth")
        if uploaded.is_file():
            ckpt = torch.load(uploaded, map_location="cpu", weights_only=False)
            saved_args = ckpt.get("args", {})
            assert saved_args.get("seed") == seed, f"Uploaded warmup seed mismatch: {saved_args.get('seed')} != {seed}"
            assert ckpt.get("epoch") == 20, "Uploaded warmup must be at epoch 20"
            del ckpt
            dest = Path(RUNS) / mc.seed_name("warmup", seed) / "Arm_V1_KIS" / "checkpoint_epoch_020.pth"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(uploaded, dest)
            print(f"Imported uploaded warmup for seed {seed}: {dest}", flush=True)
        elif MAIN_DRIVE and not mc.rt.is_done(_job, RUNS, MAIN_DRIVE) and not mc.rt.has_own_checkpoint(_job, RUNS, MAIN_DRIVE):
            local_wck = Path(RUNS) / mc.seed_name("warmup", seed) / "Arm_V1_KIS" / "checkpoint_epoch_020.pth"
            drive_wck = mc.find_warmup_ckpt(RUNS, MAIN_DRIVE, seed)
            if drive_wck and not local_wck.is_file():
                local_wck.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(drive_wck, local_wck)
                print(f"Staged existing Drive warmup for seed {seed}: {local_wck}", flush=True)
    run_command(command_for(seed))
'''

STATUS = r'''
# 5. STATUS & 3-SEED SUMMARY: report each seed (Raw & Post-LCC) and aggregate Mean ± Std across completed seeds.
import csv, math
BASE_NAME = "ours_v5" if RUN == "ours" else mc.KIND_DIRS[RUN]
ARM_NAME = "Arm_V5_Dual_CEHard" if RUN == "ours" else mc.rt.ARM_DIRS[RUN]

dice_raw_vals, hd95_raw_vals, nsd_raw_vals = [], [], []
dice_lcc_vals, hd95_lcc_vals, nsd_lcc_vals = [], [], []
per_seed_records = {}
for seed in SEEDS:
    name = mc.seed_name(BASE_NAME, seed)
    local = Path(RUNS) / name / ARM_NAME
    mirror = Path(MAIN_DRIVE) / name if MAIN_DRIVE else local
    directory = local if (local / "metrics.csv").exists() else mirror
    metric_path = directory / "metrics.csv"
    if metric_path.is_file():
        with metric_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            last = rows[-1]
            print(f"[{name}] epoch={last.get('epoch')} stage={last.get('stage', ARM_NAME)} loss={last.get('loss_total')} sec={last.get('time_sec')}")
    final = directory / "eval_final.json"
    if final.is_file():
        overall = json.loads(final.read_text(encoding="utf-8"))["overall"]
        dr = float(overall["primary_dice_raw_macro"]) * 100.0
        hr = float(overall["primary_hd95_raw_valid_macro"])
        nr = float(overall["primary_nsd_raw_macro"]) * 100.0
        dl = float(overall.get("primary_dice_lcc_macro", overall["primary_dice_raw_macro"])) * 100.0
        hl = float(overall.get("primary_hd95_lcc_valid_macro", overall["primary_hd95_raw_valid_macro"]))
        nl = float(overall.get("primary_nsd_lcc_macro", overall["primary_nsd_raw_macro"])) * 100.0
        dice_raw_vals.append(dr); hd95_raw_vals.append(hr); nsd_raw_vals.append(nr)
        dice_lcc_vals.append(dl); hd95_lcc_vals.append(hl); nsd_lcc_vals.append(nl)
        per_seed_records[str(seed)] = {
            "dice_raw_pct": dr, "hd95_raw": hr, "nsd_raw_pct": nr,
            "dice_lcc_pct": dl, "hd95_lcc": hl, "nsd_lcc_pct": nl,
        }
        print(f"  -> FINAL (seed {seed}) RAW: Dice={dr:.2f}% | HD95={hr:.2f} | NSD={nr:.2f}%")
        print(f"  -> FINAL (seed {seed}) LCC: Dice={dl:.2f}% | HD95={hl:.2f} | NSD={nl:.2f}%")

if dice_raw_vals:
    def mean_std(xs):
        m = sum(xs) / len(xs)
        s = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) if len(xs) > 1 else 0.0
        return m, s
    drm, drs = mean_std(dice_raw_vals); hrm, hrs = mean_std(hd95_raw_vals); nrm, nrs = mean_std(nsd_raw_vals)
    dlm, dls = mean_std(dice_lcc_vals); hlm, hls = mean_std(hd95_lcc_vals); nlm, nls = mean_std(nsd_lcc_vals)
    print(f"\n=== {RUN.upper()} ({len(dice_raw_vals)}/{len(SEEDS)} seeds complete) ===")
    print(f"RAW -> Dice (%): {drm:.2f} ± {drs:.2f} | HD95: {hrm:.2f} ± {hrs:.2f} | NSD (%): {nrm:.2f} ± {nrs:.2f}")
    print(f"LCC -> Dice (%): {dlm:.2f} ± {dls:.2f} | HD95: {hlm:.2f} ± {hls:.2f} | NSD (%): {nlm:.2f} ± {nls:.2f}")
    summary_payload = {
        "method": RUN,
        "arm": ARM_NAME,
        "seeds_completed": [int(s) for s in per_seed_records],
        "per_seed": per_seed_records,
        "aggregate_raw": {"dice_mean": drm, "dice_std": drs, "hd95_mean": hrm, "hd95_std": hrs, "nsd_mean": nrm, "nsd_std": nrs},
        "aggregate_lcc": {"dice_mean": dlm, "dice_std": dls, "hd95_mean": hlm, "hd95_std": hls, "nsd_mean": nlm, "nsd_std": nls},
    }
    summary_path = Path(RUNS) / f"summary_{RUN}_3seeds.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    if MAIN_DRIVE:
        Path(MAIN_DRIVE).mkdir(parents=True, exist_ok=True)
        shutil.copy2(summary_path, Path(MAIN_DRIVE) / summary_path.name)
'''

DOWNLOAD = r'''
# 6. DOWNLOAD: package evaluation JSONs, logs, final_student.pth, and final_preds for each seed
# (excludes intermediate 25-epoch optimizer checkpoints so browser download succeeds reliably).
from google.colab import files
BASE_NAME = "ours_v5" if RUN == "ours" else mc.KIND_DIRS[RUN]
for seed in SEEDS:
    name = mc.seed_name(BASE_NAME, seed)
    directory = Path(RUNS) / name
    if directory.is_dir():
        archive = f"/content/results_{name}.zip"
        has_final_student = any(directory.rglob("final_student.pth"))
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in sorted(directory.rglob("*")):
                if not item.is_file():
                    continue
                if has_final_student and (item.name.startswith("checkpoint_epoch_") or item.name == "latest_checkpoint.pth"):
                    continue
                zf.write(item, item.relative_to(Path(RUNS)).as_posix())
        files.download(archive)
    else:
        print(f"Skipping {name}: local directory {directory} not found")
'''


def build_method_notebook(kind: str) -> dict:
    spec = METHOD_SPECS[kind]
    seeds = [2345, 42, 1234]
    intro = f"""# {spec["title"]}

{spec["summary"]}

- **Seeds**: `{seeds}` (3 seeds, run sequentially in this notebook or across sessions).
- **Total Budget**: `300` epochs per seed on all 18 native BTCV training cases (`2,211` slices).
- **Evaluation Schedule**: Final epoch only (`--eval_freq 0`) on the 12 official 3D BTCV test volumes (`data/btcv/test_vol_h5`). Intermediate warmup/Stage-1 checkpoints do not run 3D volume evaluation.
- **Input Modes**:
  - `INPUT_MODE = "upload"` (default): Upload `iad_cba_v2_code.zip`, `btcv_data.zip`, and `epoch_299.pth` into `/content/`. SAM ViT-L base weights (`sam_vit_l_0b3195.pth`) download automatically if not uploaded.
  - `INPUT_MODE = "drive"`: Mounts Google Drive (`MyDrive/IAD`), reads inputs from Drive, and syncs checkpoints/metrics to `MyDrive/IAD/main/{spec["base_dir"]}*` every 5 epochs.
- **Disconnect Recovery**: Re-running cells 1–4 skips any seed that already has `eval_final.json` and resumes any interrupted seed from its latest checkpoint (`latest_checkpoint.pth`).
"""
    config = f'''# 1. CONFIG
INPUT_MODE = "upload"  # "upload" for /content files; "drive" to mount Google Drive.
RUN = "{kind}"
SEEDS = {seeds!r}
EPOCHS = 300
WORK = "/content/IAD"
RUNS = "/content/runs/main"
DRIVE_ROOT = "/content/drive/MyDrive/IAD"
'''
    cells = [
        cell("markdown", intro),
        cell("code", config),
        cell("code", SETUP),
        cell("code", VERIFY),
        cell("code", RUN_CELL),
        cell("code", STATUS),
        cell("code", DOWNLOAD),
    ]
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"gpuType": "A100", "provenance": []},
            "accelerator": "GPU",
        },
        "cells": cells,
    }


README_TEXT = """# 5-Method × 3-Seed (300-Epoch) Colab Benchmark Bundle

This folder contains the complete, reproducible 300-epoch benchmark bundle for **5 knowledge distillation methods** across **3 seeds (`2345`, `42`, `1234`)** on native BTCV (18 training cases / 2,211 slices; 12 3D test volumes).

## 1. The 5 Method Notebooks (1 Notebook per Method, 3 Seeds Each)

| Method | Notebook | Launcher Script | Arm Name | Output Directories (`seed 2345`, `42`, `1234`) |
|---|---|---|---|---|
| **1. Ours (`v5`)** | `run_cba_v2_main_colab_ours_3seeds.ipynb` | `scripts/run_main_ours.py` | `Arm_V5_Dual_CEHard` | `ours_v5`, `ours_v5_s42`, `ours_v5_s1234` |
| **2. IAD (`200 KIS + 100 KAS`)** | `run_cba_v2_main_colab_iad_3seeds.ipynb` | `scripts/run_main_iad.py` | `Arm_IAD_200_100` | `iad_200_100`, `iad_200_100_s42`, `iad_200_100_s1234` |
| **3. LSKD (CVPR 2024)** | `run_cba_v2_main_colab_lskd_3seeds.ipynb` | `scripts/run_main_lskd.py` | `Arm_LSKD` | `lskd300`, `lskd300_s42`, `lskd300_s1234` |
| **4. SinKD (Fixed OT KD)** | `run_cba_v2_main_colab_sinkd_3seeds.ipynb` | `scripts/run_main_sinkd.py` | `Arm_SinKD` | `sinkd300`, `sinkd300_s42`, `sinkd300_s1234` |
| **5. DIST+ (TPAMI 2025)** | `run_cba_v2_main_colab_dist_plus_3seeds.ipynb` | `scripts/run_main_dist_plus.py` | `Arm_DIST_Plus` | `dist_plus300`, `dist_plus300_s42`, `dist_plus300_s1234` |

## 2. Protocol & Code Audit Fixes Included in `iad_cba_v2_code.zip`

- **Matched 300-Epoch Budget**: Every method trains for exactly 300 total epochs per seed with identical data loader (`aug="full"`), optimizer (`AdamW`, `lr=1e-4`, `weight_decay=0.1`), effective batch size 16 (`batch_size=8, grad_accum_steps=2`), linear warmup (`250` iterations), polynomial LR decay (`power=0.9`), and mixed precision (`bfloat16`/`float16`).
- **Final-Epoch Evaluation Only**: All launchers pass `--eval_freq 0` so full 3D evaluation on the 12 test volumes runs only at epoch 300 (`eval_final.json`). Shared 20-epoch warmups use `--skip_eval`.
- **Verified Formulations & Bug Fixes Across Losses**:
  - **Ours (`Arm_V5_Dual_CEHard`, `CBAv2Loss` with `error_mode="ce_hard"`)**: Uses the original `ours_v5` formulation (`lambda_a=0.02, beta=0.5, tau=0.1, m=0.2, r=2`, uncapped, `flat` CPD weights, `fp_by_pred_class=False`, `fp_exclude_boundary=False`, `empty_slice_suppression=False`, `align_ce_hard_samples=True`), combining KIS (`0.8 * L_SGS + 0.2 * L_KIS`), directionally gated Boundary CBA ($L_B$, radius $r=2$), and class-balanced Error-Guided Hard-Pixel CE ($L_E$, sampling up to $K_{\\text{bg}}=512$ and $K_{\\text{fg}}=512$ misclassified pixels per active ground-truth class $c \\in \\{0..C-1\\}$ with alignment filtering against cached EMA prototypes). Any stale Solution 1+2 run directory (`fp_by_pred_class=True` or `empty_slice_suppression=True`) is automatically archived to `*_sol12_archive` so `ours_v5_s42` resumes cleanly from `warmup_s42` (`checkpoint_epoch_020.pth`) at Epoch 21.
  - **SinKD (`Synapse_Tool_KD/criterion/SinKD.py` & `SinKDLoss`)**: Removed the outer `with torch.no_grad():` wrapper that previously zeroed out all student gradients; detached only the teacher logits (`t.detach().float()`); L2-normalized flattened sample vectors so pairwise distances stay in $[0, 2]$; and subtracted the detached row minimum before `torch.exp(-lam * D)` so the Gibbs kernel cannot underflow to zero.
  - **DIST+ (`Synapse_Tool_KD/distillation_loss.py` & `DISTPlusLoss`)**: Enforced `float32` casting and teacher detachment before spatial reshaping and Pearson correlation so `intra_class_relation` over $B \\times H \\times W = 2,097,152$ pixels cannot overflow or lose precision under AMP `autocast`.
  - **LSKD (`Synapse_Tool_KD/criterion/lskd.py` & `LSKDLoss`)**: Enforced `float32` Z-score standardization and teacher detachment.
  - **IAD (`Arm_IAD_200_100` & `IADKasHead`)**: Registered `pos` ($1\\times 1$ Conv2d, $256 \\to 9$) and `neg` ($3\\times 3$ Conv2d, $9 \\to 9$) as persistent `nn.Module` parameters optimized during Stage 2 KAS (epochs 201..300) after Stage 1 KIS (epochs 1..200), fixing the inline unregistered `nn.Conv2d` bug in the original repository.

## 3. Running in Colab

1. Open one of the 5 `run_cba_v2_main_colab_<method>_3seeds.ipynb` notebooks in Google Colab (GPU runtime).
2. **Upload Mode (`INPUT_MODE = "upload"`, default)**:
   - Upload `iad_cba_v2_code.zip`, `btcv_data.zip`, and `epoch_299.pth` into `/content/`.
   - Run all cells in order (`1. CONFIG` -> `2. SETUP` -> `3. VERIFY` -> `4. RUN` -> `5. STATUS` -> `6. DOWNLOAD`).
3. **Drive Mode (`INPUT_MODE = "drive"`)**:
   - Place `iad_cba_v2_code.zip`, `btcv_data.zip`, and `teacher_weights/epoch_299.pth` in `MyDrive/IAD`.
   - Set `INPUT_MODE = "drive"` in Cell 1 and run Cells 1–5. Checkpoints sync to `MyDrive/IAD/main/` every 5 epochs, and completed seeds are automatically skipped if re-run.
"""


def main() -> None:
    UPLOAD.mkdir(exist_ok=True)
    (ROOT / "notebooks").mkdir(exist_ok=True)

    for kind, spec in METHOD_SPECS.items():
        nb = build_method_notebook(kind)
        nb_name = spec["notebook"]
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code":
                compile("".join(c["source"]), f"{nb_name}:cell{i}", "exec")
        nb_path = ROOT / "notebooks" / nb_name
        nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        shutil.copy2(nb_path, UPLOAD / nb_name)
        print(f"Wrote notebook: {nb_path} and {UPLOAD / nb_name}")

    for single_nb_name, upload_alias in (
        ("Copy of run_cba_v2_main_colab.ipynb", "run_cba_v2_main_colab_ours_s42.ipynb"),
        ("Copy(0.1) of run_cba_v2_main_colab.ipynb", "run_cba_v2_main_colab_ours_s1234.ipynb"),
    ):
        s_path = ROOT / "notebooks" / single_nb_name
        if s_path.is_file():
            s_nb = json.loads(s_path.read_text(encoding="utf-8"))
            c2 = "".join(s_nb["cells"][2]["source"])
            if "sys.modules.pop(_mod, None)" not in c2:
                c2 = c2.replace(
                    "        z.extractall(work)\n",
                    "        z.extractall(work)\n\n"
                    "for _mod in list(sys.modules):\n"
                    '    if _mod in ("main_common", "run_tuning") or _mod.startswith(("experiments.", "datasets.", "Synapse_Tool_KD.")):\n'
                    "        sys.modules.pop(_mod, None)\n",
                )
                s_nb["cells"][2]["source"] = c2.splitlines(keepends=True)
            c3 = "".join(s_nb["cells"][3]["source"])
            if "_v5_check = CBAv2Loss" not in c3:
                c3 = c3.replace(
                    "for seed in SEEDS:\n",
                    "from experiments.cba_v2.losses.cba_v2_loss import CBAv2Loss\n"
                    '_v5_check = CBAv2Loss(error_mode="ce_hard", error_gating_type="none")\n'
                    "assert (not _v5_check.fp_by_pred_class) and (not _v5_check.fp_exclude_boundary) and (not _v5_check.empty_slice_suppression) and _v5_check.align_ce_hard_samples\n"
                    "del _v5_check\n"
                    "for seed in SEEDS:\n",
                )
                s_nb["cells"][3]["source"] = c3.splitlines(keepends=True)
            s_path.write_text(json.dumps(s_nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            shutil.copy2(s_path, UPLOAD / upload_alias)

    kis_2seeds_path = UPLOAD / "run_cba_v2_main_colab_kis_2seeds.ipynb"
    if kis_2seeds_path.is_file():
        shutil.copy2(kis_2seeds_path, ROOT / "notebooks" / "run_cba_v2_main_colab.ipynb")

    readme_path = UPLOAD / "MULTISEED_README.md"
    readme_path.write_text(README_TEXT, encoding="utf-8")
    print(f"Updated {readme_path}")

    # Rebuild iad_cba_v2_code.zip with all targets in zip_code.CBA_V2_TARGETS
    entries = {}
    for rel in zip_code.CBA_V2_TARGETS:
        base_dir, target = zip_code.resolve_bundle_target(rel, ROOT)
        if target is None or not target.exists():
            if rel.startswith("tests/test_cba_v2_5methods_multiseed.py"):
                continue
            raise FileNotFoundError(f"Bundle target missing: {rel}")
        if target.is_file():
            entries[Path(rel).as_posix()] = target
        else:
            for path in target.rglob("*"):
                if (
                    path.is_file()
                    and "__pycache__" not in path.parts
                    and not any(p.startswith(".") for p in path.parts)
                    and path.suffix not in (".pyc", ".tmp")
                ):
                    entries[path.relative_to(base_dir).as_posix()] = path

    archive = UPLOAD / "iad_cba_v2_code.zip" if UPLOAD.is_dir() else ROOT / "dist" / "iad_cba_v2_code.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for name, path in sorted(entries.items()):
            z.write(path, name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    if archive.resolve() != (UPLOAD / archive.name).resolve():
        shutil.copy2(archive, UPLOAD / archive.name)
    (UPLOAD / "main_5methods_3seeds").mkdir(parents=True, exist_ok=True)
    for mirror_dir in (UPLOAD / "main_5methods_3seeds", ROOT / "colab_upload", ROOT / "colab_upload" / "main_5methods_3seeds"):
        if not mirror_dir.is_dir():
            continue
        shutil.copy2(archive, mirror_dir / archive.name)
        shutil.copy2(readme_path, mirror_dir / readme_path.name)
        for spec in METHOD_SPECS.values():
            nb_name = spec["notebook"]
            shutil.copy2(UPLOAD / nb_name, mirror_dir / nb_name)
    print(f"Prepared {len(entries)} ZIP entries; sha256={hashlib.sha256(archive.read_bytes()).hexdigest()}")

    for drive_dir in (Path(r"I:\My Drive\IAD"), Path(r"J:\My Drive\IAD"), Path(r"H:\My Drive\IAD")):
        try:
            if not Path(f"{drive_dir.drive}\\").exists():
                continue
            drive_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(archive, drive_dir / archive.name)
            shutil.copy2(readme_path, drive_dir / readme_path.name)
            boost_dir = drive_dir.parent / "boost"
            if boost_dir.is_dir():
                shutil.copy2(archive, boost_dir / archive.name)
            for kind, spec in METHOD_SPECS.items():
                nb_name = spec["notebook"]
                dst_nb = drive_dir / nb_name
                if dst_nb.is_file() and dst_nb.stat().st_size > 50_000:
                    print(f"Skipping active Colab notebook on Drive: {dst_nb}")
                    continue
                drive_nb = build_method_notebook(kind)
                drive_nb["cells"][1]["source"] = [
                    line.replace('INPUT_MODE = "upload"', 'INPUT_MODE = "drive"')
                    for line in drive_nb["cells"][1]["source"]
                ]
                dst_nb.write_text(json.dumps(drive_nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            drive_sub = drive_dir / "main_5methods_3seeds"
            drive_sub.mkdir(parents=True, exist_ok=True)
            shutil.copy2(archive, drive_sub / archive.name)
            shutil.copy2(readme_path, drive_sub / readme_path.name)
            for spec in METHOD_SPECS.values():
                nb_name = spec["notebook"]
                dst_sub_nb = drive_sub / nb_name
                if dst_sub_nb.is_file() and dst_sub_nb.stat().st_size > 50_000:
                    print(f"Skipping active Colab notebook on Drive: {dst_sub_nb}")
                    continue
                shutil.copy2(UPLOAD / nb_name, dst_sub_nb)
            for single_nb_name, upload_alias in (
                ("Copy of run_cba_v2_main_colab.ipynb", "run_cba_v2_main_colab_ours_s42.ipynb"),
                ("Copy(0.1) of run_cba_v2_main_colab.ipynb", "run_cba_v2_main_colab_ours_s1234.ipynb"),
            ):
                s_src = ROOT / "notebooks" / single_nb_name
                if s_src.is_file():
                    shutil.copy2(s_src, drive_dir / single_nb_name)
                    shutil.copy2(s_src, drive_dir / upload_alias)
            print(f"Synced code bundle and notebooks to {drive_dir}")
        except Exception as exc:
            print(f"[WARN] Could not sync to {drive_dir}: {exc}")


if __name__ == "__main__":
    main()

