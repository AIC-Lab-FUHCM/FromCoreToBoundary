"""Generate the 3 parallel Colab notebooks for ACDC + Polyp across all 5 KD methods.

Splits all 10 (dataset, method) runs across 3 balanced parallel Colab notebooks so 3 Colab GPU sessions
can execute in parallel without any shared-directory contention:

  1. run_acdc_polyp_parallel_nb1_colab.ipynb
     JOBS = [("acdc", "ours"), ("polyp", "ours"), ("acdc", "iad")]
  2. run_acdc_polyp_parallel_nb2_colab.ipynb
     JOBS = [("polyp", "iad"), ("acdc", "dist_plus"), ("polyp", "dist_plus")]
  3. run_acdc_polyp_parallel_nb3_colab.ipynb
     JOBS = [("acdc", "lskd"), ("polyp", "lskd"), ("acdc", "sinkd"), ("polyp", "sinkd")]

Strict Guards Built Into Every Notebook:
  - Dataset isolation:
      ACDC  -> runs_dir=/content/IAD/runs/acdc_main,  drive_dir=/content/drive/MyDrive/IAD/acdc_main
      Polyp -> runs_dir=/content/IAD/runs/polyp_main, drive_dir=/content/drive/MyDrive/IAD/polyp_main
  - Checkpoint verification via verify_checkpoint_compatibility():
      acdc_tea.pth  -> must have mask_tokens.weight shape (4, 256) and role="teacher"
      polyp_tea.pth -> must have mask_tokens.weight shape (1, 256) and role="teacher"
  - Automatic dataset provisioning:
      Extracts acdc_data.zip (or ACDC.zip) and polyp_data.zip (or dataset.zip) from Drive/upload,
      or automatically downloads the official Google Drive archives via gdown if not uploaded.
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


NOTEBOOK_SPECS = {
    "nb1": {
        "notebook": "run_acdc_polyp_parallel_nb1_colab.ipynb",
        "title": "Parallel Session 1/3: ACDC & Polyp — Ours (v5) + ACDC IAD",
        "jobs": [("acdc", "ours"), ("polyp", "ours"), ("acdc", "iad")],
        "summary": (
            "Runs **3 of the 10** ACDC + Polyp benchmark jobs in parallel Session 1:\n"
            "1. **ACDC — Ours (`Arm_V5_Dual_CEHard`)**: 4 classes (`BG, RV, Myo, LV`), `256×256`, 300 epochs (20-epoch KIS warmup + 280-epoch KIS + Boundary CBA + Error-Guided Hard-Pixel CE).\n"
            "2. **Polyp — Ours (`Arm_V5_Dual_CEHard`)**: 1 binary channel (`256×256`), 300 epochs (20-epoch KIS warmup + 280-epoch binary KIS + lifted $S^1$ Boundary CBA + Error-Guided Hard-Pixel CE).\n"
            "3. **ACDC — IAD (`Arm_IAD_200_100`)**: 4 classes (`256×256`), 300 epochs (200 epochs KIS + 100 epochs KAS)."
        ),
    },
    "nb2": {
        "notebook": "run_acdc_polyp_parallel_nb2_colab.ipynb",
        "title": "Parallel Session 2/3: ACDC & Polyp — Polyp IAD + DIST+ (ACDC & Polyp)",
        "jobs": [("polyp", "iad"), ("acdc", "dist_plus"), ("polyp", "dist_plus")],
        "summary": (
            "Runs **3 of the 10** ACDC + Polyp benchmark jobs in parallel Session 2:\n"
            "1. **Polyp — IAD (`Arm_IAD_200_100`)**: 1 binary channel (`256×256`), 300 epochs (200 epochs KIS + 100 epochs KAS).\n"
            "2. **ACDC — DIST+ (`Arm_DIST_Plus`)**: 4 classes (`256×256`), 300 epochs inter/intra-class Pearson correlation KD.\n"
            "3. **Polyp — DIST+ (`Arm_DIST_Plus`)**: 1 binary channel (`256×256`), 300 epochs binary 2-channel probability Pearson correlation KD."
        ),
    },
    "nb3": {
        "notebook": "run_acdc_polyp_parallel_nb3_colab.ipynb",
        "title": "Parallel Session 3/3: ACDC & Polyp — LSKD (ACDC & Polyp) + SinKD (ACDC & Polyp)",
        "jobs": [("acdc", "lskd"), ("polyp", "lskd"), ("acdc", "sinkd"), ("polyp", "sinkd")],
        "summary": (
            "Runs **4 of the 10** ACDC + Polyp benchmark jobs in parallel Session 3:\n"
            "1. **ACDC — LSKD (`Arm_LSKD`)**: 4 classes (`256×256`), 300 epochs Z-score logit standardization KD.\n"
            "2. **Polyp — LSKD (`Arm_LSKD`)**: 1 binary channel (`256×256`), 300 epochs spatial Z-score logit standardization KD.\n"
            "3. **ACDC — SinKD (`Arm_SinKD`)**: 4 classes (`256×256`), 300 epochs differentiable Sinkhorn Optimal Transport KD.\n"
            "4. **Polyp — SinKD (`Arm_SinKD`)**: 1 binary channel (`256×256`), 300 epochs binary 2-channel Sinkhorn Optimal Transport KD."
        ),
    },
}


def cell(kind: str, text: str) -> dict:
    result = {"cell_type": kind, "metadata": {}, "source": text.strip().splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


SETUP_CELL = r'''
# 2. SETUP: extract code, verify/download ACDC & Polyp datasets and Teacher checkpoints, install dependencies.
import json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

if INPUT_MODE == "drive":
    from google.colab import drive
    drive.mount("/content/drive")
    INPUT = Path(DRIVE_ROOT)
    DRIVE_BASE = Path(DRIVE_ROOT)
elif INPUT_MODE == "upload":
    INPUT = Path("/content")
    DRIVE_BASE = Path(DRIVE_ROOT) if os.path.isdir(DRIVE_ROOT) else None
else:
    raise ValueError("INPUT_MODE must be 'upload' or 'drive'")

work = Path(WORK)
work.mkdir(parents=True, exist_ok=True)

# 2a. Extract iad_cba_v2_code.zip
code_zip = INPUT / "iad_cba_v2_code.zip"
if not code_zip.is_file() and DRIVE_BASE is not None and (DRIVE_BASE / "iad_cba_v2_code.zip").is_file():
    code_zip = DRIVE_BASE / "iad_cba_v2_code.zip"
if not code_zip.is_file() or not zipfile.is_zipfile(code_zip):
    raise RuntimeError(f"Missing or incomplete code archive: {code_zip}. Upload iad_cba_v2_code.zip first.")
with zipfile.ZipFile(code_zip) as z:
    bad = z.testzip()
    if bad:
        raise RuntimeError(f"Corrupt ZIP entry: {bad} in {code_zip}")
    z.extractall(work)

for _mod in list(sys.modules):
    if _mod in ("main_common", "run_tuning") or _mod.startswith(("experiments.", "datasets.", "Synapse_Tool_KD.")):
        sys.modules.pop(_mod, None)

os.chdir(WORK)
if WORK not in sys.path:
    sys.path.insert(0, WORK)
if f"{WORK}/scripts" not in sys.path:
    sys.path.insert(0, f"{WORK}/scripts")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements_colab.txt"], check=True)

# 2b. Provision ACDC & Polyp datasets (from upload/Drive, or auto-download via gdown if missing)
NEEDED_DATASETS = sorted({ds for ds, _ in JOBS})
GDRIVE_DATA_IDS = {
    "acdc": ("1CruCQ-jjvA97BX-LIYwXaRMLmp3DN9zc", ("acdc_data.zip", "ACDC.zip"), 100_000_000),
    "polyp": ("1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC", ("polyp_data.zip", "dataset.zip"), 1_000_000_000),
}

def _acdc_ready(root: Path) -> bool:
    tr = root / "data" / "ACDC" / "train"
    te = root / "data" / "ACDC" / "test"
    return tr.is_dir() and te.is_dir() and len(list(tr.glob("*.npz"))) >= 1300 and len(list(te.glob("*.npz"))) >= 40

def _polyp_ready(root: Path) -> bool:
    tr = root / "data" / "dataset" / "TrainDataset" / "images"
    te = root / "data" / "dataset" / "TestDataset" / "Kvasir" / "images"
    return tr.is_dir() and te.is_dir() and len(list(tr.iterdir())) >= 1450 and len(list(te.iterdir())) >= 100

for ds_name in NEEDED_DATASETS:
    if ds_name == "acdc" and _acdc_ready(work):
        print("[Data] ACDC already extracted in", work / "data" / "ACDC")
        continue
    if ds_name == "polyp" and _polyp_ready(work):
        print("[Data] Polyp already extracted in", work / "data" / "dataset")
        continue

    gid, zip_names, min_bytes = GDRIVE_DATA_IDS[ds_name]
    found_zip = None
    for zn in zip_names:
        for base in ([INPUT, DRIVE_BASE, work / "data"] if DRIVE_BASE else [INPUT, work / "data"]):
            if base is not None and (base / zn).is_file() and (base / zn).stat().st_size >= min_bytes:
                found_zip = base / zn
                break
        if found_zip is not None:
            break

    if found_zip is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
        import gdown
        dl_target = (DRIVE_BASE / zip_names[0]) if (DRIVE_BASE is not None and DRIVE_BASE.is_dir()) else (work / "data" / zip_names[0])
        dl_target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[Data] Downloading official {ds_name.upper()} dataset (id={gid}) to {dl_target}...", flush=True)
        gdown.download(id=gid, output=str(dl_target), quiet=False)
        found_zip = dl_target

    assert found_zip.is_file() and zipfile.is_zipfile(found_zip), f"Invalid dataset archive: {found_zip}"
    extract_root = work / "data"
    extract_root.mkdir(parents=True, exist_ok=True)
    print(f"[Data] Extracting {found_zip} -> {extract_root} ...", flush=True)
    with zipfile.ZipFile(found_zip) as zf:
        zf.extractall(extract_root)

# 2c. Stage SAM ViT-L base checkpoint & dataset-specific Teacher checkpoints
sam_dst = work / "data" / "weights" / "sam_vit_l_0b3195.pth"
sam_dst.parent.mkdir(parents=True, exist_ok=True)
sam_cands = [INPUT / "sam_vit_l_0b3195.pth", INPUT / "data" / "weights" / "sam_vit_l_0b3195.pth"]
if DRIVE_BASE is not None:
    sam_cands += [DRIVE_BASE / "sam_vit_l_0b3195.pth", DRIVE_BASE / "data" / "weights" / "sam_vit_l_0b3195.pth"]
if not sam_dst.is_file() or sam_dst.stat().st_size < 1_200_000_000:
    sam_src = next((c for c in sam_cands if c.is_file() and c.stat().st_size >= 1_200_000_000), None)
    if sam_src is not None:
        shutil.copy2(sam_src, sam_dst)
    else:
        subprocess.run(["curl", "-fL", "--retry", "3", "-o", str(sam_dst),
                        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"], check=True)
assert sam_dst.stat().st_size >= 1_200_000_000, "Incomplete SAM ViT-L base weights"

TEACHER_FILES = {"acdc": "acdc_tea.pth", "polyp": "polyp_tea.pth"}
for ds_name in NEEDED_DATASETS:
    t_name = TEACHER_FILES[ds_name]
    t_dst = work / "teacher_weights" / t_name
    t_dst.parent.mkdir(parents=True, exist_ok=True)
    t_cands = [INPUT / t_name, INPUT / "teacher_weights" / t_name]
    if DRIVE_BASE is not None:
        t_cands += [DRIVE_BASE / t_name, DRIVE_BASE / "teacher_weights" / t_name]
    if not t_dst.is_file():
        t_src = next((c for c in t_cands if c.is_file()), None)
        if t_src is None:
            raise FileNotFoundError(
                f"Missing Teacher checkpoint {t_name} for dataset={ds_name!r}. "
                f"Place {t_name} in {INPUT} or {DRIVE_ROOT}/teacher_weights/{t_name}."
            )
        shutil.copy2(t_src, t_dst)
    print(f"[Teacher] Staged {ds_name.upper()} teacher: {t_dst} ({t_dst.stat().st_size:,d} bytes)")

STATE = str(work / "multiseed_v5_state.json")
Path(STATE).write_text(json.dumps({"lambda_a": 0.02, "tau": 0.1, "m": 0.2, "r": 2, "beta": 0.5}))

def runs_dir_for(dataset: str) -> str:
    return f"{WORK}/runs/{dataset}_main"

def drive_dir_for(dataset: str):
    if DRIVE_BASE is not None and DRIVE_BASE.is_dir():
        return str(DRIVE_BASE / f"{dataset}_main")
    return None

def run_command(command):
    print("Running:", " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=WORK, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    if code:
        raise RuntimeError(f"Training command failed with exit code {code}")

def command_for(dataset: str, method: str, seed: int, dry_run: bool = False):
    r_dir = runs_dir_for(dataset)
    d_dir = drive_dir_for(dataset)
    cmd = [
        sys.executable, f"scripts/run_main_{method}.py",
        "--dataset", dataset,
        "--runs_dir", r_dir,
        "--epochs", str(EPOCHS),
        "--seed", str(seed),
        "--eval_freq", "0",
    ]
    if d_dir:
        cmd += ["--drive_dir", d_dir]
    if method == "ours":
        cmd += ["--arm", "v5", "--state_json", STATE, "--make_warmup"]
    if dry_run:
        cmd.append("--dry_run")
    return cmd

print("Setup complete | Notebook:", NOTEBOOK_ID, "| Jobs:", JOBS, "| Seeds:", SEEDS)
'''

VERIFY_CELL = r'''
# 3. VERIFY: GPU, dataset integrity, strict Teacher checkpoint compatibility guard, and dry-run commands.
import importlib, torch
assert torch.cuda.is_available(), "Select a GPU runtime in Colab first"
print("Torch:", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
for module in ("icecream", "cv2", "einops", "medpy", "h5py", "timm", "scipy", "PIL"):
    importlib.import_module(module)

from experiments.cba_v2.models import verify_checkpoint_compatibility
from experiments.cba_v2.data import ACDCSliceDataset, PolypDataset, find_polyp_split_dir
import main_common as mc

for ds_name in NEEDED_DATASETS:
    if ds_name == "acdc":
        info = verify_checkpoint_compatibility(work / "teacher_weights" / "acdc_tea.pth", dataset="acdc", role="teacher")
        assert info["mask_tokens_shape"] == (4, 256), f"ACDC teacher shape mismatch: {info}"
        ds_tr = ACDCSliceDataset(data_dir="data/ACDC/train", list_path="data/ACDC/lists_ACDC/train.txt", iad_root=WORK, augment=False)
        assert len(ds_tr) == 1304, f"Expected 1304 ACDC train slices, found {len(ds_tr)}"
        s0 = ds_tr[0]
        assert tuple(s0["image"].shape) == (3, 256, 256) and tuple(s0["label"].shape) == (256, 256)
        print(f"[Verified ACDC] Teacher={info['mask_tokens_shape']}, train_slices={len(ds_tr)}, test_vols={len(list((work / 'data/ACDC/test').glob('*.npz')))}")
    elif ds_name == "polyp":
        info = verify_checkpoint_compatibility(work / "teacher_weights" / "polyp_tea.pth", dataset="polyp", role="teacher")
        assert info["mask_tokens_shape"] == (1, 256), f"Polyp teacher shape mismatch: {info}"
        ds_tr = PolypDataset(data_dir="data/dataset/TrainDataset", iad_root=WORK, split="train", augment=False)
        assert len(ds_tr) == 1450, f"Expected 1450 Polyp train images, found {len(ds_tr)}"
        s0 = ds_tr[0]
        assert tuple(s0["image"].shape) == (3, 256, 256) and tuple(s0["label"].shape) == (256, 256)
        sub_counts = {
            sub: len(PolypDataset(data_dir="data/dataset", iad_root=WORK, split="test", subdataset=sub, augment=False))
            for sub in ("Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "CVC-300")
        }
        assert sub_counts == {"Kvasir": 100, "CVC-ClinicDB": 62, "CVC-ColonDB": 380, "ETIS-LaribPolypDB": 196, "CVC-300": 60}, sub_counts
        print(f"[Verified Polyp] Teacher={info['mask_tokens_shape']}, train_images={len(ds_tr)}, test_sets={sub_counts}")

for ds_name, method in JOBS:
    for seed in SEEDS:
        run_command(command_for(ds_name, method, seed, dry_run=True))
'''

RUN_CELL = r'''
# 4. RUN: execute all assigned (dataset, method) jobs for this parallel session.
# Completed runs (eval_final.json present locally or on Drive) are automatically skipped;
# interrupted runs resume from their latest checkpoint.
for ds_name, method in JOBS:
    r_dir = runs_dir_for(ds_name)
    d_dir = drive_dir_for(ds_name)
    for seed in SEEDS:
        if method == "ours" and d_dir:
            _job = mc.job_for("ours", "v5", mc.rt.DEFAULT_STATE, EPOCHS, seed=seed)
            if not mc.rt.is_done(_job, r_dir, d_dir) and not mc.rt.has_own_checkpoint(_job, r_dir, d_dir):
                local_wck = Path(r_dir) / mc.seed_name("warmup", seed) / "Arm_V1_KIS" / "checkpoint_epoch_020.pth"
                drive_wck = mc.find_warmup_ckpt(r_dir, d_dir, seed)
                if drive_wck and not local_wck.is_file():
                    local_wck.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(drive_wck, local_wck)
                    print(f"Staged existing {ds_name} Drive warmup for seed {seed}: {local_wck}", flush=True)
        print(f"\n{'='*78}\n>>> STARTING DATASET={ds_name.upper()} | METHOD={method.upper()} | SEED={seed}\n{'='*78}", flush=True)
        run_command(command_for(ds_name, method, seed, dry_run=False))
'''

STATUS_CELL = r'''
# 5. STATUS & BENCHMARK SUMMARY: inspect all jobs in this notebook (plus any completed jobs on Drive).
import csv, json

ALL_METHODS = ["ours", "iad", "dist_plus", "lskd", "sinkd"]
for ds_name in ("acdc", "polyp"):
    r_dir = Path(runs_dir_for(ds_name))
    d_dir = Path(drive_dir_for(ds_name)) if drive_dir_for(ds_name) else None
    print(f"\n{'='*78}\n DATASET: {ds_name.upper()}\n{'='*78}")
    for method in ALL_METHODS:
        base_name = "ours_v5" if method == "ours" else mc.KIND_DIRS[method]
        arm_name = "Arm_V5_Dual_CEHard" if method == "ours" else mc.rt.ARM_DIRS[method]
        for seed in SEEDS:
            name = mc.seed_name(base_name, seed)
            local = r_dir / name / arm_name
            mirror = (d_dir / name) if d_dir else local
            directory = local if (local / "metrics.csv").exists() else mirror
            metric_path = directory / "metrics.csv"
            if metric_path.is_file():
                with metric_path.open(encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                if rows:
                    last = rows[-1]
                    print(f"[{ds_name}:{method}:{name}] epoch={last.get('epoch')}/{EPOCHS} stage={last.get('stage', arm_name)} loss={last.get('loss_total')} sec={last.get('time_sec')}")
            final = directory / "eval_final.json"
            if final.is_file():
                ev = json.loads(final.read_text(encoding="utf-8"))
                ov = ev["overall"]
                dr = float(ov["primary_dice_raw_macro"]) * 100.0
                jr = float(ov.get("primary_jc_raw_macro", 0.0)) * 100.0
                hr = float(ov["primary_hd95_raw_valid_macro"])
                nr = float(ov["primary_nsd_raw_macro"]) * 100.0
                dl = float(ov.get("primary_dice_lcc_macro", ov["primary_dice_raw_macro"])) * 100.0
                hl = float(ov.get("primary_hd95_lcc_valid_macro", ov["primary_hd95_raw_valid_macro"]))
                if ds_name == "acdc":
                    pc = ev.get("per_class", {})
                    rv_rec = pc.get("RV") or pc.get("rv") or {}
                    myo_rec = pc.get("Myo") or pc.get("myo") or {}
                    lv_rec = pc.get("LV") or pc.get("lv") or {}
                    rv = float(rv_rec.get("dice_clean_raw", 0.0)) * 100.0
                    myo = float(myo_rec.get("dice_clean_raw", 0.0)) * 100.0
                    lv = float(lv_rec.get("dice_clean_raw", 0.0)) * 100.0
                    print(f"  -> ACDC FINAL ({method} s{seed}): Mean DSC={dr:.2f}% (LCC {dl:.2f}%) | JC={jr:.2f}% | HD95={hr:.2f} (LCC {hl:.2f}) | NSD={nr:.2f}% | [RV={rv:.2f}%, Myo={myo:.2f}%, LV={lv:.2f}%]")
                else:
                    ps = ev.get("per_dataset") or ev.get("per_subdataset") or ev.get("per_class") or {}
                    sub_str = ", ".join(f"{k}={float(v.get('dice_clean_raw', 0.0))*100:.2f}%" for k, v in ps.items())
                    d5 = float(ov.get("polyp_5sets_dice_raw_macro", ov.get("all5_dice_raw_macro", ov["primary_dice_raw_macro"]))) * 100.0
                    print(f"  -> POLYP FINAL ({method} s{seed}): 4-Set DSC={dr:.2f}% | 5-Set DSC={d5:.2f}% | JC={jr:.2f}% | HD95={hr:.2f} | NSD={nr:.2f}% | [{sub_str}]")
'''

DOWNLOAD_CELL = r'''
# 6. DOWNLOAD: package evaluation JSONs, logs, final_student.pth, and final_preds for each job in this notebook.
from google.colab import files
for ds_name, method in JOBS:
    r_dir = Path(runs_dir_for(ds_name))
    base_name = "ours_v5" if method == "ours" else mc.KIND_DIRS[method]
    for seed in SEEDS:
        name = mc.seed_name(base_name, seed)
        directory = r_dir / name
        if directory.is_dir():
            archive = f"/content/results_{ds_name}_{name}.zip"
            has_final_student = any(directory.rglob("final_student.pth"))
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
                for item in sorted(directory.rglob("*")):
                    if not item.is_file():
                        continue
                    if has_final_student and (item.name.startswith("checkpoint_epoch_") or item.name == "latest_checkpoint.pth"):
                        continue
                    zf.write(item, item.relative_to(r_dir).as_posix())
            files.download(archive)
        else:
            print(f"Skipping {ds_name}/{name}: local directory {directory} not found")
'''


def build_parallel_notebook(nb_key: str) -> dict:
    spec = NOTEBOOK_SPECS[nb_key]
    jobs_repr = repr(spec["jobs"])
    intro = f"""# {spec["title"]}

{spec["summary"]}

---

### Strict Dataset & Checkpoint Isolation
| Dataset | Resolution | Output Channels | Teacher Checkpoint (`mask_tokens.weight`) | Local Output Root | Drive Sync Root |
|---|---:|---:|---|---|---|
| **ACDC** | `256×256` | `4` (`BG, RV, Myo, LV`) | `acdc_tea.pth` `(4, 256)` | `/content/IAD/runs/acdc_main` | `/content/drive/MyDrive/IAD/acdc_main` |
| **Polyp** | `256×256` | `1` (binary sigmoid) | `polyp_tea.pth` `(1, 256)` | `/content/IAD/runs/polyp_main` | `/content/drive/MyDrive/IAD/polyp_main` |

- **Zero cross-notebook contention**: Each of the 3 parallel notebooks writes to its own method subdirectories (`ours_v5`, `iad_200_100`, `dist_plus300`, `lskd300`, `sinkd300`) inside `acdc_main` and `polyp_main`.
- **Automatic dataset provisioning**: If `acdc_data.zip` or `polyp_data.zip` is not found in `/content` or `MyDrive/IAD`, Cell 2 automatically downloads the official archives from Google Drive (`gdown`).
"""

    config = f"""# 1. CONFIG
NOTEBOOK_ID = "{nb_key}"
JOBS = {jobs_repr}
SEEDS = [2345]         # Set to [2345, 42, 1234] if running 3 seeds
EPOCHS = 300
INPUT_MODE = "drive"   # "drive" (reads from /content/drive/MyDrive/IAD) or "upload" (reads from /content)
DRIVE_ROOT = "/content/drive/MyDrive/IAD"
WORK = "/content/IAD"
"""

    return {
        "cells": [
            cell("markdown", intro),
            cell("code", config),
            cell("code", SETUP_CELL),
            cell("code", VERIFY_CELL),
            cell("code", RUN_CELL),
            cell("code", STATUS_CELL),
            cell("code", DOWNLOAD_CELL),
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


README_TEXT = """# ACDC & Polyp 5-Method Benchmark — 3 Parallel Colab Notebooks

This directory contains **3 parallel Colab notebooks** that run all **5 Knowledge Distillation methods** (`Ours`, `IAD`, `DIST+`, `LSKD`, `SinKD`) across both **ACDC** and **Polyp** (`10` total runs = `5 methods × 2 datasets`) as fast as possible in 3 parallel Colab sessions:

| Parallel Session | Notebook File | Assigned `(dataset, method)` Runs |
|---|---|---|
| **Notebook 1** | `run_acdc_polyp_parallel_nb1_colab.ipynb` | `("acdc", "ours")`, `("polyp", "ours")`, `("acdc", "iad")` |
| **Notebook 2** | `run_acdc_polyp_parallel_nb2_colab.ipynb` | `("polyp", "iad")`, `("acdc", "dist_plus")`, `("polyp", "dist_plus")` |
| **Notebook 3** | `run_acdc_polyp_parallel_nb3_colab.ipynb` | `("acdc", "lskd")`, `("polyp", "lskd")`, `("acdc", "sinkd")`, `("polyp", "sinkd")` |

---

## Strict Dataset & Checkpoint Guards

To guarantee that **Synapse**, **ACDC**, and **Polyp** datasets or checkpoints can never be mixed up:

1. **Checkpoint Shape & Role Guard (`verify_checkpoint_compatibility`)**:
   - **Synapse**: `mask_tokens.weight` shape `(9, 256)` (`epoch_299.pth`)
   - **ACDC**: `mask_tokens.weight` shape `(4, 256)` (`acdc_tea.pth`, `sha256=b30bfb0e7ed48004...`)
   - **Polyp**: `mask_tokens.weight` shape `(1, 256)` (`polyp_tea.pth`, `sha256=a56c5f8b57d240bb...`)
   - Loading a checkpoint from the wrong dataset or wrong role (`teacher` vs `student`) raises a `ValueError` immediately before any training or evaluation step.

2. **Isolated Output & Google Drive Directories**:
   - **Synapse**: `runs/main/` and `MyDrive/IAD/main/`
   - **ACDC**: `runs/acdc_main/` and `MyDrive/IAD/acdc_main/`
   - **Polyp**: `runs/polyp_main/` and `MyDrive/IAD/polyp_main/`

---

## How to Run on Google Colab (3 Parallel Sessions)

1. **Ensure files are on Google Drive (`MyDrive/IAD/`)** (already synced if Google Drive Desktop `I:\\My Drive\\IAD` is mounted):
   - `iad_cba_v2_code.zip`
   - `acdc_tea.pth` (or inside `MyDrive/IAD/teacher_weights/acdc_tea.pth`)
   - `polyp_tea.pth` (or inside `MyDrive/IAD/teacher_weights/polyp_tea.pth`)
   - `acdc_data.zip` and `polyp_data.zip` (optional — if missing, Cell 2 automatically downloads them from the official Google Drive links via `gdown`).
2. **Open 3 Colab tabs** with GPU runtime (`A100` / `L4` / `T4`), upload one notebook per tab (`nb1`, `nb2`, `nb3`), and run all cells (`1. CONFIG` -> `2. SETUP` -> `3. VERIFY` -> `4. RUN` -> `5. STATUS` -> `6. DOWNLOAD`).
"""


def main() -> None:
    UPLOAD.mkdir(exist_ok=True)
    sub_upload = UPLOAD / "acdc_polyp_3notebooks"
    sub_upload.mkdir(parents=True, exist_ok=True)
    (ROOT / "notebooks").mkdir(exist_ok=True)

    for nb_key, spec in NOTEBOOK_SPECS.items():
        nb = build_parallel_notebook(nb_key)
        nb_name = spec["notebook"]
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code":
                compile("".join(c["source"]), f"{nb_name}:cell{i}", "exec")
        nb_path = ROOT / "notebooks" / nb_name
        nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        shutil.copy2(nb_path, UPLOAD / nb_name)
        shutil.copy2(nb_path, sub_upload / nb_name)
        print(f"Wrote parallel notebook: {nb_path}")

    readme_path = UPLOAD / "ACDC_POLYP_README.md"
    readme_path.write_text(README_TEXT, encoding="utf-8")
    shutil.copy2(readme_path, sub_upload / readme_path.name)
    print(f"Wrote {readme_path}")

    # Rebuild iad_cba_v2_code.zip with all targets in zip_code.CBA_V2_TARGETS
    entries = {}
    for rel in zip_code.CBA_V2_TARGETS:
        base_dir, target = zip_code.resolve_bundle_target(rel, ROOT)
        if target is None or not target.exists():
            if rel.startswith("tests/test_"):
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
    shutil.copy2(archive, sub_upload / archive.name)
    for extra_bundle_dir in (
        ROOT / "colab_upload",
        UPLOAD / "main_5methods_3seeds",
        ROOT / "colab_upload" / "main_5methods_3seeds",
        ROOT / "colab_upload" / "acdc_polyp_3notebooks",
    ):
        if extra_bundle_dir.is_dir():
            shutil.copy2(archive, extra_bundle_dir / archive.name)
    for t_ckpt in ("acdc_tea.pth", "polyp_tea.pth"):
        src_ckpt = ROOT / "teacher_weights" / t_ckpt
        if src_ckpt.is_file():
            shutil.copy2(src_ckpt, UPLOAD / t_ckpt)
            shutil.copy2(src_ckpt, sub_upload / t_ckpt)

    print(f"Prepared {len(entries)} ZIP entries; sha256={hashlib.sha256(archive.read_bytes()).hexdigest()}")

    for drive_dir in (Path(r"I:\My Drive\IAD"), Path(r"J:\My Drive\IAD"), Path(r"H:\My Drive\IAD")):
        try:
            if not Path(f"{drive_dir.drive}\\").exists():
                continue
            drive_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(archive, drive_dir / archive.name)
            shutil.copy2(readme_path, drive_dir / readme_path.name)
            drive_tea_dir = drive_dir / "teacher_weights"
            drive_tea_dir.mkdir(parents=True, exist_ok=True)
            for t_ckpt in ("acdc_tea.pth", "polyp_tea.pth", "acdc_stu.pth", "polyp_stu.pth", "acdc_stu_iad.pth", "polyp_stu_iad.pth"):
                src_ckpt = ROOT / "teacher_weights" / t_ckpt
                if src_ckpt.is_file():
                    shutil.copy2(src_ckpt, drive_tea_dir / t_ckpt)
                    if t_ckpt in ("acdc_tea.pth", "polyp_tea.pth"):
                        shutil.copy2(src_ckpt, drive_dir / t_ckpt)
            for d_zip in ("acdc_data.zip", "polyp_data.zip"):
                src_zip = ROOT / "data" / d_zip
                dst_zip = drive_dir / d_zip
                if src_zip.is_file() and (not dst_zip.is_file() or dst_zip.stat().st_size != src_zip.stat().st_size):
                    shutil.copy2(src_zip, dst_zip)
            drive_sub = drive_dir / "acdc_polyp_3notebooks"
            drive_sub.mkdir(parents=True, exist_ok=True)
            shutil.copy2(archive, drive_sub / archive.name)
            shutil.copy2(readme_path, drive_sub / readme_path.name)
            for spec in NOTEBOOK_SPECS.values():
                nb_name = spec["notebook"]
                shutil.copy2(UPLOAD / nb_name, drive_dir / nb_name)
                shutil.copy2(UPLOAD / nb_name, drive_sub / nb_name)
            print(f"Synced code bundle, checkpoints, datasets, and 3 parallel notebooks to {drive_dir}")
        except Exception as exc:
            print(f"[WARN] Could not sync to {drive_dir}: {exc}")


if __name__ == "__main__":
    main()
