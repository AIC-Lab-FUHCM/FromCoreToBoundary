"""Generate notebooks/run_cba_v2_main_colab.ipynb (300-epoch main runs: KIS and ours).

Run:  python scripts/make_main_notebook.py
Generated from this file so that its commands stay in sync with scripts/run_main_kis.py and run_main_ours.py;
tests/test_cba_v2_main_notebook.py checks that every code cell compiles and that every flag it uses exists.
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "notebooks" / "run_cba_v2_main_colab.ipynb"

CELLS = []


def md(text: str):
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str):
    CELLS.append(("code", text.strip("\n")))


md("""
# CBA v2: 300-epoch main runs on Colab (A100)

One notebook, two runs: set `RUN` in cell 1.

| `RUN` | trains | folder on Drive (`IAD/main/`) |
|---|---|---|
| `"kis"` | KIS baseline, 300 epochs | `kis300` |
| `"ours"` | KIS + Boundary + Error-CE-hard (`ARM="v5"`, default) or KIS + Boundary (`ARM="v2"`), 300 epochs | `ours_v5` / `ours_v2` |
| `"ours"` with `RATIO_CAP=0.1` | same, with lambda_A capped so that lambda_A * ‖g_CBA‖ / ‖g_base‖ ≤ 0.1 (logit space) | `ours_v5cap` / `ours_v2cap` |

Both train on all 18 training cases and evaluate on the 12 official test volumes every 25 epochs; the reported model is
the epoch-300 model. Method parameters of `"ours"` come from `IAD/tune/tune_state.json` (defaults when absent).

**Two sessions:** start the `"kis"` session first. It creates the shared 20-epoch checkpoint
(`IAD/main/warmup/`); the `"ours"` session waits up to 240 minutes for it. Use the same `EPOCHS` in both.

**Disconnects:** run cells 1, 2 and 3 again, then the run cell. Finished runs are skipped and interrupted runs resume from the
last checkpoint on Drive.
""")

code("""
# 1. CONFIG
RUN = "kis"            # "kis" or "ours"
ARM = "v5"             # only for RUN == "ours": "v5" (KIS + Boundary + Error-CE-hard) or "v2" (KIS + Boundary)
EPOCHS = 300           # use the same value in both sessions
MAKE_WARMUP = None     # None = script default (kis creates the shared warmup, ours waits); True / False overrides
RATIO_CAP = 0.0        # only for RUN == "ours": > 0 caps lambda_A * ||g_CBA|| / ||g_base|| (e.g. 0.1); writes to ours_<ARM>cap

DRIVE = "/content/drive/MyDrive/IAD"
WORK = "/content/IAD"
RUNS = "/content/runs/main"
MAIN_DRIVE = f"{DRIVE}/main"
ARGS = f"--runs_dir {RUNS} --drive_dir {MAIN_DRIVE} --epochs {EPOCHS}"
if RUN == "ours":
    ARGS += f" --arm {ARM}"
    if RATIO_CAP > 0:
        ARGS += f" --lambda_ratio_cap {RATIO_CAP}"
if MAKE_WARMUP is not None:
    ARGS += " --make_warmup" if MAKE_WARMUP else " --no-make_warmup"
assert RUN in ("kis", "ours"), RUN
print(f"python scripts/run_main_{RUN}.py {ARGS}")
""")

code("""
# 2. SETUP: Drive, code bundle, data, weights, dependencies
import os, shutil, subprocess, sys, zipfile
from google.colab import drive
drive.mount("/content/drive")

def sh(cmd, cwd=None):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd or WORK)
    for line in p.stdout:
        print(line, end="")
    rc = p.wait()
    assert rc == 0, f"command failed ({rc}): {cmd}"

os.makedirs(WORK, exist_ok=True)
if not os.path.isdir(f"{WORK}/data/btcv/train_npz"):
    with zipfile.ZipFile(f"{DRIVE}/btcv_data.zip") as z:      # extracts data/btcv/... and data/lists_Synapse/...
        z.extractall(WORK)
with zipfile.ZipFile(f"{DRIVE}/iad_cba_v2_code.zip") as z:
    z.extractall(WORK)
for rel in ("data/weights/sam_vit_l_0b3195.pth", "teacher_weights/epoch_299.pth"):
    src, dst = f"{DRIVE}/{rel}", f"{WORK}/{rel}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
        shutil.copy2(src, dst)
    print("ok", rel, os.path.getsize(dst))
os.chdir(WORK)
sys.path.insert(0, WORK)
sh("pip install -q -r requirements_colab.txt")
""")

code("""
# 3. VERIFY the environment and the local `datasets` package (Colab ships a different `datasets` library)
import importlib, torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0),
      "| bf16", torch.cuda.is_bf16_supported())
for mod in ("icecream", "cv2", "einops", "medpy", "h5py", "timm", "scipy"):
    importlib.import_module(mod)
import datasets.dataset_synapse as ds
assert ds.__file__.startswith(WORK), f"wrong `datasets` package: {ds.__file__} (restart the runtime and re-run)"
assert os.path.isfile(f"{WORK}/data/lists_Synapse/train.txt") and os.path.isdir(f"{WORK}/data/btcv/test_vol_h5")
# show the exact commands (nothing is trained): warmup (if needed) and the main run
sh(f"python scripts/run_main_{RUN}.py {ARGS} --dry_run")
""")

code("""
# 4. RUN (3D evaluation every 25 epochs; checkpoints sync to Drive every 5 epochs; safe to re-run after a disconnect)
sh(f"python scripts/run_main_{RUN}.py {ARGS}")
""")

code("""
# 5. STATUS (read-only): last logged epochs and the latest evaluation of this run
import csv, glob, json
name = "kis300" if RUN == "kis" else f"ours_{ARM}" + ("cap" if RATIO_CAP > 0 else "")
rows = list(csv.DictReader(open(glob.glob(f"{MAIN_DRIVE}/{name}/metrics.csv")[0])))
print(f"{name}: {len(rows)} epochs logged; last epoch time {float(rows[-1]['time_sec']):.0f}s")
evals = sorted(glob.glob(f"{MAIN_DRIVE}/{name}/eval_ep*.json"))
if evals:
    o = json.load(open(evals[-1]))["overall"]
    print(os.path.basename(evals[-1]), {k: round(v, 4) for k, v in o.items() if k.startswith("primary_") and "raw" in k and isinstance(v, float)})
""")


def build() -> dict:
    cells = []
    for kind, text in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
