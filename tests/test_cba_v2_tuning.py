"""Tests for the tuning infrastructure: validation split script and --sched_epochs."""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from make_val_split import build_split, check_contiguous, parse_slices  # noqa: E402
from experiments.cba_v2.train import build_lr_fn, build_parser, run_pilot  # noqa: E402


@pytest.fixture
def work_dir():
    d = REPO_ROOT / "runs" / "test_tmp_cba_v2_tuning" / next(tempfile._get_candidate_names())
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)
    try:
        d.parent.rmdir()
    except OSError:
        pass


def make_cases(root, cases, n_slices=4, size=32, organ_px=6, skip_slice=None):
    """Write <case>_sliceNNN.npz files; image[i] == i/10, label has organs 1..8 in every slice."""
    npz = root / "npz"
    npz.mkdir(parents=True, exist_ok=True)
    names = []
    for case in cases:
        for i in range(n_slices):
            if skip_slice == (case, i):
                continue
            lab = np.zeros((size, size), np.float32)
            for k in range(1, 9):
                lab[(k - 1) * 3:(k - 1) * 3 + 2, :organ_px] = k
            img = np.full((size, size), i / 10.0, np.float32)
            np.savez(npz / f"{case}_slice{i:03d}.npz", image=img, label=lab)
            names.append(f"{case}_slice{i:03d}")
    lst = root / "train.txt"
    lst.write_text("\n".join(names) + "\n", encoding="utf-8")
    return npz, lst, names


def test_split_produces_disjoint_lists_and_ordered_volumes(work_dir):
    cases = ["case0001", "case0002", "case0003"]
    npz, lst, names = make_cases(work_dir, cases)
    m = build_split(str(npz), str(lst), ["case0002"], str(work_dir / "h5"), str(work_dir / "lists"),
                    min_organ_vox=4)
    train = (work_dir / "lists" / "train_tune.txt").read_text(encoding="utf-8").split()
    val = (work_dir / "lists" / "val_tune_vol.txt").read_text(encoding="utf-8").split()
    assert val == ["case0002"]
    assert train == [n for n in names if not n.startswith("case0002")]
    assert not any(n.startswith("case0002") for n in train)
    with h5py.File(work_dir / "h5" / "case0002.npy.h5", "r") as f:
        image, label = f["image"][:], f["label"][:]
    assert image.shape == (4, 32, 32) and label.dtype == np.uint8 and image.dtype == np.float32
    assert [round(float(image[i, 0, 0]) * 10) for i in range(4)] == [0, 1, 2, 3]  # slice order kept
    assert set(np.unique(label)) == set(range(9))
    assert m["val_cases"]["case0002"]["slices"] == 4
    assert len(m["val_cases"]["case0002"]["sha256"]) == 64
    assert m["train_tune_cases"] == ["case0001", "case0003"]
    assert json.load(open(work_dir / "h5" / "split_manifest.json", encoding="utf-8"))["train_tune_slices"] == 8


def test_split_rejects_bad_inputs(work_dir):
    npz, lst, _ = make_cases(work_dir, ["case0001", "case0002"])
    kw = dict(npz_dir=str(npz), train_list=str(lst), h5_dir=str(work_dir / "h5"), lists_dir=str(work_dir / "l"))
    with pytest.raises(ValueError, match="not in the training list"):
        build_split(val_cases=["case0099"], min_organ_vox=4, **kw)
    with pytest.raises(ValueError, match="Duplicate"):
        build_split(val_cases=["case0001", "case0001"], min_organ_vox=4, **kw)
    with pytest.raises(ValueError, match="fewer than"):
        build_split(val_cases=["case0001"], min_organ_vox=10 ** 6, **kw)


def test_split_rejects_missing_slices(work_dir):
    npz, lst, _ = make_cases(work_dir, ["case0001"], skip_slice=("case0001", 2))
    with pytest.raises(ValueError, match="contiguous"):
        build_split(str(npz), str(lst), ["case0001"], str(work_dir / "h5"), str(work_dir / "l"), min_organ_vox=4)
    with pytest.raises(ValueError):
        parse_slices(["case0001_slice000", "oops"])
    check_contiguous({"case0001": [0, 1, 2]})


def test_split_volume_is_consumable_by_evaluate_volumes(work_dir):
    from experiments.cba_v2.evaluation import evaluate_volumes

    # 512x512 slices are required by the evaluator; image encodes the class so an oracle stub is perfect.
    npz = work_dir / "npz"
    npz.mkdir()
    names = []
    for i in range(3):
        lab = np.zeros((512, 512), np.float32)
        lab[100:200, 100:220] = 5
        lab[300:340, 300:360] = 1
        np.savez(npz / f"case0007_slice{i:03d}.npz", image=(lab / 8.0).astype(np.float32), label=lab)
        names.append(f"case0007_slice{i:03d}")
    lst = work_dir / "train.txt"
    lst.write_text("\n".join(names) + "\n", encoding="utf-8")
    build_split(str(npz), str(lst), ["case0007"], str(work_dir / "h5"), str(work_dir / "l"), min_organ_vox=0)

    class Oracle(torch.nn.Module):
        def forward(self, x, multimask=True):
            ks = torch.arange(9, dtype=x.dtype, device=x.device).view(1, 9, 1, 1)
            return -((x[:, :1] * 8.0 - ks) ** 2)

    res = evaluate_volumes(Oracle(), test_dir=str(work_dir / "h5"), test_list=str(work_dir / "l" / "val_tune_vol.txt"),
                           device=torch.device("cpu"), batch_size=2, use_amp=False, tag="t")
    from experiments.cba_v2.train import summarize_eval
    assert summarize_eval(res)["eval_dice_raw"] > 0.99


def test_real_split_files_if_present():
    h5_dir = REPO_ROOT / "data" / "btcv" / "val_vol_h5"
    mf = h5_dir / "split_manifest.json"
    if not mf.exists():
        pytest.skip("split not generated")
    m = json.load(open(mf, encoding="utf-8"))
    test_cases = {ln.strip() for ln in open(REPO_ROOT / "data" / "lists_Synapse" / "test_vol.txt") if ln.strip()}
    assert not (set(m["val_cases"]) & test_cases), "validation overlaps the official test set"
    assert len(m["val_cases"]) == 4
    train = (REPO_ROOT / "data" / "lists_Synapse" / "train_tune.txt").read_text().split()
    assert not {t.split("_")[0] for t in train} & set(m["val_cases"])


# ----------------------------------------------------------------------------- sched_epochs

def test_lr_fn_warmup_then_polynomial_decay():
    fn = build_lr_fn(total_steps=100, warmup_iters=10, mode="polynomial")
    assert fn(0) == pytest.approx(0.1) and fn(9) == pytest.approx(1.0)
    assert fn(10) > fn(50) > fn(99)
    assert fn(99) == pytest.approx(0.0, abs=1e-9)
    cos = build_lr_fn(100, 10, "cosine")
    assert cos(99) == pytest.approx(0.0, abs=1e-9)


def test_sched_epochs_keeps_lr_high_in_a_short_run(work_dir):
    def lr_after_one_epoch(extra):
        argv = ["--arm", "v1", "--epochs", "1", "--smoke_test", "--cpu", "--batch_size", "4",
                "--grad_accum_steps", "2", "--output_dir", str(work_dir / extra[-1]), "--warmup_iters", "2",
                "--lr", "1e-4"] + extra[:-1]
        return run_pilot(build_parser().parse_args(argv))["history"][0]["lr"]

    full = lr_after_one_epoch(["short"])                         # schedule ends with the run
    part = lr_after_one_epoch(["--sched_epochs", "8", "long"])   # run is the first part of a longer schedule
    assert full < 1e-5
    assert part > 5e-5
