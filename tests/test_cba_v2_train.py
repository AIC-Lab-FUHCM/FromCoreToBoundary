"""Phase 2 tests for experiments/cba_v2/train.py: eval-key handling (A1), mandatory teacher (A2),
per-epoch metrics + exact resume (A3), CLI booleans, and CBA skipped while lambda_A == 0."""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.cba_v2 import models as v2_models
from experiments.cba_v2.evaluation import evaluate_volumes
from experiments.cba_v2.train import build_parser, run_pilot, summarize_eval


@pytest.fixture
def run_dir():
    # pytest tmp_path is not writable in this environment; use an ignored scratch dir under runs/.
    d = REPO_ROOT / "runs" / "test_tmp_cba_v2_train" / next(tempfile._get_candidate_names())
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)
    try:
        d.parent.rmdir()
    except OSError:
        pass


def smoke_args(out_dir, arm="v2", epochs=2, extra=()):
    argv = [
        "--arm", arm, "--epochs", str(epochs), "--smoke_test", "--cpu",
        "--batch_size", "4", "--grad_accum_steps", "2",
        "--curriculum_warmup", "1", "--curriculum_ramp", "1",
        "--output_dir", str(out_dir), "--warmup_iters", "2",
    ] + list(extra)
    return build_parser().parse_args(argv)


# ----------------------------------------------------------------------------- CLI

def test_boolean_flags_can_be_turned_off():
    p = build_parser()
    d = p.parse_args([])
    assert d.drop_last is True and d.use_curriculum is True and d.save_final_preds is True
    assert p.parse_args(["--no-drop_last"]).drop_last is False
    assert p.parse_args(["--no-use_curriculum"]).use_curriculum is False
    assert p.parse_args(["--no_curriculum"]).use_curriculum is False


# ----------------------------------------------------------------------------- A1

class _OracleStub(nn.Module):
    """The image encodes class id / 8, so argmax recovers the label."""

    def forward(self, x, multimask=True):
        ks = torch.arange(9, dtype=x.dtype, device=x.device).view(1, 9, 1, 1)
        return -((x[:, :1] * 8.0 - ks) ** 2)


def test_summarize_eval_reads_real_evaluate_volumes_output():
    root = tempfile.mkdtemp(prefix="cba_v2_eval_")
    try:
        vol_dir = os.path.join(root, "test_vol_h5")
        os.makedirs(vol_dir)
        label = np.zeros((4, 512, 512), dtype=np.uint8)
        label[:, 100:200, 100:220] = 5
        label[:, 300:340, 300:360] = 1
        with h5py.File(os.path.join(vol_dir, "case0001.npy.h5"), "w") as f:
            f.create_dataset("image", data=(label / 8.0).astype(np.float32))
            f.create_dataset("label", data=label)
        list_file = os.path.join(root, "test_vol.txt")
        with open(list_file, "w", encoding="utf-8") as fh:
            fh.write("case0001\n")

        res = evaluate_volumes(
            _OracleStub(), test_dir=vol_dir, test_list=list_file, device=torch.device("cpu"),
            batch_size=2, use_amp=False, max_cases=1, tag="t",
        )
        s = summarize_eval(res)
        assert s["eval_dice_raw"] > 0.9, "Dice must be read from eval_res['overall'], not default to 0"
        assert s["eval_nsd_raw"] > 0.9
        assert s["distance_unit"] == "vox"  # test h5 files carry no spacing
        assert s["eval_n_units"] == 2
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_summarize_eval_raises_on_wrong_schema():
    with pytest.raises(KeyError):
        summarize_eval({"macro_clean_dice": 0.8})


@pytest.mark.parametrize("skip_eval", [False, True])
def test_final_only_evaluation_and_warmup_skip(monkeypatch, run_dir, skip_eval):
    """Exercise the normal training loop: one final eval for main runs, none for warmup."""
    from experiments.cba_v2 import data as data_mod, evaluation as eval_mod

    class TinyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {"image": torch.ones(3, 8, 8), "label": torch.zeros(8, 8, dtype=torch.long)}

    class TinyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Conv2d(3, 9, 1)

        def forward(self, image, multimask=True, **kwargs):
            return (self.head(image),)

    monkeypatch.setattr(v2_models, "build_frozen_teacher", lambda **kwargs: TinyStudent())
    monkeypatch.setattr(v2_models, "build_student", lambda *args, **kwargs: TinyStudent())
    monkeypatch.setattr(data_mod, "SliceDataset", lambda **kwargs: TinyDataset())
    evaluations = []

    def evaluate(**kwargs):
        evaluations.append(kwargs)
        overall = {name: 0.8 for name in (
            "primary_dice_raw_macro", "primary_dice_lcc_macro", "primary_nsd_raw_macro",
            "primary_nsd_lcc_macro", "primary_hd95_raw_valid_macro", "primary_hd95_lcc_valid_macro",
            "primary_recall_macro", "primary_precision_macro", "primary_vol_ratio_macro",
        )}
        overall["primary_n_units"] = 1
        return {"overall": overall, "cases": [], "per_class": {}}

    monkeypatch.setattr(eval_mod, "evaluate_volumes", evaluate)
    argv = ["--arm", "v1", "--cpu", "--epochs", "2", "--batch_size", "2",
            "--grad_accum_steps", "1", "--output_dir", str(run_dir)]
    if skip_eval:
        argv.append("--skip_eval")
    args = build_parser().parse_args(argv)
    assert args.eval_freq == 0
    run_pilot(args)
    arm_dir = run_dir / "Arm_V1_KIS"
    assert not (arm_dir / "eval_ep001.json").exists()
    assert (arm_dir / "eval_final.json").exists() is (not skip_eval)
    assert len(evaluations) == (0 if skip_eval else 1)
    if evaluations:
        assert evaluations[0]["tag"] == "final"
        assert evaluations[0]["save_preds_dir"].endswith("final_preds")
    assert (arm_dir / "latest_checkpoint.pth").exists()
    assert (arm_dir / "checkpoint_epoch_002.pth").exists()


# ----------------------------------------------------------------------------- A2

def test_missing_teacher_aborts_instead_of_ce_only(monkeypatch, run_dir):
    def boom(**kwargs):
        raise RuntimeError("teacher checkpoint missing")

    monkeypatch.setattr(v2_models, "build_frozen_teacher", boom)
    args = build_parser().parse_args(["--arm", "v1", "--epochs", "1", "--cpu", "--output_dir", str(run_dir)])
    with pytest.raises(RuntimeError, match="teacher checkpoint missing"):
        run_pilot(args)
    assert not (run_dir / "Arm_V1_KIS" / "metrics.jsonl").exists()


# ----------------------------------------------------------------------------- A3

def _read_lines(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [ln for ln in fh.read().splitlines() if ln.strip()]


def test_per_epoch_metrics_checkpoint_and_run_config(run_dir):
    run_pilot(smoke_args(run_dir, arm="v2", epochs=2))
    arm_dir = run_dir / "Arm_V2_Boundary"
    assert len(_read_lines(arm_dir / "metrics.jsonl")) == 2
    assert len(_read_lines(arm_dir / "metrics.csv")) == 3  # header + 2 epochs
    ckpt = torch.load(arm_dir / "latest_checkpoint.pth", map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 2
    assert {"student", "ema_student", "optimizer", "scheduler", "rng", "history"} <= set(ckpt)
    assert {"python", "numpy", "torch"} <= set(ckpt["rng"])
    assert (arm_dir / "checkpoint_epoch_002.pth").exists()  # final epoch is always permanent
    assert (arm_dir / "final_student.pth").exists()
    cfg = __import__("json").load(open(arm_dir / "run_config.json", encoding="utf-8"))
    assert cfg["arm"] == "Arm_V2_Boundary" and "git" in cfg and cfg["args"]["epochs"] == 2
    assert not list(arm_dir.glob("best_checkpoint*")), "no best-on-test checkpoint may be written"


def test_cba_skipped_while_lambda_is_zero(run_dir):
    res = run_pilot(smoke_args(run_dir, arm="v2", epochs=2))
    h = res["history"]
    assert h[0]["lambda_a"] == 0.0 and h[0]["loss_cba"] == 0.0 and h[0]["cache_valid_classes"] == 0
    assert h[1]["lambda_a"] > 0.0 and h[1]["loss_cba"] != 0.0


def test_resume_after_crash_matches_uninterrupted_run(monkeypatch, run_dir):
    import experiments.cba_v2.train as train_mod

    full = run_pilot(smoke_args(run_dir / "full", arm="v2", epochs=4))
    full_state = torch.load(run_dir / "full" / "Arm_V2_Boundary" / "final_student.pth",
                            map_location="cpu", weights_only=False)["student"]

    # Simulate a Colab disconnect: crash while writing epoch 3's metrics (epoch 2 is checkpointed).
    real_write = train_mod.write_metrics_files
    calls = {"n": 0}

    def crashing_write(arm_dir, history):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt("simulated disconnect")
        real_write(arm_dir, history)

    monkeypatch.setattr(train_mod, "write_metrics_files", crashing_write)
    with pytest.raises(KeyboardInterrupt):
        run_pilot(smoke_args(run_dir / "crash", arm="v2", epochs=4))
    monkeypatch.setattr(train_mod, "write_metrics_files", real_write)

    ckpt = torch.load(run_dir / "crash" / "Arm_V2_Boundary" / "latest_checkpoint.pth",
                      map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 2

    resumed = run_pilot(smoke_args(run_dir / "crash", arm="v2", epochs=4, extra=["--resume", "auto"]))
    resumed_state = torch.load(run_dir / "crash" / "Arm_V2_Boundary" / "final_student.pth",
                               map_location="cpu", weights_only=False)["student"]

    assert [r["epoch"] for r in resumed["history"]] == [1, 2, 3, 4]
    for a, b in zip(full["history"], resumed["history"]):
        assert a["loss_total"] == pytest.approx(b["loss_total"], rel=1e-6, abs=1e-8)
    for k in full_state:
        assert torch.equal(full_state[k], resumed_state[k]), f"weights differ after resume at {k}"
    cfg = __import__("json").load(open(run_dir / "crash" / "Arm_V2_Boundary" / "run_config.json", encoding="utf-8"))
    assert [i["resumed_from_epoch"] for i in cfg["invocations"]] == [0, 2]
