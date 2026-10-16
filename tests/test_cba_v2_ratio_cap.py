"""Tests for the optional gradient-ratio cap on lambda_A (--lambda_ratio_cap).

The cap bounds lambda_A * ||grad_logits(L_CBA)|| / ||grad_logits(L_base)|| per microbatch. It is off by default,
so every existing run is unchanged.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import main_common as mc  # noqa: E402
from experiments.cba_v2.train import build_parser, capped_lambda, run_pilot  # noqa: E402


@pytest.fixture
def run_dir():
    d = REPO_ROOT / "runs" / "test_tmp_ratio_cap" / next(tempfile._get_candidate_names())
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)
    try:
        d.parent.rmdir()
    except OSError:
        pass


def test_cap_is_off_by_default():
    assert build_parser().parse_args([]).lambda_ratio_cap == 0.0


def test_capped_lambda_off_returns_curriculum_value():
    assert capped_lambda(0.02, 0.0, 1.0, 10.0) == 0.02


def test_capped_lambda_binds_when_ratio_would_exceed_cap():
    lam = capped_lambda(0.02, 0.1, norm_base=0.01, norm_cba=1.0)
    assert lam == pytest.approx(0.1 * 0.01 / 1.0)
    assert lam * 1.0 / 0.01 == pytest.approx(0.1)  # resulting ratio equals the cap


def test_capped_lambda_never_raises_lambda_above_curriculum():
    assert capped_lambda(0.0033, 0.1, norm_base=1.0, norm_cba=0.1) == 0.0033


def test_capped_lambda_degenerate_norms_fall_back_to_curriculum():
    assert capped_lambda(0.02, 0.1, norm_base=1.0, norm_cba=0.0) == 0.02
    assert capped_lambda(0.02, 0.1, norm_base=0.0, norm_cba=1.0) == 0.0  # no base gradient -> no CBA push


def _smoke_args(out_dir, extra=()):
    argv = ["--arm", "v5", "--epochs", "2", "--smoke_test", "--cpu", "--batch_size", "4",
            "--grad_accum_steps", "2", "--curriculum_warmup", "1", "--curriculum_ramp", "1",
            "--output_dir", str(out_dir), "--warmup_iters", "2"] + list(extra)
    return build_parser().parse_args(argv)


def test_training_applies_the_cap_and_records_effective_lambda(run_dir):
    cap = 1e-4
    res = run_pilot(_smoke_args(run_dir / "cap", ["--lambda_ratio_cap", str(cap)]))
    h = res["history"][1]
    assert 0.0 < h["lambda_eff"] < h["lambda_a"]
    assert 0.0 < h["ratio_cba_mean"] <= cap * (1 + 1e-3)


def test_default_run_records_unchanged_lambda(run_dir):
    res = run_pilot(_smoke_args(run_dir / "nocap"))
    h = res["history"][1]
    assert h["lambda_eff"] == pytest.approx(h["lambda_a"])


def test_main_command_without_cap_is_unchanged(run_dir):
    cmd = mc.run_main("ours", ["--runs_dir", str(run_dir / "r"), "--dry_run"])
    assert "--lambda_ratio_cap" not in cmd
    assert cmd[cmd.index("--output_dir") + 1].endswith("ours_v5")


def test_main_command_with_cap_uses_separate_folder_and_flag(run_dir):
    cmd = mc.run_main("ours", ["--runs_dir", str(run_dir / "r"), "--drive_dir", str(run_dir / "d"),
                               "--lambda_ratio_cap", "0.1", "--dry_run"])
    args = build_parser().parse_args(cmd[2:])
    assert args.lambda_ratio_cap == pytest.approx(0.1)
    assert args.output_dir.endswith("ours_v5cap") and args.drive_sync_dir.endswith("ours_v5cap")
