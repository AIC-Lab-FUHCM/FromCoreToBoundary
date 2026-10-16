"""Phase 4 tests: method flags for experiments/cba_v2.

M1 feature space (raw vs centered), M2 gate reference gradient, M3 ce_hard/gate guard (D7),
M4 gradient diagnostics logging.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.cba_v2.losses.cba_v2_loss import CBAv2Loss
from experiments.cba_v2.train import build_cba_v2_loss, build_parser, run_pilot


@pytest.fixture
def run_dir():
    d = REPO_ROOT / "runs" / "test_tmp_cba_v2_phase4" / next(tempfile._get_candidate_names())
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)
    try:
        d.parent.rmdir()
    except OSError:
        pass


def _two_region_labels(h=24, w=24):
    y = torch.zeros(1, h, w, dtype=torch.long)
    y[:, :, : w // 2] = 1
    y[:, :, w // 2:] = 2
    return y


def _boundary_loss(space, gating="none", **kw):
    c = CBAv2Loss(m=0.2, tau=0.1, beta=0.5, num_classes=4, feature_space=space,
                  gating_type=gating, enable_boundary=True, enable_error=False, **kw)
    c.train()
    return c


# ----------------------------------------------------------------------------- M1

def test_feature_space_default_and_validation():
    assert CBAv2Loss(num_classes=4).feature_space == "raw"  # class default keeps legacy behaviour
    assert CBAv2Loss(num_classes=4, feature_space="centered").feature_space == "centered"
    with pytest.raises(ValueError):
        CBAv2Loss(num_classes=4, feature_space="whitened")
    assert build_parser().parse_args([]).cba_feature == "centered"  # runner default (D5)


def test_centered_space_removes_common_mode_raw_does_not():
    y = _two_region_labels()
    torch.manual_seed(1)
    f = torch.randn(1, 4, 24, 24)
    fe = torch.randn(1, 4, 24, 24)
    out = {}
    for space in ("raw", "centered"):
        l0, _ = _boundary_loss(space)(f.clone().requires_grad_(True), y, fplus_ema=fe,
                                      return_stream_losses=True)
        l1, _ = _boundary_loss(space)((f + 3.7).requires_grad_(True), y, fplus_ema=fe + 2.0,
                                      return_stream_losses=True)
        out[space] = (l0.item(), l1.item())
    assert out["centered"][0] == pytest.approx(out["centered"][1], abs=1e-4)
    assert abs(out["raw"][0] - out["raw"][1]) > 1e-2


def test_centered_gradient_is_zero_mean_over_classes():
    y = _two_region_labels()
    torch.manual_seed(2)
    for space, expect_zero in (("centered", True), ("raw", False)):
        f = torch.randn(1, 4, 24, 24, requires_grad=True)
        fe = torch.randn(1, 4, 24, 24)
        lb, _ = _boundary_loss(space)(f, y, fplus_ema=fe, return_stream_losses=True)
        lb.backward()
        resid = f.grad.sum(dim=1).abs().max().item()
        assert (resid < 1e-6) if expect_zero else (resid > 1e-3)


def test_analytic_gradient_matches_autograd():
    # No background; two adjacent classes, pixel_mean reduction: autograd norm = analytic norm / sqrt(N)
    # only if the closed-form per-pixel gradient (incl. projection) equals the autograd one.
    y = _two_region_labels()
    ratios = []
    for sp in ("raw", "centered"):
        torch.manual_seed(1)
        f = torch.randn(1, 4, 24, 24, requires_grad=True)
        fe = torch.randn(1, 4, 24, 24)
        c = _boundary_loss(sp, boundary_reduction="pixel_mean")
        _, _, d = c(f, y, fplus_ema=fe, return_stream_losses=True, return_diagnostics=True,
                    compute_autograd_norms=True)
        ratios.append((d["grad_norm_b_analytical"] / d["grad_norm_b_autograd"],
                       d["unique_boundary_pixels"]))
    assert ratios[0][0] == pytest.approx(ratios[1][0], rel=1e-4)
    n = ratios[0][1]
    assert ratios[0][0] == pytest.approx(n ** 0.5, rel=1e-4)


# ----------------------------------------------------------------------------- M2

def test_external_gate_reference_changes_the_gate():
    y = _two_region_labels()
    torch.manual_seed(3)
    f = torch.randn(1, 4, 24, 24)
    fe = torch.randn(1, 4, 24, 24)
    ref = torch.randn(1, 4, 24, 24)

    def gate_mean(g_ref):
        c = _boundary_loss("centered", gating="soft")
        _, _, d = c(f.clone().requires_grad_(True), y, fplus_ema=fe, g_ref=g_ref,
                    return_stream_losses=True, return_diagnostics=True)
        return d["boundary_gate_mean"]

    g_default = gate_mean(None)
    g_pos = gate_mean(ref)
    g_neg = gate_mean(-ref)
    g_scaled = gate_mean(10.0 * ref)
    assert g_pos == pytest.approx(g_scaled, rel=1e-4)  # cosine gate is scale-invariant
    assert abs(g_pos - g_neg) > 1e-3                    # and depends on the reference direction
    assert abs(g_pos - g_default) > 1e-4                # the external reference replaces the CE reference
    for g in (g_default, g_pos, g_neg):
        assert 0.0 <= g <= 1.0


# ----------------------------------------------------------------------------- M3

def test_ce_hard_requires_ungated_error_stream():
    with pytest.raises(ValueError):
        CBAv2Loss(num_classes=9, enable_boundary=True, enable_error=True, error_mode="ce_hard",
                  error_gating_type="soft")
    CBAv2Loss(num_classes=9, enable_boundary=True, enable_error=True, error_mode="ce_hard",
              error_gating_type="none")
    CBAv2Loss(num_classes=9, enable_boundary=True, enable_error=True, error_mode="contrastive",
              error_gating_type="none")


def test_runner_defaults_follow_decisions_d5_d6_d7():
    args = build_parser().parse_args([])
    assert (args.cba_feature, args.gate_ref, args.error_gating_type) == ("centered", "base", "none")
    v5 = build_cba_v2_loss("Arm_V5_Dual_CEHard", args)
    assert v5.feature_space == "centered" and v5.error_gating_type == "none"
    bad = build_parser().parse_args(["--error_gating_type", "soft"])
    with pytest.raises(ValueError):
        build_cba_v2_loss("Arm_V5_Dual_CEHard", bad)


# ----------------------------------------------------------------------------- M4

def _smoke(out_dir, arm, extra=()):
    argv = [
        "--arm", arm, "--epochs", "2", "--smoke_test", "--cpu", "--batch_size", "4",
        "--grad_accum_steps", "2", "--curriculum_warmup", "1", "--curriculum_ramp", "1",
        "--output_dir", str(out_dir), "--warmup_iters", "2", "--grad_log_every", "1",
    ] + list(extra)
    return run_pilot(build_parser().parse_args(argv))


def test_runner_logs_gradient_diagnostics(run_dir):
    res = _smoke(run_dir, "v5")
    h = res["history"]
    assert "grad_log" not in h[0] and "gl_ratio_b_mean" not in h[0]  # warmup epoch: CBA skipped
    assert h[1]["grad_log"], "no gradient records on an active-CBA epoch"
    for key in ("gl_ratio_b_mean", "gl_ratio_e_mean", "gl_gate_b_mean", "gl_gate_e_mean",
                "gl_h_norm_mean", "gl_conflict_b_macro", "gl_conflict_e_macro"):
        assert key in h[1] and h[1][key] == h[1][key]  # present and not NaN
    rec = h[1]["grad_log"][0]
    assert rec["norm_base"] > 0.0 and rec["ratio_b"] >= 0.0
    lines = (run_dir / "Arm_V5_Dual_CEHard" / "grad_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(h[1]["grad_log"]) and json.loads(lines[0])["epoch"] == 2
    csv_header = (run_dir / "Arm_V5_Dual_CEHard" / "metrics.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "gl_ratio_b_mean" in csv_header and "grad_log" not in csv_header.split(",")


def test_grad_logging_can_be_disabled_and_arm_v2_has_no_error_ratio(run_dir):
    res = _smoke(run_dir / "off", "v2", extra=["--grad_log_every", "0"])
    assert all("grad_log" not in r for r in res["history"])
    res = _smoke(run_dir / "on", "v2")
    assert res["history"][1]["gl_ratio_e_mean"] == 0.0
