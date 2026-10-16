"""Tests for scripts/score_tuning.py, scripts/calibrate_lambda.py and the per-unit evaluation records."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import calibrate_lambda  # noqa: E402
import score_tuning as st  # noqa: E402
from experiments.cba_v2.evaluation import evaluate_volumes  # noqa: E402
from experiments.cba_v2.train import build_parser, run_pilot  # noqa: E402


@pytest.fixture
def work_dir():
    d = REPO_ROOT / "runs" / "test_tmp_cba_v2_scripts" / next(tempfile._get_candidate_names())
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)
    try:
        d.parent.rmdir()
    except OSError:
        pass


# ----------------------------------------------------------------------------- per-unit records

def _noisy_volumes(root, n_cases=2):
    import h5py

    vol_dir = root / "vols"
    vol_dir.mkdir()
    names = []
    for ci in range(n_cases):
        label = np.zeros((3, 512, 512), np.uint8)
        label[:, 100:200, 100:220] = 5
        label[:, 300:340, 300:360] = 1
        label[:, 250:270, 50:90] = 8
        name = f"case{ci + 1:04d}"
        with h5py.File(vol_dir / f"{name}.npy.h5", "w") as f:
            f.create_dataset("image", data=(label / 8.0).astype(np.float32))
            f.create_dataset("label", data=label)
        names.append(name)
    lst = root / "list.txt"
    lst.write_text("\n".join(names) + "\n", encoding="utf-8")
    return vol_dir, lst


class _NoisyOracle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = torch.Generator().manual_seed(0)

    def forward(self, x, multimask=True):
        ks = torch.arange(9, dtype=x.dtype, device=x.device).view(1, 9, 1, 1)
        noise = torch.randn(x.shape[0], 9, x.shape[2], x.shape[3], generator=self.gen) * 4.0
        return -((x[:, :1] * 8.0 - ks) ** 2) + noise


def test_units_in_eval_output_reproduce_overall_macros(work_dir):
    vol_dir, lst = _noisy_volumes(work_dir)
    res = evaluate_volumes(_NoisyOracle(), test_dir=str(vol_dir), test_list=str(lst),
                           device=torch.device("cpu"), batch_size=3, use_amp=False, tag="t")
    assert len(res["units"]) == 2 * 8
    json.loads(json.dumps(res["units"]))  # plain python types
    ov = res["overall"]
    for variant, dice_key, nsd_key, hd_key in (
        ("raw", "primary_dice_raw_macro", "primary_nsd_raw_macro", "primary_hd95_raw_valid_macro"),
        ("lcc", "primary_dice_lcc_macro", "primary_nsd_lcc_macro", "primary_hd95_lcc_valid_macro"),
    ):
        units = st.load_units(res, variant)
        assert st.macro(units, "dice") == pytest.approx(ov[dice_key], abs=1e-9)
        assert st.macro(units, "nsd") == pytest.approx(ov[nsd_key], abs=1e-9)
        assert st.macro(units, "hd95") == pytest.approx(ov[hd_key], abs=1e-6)


def test_load_units_requires_units_key():
    with pytest.raises(KeyError):
        st.load_units({"overall": {}, "cases": []})


# ----------------------------------------------------------------------------- scorer

ORGANS = ["spleen", "liver", "aorta"]


def make_units(cases, dice=0.8, hd95=10.0, d_shift=0.0, h_shift=0.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for c in cases:
        for o in ORGANS:
            out[(c, o)] = {
                "dice": dice + d_shift + noise * rng.normal(),
                "nsd": 0.8,
                "hd95": hd95 + h_shift + 5.0 * noise * rng.normal(),
            }
    return out


CASES = [f"case{i:04d}" for i in range(1, 7)]


def test_bootstrap_diff_recovers_constant_shift():
    a = make_units(CASES, h_shift=-2.0)
    b = make_units(CASES)
    r = st.bootstrap_diff(a, b, "hd95", n_boot=200, seed=1)
    assert r["diff"] == pytest.approx(-2.0) and r["ci95"][0] == pytest.approx(-2.0)
    assert r["ci95"][1] == pytest.approx(-2.0)
    assert st.bootstrap_diff(a, b, "dice", n_boot=50)["diff"] == pytest.approx(0.0)


def test_bootstrap_ignores_units_with_nan_in_either_run():
    a = make_units(CASES)
    b = make_units(CASES)
    a[("case0001", "liver")]["hd95"] = float("nan")
    r = st.bootstrap_diff(a, b, "hd95", n_boot=20)
    assert r["n_units"] == len(CASES) * len(ORGANS) - 1


def test_select_applies_feasibility_and_noise_rule():
    ref = make_units(CASES)
    default = make_units(CASES, h_shift=-1.0)
    clear_win = make_units(CASES, h_shift=-3.0)                     # consistent HD95 gain, Dice unchanged
    dice_drop = make_units(CASES, h_shift=-8.0, d_shift=-0.02)      # biggest HD95 gain but Dice -2 pts
    res = st.select(ref, "default", {"default": default, "clear": clear_win, "drop": dice_drop},
                    dice_tol=0.003, n_boot=200, seed=0)
    assert res["rows"]["drop"]["feasible"] is False
    assert res["best_by_point_estimate"] == "clear"
    assert res["decision"] == "clear"

    # Better on average (-4 vs the default's -1) but the gain comes from alternating cases: large spread.
    noisy = make_units(CASES)
    for i, c in enumerate(CASES):
        for o in ORGANS:
            noisy[(c, o)]["hd95"] += -12.0 if i % 2 == 0 else 4.0
    res2 = st.select(ref, "default", {"default": default, "noisy": noisy}, dice_tol=0.5, n_boot=300, seed=0)
    assert res2["best_by_point_estimate"] == "noisy"
    assert res2["decision"] == "default" and "noise" in res2["reason"]


def test_select_keeps_default_when_nothing_feasible():
    ref = make_units(CASES)
    worse = make_units(CASES, d_shift=-0.05)
    res = st.select(ref, "d", {"d": worse}, dice_tol=0.003, n_boot=20, seed=0)
    assert res["decision"] == "d" and res["best_by_point_estimate"] is None


def test_average_units_and_last_eval_files(work_dir):
    a = {("c", "o"): {"dice": 0.8, "nsd": 0.7, "hd95": float("nan")}}
    b = {("c", "o"): {"dice": 0.6, "nsd": 0.9, "hd95": 4.0}}
    avg = st.average_units([a, b])[("c", "o")]
    assert avg["dice"] == pytest.approx(0.7) and avg["hd95"] == pytest.approx(4.0)
    with pytest.raises(ValueError):
        st.average_units([a, {("c", "x"): b[("c", "o")]}])

    for e in (5, 100, 25):
        (work_dir / f"eval_ep{e:03d}.json").write_text("{}", encoding="utf-8")
    assert [Path(p).name for p in st.last_eval_files(str(work_dir), 2)] == ["eval_ep025.json", "eval_ep100.json"]
    with pytest.raises(FileNotFoundError):
        st.last_eval_files(str(work_dir), 4)


# ----------------------------------------------------------------------------- calibrate_lambda

def test_calibrate_lambda_on_smoke_checkpoint(work_dir):
    argv = ["--arm", "v5", "--epochs", "2", "--smoke_test", "--cpu", "--batch_size", "4", "--grad_accum_steps", "2",
            "--curriculum_warmup", "1", "--curriculum_ramp", "1", "--output_dir", str(work_dir / "run"),
            "--warmup_iters", "2"]
    run_pilot(build_parser().parse_args(argv))
    ckpt = work_dir / "run" / "Arm_V5_Dual_CEHard" / "latest_checkpoint.pth"
    out = work_dir / "calib.json"
    res = calibrate_lambda.main(["--arm", "v5", "--smoke_test", "--cpu", "--batch_size", "4", "--batches", "4",
                                 "--ckpt", str(ckpt), "--out", str(out), "--targets", "0.05,0.1"])
    assert res["batches"] == 4 and json.load(open(out, encoding="utf-8"))["arm"] == "Arm_V5_Dual_CEHard"
    assert res["ckpt_epoch"] == 2
    assert res["rho_b"]["median"] > 0
    for t, lam in res["lambda_for_target_ratio_b"].items():
        assert lam * res["rho_b"]["median"] == pytest.approx(float(t), rel=1e-9)
    assert res["ratio_b_at_current_lambda"] == pytest.approx(0.02 * res["rho_b"]["median"])


def test_calibrate_rejects_kis_only_arm(work_dir):
    with pytest.raises(ValueError):
        calibrate_lambda.main(["--arm", "v1", "--smoke_test", "--cpu", "--ckpt", "x.pth"])


def test_lambdas_for_targets_handles_zero_gradient():
    assert calibrate_lambda.lambdas_for_targets(0.0, [0.1]) == {"0.1": None}
