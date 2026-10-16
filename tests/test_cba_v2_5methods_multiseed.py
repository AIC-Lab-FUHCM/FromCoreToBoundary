"""Comprehensive verification suite for the 5-method × 3-seed (300-epoch) benchmark.

Covers:
1. Reference loss bug fixes in Synapse_Tool_KD and ACDC_Tool (SinKD, LSKD, DIST+, contrastive_loss).
2. Unified cba_v2 baseline loss modules (LSKDLoss, SinKDLoss, DISTPlusLoss, IADKasHead, BaselineKDLoss)
   including float16/bfloat16 safety, gradient flow, teacher detachment, and B=1 edge cases.
3. End-to-end train.py smoke tests for all 5 methods (Ours v5, IAD 200+100, LSKD, SinKD, DIST+),
   including IAD Stage-1 KIS == Arm_V1_KIS bit-for-bit parity, Stage-1 -> Stage-2 KAS transition,
   and mid-Stage-2 checkpoint resume.
4. Multi-seed launcher scripts (run_main_ours.py, run_main_iad.py, run_main_lskd.py,
   run_main_sinkd.py, run_main_dist_plus.py) and the 5 method-specific 3-seed Colab notebooks.
"""

from __future__ import annotations

import ast
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_ROOT = REPO_ROOT.parent / "colab_upload"
for p in (REPO_ROOT, REPO_ROOT / "models", REPO_ROOT / "scripts", REPO_ROOT / "scripts" / "tools"):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import main_common as mc  # noqa: E402
import prepare_5methods_notebooks as p5n  # noqa: E402
import run_tuning as rt  # noqa: E402
from ACDC_Tool.distillation_loss import DIST as ACDCDIST  # noqa: E402
from ACDC_Tool.distillation_loss import contrastive_loss as acdc_contrastive_loss  # noqa: E402
from experiments.cba_v2.losses import (  # noqa: E402
    BaselineKDLoss,
    DISTPlusLoss,
    IADKasHead,
    LSKDLoss,
    SinKDLoss,
)
from experiments.cba_v2.train import build_parser, main as train_main, resolve_arm_name, run_pilot  # noqa: E402
from Synapse_Tool_KD.criterion.lskd import logit_standardization_kd_loss  # noqa: E402
from Synapse_Tool_KD.criterion.SinKD import pixel_info_nce, sinkhorn_loss  # noqa: E402
from Synapse_Tool_KD.distillation_loss import DIST as SynapseDIST  # noqa: E402
from Synapse_Tool_KD.distillation_loss import contrastive_loss as synapse_contrastive_loss  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Reference Criterion Bug-Fix Tests (Synapse_Tool_KD & ACDC_Tool)
# ---------------------------------------------------------------------------


def test_synapse_sinkd_is_differentiable_and_does_not_underflow():
    torch.manual_seed(0)
    # Use large logit magnitudes (* 25.0) that previously caused exp(-10 * cdist) to underflow to 0.0
    tea = (torch.randn(2, 9, 32, 32) * 25.0).requires_grad_(True)
    stu = (torch.randn(2, 9, 32, 32) * 25.0).requires_grad_(True)

    loss = sinkhorn_loss(tea, stu, lam=10.0, max_iter=5)
    assert loss.requires_grad, "sinkhorn_loss must remain attached to student autograd graph"
    assert torch.isfinite(loss).item() and loss.item() > 0.0

    loss.backward()
    assert stu.grad is not None, "Student logits received no gradient from sinkhorn_loss"
    assert torch.isfinite(stu.grad).all(), "Non-finite student gradient in sinkhorn_loss"
    assert stu.grad.abs().sum().item() > 0.0, "Zero student gradient in sinkhorn_loss (underflow or detached)"
    assert tea.grad is None, "Teacher logits must be detached inside sinkhorn_loss"


def test_synapse_sinkd_pixel_info_nce_stabilized():
    torch.manual_seed(1)
    pre = (torch.randn(2, 16, 8, 8) * 10.0).requires_grad_(True)
    pos = (torch.randn(2, 16, 8, 8) * 10.0).detach()
    neg = (torch.randn(2, 16, 8, 8) * 10.0).detach()
    loss = pixel_info_nce(pre, pos, neg, temperature=0.07)
    assert torch.isfinite(loss).item()
    loss.backward()
    assert pre.grad is not None and torch.isfinite(pre.grad).all() and pre.grad.abs().sum().item() > 0.0


def test_synapse_lskd_detaches_teacher_and_handles_low_precision():
    torch.manual_seed(2)
    tea = torch.randn(2, 9, 16, 16, dtype=torch.bfloat16, requires_grad=True)
    stu = torch.randn(2, 9, 16, 16, dtype=torch.bfloat16, requires_grad=True)
    loss = logit_standardization_kd_loss(stu, tea, tau=2.0)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss).item() and loss.item() > 0.0
    loss.backward()
    assert stu.grad is not None and torch.isfinite(stu.grad).all() and stu.grad.abs().sum().item() > 0.0
    assert tea.grad is None


@pytest.mark.parametrize("dist_cls", [SynapseDIST, ACDCDIST])
def test_reference_dist_casts_to_fp32_and_detaches_teacher(dist_cls):
    torch.manual_seed(3)
    criterion = dist_cls(beta=1.0, gamma=1.0, tau=1.0)
    tea = torch.randn(2, 9, 32, 32, dtype=torch.float16, requires_grad=True)
    stu = torch.randn(2, 9, 32, 32, dtype=torch.float16, requires_grad=True)
    loss = criterion(stu, tea)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss).item() and loss.item() > 0.0
    loss.backward()
    assert stu.grad is not None and torch.isfinite(stu.grad).all() and stu.grad.abs().sum().item() > 0.0
    assert tea.grad is None


@pytest.mark.parametrize("cls", [synapse_contrastive_loss, acdc_contrastive_loss])
def test_reference_contrastive_loss_is_finite_for_extreme_logits(cls):
    torch.manual_seed(4)
    criterion = cls()
    pre = (torch.randn(2, 9, 8, 8) * 15.0).requires_grad_(True)
    pos = (torch.randn(2, 9, 8, 8) * 15.0).detach()
    neg = (torch.randn(2, 9, 8, 8) * 15.0).detach()
    loss = criterion(pre, pos, neg, t=0.05)
    assert torch.isfinite(loss).item()
    loss.backward()
    assert pre.grad is not None and torch.isfinite(pre.grad).all()


# ---------------------------------------------------------------------------
# 2. cba_v2 Baseline Losses Tests (kd_baselines.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("method", ["lskd", "sinkd", "dist_plus"])
def test_baseline_kd_losses_grad_flow_and_precision_safety(method, batch_size, dtype):
    torch.manual_seed(10)
    criterion = BaselineKDLoss(method=method, num_classes=9)
    stu = (torch.randn(batch_size, 9, 24, 24, dtype=dtype) * 10.0).detach().requires_grad_(True)
    tea = (torch.randn(batch_size, 9, 24, 24, dtype=dtype) * 10.0).detach().requires_grad_(True)
    labels = torch.randint(0, 9, (batch_size, 24, 24), dtype=torch.long)

    total, parts = criterion(stu, tea, labels)
    assert total.dtype == torch.float32
    assert torch.isfinite(total).item() and total.item() > 0.0
    assert parts["loss_kd"].item() > 0.0 and parts["loss_sgs"].item() > 0.0

    total.backward()
    assert stu.grad is not None
    assert torch.isfinite(stu.grad).all()
    assert stu.grad.abs().sum().item() > 0.0
    assert tea.grad is None


def test_iad_kas_head_registered_modules_and_gradient_flow():
    torch.manual_seed(11)
    kas_head = IADKasHead(in_channels=256, num_classes=9)
    param_names = dict(kas_head.named_parameters())
    assert "pos.weight" in param_names and param_names["pos.weight"].shape == (9, 256, 1, 1)
    assert "neg.weight" in param_names and param_names["neg.weight"].shape == (9, 9, 3, 3)

    stu_logits = torch.randn(2, 9, 16, 16, requires_grad=True)
    stu_feat = torch.randn(2, 256, 8, 8, requires_grad=True)

    out = kas_head(stu_logits, stu_feat, temperature=0.5)
    loss_kas = out["loss_kas"]
    assert torch.isfinite(loss_kas).item() and loss_kas.item() > 0.0
    assert out["loss_rfc"].item() > 0.0 and out["loss_css"].item() > 0.0

    loss_kas.backward()
    assert stu_logits.grad is not None and stu_logits.grad.abs().sum().item() > 0.0
    assert stu_feat.grad is not None and stu_feat.grad.abs().sum().item() > 0.0
    assert kas_head.pos.weight.grad is not None and kas_head.pos.weight.grad.abs().sum().item() > 0.0
    assert kas_head.neg.weight.grad is not None and kas_head.neg.weight.grad.abs().sum().item() > 0.0


def test_distillation_loss_import_does_not_mutate_rng():
    import importlib
    import ACDC_Tool.distillation_loss as acdc_dl
    import Synapse_Tool_KD.distillation_loss as syn_dl
    import experiments.cba_v2.losses.kd_baselines as kd_base

    torch.manual_seed(98765)
    rng_before = torch.get_rng_state().clone()
    importlib.reload(syn_dl)
    importlib.reload(acdc_dl)
    importlib.reload(kd_base)
    rng_after = torch.get_rng_state()
    assert torch.equal(rng_before, rng_after), "Importing distillation_loss / kd_baselines mutated global PyTorch RNG state"


def test_iad_kas_head_and_baseline_kd_under_amp_autocast():
    torch.manual_seed(12)
    kas_head = IADKasHead(in_channels=256, num_classes=9)
    stu_logits = torch.randn(2, 9, 32, 32, dtype=torch.bfloat16, requires_grad=True)
    stu_feat = torch.randn(2, 256, 8, 8, dtype=torch.bfloat16, requires_grad=True)

    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        out = kas_head(stu_logits, stu_feat, temperature=0.5)
        for key in ("loss_kas", "loss_rfc", "loss_css", "f_plus", "f_neg"):
            assert out[key].dtype == torch.float32, f"{key} was downcast to {out[key].dtype} inside autocast"
            assert torch.isfinite(out[key]).all()

        for method in ("lskd", "sinkd", "dist_plus"):
            crit = BaselineKDLoss(method=method, num_classes=9)
            tea_logits = torch.randn(2, 9, 32, 32, dtype=torch.bfloat16)
            labels = torch.randint(0, 9, (2, 32, 32), dtype=torch.long)
            total, parts = crit(stu_logits, tea_logits, labels)
            assert total.dtype == torch.float32, f"{method} total loss was {total.dtype} inside autocast"
            assert parts["loss_kd"].dtype == torch.float32
            assert torch.isfinite(total).item() and total.item() > 0.0


# ---------------------------------------------------------------------------
# 3. End-to-End train.py Smoke Tests for All 5 Methods & IAD Transition
# ---------------------------------------------------------------------------


@pytest.fixture()
def scratch_dir():
    d = REPO_ROOT / "runs" / "test_tmp_5methods" / next(tempfile._get_candidate_names())
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)
    try:
        d.parent.rmdir()
    except OSError:
        pass


def _smoke_args(out_dir: Path, arm: str, epochs: int = 2, extra: list[str] | None = None):
    argv = [
        "--arm", arm,
        "--epochs", str(epochs),
        "--smoke_test",
        "--cpu",
        "--batch_size", "4",
        "--grad_accum_steps", "2",
        "--curriculum_warmup", "1",
        "--curriculum_ramp", "1",
        "--output_dir", str(out_dir),
        "--warmup_iters", "2",
        "--seed", "2345",
    ]
    if extra:
        argv.extend(extra)
    return build_parser().parse_args(argv)


@pytest.mark.parametrize("arm,canonical", [
    ("v5", "Arm_V5_Dual_CEHard"),
    ("iad", "Arm_IAD_200_100"),
    ("lskd", "Arm_LSKD"),
    ("sinkd", "Arm_SinKD"),
    ("dist_plus", "Arm_DIST_Plus"),
])
def test_train_smoke_all_5_methods(scratch_dir, arm, canonical):
    out_dir = scratch_dir / f"out_{arm}"
    drive_dir = scratch_dir / f"drive_{arm}"
    extra = ["--drive_sync_dir", str(drive_dir), "--drive_sync_every", "1"]
    if arm == "iad":
        extra += ["--kis_stage_epochs", "1"]
    run_pilot(_smoke_args(out_dir, arm, epochs=2, extra=extra))
    arm_dir = out_dir / canonical
    metrics_path = arm_dir / "metrics.csv"
    assert metrics_path.is_file()
    with metrics_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    for row in rows:
        assert np.isfinite(float(row["loss_total"])) and float(row["loss_total"]) > 0.0
    assert (arm_dir / "checkpoint_epoch_002.pth").is_file()
    assert (arm_dir / "final_student.pth").is_file()
    assert (drive_dir / "final_student.pth").is_file(), "final_student.pth must be synced to drive_sync_dir"


def test_iad_stage1_matches_kis_and_stage2_transitions_and_resumes(monkeypatch, scratch_dir):
    from experiments.cba_v2 import data as data_mod, models as v2_models

    class TinyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            g = torch.Generator().manual_seed(100 + index)
            return {
                "image": torch.randn(3, 16, 16, generator=g),
                "label": torch.randint(0, 9, (16, 16), generator=g, dtype=torch.long),
            }

    class TinyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.feat = torch.nn.Conv2d(3, 256, 1)
            self.head = torch.nn.Conv2d(256, 9, 1)

        def forward(self, image, multimask=True, **kwargs):
            f = self.feat(image)
            logits = self.head(f)
            return logits, f

    monkeypatch.setattr(v2_models, "build_frozen_teacher", lambda **kwargs: TinyNet().eval())
    monkeypatch.setattr(v2_models, "build_student", lambda *args, **kwargs: TinyNet())
    monkeypatch.setattr(data_mod, "SliceDataset", lambda **kwargs: TinyDataset())

    out_kis = scratch_dir / "out_kis_ref"
    out_iad = scratch_dir / "out_iad_transition"

    common = [
        "--cpu",
        "--batch_size", "2",
        "--grad_accum_steps", "1",
        "--sched_epochs", "4",
        "--skip_eval",
        "--no-save_final_preds",
        "--warmup_iters", "2",
        "--seed", "2345",
    ]

    # 1. Run 2 epochs of Arm_V1_KIS
    run_pilot(build_parser().parse_args(["--arm", "v1", "--epochs", "2", "--output_dir", str(out_kis)] + common))
    # 2. Run 3 epochs of Arm_IAD_200_100 with kis_stage_epochs=2 (epochs 1..2 KIS, epoch 3 KAS)
    run_pilot(
        build_parser().parse_args(
            ["--arm", "iad", "--epochs", "3", "--kis_stage_epochs", "2", "--output_dir", str(out_iad)] + common
        )
    )

    kis_dir = out_kis / "Arm_V1_KIS"
    iad_dir = out_iad / "Arm_IAD_200_100"

    # Verify Stage 1 (epoch 2) checkpoint exists and matches Arm_V1_KIS bit-for-bit
    ck_kis = torch.load(kis_dir / "checkpoint_epoch_002.pth", map_location="cpu", weights_only=False)
    ck_iad_s1 = torch.load(iad_dir / "checkpoint_epoch_002.pth", map_location="cpu", weights_only=False)
    assert ck_kis.get("kas_head") is None and ck_kis.get("kas_optimizer") is None
    for k, v in ck_kis["student"].items():
        assert torch.equal(v, ck_iad_s1["student"][k]), f"Stage 1 student weight diverged at {k}"

    # Verify Stage 2 (epoch 3) updated kas_head weights relative to Stage 1
    ck_iad_s2 = torch.load(iad_dir / "checkpoint_epoch_003.pth", map_location="cpu", weights_only=False)
    assert "kas_head" in ck_iad_s2 and "kas_optimizer" in ck_iad_s2
    pos_diff = (ck_iad_s2["kas_head"]["pos.weight"] - ck_iad_s1["kas_head"]["pos.weight"]).abs().sum().item()
    neg_diff = (ck_iad_s2["kas_head"]["neg.weight"] - ck_iad_s1["kas_head"]["neg.weight"]).abs().sum().item()
    assert pos_diff > 0.0 and neg_diff > 0.0, "IADKasHead weights must update during Stage 2 KAS"

    # 3. Resume from epoch 3 to epoch 4 and verify Stage 2 continues cleanly and matches uninterrupted 4-epoch run
    out_iad_full = scratch_dir / "out_iad_full4"
    run_pilot(
        build_parser().parse_args(
            ["--arm", "iad", "--epochs", "4", "--kis_stage_epochs", "2", "--output_dir", str(out_iad_full)] + common
        )
    )
    run_pilot(
        build_parser().parse_args(
            [
                "--arm", "iad",
                "--epochs", "4",
                "--kis_stage_epochs", "2",
                "--resume", "auto",
                "--output_dir", str(out_iad),
            ]
            + common
        )
    )
    with (iad_dir / "metrics.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["stage"] for r in rows] == ["KIS", "KIS", "KAS", "KAS"]
    assert float(rows[0]["loss_rfc"]) == 0.0 and float(rows[1]["loss_rfc"]) == 0.0
    assert float(rows[2]["loss_rfc"]) > 0.0 and float(rows[3]["loss_rfc"]) > 0.0
    assert float(rows[2]["loss_css"]) > 0.0 and float(rows[3]["loss_css"]) > 0.0

    ck_full4 = torch.load(out_iad_full / "Arm_IAD_200_100" / "final_student.pth", map_location="cpu", weights_only=False)
    ck_res4 = torch.load(iad_dir / "final_student.pth", map_location="cpu", weights_only=False)
    for k, v in ck_full4["student"].items():
        assert torch.equal(v, ck_res4["student"][k]), f"Stage 2 resumed student weight diverged at {k}"
    for k, v in ck_full4["kas_head"].items():
        assert torch.equal(v, ck_res4["kas_head"][k]), f"Stage 2 resumed kas_head weight diverged at {k}"

    # 4. Cross-arm checkpoint seeding: initialize Arm_IAD_200_100 at epoch 2 from Arm_V1_KIS checkpoint
    # (which lacks kas_head and kas_optimizer keys) and run epochs 3..4; must match uninterrupted Arm_IAD_200_100 bit-for-bit!
    out_iad_cross = scratch_dir / "out_iad_from_kis_ckpt"
    run_pilot(
        build_parser().parse_args(
            [
                "--arm", "iad",
                "--epochs", "4",
                "--kis_stage_epochs", "2",
                "--resume", str(kis_dir / "checkpoint_epoch_002.pth"),
                "--output_dir", str(out_iad_cross),
            ]
            + common
        )
    )
    ck_cross4 = torch.load(out_iad_cross / "Arm_IAD_200_100" / "final_student.pth", map_location="cpu", weights_only=False)
    for k, v in ck_full4["student"].items():
        assert torch.equal(v, ck_cross4["student"][k]), f"Cross-arm KIS->IAD student weight diverged at {k}"
    for k, v in ck_full4["kas_head"].items():
        assert torch.equal(v, ck_cross4["kas_head"][k]), f"Cross-arm KIS->IAD kas_head weight diverged at {k}"


# ---------------------------------------------------------------------------
# 4. Multi-Seed Launchers & 5 Colab Notebooks Verification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind,script,arm_name,base_dir", [
    ("ours", "run_main_ours.py", "Arm_V5_Dual_CEHard", "ours_v5"),
    ("iad", "run_main_iad.py", "Arm_IAD_200_100", "iad_200_100"),
    ("lskd", "run_main_lskd.py", "Arm_LSKD", "lskd300"),
    ("sinkd", "run_main_sinkd.py", "Arm_SinKD", "sinkd300"),
    ("dist_plus", "run_main_dist_plus.py", "Arm_DIST_Plus", "dist_plus300"),
])
@pytest.mark.parametrize("seed", [2345, 42, 1234])
def test_5methods_launchers_across_3_seeds(kind, script, arm_name, base_dir, seed):
    root = Path(tempfile.mkdtemp(prefix=f"launcher_{kind}_{seed}_"))
    try:
        runs = root / "runs"
        drive = root / "drive"
        argv = ["--runs_dir", str(runs), "--drive_dir", str(drive), "--seed", str(seed), "--dry_run"]
        if kind == "ours":
            argv.append("--make_warmup")

        cmd = mc.run_main(kind, argv)
        parsed = build_parser().parse_args(cmd[2:])
        assert resolve_arm_name(parsed.arm) == arm_name
        assert parsed.epochs == 300 and parsed.sched_epochs == 300
        assert parsed.seed == seed and parsed.eval_freq == 0
        expected_dir = base_dir if seed == 2345 else f"{base_dir}_s{seed}"
        assert Path(parsed.output_dir).name == expected_dir
        assert Path(parsed.drive_sync_dir).name == expected_dir

        if kind == "ours":
            assert parsed.resume == "<warmup checkpoint>"
            assert (parsed.lambda_a, parsed.tau, parsed.m, parsed.r, parsed.beta) == (0.02, 0.1, 0.2, 2, 0.5)
        else:
            assert parsed.resume == "auto"

        # Also verify CLI script invocation
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / script)] + argv,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert proc.returncode == 0, proc.stderr
        assert "[run] " in proc.stdout
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize("kind", ["ours", "iad", "lskd", "sinkd", "dist_plus"])
def test_5_generated_notebooks_compile_and_build_valid_3seed_commands(kind):
    spec = p5n.METHOD_SPECS[kind]
    nb_name = spec["notebook"]
    repo_nb_path = REPO_ROOT / "notebooks" / nb_name
    upload_nb_path = UPLOAD_ROOT / nb_name
    assert repo_nb_path.is_file(), f"Missing notebook in repo: {repo_nb_path}"
    assert upload_nb_path.is_file(), f"Missing notebook in colab_upload: {upload_nb_path}"
    assert repo_nb_path.read_text(encoding="utf-8") == upload_nb_path.read_text(encoding="utf-8")

    nb = json.loads(repo_nb_path.read_text(encoding="utf-8"))
    assert nb == p5n.build_method_notebook(kind)

    codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for idx, code in enumerate(codes):
        compile(code, f"{nb_name}:cell{idx}", "exec")

    # STATUS cell (codes[4]) must report both Raw and LCC metrics
    assert "primary_dice_raw_macro" in codes[4] and "primary_dice_lcc_macro" in codes[4]
    assert "primary_hd95_raw_valid_macro" in codes[4] and "primary_hd95_lcc_valid_macro" in codes[4]
    assert "primary_nsd_raw_macro" in codes[4] and "primary_nsd_lcc_macro" in codes[4]
    # DOWNLOAD cell (codes[5]) must skip intermediate checkpoint_epoch_* when final_student.pth exists
    assert "final_student.pth" in codes[5] and "checkpoint_epoch_" in codes[5]

    config: dict = {}
    exec(codes[0], config)
    assert config["RUN"] == kind
    assert config["SEEDS"] == [2345, 42, 1234]
    assert config["EPOCHS"] == 300
    assert config["INPUT_MODE"] == "upload"

    root = Path(tempfile.mkdtemp(prefix=f"nb_{kind}_"))
    try:
        state_file = root / "multiseed_v5_state.json"
        state_file.write_text(json.dumps({"lambda_a": 0.02, "tau": 0.1, "m": 0.2, "r": 2, "beta": 0.5}))
        fn_node = next(
            n for n in ast.parse(codes[1]).body if isinstance(n, ast.FunctionDef) and n.name == "command_for"
        )
        config.update(sys=sys, MAIN_DRIVE=str(root / "drive"), STATE=str(state_file), RUNS=str(root / "runs"))
        exec(compile(ast.Module(body=[fn_node], type_ignores=[]), nb_name, "exec"), config)

        for seed in [2345, 42, 1234]:
            cmd = config["command_for"](seed, dry_run=True)
            launcher_args = mc.build_parser(kind).parse_args(cmd[2:])
            assert launcher_args.seed == seed
            assert launcher_args.epochs == 300
            assert launcher_args.eval_freq == 0
            assert launcher_args.dry_run is True
            if kind == "ours":
                assert launcher_args.make_warmup is True
                assert launcher_args.arm == "v5"
                assert launcher_args.state_json == str(state_file)

            train_cmd = mc.run_main(kind, cmd[2:])
            train_args = build_parser().parse_args(train_cmd[2:])
            assert resolve_arm_name(train_args.arm) == spec["arm_dir"]
            expected_name = spec["base_dir"] if seed == 2345 else f"{spec['base_dir']}_s{seed}"
            assert Path(train_args.output_dir).name == expected_name
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_code_zip_bundle_contains_all_5method_files():
    archives = [UPLOAD_ROOT / "iad_cba_v2_code.zip"]
    if (REPO_ROOT / "dist" / "iad_cba_v2_code.zip").is_file():
        archives.append(REPO_ROOT / "dist" / "iad_cba_v2_code.zip")
    if (REPO_ROOT / "colab_upload").is_dir():
        archives.append(REPO_ROOT / "colab_upload" / "iad_cba_v2_code.zip")
    for archive in archives:
        assert archive.is_file(), f"Missing archive: {archive}"
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            names = set(z.namelist())
            required = {
                "experiments/cba_v2/losses/kd_baselines.py",
                "experiments/cba_v2/train.py",
                "scripts/main_common.py",
                "scripts/run_main_ours.py",
                "scripts/run_main_iad.py",
                "scripts/run_main_lskd.py",
                "scripts/run_main_sinkd.py",
                "scripts/run_main_dist_plus.py",
                "scripts/prepare_5methods_notebooks.py",
                "Synapse_Tool_KD/criterion/SinKD.py",
                "Synapse_Tool_KD/criterion/lskd.py",
                "Synapse_Tool_KD/distillation_loss.py",
                "ACDC_Tool/distillation_loss.py",
            }
            missing = required - names
            assert not missing, f"Missing from {archive}: {missing}"


# ---------------------------------------------------------------------------
# 5. Solution 1 & Solution 2 Unit Tests (CBAv2Loss & Arm_V5_Dual_CEHard)
# ---------------------------------------------------------------------------


def test_solution1_boundary_band_exclusion_for_background_fp():
    from experiments.cba_v2.losses.cba_v2_loss import CBAv2Loss

    # Create a 16x16 slice with organ 1 occupying [6:10, 6:10]
    y_gt = torch.zeros(1, 16, 16, dtype=torch.long)
    y_gt[0, 6:10, 6:10] = 1

    # Base logits strongly predict GT everywhere (no FN, no far FP)
    logits_near_fp = torch.full((1, 9, 16, 16), -10.0)
    logits_near_fp[0, 0] = 6.0
    logits_near_fp[0, 0, 6:10, 6:10] = -6.0
    logits_near_fp[0, 1, 6:10, 6:10] = 6.0

    # Inject background FP (y=0, pred=1) ONLY inside the r<=2 boundary band at row 5 (distance 1 from row 6)
    logits_near_fp[0, 0, 5, 6:10] = -2.0
    logits_near_fp[0, 1, 5, 6:10] = 4.0
    logits_near_fp.requires_grad_(True)

    crit_excl = CBAv2Loss(
        num_classes=9,
        r=2,
        enable_boundary=False,
        enable_error=True,
        error_mode="ce_hard",
        error_gating_type="none",
        fp_by_pred_class=True,
        fp_exclude_boundary=True,
        empty_slice_suppression=False,
    )
    _, loss_e_excl, diag_excl = crit_excl(
        logits_near_fp, y_gt, return_diagnostics=True, return_stream_losses=True
    )
    assert diag_excl["bg_fp_excluded_boundary_pixels"] == 4
    assert diag_excl["active_fp_pred_classes"] == []
    assert diag_excl["loss_e_fp"] == 0.0
    assert float(loss_e_excl.item()) == 0.0

    # When fp_exclude_boundary=False, those 4 boundary-band FPs are penalized
    crit_incl = CBAv2Loss(
        num_classes=9,
        r=2,
        enable_boundary=False,
        enable_error=True,
        error_mode="ce_hard",
        error_gating_type="none",
        fp_by_pred_class=True,
        fp_exclude_boundary=False,
        empty_slice_suppression=False,
    )
    _, loss_e_incl, diag_incl = crit_incl(
        logits_near_fp, y_gt, return_diagnostics=True, return_stream_losses=True
    )
    assert diag_incl["bg_fp_excluded_boundary_pixels"] == 0
    assert diag_incl["active_fp_pred_classes"] == [1]
    assert diag_incl["fp_pixels_per_pred_class"][1] == 4
    assert float(loss_e_incl.item()) > 0.0

    # Now add a far background FP outside r<=2 at [0, 0] (distance > 2 from [6:10, 6:10])
    logits_far_fp = logits_near_fp.detach().clone()
    logits_far_fp[0, 0, 0, 0] = -2.0
    logits_far_fp[0, 1, 0, 0] = 4.0
    logits_far_fp.requires_grad_(True)

    _, loss_e_far, diag_far = crit_excl(
        logits_far_fp, y_gt, return_diagnostics=True, return_stream_losses=True
    )
    assert diag_far["bg_fp_excluded_boundary_pixels"] == 4
    assert diag_far["active_fp_pred_classes"] == [1]
    assert diag_far["fp_pixels_per_pred_class"][1] == 1
    assert float(loss_e_far.item()) > 0.0
    loss_e_far.backward()
    # Gradient from L_E must be non-zero at the far FP [0, 0] and strictly zero at the r<=2 boundary band [5, 6:10]
    assert logits_far_fp.grad[0, :, 0, 0].abs().sum().item() > 0.0
    assert logits_far_fp.grad[0, :, 5, 6:10].abs().sum().item() == 0.0


def test_solution1_fp_by_pred_class_balances_fn_and_fp_and_handles_absent_cache():
    from experiments.cba_v2.losses.cba_v2_loss import CBAv2Loss

    # Put all 8 foreground organs (1..8) in y_gt so all 8 can have FN errors
    y_gt = torch.zeros(1, 16, 16, dtype=torch.long)
    for c in range(1, 9):
        y_gt[0, c, 0:4] = c

    # Construct logits where:
    # - Each foreground organ c=1..8 has 1 FN pixel at (c, 0) where class 0 beats class c
    # - Background at (15, 15) (far from all organs, r<=2 is rows 0..10, cols 0..6) has a false positive for organ 7
    logits = torch.full((1, 9, 16, 16), -8.0)
    logits[0, 0] = 5.0
    for c in range(1, 9):
        logits[0, 0, c, 1:4] = -5.0
        logits[0, c, c, 1:4] = 5.0
        # FN at (c, 0): pred=0, gt=c
        logits[0, 0, c, 0] = 3.0
        logits[0, c, c, 0] = 1.0

    # Far FP at (15, 15): y=0, pred=7
    logits[0, 0, 15, 15] = 0.5
    logits[0, 7, 15, 15] = 4.5
    logits.requires_grad_(True)

    crit = CBAv2Loss(
        num_classes=9,
        r=2,
        enable_boundary=False,
        enable_error=True,
        error_mode="ce_hard",
        error_gating_type="none",
        fp_by_pred_class=True,
        fp_exclude_boundary=True,
        empty_slice_suppression=False,
    )
    _, loss_e, diag = crit(logits, y_gt, return_diagnostics=True, return_stream_losses=True)
    assert diag["active_fn_classes"] == list(range(1, 9))
    assert diag["active_fp_pred_classes"] == [7]
    assert diag["loss_e_fn"] > 0.0 and diag["loss_e_fp"] > 0.0
    expected_balanced = 0.5 * diag["loss_e_fn"] + 0.5 * diag["loss_e_fp"]
    assert np.isclose(diag["loss_e_ce"], expected_balanced, atol=1e-6)
    assert np.isclose(float(loss_e.item()), expected_balanced, atol=1e-6)

    # Verify cold-cache absent-organ FP: organ 7 is completely absent from y_gt and cache is empty (valid_mask=False),
    # yet predicting organ 7 on background is still penalized by L_FP in ce_hard mode!
    y_empty = torch.zeros(1, 16, 16, dtype=torch.long)
    logits_absent_fp = torch.full((1, 9, 16, 16), -8.0)
    logits_absent_fp[0, 0] = 5.0
    logits_absent_fp[0, 0, 10, 10] = 0.0
    logits_absent_fp[0, 7, 10, 10] = 4.0
    logits_absent_fp.requires_grad_(True)

    crit_cold = CBAv2Loss(
        num_classes=9,
        r=2,
        enable_boundary=False,
        enable_error=True,
        error_mode="ce_hard",
        error_gating_type="none",
        fp_by_pred_class=True,
        fp_exclude_boundary=True,
        empty_slice_suppression=False,
    )
    assert not bool(crit_cold.valid_mask.any().item())
    _, loss_e_cold, diag_cold = crit_cold(
        logits_absent_fp, y_empty, return_diagnostics=True, return_stream_losses=True
    )
    assert diag_cold["active_fn_classes"] == []
    assert diag_cold["active_fp_pred_classes"] == [7]
    assert diag_cold["fp_pixels_per_pred_class"][7] == 1
    assert np.isclose(float(loss_e_cold.item()), diag_cold["loss_e_fp"], atol=1e-6)
    assert float(loss_e_cold.item()) > 0.0


def test_solution2_empty_slice_topk_suppression_and_clean_slice_guard():
    from experiments.cba_v2.losses.cba_v2_loss import CBAv2Loss

    crit = CBAv2Loss(
        num_classes=9,
        r=2,
        enable_boundary=False,
        enable_error=True,
        error_mode="ce_hard",
        error_gating_type="none",
        fp_by_pred_class=True,
        fp_exclude_boundary=True,
        empty_slice_suppression=True,
        empty_slice_topk=64,
        empty_slice_prob_thresh=0.05,
        empty_slice_focal_gamma=2.0,
        empty_slice_weight=1.0,
    )

    # 1. Clean absent slice: all foreground probabilities p_{b,c}(x) <= 0.05 -> exact 0 loss and 0 gradient
    y_clean = torch.zeros(2, 16, 16, dtype=torch.long)
    logits_clean = torch.full((2, 9, 16, 16), -4.0)
    logits_clean[:, 0] = 4.0  # p_fg = exp(-4) / (exp(4) + 8*exp(-4)) ~ 0.00033 << 0.05
    logits_clean.requires_grad_(True)

    _, loss_clean, diag_clean = crit(
        logits_clean, y_clean, return_diagnostics=True, return_stream_losses=True
    )
    assert diag_clean["empty_slice_total_absent_pairs"] == 16  # 2 slices * 8 absent foreground classes
    assert diag_clean["empty_slice_active_pairs"] == 0
    assert diag_clean["empty_slice_active_pixels"] == 0
    assert diag_clean["loss_e_empty"] == 0.0
    assert float(loss_clean.item()) == 0.0
    loss_clean.backward()
    assert logits_clean.grad.abs().max().item() == 0.0

    # 2. Spurious sub-argmax or super-argmax island of 10 pixels on slice 0 for absent organ c=3
    # Even if p_{0,3}(x) = 0.35 < p_{0,0}(x) = 0.65 (so argmax is still 0 and hard-error CE misses it!),
    # Solution 2 catches and suppresses those 10 pixels while leaving clean pixels untouched.
    logits_island = torch.full((2, 9, 16, 16), -6.0)
    logits_island[:, 0] = 4.0
    # On slice 0, organ 3 is absent; create 10 pixels with p_{0,3} ~ 0.35 (sub-argmax soft island > tau_p=0.05)
    logits_island[0, 0, 2, 2:12] = 1.0
    logits_island[0, 3, 2, 2:12] = 0.4  # softmax: exp(0.4)/(exp(1.0)+exp(0.4)+7*exp(-6)) ~ 0.354 > 0.05, argmax=0
    logits_island.requires_grad_(True)

    _, loss_island, diag_island = crit(
        logits_island, y_clean, return_diagnostics=True, return_stream_losses=True
    )
    # Because argmax is 0 everywhere, hard-error CE is 0, so loss_e comes purely from Solution 2!
    assert diag_island["loss_e_ce"] == 0.0
    assert diag_island["empty_slice_active_pairs"] == 1
    assert diag_island["empty_slice_active_pixels"] == 10
    assert diag_island["active_empty_classes"] == [3]
    assert diag_island["loss_e_empty"] > 0.0
    assert np.isclose(float(loss_island.item()), diag_island["loss_e_empty"], atol=1e-6)

    loss_island.backward()
    # On the 10 island pixels, dL/dz_{0,3} must be positive (pushes organ 3 logit down)
    grad_island_c3 = logits_island.grad[0, 3, 2, 2:12]
    assert (grad_island_c3 > 0.0).all(), "Empty-slice suppression must push spurious organ logit down"
    # Clean pixels on slice 0 outside [2, 2:12] and all pixels on slice 1 must have exact zero gradient!
    mask_clean = torch.ones_like(logits_island.grad, dtype=torch.bool)
    mask_clean[0, :, 2, 2:12] = False
    assert logits_island.grad[mask_clean].abs().max().item() == 0.0

    # 3. Present-organ slice is completely untouched by Solution 2:
    # If organ 3 has at least 1 GT pixel on slice 0, slice 0 is NOT an absent slice for organ 3.
    y_present = y_clean.clone()
    y_present[0, 14, 14] = 3
    logits_present = logits_island.detach().clone()
    # Make (0, 14, 14) correctly predicted as class 3 so hard-error CE is also 0
    logits_present[0, 0, 14, 14] = -6.0
    logits_present[0, 3, 14, 14] = 6.0
    logits_present.requires_grad_(True)

    _, loss_present, diag_present = crit(
        logits_present, y_present, return_diagnostics=True, return_stream_losses=True
    )
    assert diag_present["empty_slice_total_absent_pairs"] == 15  # (slice 0, organ 3) is no longer absent
    assert diag_present["empty_slice_active_pairs"] == 0
    assert diag_present["loss_e_empty"] == 0.0
    assert float(loss_present.item()) == 0.0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_solution1_and_solution2_amp_stability(dtype):
    from experiments.cba_v2.losses.cba_v2_loss import CBAv2Loss

    torch.manual_seed(42)
    crit = CBAv2Loss(
        num_classes=9,
        r=2,
        enable_boundary=True,
        enable_error=True,
        error_mode="ce_hard",
        error_gating_type="none",
    )
    # Create batch with present organs on slice 0 and absent organs on slice 1, with extreme logit values
    y_gt = torch.zeros(2, 24, 24, dtype=torch.long)
    y_gt[0, 4:10, 4:10] = 1
    y_gt[0, 14:20, 14:20] = 2
    logits = (torch.randn(2, 9, 24, 24, dtype=dtype) * 20.0).detach().requires_grad_(True)
    logits_ema = (torch.randn(2, 9, 24, 24, dtype=dtype) * 20.0).detach()

    with torch.amp.autocast(device_type="cpu", dtype=dtype, enabled=True):
        loss_b, loss_e, diag = crit(
            logits, y_gt, fplus_ema=logits_ema, return_diagnostics=True, return_stream_losses=True
        )
        total = loss_b + crit.beta * loss_e

    assert loss_b.dtype == torch.float32 and loss_e.dtype == torch.float32
    assert torch.isfinite(total).item() and total.item() > 0.0
    total.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0.0


@pytest.mark.parametrize("zc_val", [2.0, 10.0, 15.0, 25.0, 40.0])
def test_solution2_confident_fp_gradient_and_exact_analytical_norm(zc_val):
    from experiments.cba_v2.losses.cba_v2_loss import CBAv2Loss

    # Isolate Solution 2 by setting align_ce_hard_samples=True (with cold cache so L_ErrorCE skips)
    # and empty_slice_suppression=True
    crit = CBAv2Loss(
        num_classes=9,
        enable_boundary=False,
        enable_error=True,
        error_mode="ce_hard",
        error_gating_type="none",
        align_ce_hard_samples=True,
        fp_by_pred_class=False,
        empty_slice_suppression=True,
        empty_slice_topk=64,
        empty_slice_prob_thresh=0.05,
        empty_slice_focal_gamma=2.0,
    )
    y_empty = torch.zeros(1, 8, 8, dtype=torch.long)
    logits = torch.full((1, 9, 8, 8), -10.0, dtype=torch.float32)
    logits[0, 0] = 5.0
    # Single confident false positive for absent organ 3 at (0, 2, 2)
    logits[0, 0, 2, 2] = 0.0
    logits[0, 3, 2, 2] = float(zc_val)
    logits.requires_grad_(True)

    _, loss_e, diag = crit(
        logits,
        y_empty,
        return_diagnostics=True,
        return_stream_losses=True,
        compute_autograd_norms=True,
        update_cache=False,
    )
    assert diag["loss_e_ce"] == 0.0
    assert diag["empty_slice_active_pixels"] == 1
    assert diag["loss_e_empty"] > 0.0
    # Autograd gradient must remain non-zero (~1.0 on z_c) even for extreme confident FPs (zc=15..40)
    loss_e.backward()
    assert float(logits.grad[0, 3, 2, 2].item()) > 0.99
    assert float(logits.grad[0, 0, 2, 2].item()) < -0.99
    assert diag["grad_norm_e_autograd"] > 1.0
    assert diag["grad_norm_empty_analytical"] == pytest.approx(diag["grad_norm_e_autograd"], rel=1e-5)


def test_v2_ce_hard_align_samples_matches_contrastive_and_legacy_loss_e_fn():
    import torch.nn.functional as F
    from experiments.cba_v2.losses.cba_v2_loss import CBAv2Loss

    c_contra = CBAv2Loss(
        num_classes=9,
        error_mode="contrastive",
        error_gating_type="none",
        k_bg=32,
        k_fg=32,
        align_ce_hard_samples=True,
    )
    c_ce = CBAv2Loss(
        num_classes=9,
        error_mode="ce_hard",
        error_gating_type="none",
        k_bg=32,
        k_fg=32,
        align_ce_hard_samples=True,
    )
    assert not c_ce.fp_by_pred_class and not c_ce.fp_exclude_boundary and not c_ce.empty_slice_suppression

    for c in (0, 1, 2):
        p = F.normalize(torch.randn(9), p=2, dim=0)
        c_contra.set_prototype(c, p)
        c_ce.set_prototype(c, p)

    torch.manual_seed(999)
    fplus = torch.randn(2, 9, 32, 32, requires_grad=True)
    y_gt = torch.zeros(2, 32, 32, dtype=torch.long)
    y_gt[:, 8:16, 8:16] = 1
    y_gt[:, 16:24, 16:24] = 2
    with torch.no_grad():
        fplus[:, 3, 8:12, 8:12] = 10.0   # Unseen opponent class 3
        fplus[:, 2, 12:16, 8:16] = 10.0  # Seen opponent class 2

    _, diag_contra = c_contra(fplus, y_gt, return_diagnostics=True, sample_seed=42, update_cache=False)
    _, diag_ce = c_ce(fplus, y_gt, return_diagnostics=True, sample_seed=42, update_cache=False)

    assert diag_contra["sampled_error_pixels"] == diag_ce["sampled_error_pixels"]
    assert diag_contra["skipped_error_pixels"] == diag_ce["skipped_error_pixels"]
    assert diag_contra["sampled_pixels_per_class"] == diag_ce["sampled_pixels_per_class"]
    assert diag_contra["error_pixels_per_class"] == diag_ce["error_pixels_per_class"]
    assert diag_contra["loss_e_fn"] > 0.0 and diag_ce["loss_e_fn"] > 0.0


def test_ours_v5_reverted_defaults_match_original_formulation():
    from experiments.cba_v2.losses.cba_v2_loss import CBAv2Loss
    from experiments.cba_v2.train import build_cba_v2_loss, build_parser

    args = build_parser().parse_args(["--arm", "v5"])
    assert args.fp_by_pred_class is False
    assert args.fp_exclude_boundary is False
    assert args.empty_slice_suppression is False

    crit = build_cba_v2_loss("Arm_V5_Dual_CEHard", args)
    assert isinstance(crit, CBAv2Loss)
    assert crit.error_mode == "ce_hard"
    assert crit.fp_by_pred_class is False
    assert crit.fp_exclude_boundary is False
    assert crit.empty_slice_suppression is False
    assert crit.align_ce_hard_samples is True

    torch.manual_seed(42)
    y_gt = torch.zeros(2, 16, 16, dtype=torch.long)
    y_gt[0, 4:10, 4:10] = 1
    y_gt[0, 10:14, 10:14] = 2
    logits = torch.randn(2, 9, 16, 16, requires_grad=True)
    logits_ema = torch.randn(2, 9, 16, 16)

    loss_b, loss_e, diag = crit(
        logits, y_gt, fplus_ema=logits_ema, return_diagnostics=True, return_stream_losses=True
    )
    assert diag["loss_e_empty"] == 0.0
    assert diag["active_fp_pred_classes"] == []
    assert diag["active_error_classes"] == [0, 1, 2]
    assert np.isclose(
        float(loss_e.item()),
        (diag["loss_e_fp"] + 2.0 * diag["loss_e_fn"]) / 3.0,
        atol=1e-6,
    )
    assert np.isclose(float(loss_e.item()), diag["loss_e_ce"], atol=1e-6)
    (loss_b + crit.beta * loss_e).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_archive_sol12_run_if_needed_archives_stale_sol12_and_resumes_from_warmup(scratch_dir):
    import main_common as mc

    runs_dir = scratch_dir / "runs"
    drive_dir = scratch_dir / "drive_main"
    runs_dir.mkdir(parents=True, exist_ok=True)
    drive_dir.mkdir(parents=True, exist_ok=True)

    # 1. Seed 2345 original ours_v5 (no Sol 1+2 flags) with eval_final.json -> must be preserved and skipped
    og_drive = drive_dir / "ours_v5"
    og_drive.mkdir(parents=True, exist_ok=True)
    (og_drive / "run_config.json").write_text(
        json.dumps({"arm_name": "Arm_V5_Dual_CEHard", "args": {"seed": 2345, "lambda_a": 0.02}}),
        encoding="utf-8",
    )
    (og_drive / "eval_final.json").write_text(
        json.dumps({"overall": {"primary_dice_raw_macro": 0.8185}}),
        encoding="utf-8",
    )
    cmd_2345 = mc.run_main(
        "ours",
        ["--runs_dir", str(runs_dir), "--drive_dir", str(drive_dir), "--seed", "2345", "--dry_run"],
    )
    assert cmd_2345 == []
    assert og_drive.is_dir()
    assert not (drive_dir / "ours_v5_sol12_archive").exists()

    # 2. Seed 42 stale Solution 1+2 ours_v5_s42 + existing warmup_s42 checkpoint -> must archive and resume from warmup_s42
    warmup_s42 = drive_dir / "warmup_s42"
    warmup_s42.mkdir(parents=True, exist_ok=True)
    warmup_ckpt = warmup_s42 / "checkpoint_epoch_020.pth"
    warmup_ckpt.write_bytes(b"warmup_s42_ckpt")

    sol12_drive = drive_dir / "ours_v5_s42"
    sol12_drive.mkdir(parents=True, exist_ok=True)
    (sol12_drive / "run_config.json").write_text(
        json.dumps({
            "arm_name": "Arm_V5_Dual_CEHard",
            "args": {"seed": 42, "fp_by_pred_class": True, "empty_slice_suppression": True},
        }),
        encoding="utf-8",
    )
    (sol12_drive / "eval_final.json").write_text(
        json.dumps({"overall": {"primary_dice_raw_macro": 0.8081}}),
        encoding="utf-8",
    )
    (sol12_drive / "latest_checkpoint.pth").write_bytes(b"bad_sol12_ckpt")

    sol12_local = runs_dir / "ours_v5_s42" / "Arm_V5_Dual_CEHard"
    sol12_local.mkdir(parents=True, exist_ok=True)
    (sol12_local / "run_config.json").write_text(
        json.dumps({
            "arm_name": "Arm_V5_Dual_CEHard",
            "args": {"seed": 42, "fp_by_pred_class": True, "empty_slice_suppression": True},
        }),
        encoding="utf-8",
    )
    (sol12_local / "eval_final.json").write_text(
        json.dumps({"overall": {"primary_dice_raw_macro": 0.8081}}),
        encoding="utf-8",
    )

    cmd_42 = mc.run_main(
        "ours",
        ["--runs_dir", str(runs_dir), "--drive_dir", str(drive_dir), "--seed", "42", "--dry_run"],
    )
    assert not sol12_drive.exists()
    assert (drive_dir / "ours_v5_s42_sol12_archive" / "eval_final.json").is_file()
    assert not (runs_dir / "ours_v5_s42").exists()
    assert (runs_dir / "ours_v5_s42_sol12_archive" / "Arm_V5_Dual_CEHard" / "eval_final.json").is_file()
    assert cmd_42 != []
    resume_idx = cmd_42.index("--resume")
    assert cmd_42[resume_idx + 1] == str(warmup_ckpt)

    # 3. Also detect fp_exclude_boundary=True as a Solution 1+2 run directory
    sol1_only = drive_dir / "ours_v5_s1234"
    sol1_only.mkdir(parents=True, exist_ok=True)
    (sol1_only / "run_config.json").write_text(
        json.dumps({"arm_name": "Arm_V5_Dual_CEHard", "args": {"seed": 1234, "fp_exclude_boundary": True}}),
        encoding="utf-8",
    )
    assert mc._is_sol12_run_dir(str(sol1_only)) is True


def test_reverted_v5_matches_legacy_cba_v2_loss_bit_for_bit():
    from experiments.cba_v2.losses.cba_v2_loss import CBAv2Loss as V2CBAv2Loss

    torch.manual_seed(123)
    x1 = torch.randn(2, 9, 16, 16, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    ema = torch.randn(2, 9, 16, 16)
    y = torch.randint(0, 9, (2, 16, 16))

    v2 = V2CBAv2Loss(error_mode="ce_hard", error_gating_type="none", k_bg=16, k_fg=16)
    leg = V2CBAv2Loss(
        error_mode="ce_hard",
        error_gating_type="none",
        k_bg=16,
        k_fg=16,
        fp_by_pred_class=False,
        fp_exclude_boundary=False,
        empty_slice_suppression=False,
        align_ce_hard_samples=True,
    )

    lb1, le1, _ = v2(x1, y, fplus_ema=ema, return_diagnostics=True, return_stream_losses=True, sample_seed=7)
    lb2, le2, _ = leg(x2, y, fplus_ema=ema, return_diagnostics=True, return_stream_losses=True, sample_seed=7)

    assert torch.allclose(lb1, lb2, atol=0.0, rtol=0.0)
    assert torch.allclose(le1, le2, atol=0.0, rtol=0.0)
    (lb1 + 0.5 * le1).backward()
    (lb2 + 0.5 * le2).backward()
    assert torch.allclose(x1.grad, x2.grad, atol=0.0, rtol=0.0)


def test_main_5methods_3seeds_directories_and_zip_sha256_synced():
    import hashlib
    import prepare_5methods_notebooks as prep

    dist_zip = REPO_ROOT / "dist" / "iad_cba_v2_code.zip"
    if not dist_zip.is_file():
        dist_zip = REPO_ROOT.parent / "colab_upload" / "iad_cba_v2_code.zip"
    assert dist_zip.is_file()
    expected_sha = hashlib.sha256(dist_zip.read_bytes()).hexdigest()

    bundle_dirs = [
        REPO_ROOT.parent / "colab_upload",
        REPO_ROOT.parent / "colab_upload" / "main_5methods_3seeds",
    ]
    for opt_dir in (REPO_ROOT / "colab_upload", REPO_ROOT / "colab_upload" / "main_5methods_3seeds"):
        if opt_dir.is_dir():
            bundle_dirs.append(opt_dir)
    for bdir in bundle_dirs:
        assert bdir.is_dir(), f"Missing bundle directory: {bdir}"
        zpath = bdir / "iad_cba_v2_code.zip"
        assert zpath.is_file(), f"Missing zip in {bdir}"
        assert hashlib.sha256(zpath.read_bytes()).hexdigest() == expected_sha, f"Stale zip in {bdir}"
        assert (bdir / "MULTISEED_README.md").is_file(), f"Missing README in {bdir}"
        for spec in prep.METHOD_SPECS.values():
            nb_file = bdir / spec["notebook"]
            assert nb_file.is_file(), f"Missing notebook {nb_file}"



