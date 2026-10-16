"""Comprehensive verification suite for ACDC (4 classes, 256x256) and Polyp (1 binary channel, 256x256)
across all 5 Knowledge Distillation methods (Ours v5, IAD, DIST+, LSKD, SinKD) and the 3 parallel Colab notebooks."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import main_common as mc  # noqa: E402
from experiments.cba_v2.data import ACDCSliceDataset, PolypDataset  # noqa: E402
from experiments.cba_v2.evaluation import evaluate_volumes  # noqa: E402
from experiments.cba_v2.losses.cba_v2_loss import CBAv2Loss  # noqa: E402
from experiments.cba_v2.losses.kd_baselines import BaselineKDLoss, IADKasHead  # noqa: E402
from experiments.cba_v2.losses.kis_loss import KISLoss, lift_binary_logits_for_cba, structure_loss  # noqa: E402
from experiments.cba_v2.models import verify_checkpoint_compatibility  # noqa: E402
from experiments.cba_v2.train import apply_dataset_defaults, build_parser, run_pilot  # noqa: E402


@pytest.fixture()
def tmp_work():
    d = Path(tempfile.mkdtemp(prefix="acdc_polyp_test_"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_checkpoint_compatibility_guard_strictly_isolates_datasets_and_roles():
    """Ensure Synapse (9, 256), ACDC (4, 256), and Polyp (1, 256) checkpoints and Teacher/Student roles cannot be mixed."""
    syn_tea = REPO_ROOT / "teacher_weights" / "epoch_299.pth"
    acdc_tea = REPO_ROOT / "teacher_weights" / "acdc_tea.pth"
    acdc_stu = REPO_ROOT / "teacher_weights" / "acdc_stu.pth"
    polyp_tea = REPO_ROOT / "teacher_weights" / "polyp_tea.pth"
    polyp_stu = REPO_ROOT / "teacher_weights" / "polyp_stu.pth"

    assert verify_checkpoint_compatibility(syn_tea, dataset="synapse", role="teacher")["mask_tokens_shape"] == (9, 256)
    assert verify_checkpoint_compatibility(acdc_tea, dataset="acdc", role="teacher")["mask_tokens_shape"] == (4, 256)
    assert verify_checkpoint_compatibility(acdc_stu, dataset="acdc", role="student")["mask_tokens_shape"] == (4, 256)
    assert verify_checkpoint_compatibility(polyp_tea, dataset="polyp", role="teacher")["mask_tokens_shape"] == (1, 256)
    assert verify_checkpoint_compatibility(polyp_stu, dataset="polyp", role="student")["mask_tokens_shape"] == (1, 256)

    # Cross-dataset must fail
    for bad_ds in ("synapse", "polyp"):
        with pytest.raises(ValueError, match="Dataset mismatch"):
            verify_checkpoint_compatibility(acdc_tea, dataset=bad_ds, role="teacher")
    for bad_ds in ("synapse", "acdc"):
        with pytest.raises(ValueError, match="Dataset mismatch"):
            verify_checkpoint_compatibility(polyp_tea, dataset=bad_ds, role="teacher")
    for bad_ds in ("acdc", "polyp"):
        with pytest.raises(ValueError, match="Dataset mismatch"):
            verify_checkpoint_compatibility(syn_tea, dataset=bad_ds, role="teacher")

    # Cross-role must fail
    with pytest.raises(ValueError, match="Role mismatch"):
        verify_checkpoint_compatibility(acdc_stu, dataset="acdc", role="teacher")
    with pytest.raises(ValueError, match="Role mismatch"):
        verify_checkpoint_compatibility(acdc_tea, dataset="acdc", role="student")
    with pytest.raises(ValueError, match="Role mismatch"):
        verify_checkpoint_compatibility(polyp_stu, dataset="polyp", role="teacher")
    with pytest.raises(ValueError, match="Role mismatch"):
        verify_checkpoint_compatibility(polyp_tea, dataset="polyp", role="student")


def test_dataset_defaults_and_class_count_guard():
    acdc_args = build_parser().parse_args(["--dataset", "acdc"])
    assert apply_dataset_defaults(acdc_args) == "acdc"
    assert acdc_args.num_classes == 4 and acdc_args.img_size == 256
    assert acdc_args.teacher_ckpt == "teacher_weights/acdc_tea.pth"

    polyp_args = build_parser().parse_args(["--dataset", "polyp"])
    assert apply_dataset_defaults(polyp_args) == "polyp"
    assert polyp_args.num_classes == 1 and polyp_args.img_size == 256
    assert polyp_args.teacher_ckpt == "teacher_weights/polyp_tea.pth"

    bad_acdc = build_parser().parse_args(["--dataset", "acdc", "--num_classes", "1"])
    with pytest.raises(ValueError, match="ACDC requires num_classes=4"):
        apply_dataset_defaults(bad_acdc)

    bad_polyp = build_parser().parse_args(["--dataset", "polyp", "--num_classes", "4"])
    with pytest.raises(ValueError, match="Polyp requires num_classes=1"):
        apply_dataset_defaults(bad_polyp)


def test_real_acdc_and_polyp_datasets_and_label_guard(tmp_work):
    # Real ACDC train & valid
    acdc_tr = ACDCSliceDataset(
        data_dir="data/ACDC/train",
        list_path="data/ACDC/lists_ACDC/train.txt",
        iad_root=str(REPO_ROOT),
        augment=True,
        aug="full",
    )
    assert len(acdc_tr) == 1304
    s0 = acdc_tr[0]
    assert tuple(s0["image"].shape) == (3, 256, 256)
    assert tuple(s0["label"].shape) == (256, 256)
    assert int(s0["label"].min()) >= 0 and int(s0["label"].max()) <= 3

    # Corrupt ACDC slice with Synapse label 8 must be rejected by ACDCSliceDataset guard
    bad_dir = tmp_work / "bad_acdc"
    bad_dir.mkdir(parents=True)
    np.savez(bad_dir / "patient001_frame01_slice_0.npz", img=np.zeros((256, 256), dtype=np.float32), label=np.full((256, 256), 8, dtype=np.uint8))
    bad_list = tmp_work / "bad_train.txt"
    bad_list.write_text("patient001_frame01_slice_0.npz\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Dataset Guard"):
        ACDCSliceDataset(data_dir=str(bad_dir), list_path=str(bad_list), iad_root=str(REPO_ROOT), augment=False)

    # Real Polyp train & 5 test subdatasets
    polyp_tr = PolypDataset(
        data_dir="data/dataset/TrainDataset",
        iad_root=str(REPO_ROOT),
        split="train",
        augment=True,
        aug="full",
    )
    assert len(polyp_tr) == 1450
    p0 = polyp_tr[0]
    assert tuple(p0["image"].shape) == (3, 256, 256)
    assert tuple(p0["label"].shape) == (256, 256)
    assert set(np.unique(p0["label"].numpy())).issubset({0, 1})

    expected_test = {"Kvasir": 100, "CVC-ClinicDB": 62, "CVC-ColonDB": 380, "ETIS-LaribPolypDB": 196, "CVC-300": 60}
    for sub, cnt in expected_test.items():
        sub_ds = PolypDataset(data_dir="data/dataset", iad_root=str(REPO_ROOT), split="test", subdataset=sub, augment=False)
        assert len(sub_ds) == cnt


@pytest.mark.parametrize("dataset,num_classes", [("acdc", 4), ("polyp", 1)])
@pytest.mark.parametrize("method", ["ours", "iad_s1", "iad_s2", "dist_plus", "lskd", "sinkd"])
def test_all_5_methods_losses_and_gradients_on_acdc_and_polyp(dataset, num_classes, method):
    torch.manual_seed(2345)
    b, h, w = 2, 32, 32
    stu_logits = torch.randn(b, num_classes, h, w, requires_grad=True)
    tea_logits = torch.randn(b, num_classes, h, w)
    ema_logits = torch.randn(b, num_classes, h, w)
    if num_classes == 1:
        labels = torch.randint(0, 2, (b, h, w), dtype=torch.long)
    else:
        labels = torch.randint(0, num_classes, (b, h, w), dtype=torch.long)
        for c in range(num_classes):
            labels[0, (c * 4) : (c * 4 + 3), 2:8] = c

    if method == "ours":
        kis = KISLoss(num_classes=num_classes, cpd_weight_mode="flat")
        loss_base, _ = kis(stu_logits, tea_logits, labels)
        cba_c = 2 if num_classes == 1 else num_classes
        cba_feat = "raw" if num_classes == 1 else "centered"
        cba = CBAv2Loss(
            num_classes=cba_c,
            feature_space=cba_feat,
            enable_boundary=True,
            enable_error=True,
            error_mode="ce_hard",
            error_gating_type="none",
        )
        g_base = torch.autograd.grad(loss_base, stu_logits, retain_graph=True)[0].detach()
        if num_classes == 1:
            fplus = lift_binary_logits_for_cba(stu_logits)
            fplus_ema = lift_binary_logits_for_cba(ema_logits)
            g_ref = torch.cat([-g_base, g_base], dim=1)
        else:
            fplus = stu_logits
            fplus_ema = ema_logits
            g_ref = g_base
        lb, le, diag = cba(fplus=fplus, y_gt=labels, fplus_ema=fplus_ema, return_diagnostics=True, return_stream_losses=True, g_ref=g_ref)
        total = loss_base + 0.02 * (lb + 0.5 * le)
        assert float(lb.item()) > 0.0 and float(le.item()) > 0.0
    elif method == "iad_s1":
        kis = KISLoss(num_classes=num_classes, cpd_weight_mode="flat")
        total, metrics = kis(stu_logits, tea_logits, labels)
        assert float(metrics["loss_kis"].item()) > 0.0
    elif method == "iad_s2":
        sgs = BaselineKDLoss(method="sgs_only", num_classes=num_classes)
        kas = IADKasHead(in_channels=16, num_classes=num_classes)
        feat = torch.randn(b, 16, h // 4, w // 4, requires_grad=True)
        loss_sgs, _ = sgs.forward_sgs(stu_logits, labels)
        kas_res = kas(stu_logits, feat)
        total = loss_sgs + 0.1 * kas_res["loss_kas"]
    else:
        kd = BaselineKDLoss(method=method, num_classes=num_classes)
        total, metrics = kd(stu_logits, tea_logits, labels)
        assert float(metrics["loss_kd"].item()) > 0.0

    assert torch.isfinite(total)
    total.backward()
    assert stu_logits.grad is not None
    assert torch.isfinite(stu_logits.grad).all()
    assert float(stu_logits.grad.norm().item()) > 0.0


@pytest.mark.parametrize("dataset", ["acdc", "polyp"])
@pytest.mark.parametrize("arm", ["v5", "iad", "dist_plus", "lskd", "sinkd"])
def test_smoke_train_all_5_methods_and_determinism(dataset, arm, tmp_work):
    out1 = tmp_work / f"{dataset}_{arm}_run1"
    out2 = tmp_work / f"{dataset}_{arm}_run2"
    common = [
        "--dataset", dataset,
        "--arm", arm,
        "--epochs", "2",
        "--kis_stage_epochs", "1",
        "--curriculum_warmup", "0",
        "--curriculum_ramp", "1",
        "--batch_size", "2",
        "--grad_accum_steps", "1",
        "--max_train_batches", "2",
        "--img_size", "64",
        "--smoke_test",
        "--cpu",
        "--seed", "2345",
    ]
    r1 = run_pilot(build_parser().parse_args(common + ["--output_dir", str(out1)]))
    r2 = run_pilot(build_parser().parse_args(common + ["--output_dir", str(out2)]))
    assert np.isclose(r1["final_loss"], r2["final_loss"], rtol=1e-6, atol=1e-6)
    assert len(r1["history"]) == 2

    # Cross-dataset resume must be blocked by Checkpoint Guard
    other_ds = "polyp" if dataset == "acdc" else "acdc"
    ckpt_path = out1 / r1["arm_name"] / "latest_checkpoint.pth"
    assert ckpt_path.is_file()
    with pytest.raises(ValueError, match="Checkpoint Guard"):
        run_pilot(
            build_parser().parse_args(
                [
                    "--dataset", other_ds,
                    "--arm", arm,
                    "--epochs", "3",
                    "--smoke_test",
                    "--cpu",
                    "--output_dir", str(tmp_work / "bad_resume"),
                    "--resume", str(ckpt_path),
                ]
            )
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for real SAM ViT-L Teacher + Student smoke test")
@pytest.mark.parametrize("dataset", ["acdc", "polyp"])
def test_real_gpu_teacher_student_train_and_eval_on_acdc_and_polyp(dataset, tmp_work):
    out_dir = tmp_work / f"real_{dataset}"
    args = build_parser().parse_args(
        [
            "--dataset", dataset,
            "--arm", "v5",
            "--epochs", "1",
            "--curriculum_warmup", "0",
            "--curriculum_ramp", "1",
            "--batch_size", "2",
            "--grad_accum_steps", "1",
            "--max_train_batches", "2",
            "--eval_max_cases", "2",
            "--no-save_final_preds",
            "--output_dir", str(out_dir),
            "--seed", "2345",
        ]
    )
    res = run_pilot(args)
    assert np.isfinite(res["final_loss"])
    assert res["final_eval"] is not None
    assert 0.0 <= res["final_eval"]["eval_dice_raw"] <= 1.0
    assert (out_dir / "Arm_V5_Dual_CEHard" / "eval_final.json").is_file()


def test_parallel_3_notebooks_cover_all_10_jobs_and_compile(tmp_work):
    import scripts.prepare_acdc_polyp_3notebooks as prep

    all_jobs = []
    for nb_key, spec in prep.NOTEBOOK_SPECS.items():
        nb_path = REPO_ROOT / "notebooks" / spec["notebook"]
        assert nb_path.is_file(), f"Missing generated notebook: {nb_path}"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        for idx, code in enumerate(codes):
            compile(code, f"{spec['notebook']}:cell{idx}", "exec")
        ns = {}
        exec(codes[0], ns)
        assert ns["JOBS"] == spec["jobs"]
        all_jobs.extend(spec["jobs"])

    expected_jobs = {
        (ds, m)
        for ds in ("acdc", "polyp")
        for m in ("ours", "iad", "dist_plus", "lskd", "sinkd")
    }
    assert len(all_jobs) == 10
    assert set(all_jobs) == expected_jobs

    # Verify dry-run CLI commands via main_common for all 10 jobs
    state_file = tmp_work / "state.json"
    state_file.write_text(json.dumps({"lambda_a": 0.02, "tau": 0.1, "m": 0.2, "r": 2, "beta": 0.5}), encoding="utf-8")
    for ds_name, method in all_jobs:
        r_dir = tmp_work / f"{ds_name}_main"
        argv = ["--dataset", ds_name, "--runs_dir", str(r_dir), "--epochs", "300", "--seed", "2345", "--dry_run"]
        if method == "ours":
            argv += ["--arm", "v5", "--state_json", str(state_file), "--make_warmup"]
        cmd = mc.run_main(method, argv)
        parsed = build_parser().parse_args(cmd[2:])
        assert parsed.dataset == ds_name
        assert str(r_dir) in parsed.output_dir


def test_status_cell_executes_against_acdc_and_polyp_eval_final_json(tmp_work, capsys):
    import csv
    import scripts.prepare_acdc_polyp_3notebooks as prep

    acdc_dir = tmp_work / "acdc_main" / "ours_v5" / "Arm_V5_Dual_CEHard"
    polyp_dir = tmp_work / "polyp_main" / "ours_v5" / "Arm_V5_Dual_CEHard"
    acdc_dir.mkdir(parents=True, exist_ok=True)
    polyp_dir.mkdir(parents=True, exist_ok=True)

    acdc_eval = {
        "overall": {
            "primary_dice_raw_macro": 0.9142,
            "primary_dice_lcc_macro": 0.9148,
            "primary_jc_raw_macro": 0.8467,
            "primary_hd95_raw_valid_macro": 1.19,
            "primary_hd95_lcc_valid_macro": 1.12,
            "primary_nsd_raw_macro": 0.9410,
        },
        "per_class": {
            "RV": {"dice_clean_raw": 0.9032},
            "Myo": {"dice_clean_raw": 0.8886},
            "LV": {"dice_clean_raw": 0.9508},
        },
    }
    polyp_eval = {
        "overall": {
            "primary_dice_raw_macro": 0.8034,
            "polyp_5sets_dice_raw_macro": 0.8149,
            "primary_jc_raw_macro": 0.7285,
            "primary_hd95_raw_valid_macro": 14.50,
            "primary_nsd_raw_macro": 0.6500,
        },
        "per_dataset": {
            "Kvasir": {"dice_clean_raw": 0.8960},
            "CVC-ClinicDB": {"dice_clean_raw": 0.8486},
            "CVC-ColonDB": {"dice_clean_raw": 0.7781},
            "ETIS-LaribPolypDB": {"dice_clean_raw": 0.6910},
            "CVC-300": {"dice_clean_raw": 0.8607},
        },
    }
    (acdc_dir / "eval_final.json").write_text(json.dumps(acdc_eval), encoding="utf-8")
    (polyp_dir / "eval_final.json").write_text(json.dumps(polyp_eval), encoding="utf-8")

    ns = {
        "Path": Path,
        "csv": csv,
        "json": json,
        "mc": mc,
        "SEEDS": [2345],
        "EPOCHS": 300,
        "runs_dir_for": lambda ds: str(tmp_work / f"{ds}_main"),
        "drive_dir_for": lambda ds: None,
    }
    exec(prep.STATUS_CELL, ns)
    out = capsys.readouterr().out
    assert "RV=90.32%, Myo=88.86%, LV=95.08%" in out
    assert "4-Set DSC=80.34% | 5-Set DSC=81.49%" in out
    assert "Kvasir=89.60%" in out and "CVC-300=86.07%" in out


def test_seed_worker_and_real_augmentation_determinism():
    from experiments.cba_v2.train import seed_worker, set_seed

    set_seed(2345)
    if hasattr(torch.backends, "cudnn"):
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    # Worker 0 and Worker 1 receive distinct torch initial seeds from DataLoader
    torch.manual_seed(2345 + 0)
    seed_worker(0)
    ds_acdc = ACDCSliceDataset(iad_root=str(REPO_ROOT), split="train", augment=True, aug="full")
    s0_a = ds_acdc[0]["image"].clone()

    torch.manual_seed(2345 + 0)
    seed_worker(0)
    s0_b = ds_acdc[0]["image"].clone()
    assert torch.allclose(s0_a, s0_b), "Same worker seed must produce identical ACDC augmented slice"

    torch.manual_seed(2345 + 0)
    seed_worker(0)
    ds_polyp = PolypDataset(iad_root=str(REPO_ROOT), split="train", augment=True, aug="full")
    p0_a = ds_polyp[0]["image"].clone()

    torch.manual_seed(2345 + 0)
    seed_worker(0)
    p0_b = ds_polyp[0]["image"].clone()
    assert torch.allclose(p0_a, p0_b), "Same worker seed must produce identical Polyp augmented image"


def test_polyp_final_preds_subdirectories_synced_to_drive(tmp_work, monkeypatch):
    import experiments.cba_v2.models as models_mod

    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 1, kernel_size=1)

        def forward(self, x, *args, **kwargs):
            h, w = x.shape[-2], x.shape[-1]
            out = self.conv(x)
            feat = torch.zeros(x.shape[0], 256, h // 16, w // 16, device=x.device, dtype=x.dtype)
            return out, out, feat

    monkeypatch.setattr(
        models_mod,
        "build_frozen_teacher",
        lambda iad_root, teacher_ckpt, base_ckpt, device, dataset="synapse": _TinyModel().to(device).eval(),
    )
    monkeypatch.setattr(
        models_mod,
        "build_student",
        lambda iad_root, student_ckpt, device, strict_load=True, dataset="synapse": _TinyModel().to(device),
    )

    out_dir = tmp_work / "polyp_preds_local"
    drive_dir = tmp_work / "polyp_preds_drive"
    args = build_parser().parse_args(
        [
            "--dataset", "polyp",
            "--arm", "v1",
            "--epochs", "1",
            "--batch_size", "2",
            "--grad_accum_steps", "1",
            "--max_train_batches", "1",
            "--eval_max_cases", "1",
            "--save_final_preds",
            "--drive_sync_dir", str(drive_dir),
            "--drive_sync_every", "1",
            "--output_dir", str(out_dir),
            "--seed", "2345",
        ]
    )
    run_pilot(args)
    synced_npy = list((drive_dir / "final_preds").rglob("*.npy"))
    assert len(synced_npy) == 5, f"Expected 5 synced Polyp prediction .npy files across 5 subdatasets, got {len(synced_npy)}"



def test_short_epochs_ours_disables_warmup():
    state = {"lambda_a": 0.02, "tau": 0.1, "m": 0.2, "r": 2, "beta": 0.5}
    job_short = mc.job_for("ours", "v5", state, epochs=2, seed=2345)
    assert job_short["needs_warmup"] is False
    job_full = mc.job_for("ours", "v5", state, epochs=300, seed=2345)
    assert job_full["needs_warmup"] is True

