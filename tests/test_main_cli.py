"""Comprehensive unit and integration tests for the unified top-level CLI entry point (main.py)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
for p in (REPO_ROOT, REPO_ROOT / "models", REPO_ROOT / "scripts", REPO_ROOT / "scripts" / "tools"):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import main as top_main  # noqa: E402
import main_common as mc  # noqa: E402
from experiments.cba_v2 import train as train_mod  # noqa: E402


@pytest.fixture()
def tmp_dir():
    d = Path(tempfile.mkdtemp(prefix="test_main_cli_"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_main_parser_is_superset_of_train_and_launcher_parsers():
    """Verify main.py exposes every parameter from train.py and main_common.py with identical defaults."""
    main_parser = top_main.build_parser()
    train_parser = train_mod.build_parser()

    main_opts = set(main_parser._option_string_actions.keys())
    train_opts = set(train_parser._option_string_actions.keys())
    missing_from_train = train_opts - main_opts
    assert not missing_from_train, f"Flags in train.py missing from main.py: {missing_from_train}"

    main_defaults = vars(main_parser.parse_args([]))
    train_defaults = vars(train_parser.parse_args([]))
    for key, val in train_defaults.items():
        assert key in main_defaults, f"Dest '{key}' missing from main.py"
        assert main_defaults[key] == val, f"Default mismatch for '{key}': main={main_defaults[key]!r} vs train={val!r}"

    for kind in ("kis", "ours", "iad", "lskd", "sinkd", "dist_plus", "sgs_only"):
        mc_opts = set(mc.build_parser(kind)._option_string_actions.keys())
        missing_from_mc = mc_opts - main_opts
        assert not missing_from_mc, f"Flags in main_common({kind}) missing from main.py: {missing_from_mc}"


def test_list_methods_and_list_datasets(capsys):
    res_m = top_main.main(["--list_methods"])
    out_m = capsys.readouterr().out
    assert res_m["mode"] == "list_methods"
    for m in ("ours", "iad", "dist_plus", "lskd", "sinkd", "kis", "v2", "v3", "v4", "sgs_only"):
        assert m in res_m["methods"]
        assert m in out_m

    res_d = top_main.main(["--list_datasets"])
    out_d = capsys.readouterr().out
    assert res_d["mode"] == "list_datasets"
    assert res_d["datasets"] == ["synapse", "acdc", "polyp"]
    for ds in ("synapse", "acdc", "polyp"):
        assert ds in out_d


def test_parse_seeds_and_datasets_edge_cases():
    assert top_main.parse_seeds(2345, None) == [2345]
    assert top_main.parse_seeds(2345, []) == [2345]
    assert top_main.parse_seeds(2345, ["42", "1234", "42"]) == [42, 1234]
    assert top_main.parse_seeds(2345, ["2345,42,1234"]) == [2345, 42, 1234]
    assert top_main.parse_seeds(99, ["all"]) == [2345, 42, 1234]
    assert top_main.parse_seeds(99, ["3seeds"]) == [2345, 42, 1234]

    assert top_main.parse_datasets("synapse") == ["synapse"]
    assert top_main.parse_datasets("btcv") == ["synapse"]
    assert top_main.parse_datasets("acdc,polyp") == ["acdc", "polyp"]
    assert top_main.parse_datasets("all") == ["synapse", "acdc", "polyp"]

    with pytest.raises(ValueError, match="Unsupported dataset"):
        top_main.parse_datasets("unknown_ds")


def test_resolve_effective_arm_and_reproduce_method_parsing():
    p = top_main.build_parser()

    for alias, expected in [
        ("ours", "Arm_V5_Dual_CEHard"),
        ("v5", "Arm_V5_Dual_CEHard"),
        ("iad", "Arm_IAD_200_100"),
        ("dist_plus", "Arm_DIST_Plus"),
        ("lskd", "Arm_LSKD"),
        ("sinkd", "Arm_SinKD"),
        ("kis", "Arm_V1_KIS"),
        ("v1", "Arm_V1_KIS"),
        ("v2", "Arm_V2_Boundary"),
        ("v3", "Arm_V3_Error"),
        ("v4", "Arm_V4_Dual_Contrastive"),
        ("sgs_only", "Arm_SGS_Only"),
        ("sgs", "Arm_SGS_Only"),
    ]:
        argv = ["--method", alias]
        args = p.parse_args(argv)
        assert top_main.resolve_effective_arm(args, argv) == expected

    # --method ours with explicit --arm v2 selects Arm_V2_Boundary
    argv_v2 = ["--method", "ours", "--arm", "v2"]
    assert top_main.resolve_effective_arm(p.parse_args(argv_v2), argv_v2) == "Arm_V2_Boundary"

    # --method all in train mode must raise a helpful ValueError
    argv_all = ["--method", "all"]
    with pytest.raises(ValueError, match="--method all is supported in '--mode reproduce'"):
        top_main.resolve_effective_arm(p.parse_args(argv_all), argv_all)

    # parse_methods_for_reproduce
    assert top_main.parse_methods_for_reproduce("all", "Arm_V5_Dual_CEHard") == [
        ("ours", "v5"),
        ("iad", None),
        ("dist_plus", None),
        ("lskd", None),
        ("sinkd", None),
    ]
    assert top_main.parse_methods_for_reproduce("ours,iad,sgs_only", "Arm_V5_Dual_CEHard") == [
        ("ours", "v5"),
        ("iad", None),
        ("sgs_only", None),
    ]
    assert top_main.parse_methods_for_reproduce("ours", "v2", ["--method", "ours", "--arm", "v2"]) == [
        ("ours", "v2")
    ]
    with pytest.raises(ValueError, match="Unsupported method/arm"):
        top_main.parse_methods_for_reproduce("nonexistent_method", "v5")


@pytest.mark.parametrize(
    "method,expected_arm",
    [
        ("ours", "Arm_V5_Dual_CEHard"),
        ("iad", "Arm_IAD_200_100"),
        ("dist_plus", "Arm_DIST_Plus"),
        ("lskd", "Arm_LSKD"),
        ("sinkd", "Arm_SinKD"),
        ("kis", "Arm_V1_KIS"),
        ("v2", "Arm_V2_Boundary"),
        ("v3", "Arm_V3_Error"),
        ("v4", "Arm_V4_Dual_Contrastive"),
        ("sgs_only", "Arm_SGS_Only"),
    ],
)
def test_main_train_smoke_all_methods(tmp_dir, method, expected_arm):
    out_dir = tmp_dir / f"smoke_{method}"
    res = top_main.main(
        [
            "--mode", "train",
            "--method", method,
            "--dataset", "acdc",
            "--epochs", "1",
            "--batch_size", "2",
            "--grad_accum_steps", "1",
            "--max_train_batches", "1",
            "--img_size", "64",
            "--smoke_test",
            "--cpu",
            "--output_dir", str(out_dir),
        ]
    )
    assert res["arm_name"] == expected_arm
    assert res["dataset"] == "acdc"
    assert len(res["history"]) == 1
    assert (out_dir / expected_arm / "final_student.pth").is_file()


def test_main_train_multiseed_and_state_json_override(tmp_dir):
    state_file = tmp_dir / "tune_state.json"
    state_file.write_text(
        json.dumps({"lambda_a": 0.03, "tau": 0.15, "m": 0.25, "r": 3, "beta": 0.75}),
        encoding="utf-8",
    )

    # 1. Dry-run checks state_json loading and explicit CLI flag precedence
    res_dry = top_main.main(
        [
            "--mode", "train",
            "--method", "ours",
            "--state_json", str(state_file),
            "--lambda_a", "0.05",  # explicit CLI override must win over state_json's 0.03
            "--seeds", "2345", "42",
            "--output_dir", str(tmp_dir / "dry_ms"),
            "--dry_run",
        ]
    )
    assert res_dry["dry_run"] is True
    assert res_dry["seeds"] == [2345, 42]
    assert len(res_dry["runs"]) == 2

    # 2. Real 2-seed smoke run of sgs_only writes multiseed_summary.json
    ms_out = tmp_dir / "real_ms"
    res_ms = top_main.main(
        [
            "--mode", "train",
            "--method", "sgs_only",
            "--dataset", "polyp",
            "--seeds", "2345,42",
            "--epochs", "1",
            "--batch_size", "2",
            "--grad_accum_steps", "1",
            "--max_train_batches", "1",
            "--img_size", "64",
            "--smoke_test",
            "--cpu",
            "--output_dir", str(ms_out),
        ]
    )
    assert res_ms["arm_name"] == "Arm_SGS_Only"
    assert res_ms["seeds"] == [2345, 42]
    assert len(res_ms["runs"]) == 2
    assert Path(res_ms["summary_path"]).is_file()
    assert (ms_out / "seed_2345" / "Arm_SGS_Only" / "final_student.pth").is_file()
    assert (ms_out / "seed_42" / "Arm_SGS_Only" / "final_student.pth").is_file()


def test_main_reproduce_dry_run_all_methods_and_datasets(tmp_dir):
    runs_dir = tmp_dir / "runs_main"
    drive_dir = tmp_dir / "drive_main"
    res = top_main.main(
        [
            "--mode", "reproduce",
            "--method", "all",
            "--dataset", "all",
            "--seeds", "2345", "42",
            "--runs_dir", str(runs_dir),
            "--drive_dir", str(drive_dir),
            "--make_warmup",
            "--dry_run",
        ]
    )
    assert res["mode"] == "reproduce"
    assert res["datasets"] == ["synapse", "acdc", "polyp"]
    assert res["seeds"] == [2345, 42]
    # 3 datasets * 5 methods * 2 seeds = 30 jobs
    assert len(res["jobs"]) == 30
    assert len(res["commands"]) == 30

    train_parser = train_mod.build_parser()
    for cmd in res["commands"]:
        parsed = train_parser.parse_args(cmd[2:])
        assert parsed.epochs == 300
        assert parsed.seed in (2345, 42)


def test_main_eval_mode_smoke_and_dry_run(tmp_dir):
    # Dry-run teacher evaluation on ACDC
    res_dry = top_main.main(
        [
            "--mode", "eval",
            "--dataset", "acdc",
            "--eval_role", "teacher",
            "--output_dir", str(tmp_dir / "eval_dry"),
            "--dry_run",
        ]
    )
    assert res_dry["mode"] == "eval" and res_dry["dry_run"] is True
    assert res_dry["dataset"] == "acdc" and res_dry["role"] == "teacher"
    assert res_dry["ckpt"] == "teacher_weights/acdc_tea.pth"

    # Real smoke evaluation via --eval_only
    eval_out = tmp_dir / "eval_smoke" / "metrics.json"
    res_smoke = top_main.main(
        [
            "--eval_only",
            "--dataset", "synapse",
            "--img_size", "64",
            "--batch_size", "2",
            "--test_dir", str(tmp_dir / "nonexistent_test_dir"),
            "--smoke_test",
            "--cpu",
            "--no-save_final_preds",
            "--eval_output", str(eval_out),
        ]
    )
    assert res_smoke["mode"] == "eval"
    assert eval_out.is_file()
    assert "eval_dice_raw" in res_smoke["summary"]

    # Missing student ckpt outside smoke_test raises ValueError
    with pytest.raises(ValueError, match="requires a checkpoint via --ckpt"):
        top_main.main(["--mode", "eval", "--dataset", "synapse", "--eval_role", "student", "--cpu"])

    # Explicit nonexistent --ckpt in smoke_test mode raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        top_main.main(
            [
                "--mode", "eval",
                "--dataset", "synapse",
                "--ckpt", str(tmp_dir / "missing_student.pth"),
                "--smoke_test",
                "--cpu",
            ]
        )

    # --dataset all in eval mode raises ValueError
    with pytest.raises(ValueError, match="requires a single dataset"):
        top_main.main(["--mode", "eval", "--dataset", "all", "--dry_run"])


def test_main_train_all_datasets_multiseed_isolates_drive_and_groups_by_dataset(tmp_dir):
    out_dir = tmp_dir / "all_ds_out"
    drive_dir = tmp_dir / "all_ds_drive"

    res_dry = top_main.main(
        [
            "--mode", "train",
            "--method", "sgs_only",
            "--dataset", "all",
            "--seeds", "2345", "42",
            "--output_dir", str(out_dir),
            "--drive_sync_dir", str(drive_dir),
            "--dry_run",
        ]
    )
    assert res_dry["datasets"] == ["synapse", "acdc", "polyp"]
    assert res_dry["seeds"] == [2345, 42]
    assert len(res_dry["runs"]) == 6
    drive_dirs = [r["drive_sync_dir"] for r in res_dry["runs"]]
    assert len(set(drive_dirs)) == 6, f"Expected 6 distinct drive_sync_dir paths, got {drive_dirs}"

    # Real smoke execution across 2 datasets x 2 seeds
    real_out = tmp_dir / "real_2ds_2seeds"
    res_real = top_main.main(
        [
            "--mode", "train",
            "--method", "sgs_only",
            "--dataset", "acdc,polyp",
            "--seeds", "2345,42",
            "--epochs", "1",
            "--batch_size", "2",
            "--grad_accum_steps", "1",
            "--max_train_batches", "1",
            "--img_size", "64",
            "--smoke_test",
            "--cpu",
            "--output_dir", str(real_out),
        ]
    )
    assert res_real["datasets"] == ["acdc", "polyp"]
    assert "by_dataset" in res_real
    assert res_real["by_dataset"]["acdc"]["n_seeds"] == 2
    assert res_real["by_dataset"]["polyp"]["n_seeds"] == 2
    assert (real_out / "acdc" / "multiseed_summary.json").is_file()
    assert (real_out / "polyp" / "multiseed_summary.json").is_file()
    assert (real_out / "acdc" / "seed_2345" / "Arm_SGS_Only" / "final_student.pth").is_file()
    assert (real_out / "polyp" / "seed_42" / "Arm_SGS_Only" / "final_student.pth").is_file()


def test_main_reproduce_forwards_explicit_cli_hyperparameters_and_executes_smoke(tmp_dir):
    runs_dir = tmp_dir / "repro_runs"
    state_file = tmp_dir / "state.json"
    state_file.write_text(
        json.dumps({"lambda_a": 0.01, "tau": 0.1, "m": 0.2, "r": 2, "beta": 0.25}),
        encoding="utf-8",
    )

    # 1. Explicit CLI ours hyperparameters must override state_json / DEFAULT_STATE in reproduce mode
    res_ours = top_main.main(
        [
            "--mode", "reproduce",
            "--method", "ours",
            "--dataset", "synapse",
            "--state_json", str(state_file),
            "--lambda_a", "0.05",
            "--beta", "0.75",
            "--tau", "0.2",
            "--m", "0.15",
            "--r", "3",
            "--batch_size", "4",
            "--runs_dir", str(runs_dir),
            "--dry_run",
        ]
    )
    parsed_ours = train_mod.build_parser().parse_args(res_ours["commands"][0][2:])
    assert parsed_ours.lambda_a == pytest.approx(0.05)
    assert parsed_ours.beta == pytest.approx(0.75)
    assert parsed_ours.tau == pytest.approx(0.2)
    assert parsed_ours.m == pytest.approx(0.15)
    assert parsed_ours.r == 3
    assert parsed_ours.batch_size == 4

    # 2. Explicit baseline KD flags and boolean flags are forwarded in reproduce mode
    res_lskd = top_main.main(
        [
            "--mode", "reproduce",
            "--method", "lskd",
            "--dataset", "acdc",
            "--lskd_tau", "3.5",
            "--no-drop_last",
            "--smoke_test",
            "--cpu",
            "--runs_dir", str(runs_dir),
            "--dry_run",
        ]
    )
    parsed_lskd = train_mod.build_parser().parse_args(res_lskd["commands"][0][2:])
    assert parsed_lskd.lskd_tau == pytest.approx(3.5)
    assert parsed_lskd.drop_last is False
    assert parsed_lskd.smoke_test is True
    assert parsed_lskd.cpu is True

    # 3. End-to-end reproduce execution (non-dry-run) in smoke_test mode
    exec_runs = tmp_dir / "repro_exec"
    res_exec = top_main.main(
        [
            "--mode", "reproduce",
            "--method", "sgs_only",
            "--dataset", "acdc",
            "--epochs", "1",
            "--batch_size", "2",
            "--grad_accum_steps", "1",
            "--max_train_batches", "1",
            "--img_size", "64",
            "--smoke_test",
            "--cpu",
            "--runs_dir", str(exec_runs),
        ]
    )
    assert len(res_exec["commands"]) == 1
    assert (exec_runs / "sgs300" / "Arm_SGS_Only" / "final_student.pth").is_file()


def test_main_tune_and_package_modes(tmp_dir):
    res_plan = top_main.main(["--mode", "tune", "--tune_action", "plan", "--tune_L", "60"])
    assert res_plan["mode"] == "tune" and res_plan["action"] == "plan"
    assert "total_h" in res_plan["cost"]

    custom_state = tmp_dir / "custom_tune_state.json"
    custom_state.write_text(
        json.dumps({"lambda_a": 0.03, "tau": 0.2, "m": 0.15, "r": 2, "beta": 0.5}),
        encoding="utf-8",
    )
    res_tune_run = top_main.main(
        [
            "--mode", "tune",
            "--tune_action", "run",
            "--tune_stage", "B",
            "--runs_dir", str(tmp_dir / "tune_runs"),
            "--state_json", str(custom_state),
            "--dry_run",
        ]
    )
    assert res_tune_run["mode"] == "tune" and res_tune_run["action"] == "run"
    assert len(res_tune_run["out"]) == 6
    assert all(j["params"]["lambda_a"] == pytest.approx(0.03) for j in res_tune_run["out"])

    res_pkg = top_main.main(["--mode", "package", "--bundle", "cba_v2", "--dry_run"])
    assert res_pkg["mode"] == "package" and res_pkg["dry_run"] is True


def test_compute_boundary_bands_and_diagnostic_tools_import():
    import torch
    from experiments.cba_v2.losses.cba_v2_loss import compute_boundary_bands
    import tools.run_lcc_postprocess_diagnostic as rld

    y_gt = torch.zeros((1, 16, 16), dtype=torch.long)
    y_gt[0, 4:12, 4:12] = 1
    inner, outer, full, per_class = compute_boundary_bands(y_gt, num_classes=9, radius=2)
    assert inner.shape == (1, 16, 16) and outer.shape == (1, 16, 16) and full.shape == (1, 16, 16)
    assert bool(inner.any()) and bool(outer.any()) and bool((inner | outer == full).all())
    assert 1 in per_class and bool(per_class[1]["full"].any())
    assert callable(rld.main)


def test_polyp_legacy_test_dataset_accepts_jpg_png_tif(tmp_dir):
    from PIL import Image
    from datasets.dataset_polyp import test_dataset

    img_dir = tmp_dir / "images"
    gt_dir = tmp_dir / "masks"
    img_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    for idx, ext in enumerate((".png", ".jpg", ".tif")):
        Image.new("RGB", (32, 32)).save(img_dir / f"sample_{idx}.png")
        Image.new("L", (32, 32)).save(gt_dir / f"sample_{idx}{ext}")

    ds = test_dataset(str(img_dir) + "/", str(gt_dir) + "/", testsize=32)
    assert ds.size == 3
    assert len(ds.gts) == 3


@pytest.mark.parametrize("entry_script", [REPO_ROOT / "main.py", WORKSPACE_ROOT / "main.py"])
def test_main_cli_subprocess_execution(entry_script, tmp_dir):
    if not entry_script.is_file():
        pytest.skip(f"{entry_script} not present")
    proc = subprocess.run(
        [
            sys.executable,
            str(entry_script),
            "--mode", "reproduce",
            "--method", "sgs_only",
            "--dataset", "acdc",
            "--seed", "2345",
            "--runs_dir", str(tmp_dir / "r"),
            "--dry_run",
        ],
        capture_output=True,
        text=True,
        cwd=str(entry_script.parent),
    )
    assert proc.returncode == 0, proc.stderr
    assert "[run] " in proc.stdout
    assert "sgs_only" in proc.stdout


def test_workspace_root_wrappers_import_cleanly_without_circular_import():
    if not (WORKSPACE_ROOT / "main.py").is_file():
        pytest.skip("Workspace root wrappers not present")
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import main, zip_code, tools.measure_far_fp_benchmarks as m; "
                "assert tuple(main.CANONICAL_DATASETS) == ('synapse', 'acdc', 'polyp'); "
                "assert 'main.py' in zip_code.CBA_V2_TARGETS; "
                "assert callable(m.main)"
            ),
        ],
        capture_output=True,
        text=True,
        cwd=str(WORKSPACE_ROOT),
    )
    assert proc.returncode == 0, proc.stderr


def test_readme_and_requirements_integrity():
    colab_req = REPO_ROOT / "scripts" / "requirements_colab.txt"
    if not colab_req.is_file():
        colab_req = REPO_ROOT / "requirements_colab.txt"
    for req_path in (
        REPO_ROOT / "requirements.txt",
        colab_req,
    ):
        assert req_path.is_file(), f"Missing {req_path}"
        content = req_path.read_text(encoding="utf-8")
        for pkg in ("timm", "MedPy", "SimpleITK", "nibabel", "h5py", "Pillow", "pytest", "scipy", "opencv-python-headless"):
            assert pkg in content, f"Missing {pkg} in {req_path}"

    assert "torch>=" in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")

    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for token in (
        "python main.py --help",
        "python main.py --list_methods",
        "python main.py --list_datasets",
        "--mode train",
        "--mode reproduce",
        "--mode eval",
        "--mode package",
        "requirements.txt",
        "requirements_colab.txt",
        "python -m pytest",
    ):
        assert token in readme_text, f"Missing '{token}' in IAD/README.md"

    import zip_code

    for target in ("main.py", "README.md", "requirements.txt", "requirements_colab.txt"):
        assert target in zip_code.CBA_V2_TARGETS


