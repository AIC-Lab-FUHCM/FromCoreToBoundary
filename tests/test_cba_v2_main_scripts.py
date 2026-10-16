"""Tests for the 300-epoch main-run launchers (scripts/main_common.py, run_main_kis.py, run_main_ours.py)."""
import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import main_common as mc  # noqa: E402
import run_tuning as rt  # noqa: E402
from experiments.cba_v2.train import build_parser, resolve_arm_name  # noqa: E402


@pytest.fixture()
def work_dir():
    d = REPO_ROOT / "runs" / "test_tmp_main"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _make_warmup(runs):
    ck = Path(runs) / "warmup" / rt.ARM_DIRS["v1"] / "checkpoint_epoch_020.pth"
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"x")
    return ck


def test_kis_command_parses_and_uses_full_data_defaults(work_dir):
    cmd = mc.run_main("kis", ["--runs_dir", str(work_dir / "r"), "--drive_dir", str(work_dir / "d"), "--dry_run"])
    args = build_parser().parse_args(cmd[2:])
    assert resolve_arm_name(args.arm) == "Arm_V1_KIS"
    assert args.epochs == 300 and args.sched_epochs == 300 and args.seed == 2345 and args.eval_freq == 0
    # no tuning split flags: the runner defaults (train.txt / official test volumes) apply
    assert "--list_path" not in cmd and "--test_dir" not in cmd and "--no-save_final_preds" not in cmd
    assert args.drive_sync_dir.endswith("kis300")
    assert args.resume == "<warmup checkpoint>"


def test_ours_command_uses_arm5_and_tuned_state(work_dir):
    state = work_dir / "state.json"
    state.write_text(json.dumps({"lambda_a": 0.01, "tau": 0.2, "m": 0.1, "r": 1, "beta": 0.25}))
    cmd = mc.run_main("ours", ["--runs_dir", str(work_dir / "r"), "--state_json", str(state), "--dry_run"])
    args = build_parser().parse_args(cmd[2:])
    assert resolve_arm_name(args.arm) == "Arm_V5_Dual_CEHard"
    assert (args.lambda_a, args.tau, args.m, args.r, args.beta) == (0.01, 0.2, 0.1, 1, 0.25)
    assert args.output_dir.endswith("ours_v5")


def test_ours_arm2_has_no_beta_and_defaults_without_state(work_dir):
    cmd = mc.run_main("ours", ["--arm", "v2", "--runs_dir", str(work_dir / "r"), "--dry_run"])
    assert "--beta" not in cmd
    args = build_parser().parse_args(cmd[2:])
    assert resolve_arm_name(args.arm) == "Arm_V2_Boundary"
    assert (args.lambda_a, args.tau, args.m, args.r) == (0.02, 0.1, 0.2, 2)


def test_state_path_is_next_to_the_main_folder_on_drive():
    assert Path(mc.default_state_path("/content/drive/MyDrive/IAD/main")) == Path("/content/drive/MyDrive/IAD/tune/tune_state.json")
    assert mc.default_state_path(None) is None


def test_forks_from_the_shared_warmup_then_auto_resumes(work_dir):
    runs = work_dir / "r"
    ck = _make_warmup(runs)
    cmd = mc.run_main("kis", ["--runs_dir", str(runs), "--dry_run"])
    assert cmd[cmd.index("--resume") + 1] == str(ck)
    own = runs / "kis300" / rt.ARM_DIRS["v1"] / "latest_checkpoint.pth"
    own.parent.mkdir(parents=True)
    own.write_bytes(b"x")
    cmd = mc.run_main("kis", ["--runs_dir", str(runs), "--dry_run"])
    assert cmd[cmd.index("--resume") + 1] == "auto"


def test_finished_run_is_skipped(work_dir, capsys):
    runs = work_dir / "r"
    done = runs / "kis300" / rt.ARM_DIRS["v1"] / "eval_final.json"
    done.parent.mkdir(parents=True)
    done.write_text("{}")
    assert mc.run_main("kis", ["--runs_dir", str(runs), "--dry_run"]) == []
    assert "[skip]" in capsys.readouterr().out


def test_warmup_command_matches_tuning_warmup_and_uses_the_full_schedule(work_dir):
    cmd = mc.warmup_command(300, str(work_dir / "r"), str(work_dir / "d"), 2345)
    args = build_parser().parse_args(cmd[2:])
    assert resolve_arm_name(args.arm) == "Arm_V1_KIS"
    assert args.epochs == rt.WARMUP_EPOCHS == 20 and args.sched_epochs == 300 and args.eval_freq == 0
    assert args.skip_eval is True
    assert args.drive_sync_dir.endswith("warmup")


def test_make_warmup_default_differs_by_kind():
    assert mc.build_parser("kis").parse_args([]).make_warmup is True
    assert mc.build_parser("ours").parse_args([]).make_warmup is False
    assert mc.build_parser("ours").parse_args(["--make_warmup"]).make_warmup is True


def test_missing_warmup_without_make_raises(work_dir):
    with pytest.raises(RuntimeError):
        mc.run_main("ours", ["--runs_dir", str(work_dir / "r"), "--wait_min", "0"])


@pytest.mark.parametrize("seed", [42, 1234])
def test_additional_seed_does_not_reuse_original_warmup_or_finished_run(work_dir, seed):
    runs = work_dir / "r"
    original = _make_warmup(runs)
    done = runs / "kis300" / rt.ARM_DIRS["v1"] / "eval_final.json"
    done.parent.mkdir(parents=True)
    done.write_text("{}")
    assert mc.find_warmup_ckpt(str(runs), None, seed) is None
    cmd = mc.run_main("kis", ["--runs_dir", str(runs), "--seed", str(seed), "--dry_run"])
    args = build_parser().parse_args(cmd[2:])
    assert Path(args.output_dir).name == f"kis300_s{seed}"
    assert args.resume == "<warmup checkpoint>"
    assert original.exists()


@pytest.mark.parametrize("seed", [42, 1234])
def test_seed_warmup_is_shared_by_matched_arms_and_resume_is_isolated(work_dir, seed):
    runs = work_dir / "r"
    drive = work_dir / "d"
    ck = runs / f"warmup_s{seed}" / rt.ARM_DIRS["v1"] / "checkpoint_epoch_020.pth"
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"x")
    for kind, name in (("kis", "kis300"), ("ours", "ours_v5")):
        argv = ["--runs_dir", str(runs), "--drive_dir", str(drive), "--seed", str(seed), "--dry_run"]
        cmd = mc.run_main(kind, argv)
        args = build_parser().parse_args(cmd[2:])
        assert args.resume == str(ck)
        assert Path(args.output_dir).name == f"{name}_s{seed}"
        assert Path(args.drive_sync_dir).name == f"{name}_s{seed}"
        own = Path(args.output_dir) / rt.ARM_DIRS[args.arm] / "latest_checkpoint.pth"
        own.parent.mkdir(parents=True)
        own.write_bytes(b"x")
        cmd = mc.run_main(kind, argv)
        assert cmd[cmd.index("--resume") + 1] == "auto"
    warmup = build_parser().parse_args(mc.warmup_command(300, str(runs), str(drive), seed)[2:])
    assert Path(warmup.output_dir).name == f"warmup_s{seed}"
    assert Path(warmup.drive_sync_dir).name == f"warmup_s{seed}"


def test_drive_warmup_lookup_and_finished_run_are_seed_specific(work_dir):
    drive = work_dir / "d"
    ck = drive / "warmup_s1234" / "checkpoint_epoch_020.pth"
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"x")
    assert mc.find_warmup_ckpt(str(work_dir / "r"), str(drive), 42) is None
    assert mc.find_warmup_ckpt(str(work_dir / "r"), str(drive), 1234) == str(ck)
    done = drive / "ours_v5_s42" / "eval_final.json"
    done.parent.mkdir(parents=True)
    done.write_text("{}")
    argv = ["--runs_dir", str(work_dir / "r"), "--drive_dir", str(drive), "--dry_run"]
    assert mc.run_main("ours", argv + ["--seed", "42"]) == []
    assert mc.run_main("ours", argv + ["--seed", "1234"])


@pytest.mark.parametrize("name,kind,seeds", [
    ("run_cba_v2_main_colab.ipynb", "kis", [42, 1234]),
])
def test_multiseed_notebook_commands_match_launcher(name, kind, seeds, work_dir):
    notebook = json.loads((REPO_ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    codes = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    for index, code in enumerate(codes):
        compile(code, f"{name}:cell{index}", "exec")
    config = {}
    exec(codes[0], config)
    assert config["RUN"] == kind and config["SEEDS"] == seeds
    assert config["INPUT_MODE"] == "upload"
    assert 'INPUT_MODE == "drive"' in codes[1] and 'INPUT_MODE == "upload"' in codes[1]
    function = next(n for n in ast.parse(codes[1]).body if isinstance(n, ast.FunctionDef) and n.name == "command_for")
    config.update(sys=sys, MAIN_DRIVE=None, STATE=str(work_dir / "frozen.json"), RUNS=str(work_dir / "r"))
    exec(compile(ast.Module(body=[function], type_ignores=[]), name, "exec"), config)
    for seed in seeds:
        command = config["command_for"](seed, dry_run=True)
        parsed = mc.build_parser(kind).parse_args(command[2:])
        assert parsed.seed == seed and parsed.epochs == 300
        assert parsed.eval_freq == 0
        if kind == "ours":
            assert parsed.make_warmup is False and parsed.arm == "v5"
            assert parsed.state_json == config["STATE"] and parsed.lambda_ratio_cap == 0
        launched = mc.run_main(kind, command[2:])
        assert Path(launched[launched.index("--output_dir") + 1]).name == mc.seed_name(
            "kis300" if kind == "kis" else "ours_v5", seed
        )


@pytest.mark.parametrize("script", ["run_main_kis.py", "run_main_ours.py"])
def test_scripts_run_from_the_command_line(script, work_dir):
    out = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / script), "--runs_dir", str(work_dir / "r"),
                          "--dry_run"], capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert out.returncode == 0, out.stderr
    assert "[run] " in out.stdout
