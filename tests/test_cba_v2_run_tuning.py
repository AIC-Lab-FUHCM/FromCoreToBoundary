"""Tests for scripts/run_tuning.py (job construction, command lines, resume/skip logic, scoring, cost plan)."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_tuning as rt  # noqa: E402
from experiments.cba_v2.train import build_parser, resolve_arm_name  # noqa: E402

ORGANS = ["spleen", "liver", "aorta"]
CASES = [f"case{i:04d}" for i in range(1, 7)]


@pytest.fixture
def work_dir():
    d = REPO_ROOT / "runs" / "test_tmp_run_tuning" / next(tempfile._get_candidate_names())
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)
    try:
        d.parent.rmdir()
    except OSError:
        pass


def touch(path: Path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_eval(dir_: Path, epoch: int, dice=0.8, hd95=10.0, d_shift=0.0, h_shift=0.0):
    units = []
    for c in CASES:
        for o in ORGANS:
            units.append({"case": c, "organ": o, "gt_present": True,
                          "clean_dice_raw": dice + d_shift, "clean_dice_lcc": dice + d_shift,
                          "nsd_raw": 0.8, "nsd_lcc": 0.8,
                          "hd95_raw": hd95 + h_shift, "hd95_lcc": hd95 + h_shift})
    touch(dir_ / f"eval_ep{epoch:03d}.json", json.dumps({"units": units}))


# ----------------------------------------------------------------------------- jobs

def test_arm_dirs_match_the_runner():
    for short, name in rt.ARM_DIRS.items():
        assert resolve_arm_name(short) == name


def test_build_jobs_per_stage():
    st = rt.load_state("does-not-exist.json")
    assert [j["name"] for j in rt.build_jobs("warmup", 60, st)] == ["warmup"]
    assert rt.build_jobs("warmup", 60, st)[0]["epochs"] == 20
    assert rt.build_jobs("ref", 60, st)[0]["arm"] == "v1"
    a = rt.build_jobs("A", 60, st, [0.05, 0.1])
    assert [j["name"] for j in a] == ["a_lam0.02", "a_lam0.05", "a_lam0.1"]  # default always included
    assert all(j["arm"] == "v2" and j["epochs"] == 60 for j in a)
    b = rt.build_jobs("B", 60, st)
    assert [j["name"] for j in b] == ["b_tau0.07", "b_tau0.2", "b_m0.1", "b_m0.3", "b_r1", "b_r3"]
    assert b[0]["params"] == {"tau": 0.07, "m": 0.2, "r": 2, "lambda_a": 0.02}  # one parameter varied
    st2 = dict(st, lambda_a=0.05, tau=0.2, m=0.3, r=1)
    c = rt.build_jobs("C", 60, st2)
    assert [j["name"] for j in c] == ["c_beta0.25", "c_beta0.5", "c_beta1"]
    assert c[0]["arm"] == "v5" and c[0]["params"]["tau"] == 0.2 and c[0]["params"]["lambda_a"] == 0.05


def test_length_validation():
    for bad in (50, 65, 0):
        with pytest.raises(ValueError):
            rt.build_jobs("A", bad, rt.DEFAULT_STATE)
    with pytest.raises(ValueError):
        rt.build_jobs("Z", 60, rt.DEFAULT_STATE)


def test_shards_partition_the_jobs():
    jobs = rt.build_jobs("B", 60, rt.DEFAULT_STATE)
    s1, s2 = rt.shard_jobs(jobs, "1/2"), rt.shard_jobs(jobs, "2/2")
    assert sorted(j["name"] for j in s1 + s2) == sorted(j["name"] for j in jobs)
    assert not {j["name"] for j in s1} & {j["name"] for j in s2}
    with pytest.raises(ValueError):
        rt.shard_jobs(jobs, "3/2")


def test_lambda_grid_from_calibration(work_dir):
    f = work_dir / "calib.json"
    f.write_text(json.dumps({"lambda_for_target_ratio_b": {"0.05": 0.01307, "0.1": 0.02615, "0.2": None}}))
    assert rt.lambda_grid_from_calibration(str(f)) == [0.013, 0.02, 0.026]


# ----------------------------------------------------------------------------- command lines

def test_commands_are_accepted_by_the_runner_parser(work_dir):
    st = dict(rt.DEFAULT_STATE, lambda_a=0.05, tau=0.2, m=0.3, r=1, beta=1.0)
    runs, drive = str(work_dir / "runs"), str(work_dir / "drive")
    wck = "W.pth"
    for stage in ("warmup", "ref", "A", "B", "C"):
        for job in rt.build_jobs(stage, 60, st):
            cmd = rt.job_command(job, 60, runs, drive, wck, python="py")
            assert cmd[:2] == ["py", "run_cba_v2.py"]
            args = build_parser().parse_args(cmd[2:])
            assert resolve_arm_name(args.arm) == rt.ARM_DIRS[job["arm"]]
            assert args.epochs == job["epochs"] and args.sched_epochs == 60
            assert args.list_path.endswith("train_tune.txt") and args.test_dir.endswith("val_vol_h5")
            assert args.test_list.endswith("val_tune_vol.txt")
            assert args.save_final_preds is False
            assert args.drive_sync_dir == str(Path(drive) / job["name"])
            for k, flag_dest in (("lambda_a", "lambda_a"), ("tau", "tau"), ("m", "m"), ("r", "r"), ("beta", "beta")):
                if k in job["params"]:
                    assert getattr(args, flag_dest) == pytest.approx(job["params"][k])
    # warmup evaluates only at its final epoch; forks every 10 epochs
    w = build_parser().parse_args(rt.job_command(rt.build_jobs("warmup", 60, st)[0], 60, runs, drive, None)[2:])
    assert w.eval_freq == 0
    f = build_parser().parse_args(rt.job_command(rt.build_jobs("A", 60, st)[0], 60, runs, drive, wck)[2:])
    assert f.eval_freq == 10 and f.resume == wck


def test_fork_resumes_from_own_checkpoint_when_present(work_dir):
    runs, drive = str(work_dir / "runs"), str(work_dir / "drive")
    job = rt.build_jobs("A", 60, rt.DEFAULT_STATE)[0]
    cmd = rt.job_command(job, 60, runs, drive, "W.pth")
    assert cmd[cmd.index("--resume") + 1] == "W.pth"
    touch(work_dir / "drive" / job["name"] / "latest_checkpoint.pth")  # e.g. restored after a Colab disconnect
    cmd = rt.job_command(job, 60, runs, drive, "W.pth")
    assert cmd[cmd.index("--resume") + 1] == "auto"


def test_done_and_warmup_discovery(work_dir):
    runs, drive = str(work_dir / "runs"), str(work_dir / "drive")
    job = rt.build_jobs("A", 60, rt.DEFAULT_STATE)[0]
    assert not rt.is_done(job, runs, drive)
    touch(work_dir / "drive" / job["name"] / "eval_final.json")
    assert rt.is_done(job, runs, drive)
    assert rt.find_warmup_ckpt(runs, drive) is None
    assert rt.wait_for_warmup(runs, drive, wait_min=0) is None
    touch(work_dir / "drive" / "warmup" / "checkpoint_epoch_020.pth")
    assert rt.find_warmup_ckpt(runs, drive).endswith("checkpoint_epoch_020.pth")


def test_run_dry_run_prints_commands_and_skips_finished(work_dir, capsys):
    runs, drive = work_dir / "runs", work_dir / "drive"
    touch(drive / "warmup" / "checkpoint_epoch_020.pth")
    touch(drive / "a_lam0.02" / "eval_final.json")
    rt.main(["run", "--stage", "A", "--L", "60", "--runs_dir", str(runs), "--drive_dir", str(drive),
             "--lambda_grid", "0.05", "--dry_run"])
    out = capsys.readouterr().out
    assert "[skip] a_lam0.02" in out and "a_lam0.05" in out and "--sched_epochs 60" in out


# ----------------------------------------------------------------------------- plan

def test_plan_cost_arithmetic():
    c = rt.plan_cost(epoch_sec=100.0, eval_sec=60.0, L=60)
    fork_h = (40 * 100 + 4 * 60) / 3600
    assert c["fork_h_each"] == pytest.approx(fork_h)
    assert c["n_fork_runs"] == 1 + 3 + 6 + 3
    assert c["total_h"] == pytest.approx((20 * 100 + 60) / 3600 + 13 * fork_h)
    # KIS-only epochs (warmup + reference) are cheaper than CBA epochs
    c2 = rt.plan_cost(epoch_sec=100.0, eval_sec=60.0, L=60, kis_epoch_sec=60.0)
    ref_h = (40 * 60 + 4 * 60) / 3600
    assert c2["stages"]["ref_kis"] == pytest.approx(ref_h)
    assert c2["warmup_h"] == pytest.approx((20 * 60 + 60) / 3600)
    assert c2["total_h"] == pytest.approx(c2["warmup_h"] + ref_h + 12 * fork_h)


# ----------------------------------------------------------------------------- scoring + state

def build_stage_a_dirs(runs: Path):
    write_eval(runs / "ref_kis" / "Arm_V1_KIS", 50)
    write_eval(runs / "ref_kis" / "Arm_V1_KIS", 60)
    for name, hs in (("a_lam0.01", -0.5), ("a_lam0.02", -1.0), ("a_lam0.05", -4.0)):
        for ep in (50, 60):
            write_eval(runs / name / "Arm_V2_Boundary", ep, h_shift=hs)


def test_score_stage_a_updates_lambda(work_dir):
    runs = work_dir / "runs"
    build_stage_a_dirs(runs)
    out = rt.score_stage("A", 60, str(runs), None, str(work_dir / "state.json"), lambda_grid=[0.01, 0.05], n_boot=100)
    assert out["state"]["lambda_a"] == 0.05
    assert json.load(open(work_dir / "state.json", encoding="utf-8"))["lambda_a"] == 0.05
    assert out["report"]["A"]["decision"] == "a_lam0.05"


def test_score_stage_a_keeps_default_when_nothing_beats_it(work_dir):
    runs = work_dir / "runs"
    write_eval(runs / "ref_kis" / "Arm_V1_KIS", 50)
    write_eval(runs / "ref_kis" / "Arm_V1_KIS", 60)
    for name, hs in (("a_lam0.01", 0.5), ("a_lam0.02", 0.0), ("a_lam0.05", 1.0)):
        for ep in (50, 60):
            write_eval(runs / name / "Arm_V2_Boundary", ep, h_shift=hs)
    out = rt.score_stage("A", 60, str(runs), None, str(work_dir / "state.json"), lambda_grid=[0.01, 0.05], n_boot=50)
    assert out["state"]["lambda_a"] == 0.02


def test_score_stage_b_selects_each_parameter_independently(work_dir):
    runs = work_dir / "runs"
    build_stage_a_dirs(runs)
    state_path = work_dir / "state.json"
    rt.score_stage("A", 60, str(runs), None, str(state_path), lambda_grid=[0.01, 0.05], n_boot=50)  # lambda=0.05
    shifts = {"b_tau0.07": -9.0, "b_tau0.2": 2.0,       # tau 0.07 clearly better than the default (-4)
              "b_m0.1": 3.0, "b_m0.3": 4.0,             # m: both worse -> default
              "b_r1": -4.0, "b_r3": -4.0}               # r: equal to default -> default
    for name, hs in shifts.items():
        for ep in (50, 60):
            write_eval(runs / name / "Arm_V2_Boundary", ep, h_shift=hs)
    out = rt.score_stage("B", 60, str(runs), None, str(state_path), n_boot=100)
    st = out["state"]
    assert (st["tau"], st["m"], st["r"]) == (0.07, 0.2, 2)
    assert st["lambda_a"] == 0.05


def test_score_stage_c_and_missing_runs(work_dir):
    runs = work_dir / "runs"
    state_path = work_dir / "state.json"
    write_eval(runs / "ref_kis" / "Arm_V1_KIS", 50)
    write_eval(runs / "ref_kis" / "Arm_V1_KIS", 60)
    with pytest.raises(FileNotFoundError):
        rt.score_stage("C", 60, str(runs), None, str(state_path), n_boot=20)
    for name, hs in (("c_beta0.25", -1.0), ("c_beta0.5", -2.0), ("c_beta1", -6.0)):
        for ep in (50, 60):
            write_eval(runs / name / "Arm_V5_Dual_CEHard", ep, h_shift=hs)
    out = rt.score_stage("C", 60, str(runs), None, str(state_path), n_boot=100)
    assert out["state"]["beta"] == 1.0
