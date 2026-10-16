"""Ours, 300 epochs, all 18 training cases, evaluated on the 12 official test volumes.

Default arm v5 = KIS + Boundary + Error-CE-hard; `--arm v2` runs KIS + Boundary instead.

    python scripts/run_main_ours.py --runs_dir /content/runs/main --drive_dir /content/drive/MyDrive/IAD/main

Method parameters come from the tuning state (Drive `tune/tune_state.json`, defaults when absent).
Waits for the shared 20-epoch warmup unless --make_warmup is given. Re-running the same command resumes an
interrupted run. See scripts/main_common.py for details.
"""
from main_common import run_main

if __name__ == "__main__":
    run_main("ours")
