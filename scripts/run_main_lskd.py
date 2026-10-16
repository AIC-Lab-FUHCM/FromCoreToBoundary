"""LSKD (CVPR 2024), 300 epochs, all 18 training cases, evaluated on the 12 official test volumes at epoch 300.

    python scripts/run_main_lskd.py --runs_dir /content/runs/main --drive_dir /content/drive/MyDrive/IAD/main

Re-running the same command resumes an interrupted run or skips a finished run.
"""
from main_common import run_main

if __name__ == "__main__":
    run_main("lskd")
