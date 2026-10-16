"""KIS baseline, 300 epochs, all 18 training cases, evaluated on the 12 official test volumes.

    python scripts/run_main_kis.py --runs_dir /content/runs/main --drive_dir /content/drive/MyDrive/IAD/main

Creates the shared 20-epoch warmup checkpoint when it is missing, then continues to --epochs.
Re-running the same command resumes an interrupted run. See scripts/main_common.py for details.
"""
from main_common import run_main

if __name__ == "__main__":
    run_main("kis")
