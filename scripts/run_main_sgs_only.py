"""SGS-Only (Supervised Ground-Truth Segmentation without teacher KD), 300 epochs.

    python scripts/run_main_sgs_only.py --runs_dir /content/runs/main --drive_dir /content/drive/MyDrive/IAD/main

Re-running the same command resumes an interrupted run or skips a finished run.
"""
from main_common import run_main

if __name__ == "__main__":
    run_main("sgs_only")
