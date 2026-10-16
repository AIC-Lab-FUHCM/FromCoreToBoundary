"""IAD (200 epochs KIS + 100 epochs KAS = 300 epochs total), evaluated on the 12 official test volumes at epoch 300.

    python scripts/run_main_iad.py --runs_dir /content/runs/main --drive_dir /content/drive/MyDrive/IAD/main

Stage 1 (epochs 1..200) trains with KIS (0.8 * L_SGS + 0.2 * L_KIS, flat CPD weights).
Stage 2 (epochs 201..300) trains with KAS (L_SGS + 0.1 * (L_RFC + L_CSS)) using registered KasHead parameters.
Re-running the same command resumes an interrupted run or skips a finished run.
"""
from main_common import run_main

if __name__ == "__main__":
    run_main("iad")
