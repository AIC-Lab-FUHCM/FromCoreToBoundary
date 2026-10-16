"""Root entry point for the CBA v2 experiment (implementation: experiments/cba_v2/train.py)."""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent if _HERE.name == "scripts" else _HERE
for _p in (REPO_ROOT, REPO_ROOT / "models", REPO_ROOT / "scripts", REPO_ROOT / "scripts" / "tools"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.cba_v2.train import main  # noqa: E402

if __name__ == "__main__":
    main()
