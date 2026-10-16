import atexit
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT, _REPO_ROOT / "models", _REPO_ROOT / "scripts", _REPO_ROOT / "scripts" / "tools"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _cleanup_top_level_caches():
    runs_dir = _REPO_ROOT / "runs"
    if runs_dir.is_dir():
        for child in list(runs_dir.iterdir()):
            if child.is_dir() and child.name.startswith("test_tmp_"):
                try:
                    child.rmdir()
                except OSError:
                    pass
        try:
            runs_dir.rmdir()
        except OSError:
            pass
    for root in (_REPO_ROOT, _REPO_ROOT.parent):
        for cache_name in ("__pycache__", ".pytest_cache"):
            cache_dir = root / cache_name
            if cache_dir.is_dir():
                shutil.rmtree(cache_dir, ignore_errors=True)


atexit.register(_cleanup_top_level_caches)


def pytest_sessionfinish(session, exitstatus):
    _cleanup_top_level_caches()
