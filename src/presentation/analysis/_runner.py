from __future__ import annotations

from pathlib import Path
import runpy
import sys
import traceback
from typing import Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[3]


def ensure_repo_on_path() -> None:
    """Ensure repo root is on sys.path so src.* imports work."""
    repo_str = str(REPO_ROOT)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def run_script(path: Path, argv: List[str] | None = None) -> bool:
    """Run a Python script path in-process, returning True on success."""
    old_argv = sys.argv[:]
    script_dir = str(path.parent)
    added_path = False
    try:
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
            added_path = True
        sys.argv = argv or [str(path)]
        runpy.run_path(str(path), run_name="__main__")
        return True
    except SystemExit as exc:
        code = getattr(exc, "code", 0)
        if code not in (0, None):
            print(f"[runner] Script exited with code {code}: {path}")
            return False
        return True
    except Exception:  # noqa: BLE001
        print(f"[runner] Script failed: {path}")
        traceback.print_exc()
        return False
    finally:
        if added_path:
            try:
                sys.path.remove(script_dir)
            except ValueError:
                pass
        sys.argv = old_argv


def run_scripts(paths: Iterable[Path], label: str) -> None:
    """Run a sequence of scripts, raising SystemExit on failure."""
    ensure_repo_on_path()
    failures: List[Path] = []
    for script_path in paths:
        print(f"[{label}] Running {script_path}")
        if not run_script(script_path):
            failures.append(script_path)

    if failures:
        raise SystemExit(f"{label} failed: {len(failures)} script(s)")
