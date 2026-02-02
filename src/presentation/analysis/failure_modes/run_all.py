from __future__ import annotations

from pathlib import Path

from src.presentation.analysis._runner import run_scripts


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    base_dir = repo_root / "src" / "presentation" / "analysis" / "failure_modes"

    scripts = [
        base_dir / "hallucination" / "run_all.py",
        base_dir / "quality" / "run_all.py",
        base_dir / "coverage_gap" / "run_all.py",
        base_dir / "cross_system" / "run_all.py",
        base_dir / "advanced" / "run_all.py",
    ]
    run_scripts(scripts, "failure_modes.all")


if __name__ == "__main__":
    main()
