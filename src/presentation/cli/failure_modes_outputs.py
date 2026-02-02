from __future__ import annotations

from pathlib import Path

from src.presentation.analysis._runner import run_scripts


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    scripts = [
        repo_root / "src" / "service" / "failure_modes_analysis" / "hallucination_judgment.py",
        repo_root / "src" / "service" / "failure_modes_analysis" / "coverage_gap_judgments.py",
        repo_root / "src" / "service" / "failure_modes_analysis" / "quality_judgement.py",
    ]
    run_scripts(scripts, "failure_modes.outputs")


if __name__ == "__main__":
    main()
