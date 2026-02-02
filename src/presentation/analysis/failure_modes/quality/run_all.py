from __future__ import annotations

from pathlib import Path

from src.presentation.analysis._runner import run_scripts


def main() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    category = "quality"
    category_dir = repo_root / "src" / "service" / "plot_codes" / "failure_modes" / category
    plot_dir = repo_root / "data" / "plots" / "failure_modes" / category
    plot_dir.mkdir(parents=True, exist_ok=True)

    skip = {"__init__.py"}
    scripts = sorted(p for p in category_dir.glob("*.py") if p.name not in skip)
    run_scripts(scripts, f"failure_modes.{category}")


if __name__ == "__main__":
    main()
