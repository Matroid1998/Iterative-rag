"""Run the cross-system RAG plots."""
from __future__ import annotations

import subprocess
from pathlib import Path

PLOTS = [
    "1_error_cascade.py",
    "2_correctness_problem_heatmap.py",
    "3_efficiency_quality_tradeoff.py",
    "4_carry_drop_accuracy.py",
    "5_coverage_to_hallucination.py",
    "6_carry_vs_anchoring.py",
    "7_planning_vs_confidence.py",
]


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    for script in PLOTS:
        path = script_dir / script
        if not path.exists():
            print(f"⚠️  Missing {script}")
            continue
        print(f"\n=== Running {script} ===")
        result = subprocess.run(["python3", str(path)], cwd=script_dir)
        if result.returncode != 0:
            print(f"❌ {script} exited with code {result.returncode}")
        else:
            print(f"✅ Completed {script}")


if __name__ == "__main__":
    main()
