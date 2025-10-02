"""Run the full Quality (Query Audit) plotting suite."""
from __future__ import annotations

import subprocess
from pathlib import Path

PLOTS = [
    "1_query_degradation_over_steps.py",
    "2_fusion_skip_effectiveness.py",
    "3_query_flag_cooccurrence.py",
    "4_distractor_vs_accuracy.py",
    "5_step_alignment.py",
    "6_query_flags_stacked.py",
    "7_score_distribution_trends.py",
    "8_fusion_skip_by_step.py",
    "9_stability_indicators.py",
]


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    for script in PLOTS:
        path = script_dir / script
        if not path.exists():
            print(f"⚠️  Missing script {script}")
            continue
        print(f"\n=== Running {script} ===")
        result = subprocess.run(["python3", str(path)], cwd=script_dir)
        if result.returncode != 0:
            print(f"❌ {script} exited with code {result.returncode}")
        else:
            print(f"✅ Completed {script}")


if __name__ == "__main__":
    main()
