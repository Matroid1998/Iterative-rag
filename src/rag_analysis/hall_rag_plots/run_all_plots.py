"""Run the entire hallucination plotting suite."""
from __future__ import annotations

import subprocess
from pathlib import Path

PLOTS = [
    "1_miscalibration_by_hop.py",
    "2_sufficiency_vs_coverage.py",
    "3_unsupported_claims_distribution.py",
    "4_composition_failure_root_causes.py",
    "5_composition_failure_rate.py",
    "6_sufficiency_distribution.py",
    "7_miscalibration_mix.py",
    "8_coverage_vs_confidence.py",
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
