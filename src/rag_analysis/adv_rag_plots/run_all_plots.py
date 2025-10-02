"""Run all advanced plots."""
from __future__ import annotations

import subprocess
from pathlib import Path

PLOTS = [
    "1_stepwise_error_alluvial.py",
    "2_model_radar_profile.py",
    "3_hop_count_effects.py",
    "4_steps_vs_retrieval_efficiency.py",
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
