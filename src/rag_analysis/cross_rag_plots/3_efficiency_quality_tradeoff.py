"""
Plot 3: Efficiency vs Quality Trade-off
Scatter with avg steps vs accuracy per model, marker size = avg specificity.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from cross_plot_utils import load_cross_data


def build_plot(run_df, quality_step_df, output_path: Path) -> None:
    if run_df.empty or quality_step_df.empty:
        print("Insufficient data for trade-off plot.")
        return

    # Average steps per run per system
    step_counts = quality_step_df.groupby(["system", "question"])["step"].max().reset_index(name="steps")
    avg_steps = step_counts.groupby("system")["steps"].mean()

    # Accuracy per system
    run_df = run_df.copy()
    run_df["is_correct"] = run_df["is_correct"].astype(float)
    accuracy = run_df.groupby("system")[["is_correct"]].mean()["is_correct"]

    # Average specificity per system
    specificity = quality_step_df.groupby("system")["specificity_score"].mean()

    systems = sorted(set(avg_steps.index) & set(accuracy.index))
    if not systems:
        print("No overlapping systems for trade-off plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    for system in systems:
        x = avg_steps.get(system, float('nan'))
        y = accuracy.get(system, float('nan'))
        size = specificity.get(system, 0.0)
        if not (float(x) == x and float(y) == y):
            continue
        ax.scatter(x, y, s=200 * max(size, 0.1), alpha=0.7, edgecolors='k', label=system)
        ax.text(x + 0.02, y + 0.01, system, fontsize=10)

    ax.set_xlabel('Average planner steps per run', fontsize=12)
    ax.set_ylabel('Accuracy (share correct)', fontsize=12)
    ax.set_title('Efficiency vs quality trade-off', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()


def main():
    run_df, _cov_df, _cov_step_df, quality_step_df, _hall_df, _late_df = load_cross_data()
    plot_dir = Path(__file__).resolve().parent
    output_path = plot_dir / "efficiency_quality_tradeoff.png"
    build_plot(run_df, quality_step_df, output_path)


if __name__ == "__main__":
    main()
