"""
Plot 7: Planning Alignment vs Confidence Calibration per model.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from cross_plot_utils import load_cross_data


def build_plot(quality_step_df, hall_df, output_path: Path) -> None:
    if quality_step_df.empty or hall_df.empty:
        print("Insufficient data for planning/confidence plot.")
        return

    alignment = quality_step_df.copy()
    alignment['is_next_logical_hop'] = alignment['is_next_logical_hop'].astype(float)
    alignment = alignment.groupby('system')['is_next_logical_hop'].mean()
    overconf = hall_df.groupby('system')['direction'].apply(lambda s: (s == 'overconfident_finalize').mean())

    systems = sorted(set(alignment.index) & set(overconf.index))
    if not systems:
        print("No overlapping systems between planning and confidence data.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    axes[0].bar(systems, [alignment.get(sys, 0.0) for sys in systems], color='#4c72b0', alpha=0.85)
    axes[0].set_title('Step alignment (is_next_logical_hop)', fontsize=14)
    axes[0].set_ylabel('Share of steps')
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, axis='y', alpha=0.3)
    axes[0].tick_params(axis='x', rotation=20)

    axes[1].bar(systems, [overconf.get(sys, 0.0) for sys in systems], color='#c44e52', alpha=0.85)
    axes[1].set_title('Overconfident finalize rate', fontsize=14)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, axis='y', alpha=0.3)
    axes[1].tick_params(axis='x', rotation=20)

    fig.suptitle('Planning alignment vs confidence calibration', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()


def main():
    _run_df, _coverage_df, _coverage_step_df, quality_step_df, hall_df, _late_df = load_cross_data()
    plot_dir = Path(__file__).resolve().parent
    output_path = plot_dir / "planning_vs_confidence.png"
    build_plot(quality_step_df, hall_df, output_path)


if __name__ == "__main__":
    main()
