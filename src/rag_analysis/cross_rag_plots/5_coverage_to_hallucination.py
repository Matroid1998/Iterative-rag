"""
Plot 5: Coverage issues vs composition failure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from cross_plot_utils import load_cross_data


def build_plot(run_df, output_path: Path) -> None:
    if run_df.empty:
        print("No joined run data available.")
        return
    df = run_df.copy()
    df['composition_failure'] = df['composition_failure'].fillna(False).astype(bool)
    df['has_gap'] = df['has_gap'].fillna(False).astype(bool)
    df['any_late_hit'] = df['any_late_hit'].fillna(False).astype(bool)

    def rate_for(mask):
        subset = df[mask]
        if subset.empty:
            return 0.0
        return subset['composition_failure'].mean()

    rates_gap = [
        rate_for(df['has_gap']),
        rate_for(~df['has_gap'])
    ]
    rates_late = [
        rate_for(df['any_late_hit']),
        rate_for(~df['any_late_hit'])
    ]

    labels = ['Has gap', 'No gap']
    labels_late = ['Late hit', 'No late hit']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    axes[0].bar(labels, rates_gap, color=['#c44e52', '#55a868'], alpha=0.85)
    axes[1].bar(labels_late, rates_late, color=['#c44e52', '#55a868'], alpha=0.85)

    for ax, rates in zip(axes, [rates_gap, rates_late]):
        for idx, val in enumerate(rates):
            ax.text(idx, val + 0.01, f"{val:.2f}", ha='center', va='bottom', fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylabel('Composition failure rate')

    axes[0].set_title('Conditioned on coverage gap')
    axes[1].set_title('Conditioned on late hit')
    fig.suptitle('Do coverage issues drive composition failures?', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()


def main():
    run_df, _cov_df, _cov_step_df, _quality_step_df, _hall_df, _late_df = load_cross_data()
    plot_dir = Path(__file__).resolve().parent
    output_path = plot_dir / "coverage_to_hallucination.png"
    build_plot(run_df, output_path)


if __name__ == "__main__":
    main()
