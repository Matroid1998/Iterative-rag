"""
Plot 8: Coverage vs Confidence Scatter (coverage on X, sufficiency on Y).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from hall_plot_utils import PALETTE, load_hallucination_data


def build_plot(df, output_path: Path) -> None:
    usable = df.dropna(subset=["hop_coverage_est", "sufficiency_score_est"])
    if usable.empty:
        print("No data for coverage vs confidence plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for direction, color in PALETTE.items():
        subset = usable[usable["direction"] == direction]
        if subset.empty:
            continue
        ax.scatter(
            subset["hop_coverage_est"],
            subset["sufficiency_score_est"],
            s=60,
            color=color,
            alpha=0.65,
            edgecolors='k',
            linewidths=0.4,
            label=direction,
        )

    ax.set_xlabel('Hop coverage estimate', fontsize=12)
    ax.set_ylabel('Sufficiency score estimate', fontsize=12)
    ax.set_title('Coverage vs sufficiency by miscalibration direction', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(title='Direction', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()


def main():
    df, _output_dir, _csv_dir = load_hallucination_data()
    plot_dir = Path(__file__).resolve().parent
    output_path = plot_dir / "coverage_vs_confidence.png"
    build_plot(df, output_path)


if __name__ == "__main__":
    main()
