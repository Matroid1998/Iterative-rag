"""
Plot 6: Evidence Sufficiency Distribution
Histogram of sufficiency_score_est with 0.6 threshold line.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from hall_plot_utils import load_hallucination_data


def build_plot(df, output_path: Path) -> None:
    series = df["sufficiency_score_est"].dropna()
    if series.empty:
        print("No sufficiency scores available.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(series, bins=20, color="#55a868", alpha=0.8, edgecolor='black')
    ax.axvline(0.6, color='red', linestyle='--', linewidth=2, label='0.60 threshold')
    ax.set_xlabel('Sufficiency score estimate')
    ax.set_ylabel('Run count')
    ax.set_title('Evidence sufficiency distribution', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()


def main():
    df, _output_dir, _csv_dir = load_hallucination_data()
    plot_dir = Path(__file__).resolve().parent
    output_path = plot_dir / "sufficiency_distribution.png"
    build_plot(df, output_path)


if __name__ == "__main__":
    main()
