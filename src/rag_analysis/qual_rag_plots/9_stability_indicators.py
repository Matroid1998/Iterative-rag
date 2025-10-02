"""
Plot 9: Stability Indicators
Grouped bars for partial contradictions and distractor latch rates per system.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quality_plot_utils import load_quality_data


def build_plot(run_df, output_path: Path) -> None:
    if run_df.empty:
        print("No run-level data available.")
        return

    df = run_df.copy()
    df["any_partial_contradiction"] = df["any_partial_contradiction"].fillna(False).astype(bool)
    df["distractor_latch"] = df["distractor_latch"].fillna(False).astype(bool)

    grouped = df.groupby("system")[["any_partial_contradiction", "distractor_latch"]].mean()
    if grouped.empty:
        print("No stability metrics available.")
        return

    systems = grouped.index.tolist()
    rates1 = grouped["any_partial_contradiction"].values
    rates2 = grouped["distractor_latch"].values

    x = np.arange(len(systems))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, rates1, width, label='Partial contradiction', color='#4c72b0', alpha=0.85)
    ax.bar(x + width/2, rates2, width, label='Distractor latch', color='#c44e52', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=20, ha='right')
    ax.set_ylabel('Proportion of runs')
    ax.set_ylim(0, 1)
    ax.set_title('Stability indicators by system', fontsize=16, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='upper right')

    for xi, val in zip(x - width/2, rates1):
        ax.text(xi, val + 0.01, f"{val:.2f}", ha='center', va='bottom', fontsize=10)
    for xi, val in zip(x + width/2, rates2):
        ax.text(xi, val + 0.01, f"{val:.2f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()


def main():
    _step_df, run_df, _output_dir, _csv_dir = load_quality_data()
    plot_dir = Path(__file__).resolve().parent
    output_path = plot_dir / "stability_indicators.png"
    build_plot(run_df, output_path)


if __name__ == "__main__":
    main()
