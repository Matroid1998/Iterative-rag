"""
Plot 1: Miscalibration Direction by Hop Count
Stacked bar chart of miscalibration direction counts across question complexities.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hall_plot_utils import load_hallucination_data

PALETTE = {
    "overconfident_finalize": "#c44e52",
    "underconfident_continue": "#55a868",
    "ok": "#4c72b0",
    "unknown": "#8172b2",
}


def build_plot(df, output_path: Path) -> None:
    if df.empty:
        print("No hallucination records found.")
        return
    counts = df.groupby(["number_of_hops", "direction"]).size().unstack(fill_value=0)
    counts = counts[[col for col in PALETTE if col in counts.columns]]

    complexities = counts.index.tolist()
    cumulative = np.zeros(len(complexities))

    fig, ax = plt.subplots(figsize=(10, 6))
    for direction, color in PALETTE.items():
        if direction not in counts.columns:
            continue
        values = counts[direction].values
        ax.bar(
            complexities,
            values,
            bottom=cumulative,
            color=color,
            label=direction,
            alpha=0.85,
        )
        cumulative += values

    totals = counts.sum(axis=1)
    for x, total in zip(complexities, totals):
        ax.text(x, total + max(totals) * 0.02, f"n={int(total)}", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Number of hops", fontsize=12)
    ax.set_ylabel("Run count", fontsize=12)
    ax.set_title("Miscalibration direction by question hop count", fontsize=16, fontweight="bold")
    ax.set_xticks(complexities)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Direction", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved plot to {output_path}")
    plt.close()


def main():
    df, _output_dir, _csv_dir = load_hallucination_data()
    plot_dir = Path(__file__).resolve().parent
    output_path = plot_dir / "miscalibration_by_hop.png"
    build_plot(df, output_path)


if __name__ == "__main__":
    main()
