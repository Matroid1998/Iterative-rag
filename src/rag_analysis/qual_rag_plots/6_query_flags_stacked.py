"""
Plot 6: Query Flag Composition per Model
Stacked bars of flag prevalence (vague / over-broad / compound / off-topic / anchored).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quality_plot_utils import load_quality_data

FLAG_ORDER = [
    ("query_vague", "Vague", "#c44e52"),
    ("query_over_broad", "Over-broad", "#dd8452"),
    ("query_compound", "Compound", "#55a868"),
    ("query_off_topic", "Off-topic", "#4c72b0"),
    ("query_anchored", "Anchored", "#8172b2"),
]


def build_plot(step_df, output_path: Path) -> None:
    if step_df.empty:
        print("No step data for flag analysis.")
        return

    systems = sorted(step_df["system"].unique())
    if not systems:
        print("No systems found.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(systems))
    heights = np.zeros(len(systems))

    for col, label, color in FLAG_ORDER:
        values = []
        for idx, system in enumerate(systems):
            subset = step_df[step_df["system"] == system][col]
            if subset.empty:
                rate = 0.0
            else:
                rate = subset.fillna(False).astype(bool).mean()
            values.append(rate)
        ax.bar(x, values, bottom=heights, label=label, color=color, alpha=0.85)
        heights += np.array(values)

    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=20, ha='right')
    ax.set_ylabel('Proportion of steps', fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('Query flag prevalence by system', fontsize=16, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()


def main():
    step_df, _run_df, _output_dir, _csv_dir = load_quality_data()
    plot_dir = Path(__file__).resolve().parent
    output_path = plot_dir / "query_flags_stacked.png"
    build_plot(step_df, output_path)


if __name__ == "__main__":
    main()
