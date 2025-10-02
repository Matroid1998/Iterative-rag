"""
Plot 1: Query Degradation Over Steps
Multi-line chart showing specificity/on-topic score trends per planner step, faceted by system.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quality_plot_utils import load_quality_data


def build_plot(step_df, output_path: Path) -> None:
    if step_df.empty:
        print("No step-level data found. Skipping plot.")
        return

    systems = sorted(step_df["system"].unique())
    max_step = int(step_df["step"].max())

    cols = 2
    rows = math.ceil(len(systems) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows), sharey=True)
    axes = np.array(axes).reshape(rows, cols)

    for idx, system in enumerate(systems):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        subset = step_df[step_df["system"] == system]
        grouped = subset.groupby("step")[
            ["specificity_score", "on_topic_score"]
        ].mean().reindex(range(1, max_step + 1))

        ax.plot(grouped.index, grouped["specificity_score"], marker="o", color="#4c72b0", label="Specificity")
        ax.plot(grouped.index, grouped["on_topic_score"], marker="s", color="#c44e52", label="On-topic")

        ax.set_title(system, fontsize=14, fontweight="bold")
        ax.set_xlabel("Planner step", fontsize=12)
        ax.set_ylabel("Average score", fontsize=12)
        ax.set_xticks(range(1, max_step + 1))
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=10)

    # Hide unused axes
    for j in range(len(systems), rows * cols):
        r = j // cols
        c = j % cols
        axes[r, c].axis("off")

    fig.suptitle("Query score trends across planner steps", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved plot to {output_path}")
    plt.close()


def main():
    step_df, _run_df, _output_dir, _csv_dir = load_quality_data()
    plot_dir = Path(__file__).resolve().parent
    output_path = plot_dir / "query_degradation_over_steps.png"
    build_plot(step_df, output_path)


if __name__ == "__main__":
    main()
