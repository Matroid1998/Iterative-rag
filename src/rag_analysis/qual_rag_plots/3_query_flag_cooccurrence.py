"""
Plot 3: Query Flag Co-occurrence Matrix
Heatmap visualising how often query-quality flags appear together.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quality_plot_utils import load_quality_data

FLAG_COLUMNS = [
    ("query_vague", "Vague"),
    ("query_over_broad", "Over-broad"),
    ("query_compound", "Compound"),
    ("query_off_topic", "Off-topic"),
]


def build_matrix(step_df):
    if step_df.empty:
        return None
    df = step_df.copy()
    for col, _ in FLAG_COLUMNS:
        df[col] = df[col].fillna(False).astype(bool)
    matrix = np.zeros((len(FLAG_COLUMNS), len(FLAG_COLUMNS)))

    for i, (col_i, _) in enumerate(FLAG_COLUMNS):
        for j, (col_j, _) in enumerate(FLAG_COLUMNS):
            if i <= j:
                mask = df[col_i] & df[col_j]
            else:
                mask = df[col_j] & df[col_i]
            matrix[i, j] = mask.mean() if len(df) else 0.0
            matrix[j, i] = matrix[i, j]
    return matrix


def plot_heatmap(matrix: np.ndarray, output_path: Path) -> None:
    if matrix is None:
        print("No data for co-occurrence heatmap.")
        return
    labels = [label for _, label in FLAG_COLUMNS]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=matrix.max() if matrix.max() > 0 else 1)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Query flag co-occurrence (share of steps)", fontsize=16, fontweight='bold', pad=15)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)
    cbar.ax.set_ylabel("Proportion of steps", rotation=-90, va="bottom")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved heatmap to {output_path}")
    plt.close()


def main():
    step_df, _run_df, _output_dir, _csv_dir = load_quality_data()
    matrix = build_matrix(step_df)
    plot_dir = Path(__file__).resolve().parent
    output_path = plot_dir / "query_flag_cooccurrence.png"
    plot_heatmap(matrix, output_path)


if __name__ == "__main__":
    main()
