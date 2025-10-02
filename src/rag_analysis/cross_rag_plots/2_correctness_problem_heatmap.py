"""
Plot 2: Correctness vs Problem Type Heatmap.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cross_plot_utils import load_cross_data

try:
    import seaborn as sns
except ImportError:  # optional dependency
    sns = None  # type: ignore

PROBLEMS = [
    ("has_gap", "Coverage gap"),
    ("any_carry_drop", "Carry drop"),
    ("any_late_hit", "Late hit"),
    ("composition_failure", "Composition failure"),
    ("is_miscalibrated", "Miscalibration"),
]


def build_plot(run_df, output_path: Path) -> None:
    if run_df.empty:
        print("No run-level data available.")
        return

    df = run_df.copy()
    df["is_correct"] = df["is_correct"].astype('float')
    incorrect = df[df["is_correct"] == 0.0]
    if incorrect.empty:
        print("No incorrect runs to analyse.")
        return

    systems = sorted(incorrect["system"].dropna().unique())
    matrix = np.zeros((len(systems), len(PROBLEMS)))

    for i, system in enumerate(systems):
        subset = incorrect[incorrect["system"] == system]
        if subset.empty:
            continue
        for j, (column, _label) in enumerate(PROBLEMS):
            if column in subset.columns:
                matrix[i, j] = subset[column].fillna(False).astype(bool).mean()

    fig, ax = plt.subplots(figsize=(10, max(4, len(systems) * 0.6)))
    if sns is not None:
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlOrRd", xticklabels=[label for _, label in PROBLEMS], yticklabels=systems, vmin=0, vmax=1, ax=ax)
    else:
        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1)
        for (i_idx, j_idx), value in np.ndenumerate(matrix):
            ax.text(j_idx, i_idx, f"{value:.2f}", ha='center', va='center', color='black', fontsize=10)
        ax.set_xticks(np.arange(len(PROBLEMS)))
        ax.set_xticklabels([label for _, label in PROBLEMS], rotation=45, ha='right')
        ax.set_yticks(np.arange(len(systems)))
        ax.set_yticklabels(systems)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Share of incorrect runs', rotation=-90, va='bottom')

    ax.set_title('Failure-mode prevalence among incorrect answers', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()


def main():
    run_df, coverage_df, coverage_step_df, quality_step_df, hall_df, _late_df = load_cross_data()
    plot_dir = Path(__file__).resolve().parent
    output_path = plot_dir / "correctness_problem_heatmap.png"
    build_plot(run_df, output_path)


if __name__ == "__main__":
    main()
