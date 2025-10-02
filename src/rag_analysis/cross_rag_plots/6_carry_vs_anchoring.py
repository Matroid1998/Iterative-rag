"""
Plot 6: Carry-drop vs query anchoring correlation per planner step.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cross_plot_utils import load_cross_data


def build_plot(coverage_step_df, quality_step_df, output_path: Path) -> None:
    if coverage_step_df.empty or quality_step_df.empty:
        print("Need both coverage and quality step data.")
        return

    cov_steps = coverage_step_df.copy()
    cov_steps["carry_drop"] = cov_steps["carry_drop"].fillna(False).astype(bool)

    qual = quality_step_df[['system', 'question', 'step', 'query_anchored']].copy()
    qual['query_anchored'] = qual['query_anchored'].fillna(False).astype(bool)

    merged = cov_steps.merge(qual, on=['system', 'question', 'step'], how='inner')
    if merged.empty:
        print("No overlapping steps between coverage and quality.")
        return

    grouped = merged.groupby('step')[['carry_drop', 'query_anchored']].mean()
    steps = grouped.index.values

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, grouped['carry_drop'], marker='o', color='#c44e52', label='Carry-drop rate')
    ax.plot(steps, grouped['query_anchored'], marker='s', color='#4c72b0', label='Anchored rate')

    ax.set_xlabel('Planner step')
    ax.set_ylabel('Share of steps')
    ax.set_ylim(0, 1)
    ax.set_xticks(steps)
    ax.set_title('Carry-drop vs anchoring by planner step', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()


def main():
    _run_df, _coverage_df, coverage_step_df, quality_step_df, _hall_df, _late_df = load_cross_data()
    plot_dir = Path(__file__).resolve().parent
    output_path = plot_dir / "carry_vs_anchoring.png"
    build_plot(coverage_step_df, quality_step_df, output_path)


if __name__ == "__main__":
    main()
