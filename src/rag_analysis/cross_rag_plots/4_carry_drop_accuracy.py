"""
Plot 4: Anchor Carry-Drop Impact on Accuracy.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cross_plot_utils import load_cross_data


def build_plot(run_df, output_path: Path) -> None:
    if run_df.empty:
        print("No run data available for carry-drop analysis.")
        return
    df = run_df.dropna(subset=["any_carry_drop", "is_correct"]).copy()
    if df.empty:
        print("Carry-drop data unavailable.")
        return
    df["is_correct"] = df["is_correct"].astype(float)
    grouped = df.groupby(["system", "any_carry_drop"])["is_correct"].mean().unstack(fill_value=np.nan)

    systems = grouped.index.tolist()
    x = np.arange(len(systems))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    no_drop = grouped.get(False, pd.Series(index=systems, dtype=float))
    with_drop = grouped.get(True, pd.Series(index=systems, dtype=float))

    ax.bar(x - width/2, no_drop.values, width, label='No carry drop', color='#55a868', alpha=0.85)
    ax.bar(x + width/2, with_drop.values, width, label='Carry drop', color='#c44e52', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=20, ha='right')
    ax.set_ylabel('Accuracy (share correct)')
    ax.set_ylim(0, 1)
    ax.set_title('Impact of anchor carry-drop on accuracy', fontsize=16, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='lower left')

    for xi, val in zip(x - width/2, no_drop.values):
        if np.isfinite(val):
            ax.text(xi, val + 0.01, f"{val:.2f}", ha='center', va='bottom', fontsize=10)
    for xi, val in zip(x + width/2, with_drop.values):
        if np.isfinite(val):
            ax.text(xi, val + 0.01, f"{val:.2f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()


def main():
    run_df, _cov_df, _cov_step_df, _quality_step_df, _hall_df, _late_df = load_cross_data()
    plot_dir = Path(__file__).resolve().parent
    output_path = plot_dir / "carry_drop_accuracy.png"
    build_plot(run_df, output_path)


if __name__ == "__main__":
    main()
