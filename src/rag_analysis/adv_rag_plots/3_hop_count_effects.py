"""
Plot 3: Hop count effects on key failure rates.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adv_plot_utils import load_all_data

METRICS = [
    ("is_miscalibrated", "Miscalibration rate"),
    ("any_late_hit", "Late hit rate"),
    ("composition_failure", "Composition failure rate"),
]


def build_plot(run_df, hall_df, output_path: Path) -> None:
    if run_df.empty and hall_df.empty:
        print("No data available for hop analysis.")
        return
    df = run_df.merge(hall_df[["system", "question", "number_of_hops", "is_miscalibrated", "composition_failure"]], on=["system", "question", "number_of_hops"], how='outer')
    if df["number_of_hops"].isna().all():
        print("No hop counts present.")
        return
    df["number_of_hops"] = df["number_of_hops"].fillna(method='ffill').fillna(method='bfill')
    df = df.dropna(subset=["number_of_hops"])
    df["number_of_hops"] = df["number_of_hops"].astype(int)

    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4), sharey=False)
    if len(METRICS) == 1:
        axes = [axes]

    for ax, (column, title) in zip(axes, METRICS):
        if column not in df.columns:
            ax.set_visible(False)
            continue
        grouped = df.groupby("number_of_hops")[column].mean()
        ax.plot(grouped.index, grouped.values, marker='o', color='#4c72b0')
        ax.set_title(title)
        ax.set_xlabel('Number of hops')
        ax.set_ylabel('Rate')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Failure rates vs question hop count', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()


def main():
    run_df, _coverage_df, _coverage_step_df, _quality_step_df, hall_df, _late_df = load_all_data()
    plot_dir = Path(__file__).resolve().parent
    output_path = plot_dir / "hop_count_effects.png"
    build_plot(run_df, hall_df, output_path)


if __name__ == "__main__":
    main()
