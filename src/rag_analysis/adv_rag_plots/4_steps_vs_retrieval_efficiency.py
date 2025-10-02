"""
Plot 4: Steps-per-run distribution per model, with retrieval efficiency overlay.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adv_plot_utils import load_all_data


def compute_step_lengths(quality_step_df: pd.DataFrame) -> pd.DataFrame:
    if quality_step_df.empty:
        return pd.DataFrame(columns=["system", "question", "steps"])
    return quality_step_df.groupby(["system", "question"])["step"].max().reset_index(name="steps")


def compute_late_deltas(late_hit_df: pd.DataFrame) -> pd.Series:
    if late_hit_df.empty:
        return pd.Series(dtype=float)
    deltas = late_hit_df.copy()
    deltas["delta"] = deltas["first_hit_step"] - deltas["hop_index"]
    return deltas.groupby("system")["delta"].mean()


def build_plot(quality_step_df, late_hit_df, output_path: Path) -> None:
    lengths = compute_step_lengths(quality_step_df)
    if lengths.empty:
        print("No step-length data available.")
        return
    systems = sorted(lengths["system"].unique())
    late_means = compute_late_deltas(late_hit_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    data = [lengths[lengths["system"] == system]["steps"].values for system in systems]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b2', '#ccb974', '#64b5cd']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticklabels(systems, rotation=20, ha='right')
    ax.set_ylabel('Steps per run')
    ax.set_title('Planner depth distribution vs retrieval efficiency', fontsize=16, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    # Overlay retrieval efficiency (lower delta better)
    if not late_means.empty:
        ax2 = ax.twinx()
        ax2.plot(range(1, len(systems) + 1), [late_means.get(system, np.nan) for system in systems],
                 marker='o', color='#c44e52', linewidth=2, label='Avg (first_hit - hop)')
        ax2.set_ylabel('Average late-hit delta', color='#c44e52')
        ax2.tick_params(axis='y', labelcolor='#c44e52')
    else:
        ax.text(0.95, 0.9, 'No late-hit data', transform=ax.transAxes, ha='right', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()


def main():
    _run_df, _coverage_df, _coverage_step_df, quality_step_df, _hall_df, late_hit_df = load_all_data()
    plot_dir = Path(__file__).resolve().parent
    output_path = plot_dir / "steps_vs_retrieval_efficiency.png"
    build_plot(quality_step_df, late_hit_df, output_path)


if __name__ == "__main__":
    main()
