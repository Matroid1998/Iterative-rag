"""
Plot 5: Step Alignment Metrics
Panel A: per-step is_next_logical_hop rates by system.
Panel B: overall alignment vs exact step=predicted-hop match per system.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quality_plot_utils import load_quality_data


def build_plot(step_df, output_path: Path) -> None:
    if step_df.empty:
        print("No step-level records available.")
        return

    step_df = step_df.copy()
    step_df["is_next_logical_hop"] = step_df["is_next_logical_hop"].astype(float)
    step_df["predicted_equals_step"] = (
        (step_df["predicted_hop"].notna()) & (step_df["predicted_hop"] == step_df["step"])
    ).astype(float)

    # Panel A data
    per_step = (
        step_df.dropna(subset=["is_next_logical_hop"])
        .groupby(["system", "step"])["is_next_logical_hop"]
        .mean()
        .reset_index()
    )
    systems = sorted(per_step["system"].unique())
    steps = sorted(per_step["step"].unique())

    # Panel B data
    overall = step_df.groupby("system")["is_next_logical_hop"].mean()
    exact = step_df.groupby("system")["predicted_equals_step"].mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [2, 1]})

    # Panel A: grouped bars by step
    total_width = 0.7
    bar_width = total_width / max(len(systems), 1)
    x_ticks = np.arange(len(steps))

    for idx, system in enumerate(systems):
        offsets = x_ticks - total_width/2 + idx * bar_width + bar_width/2
        system_vals = [
            per_step[(per_step["system"] == system) & (per_step["step"] == step)]["is_next_logical_hop"].mean()
            if not per_step[(per_step["system"] == system) & (per_step["step"] == step)].empty else 0.0
            for step in steps
        ]
        ax1.bar(offsets, system_vals, width=bar_width, label=system)

    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([f"Step {s}" for s in steps])
    ax1.set_ylabel("Alignment rate", fontsize=12)
    ax1.set_title("Per-step logical alignment by system", fontsize=16, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)

    # Panel B: overall vs exact
    bar_positions = np.arange(len(systems))
    bar_width2 = 0.35
    ax2.bar(bar_positions - bar_width2/2, [overall.get(sys, 0.0) for sys in systems],
            width=bar_width2, label='Standard alignment', color='#4c72b0', alpha=0.8)
    ax2.bar(bar_positions + bar_width2/2, [exact.get(sys, 0.0) for sys in systems],
            width=bar_width2, label='Exact step match', color='#c44e52', alpha=0.8)

    ax2.set_xticks(bar_positions)
    ax2.set_xticklabels(systems, rotation=20, ha='right')
    ax2.set_ylabel('Rate', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_title('Overall alignment vs exact step match', fontsize=14)
    ax2.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()


def main():
    step_df, _run_df, _output_dir, _csv_dir = load_quality_data()
    plot_dir = Path(__file__).resolve().parent
    output_path = plot_dir / "step_alignment_metrics.png"
    build_plot(step_df, output_path)


if __name__ == "__main__":
    main()
