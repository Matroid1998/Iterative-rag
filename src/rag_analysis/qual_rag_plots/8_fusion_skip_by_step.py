"""
Plot 8: Fusion/Skip activation by planner step.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from quality_plot_utils import load_quality_data


def build_plot(step_df, output_path: Path) -> None:
    if step_df.empty:
        print("No step data available.")
        return
    df = step_df.dropna(subset=["fusion_or_skip"]).copy()
    if df.empty:
        print("No fusion/skip annotations found.")
        return
    df["fusion_or_skip"] = df["fusion_or_skip"].astype(float)
    aggregated = df.groupby("step")["fusion_or_skip"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(aggregated["step"], aggregated["fusion_or_skip"], color="#c44e52", alpha=0.8)
    ax.set_xlabel('Planner step')
    ax.set_ylabel('Share of steps with fusion/skip')
    ax.set_ylim(0, 1)
    ax.set_title('Fusion / skip behaviour by planner step', fontsize=16, fontweight='bold')
    ax.set_xticks(aggregated["step"].tolist())
    for x, y in zip(aggregated["step"], aggregated["fusion_or_skip"]):
        ax.text(x, y + 0.02, f"{y:.2f}", ha='center', va='bottom', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()


def main():
    step_df, _run_df, _output_dir, _csv_dir = load_quality_data()
    plot_dir = Path(__file__).resolve().parent
    output_path = plot_dir / "fusion_skip_by_step.png"
    build_plot(step_df, output_path)


if __name__ == "__main__":
    main()
