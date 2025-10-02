"""
Plot 7: Score Distributions & Trends
Left: box plots of specificity/on-topic scores per system. Right: average trend by step.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quality_plot_utils import load_quality_data


def build_plot(step_df, output_path: Path) -> None:
    valid = step_df.dropna(subset=["specificity_score", "on_topic_score"])
    if valid.empty:
        print("No score data available.")
        return

    systems = sorted(valid["system"].unique())

    fig, (ax_box, ax_trend) = plt.subplots(1, 2, figsize=(14, 6))

    # Box plots
    data_spec = [valid[valid["system"] == system]["specificity_score"].dropna().values for system in systems]
    data_topic = [valid[valid["system"] == system]["on_topic_score"].dropna().values for system in systems]

    positions = np.arange(len(systems))
    box1 = ax_box.boxplot(data_spec, positions=positions - 0.15, widths=0.25, patch_artist=True, showfliers=False)
    box2 = ax_box.boxplot(data_topic, positions=positions + 0.15, widths=0.25, patch_artist=True, showfliers=False)

    for box in box1['boxes']:
        box.set_facecolor('#4c72b0')
        box.set_alpha(0.7)
    for box in box2['boxes']:
        box.set_facecolor('#c44e52')
        box.set_alpha(0.7)

    for component in ['medians']:
        for line in box1.get(component, []):
            line.set_color('black')
        for line in box2.get(component, []):
            line.set_color('black')

    ax_box.set_xticks(positions)
    ax_box.set_xticklabels(systems, rotation=20, ha='right')
    ax_box.set_ylim(0, 1)
    ax_box.set_ylabel('Score distribution')
    ax_box.set_title('Score distributions per system')
    ax_box.grid(True, axis='y', alpha=0.3)
    ax_box.legend([box1['boxes'][0], box2['boxes'][0]], ['Specificity', 'On-topic'], loc='upper right')

    # Trend lines (aggregate across systems)
    grouped = valid.groupby('step')[['specificity_score', 'on_topic_score']].mean()
    steps = grouped.index.values
    ax_trend.plot(steps, grouped['specificity_score'], marker='o', color='#4c72b0', label='Specificity')
    ax_trend.plot(steps, grouped['on_topic_score'], marker='s', color='#c44e52', label='On-topic')
    ax_trend.set_ylim(0, 1)
    ax_trend.set_xlabel('Planner step')
    ax_trend.set_ylabel('Average score')
    ax_trend.set_title('Average score trend by step')
    ax_trend.set_xticks(steps)
    ax_trend.grid(True, alpha=0.3)
    ax_trend.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()


def main():
    step_df, _run_df, _output_dir, _csv_dir = load_quality_data()
    plot_dir = Path(__file__).resolve().parent
    output_path = plot_dir / "score_distribution_trends.png"
    build_plot(step_df, output_path)


if __name__ == "__main__":
    main()
