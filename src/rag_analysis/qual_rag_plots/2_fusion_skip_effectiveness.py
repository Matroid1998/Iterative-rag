"""
Plot 2: Fusion/Skip Effectiveness
Box plot comparing run accuracy for traces that used fusion/skip vs those that did not, grouped by hop count.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quality_plot_utils import load_quality_data


def build_plot(run_df, output_path: Path) -> None:
    df = run_df.dropna(subset=["is_correct"])
    if df.empty:
        print("No accuracy-labelled runs found. Skipping plot.")
        return

    df = df.copy()
    df["accuracy"] = df["is_correct"].astype(float)
    hop_groups = sorted(df["number_of_hops"].unique())

    data = []
    labels = []
    colors = []
    positions = []
    base_pos = 1
    spacing = 1.5

    for hop in hop_groups:
        sub = df[df["number_of_hops"] == hop]
        with_fusion = sub[sub["has_fusion"]]["accuracy"].values
        without_fusion = sub[~sub["has_fusion"]]["accuracy"].values

        if len(without_fusion) > 0:
            data.append(without_fusion)
            labels.append(f"Hop {hop}\n(no fusion)")
            positions.append(base_pos)
            colors.append("#4c72b0")
        if len(with_fusion) > 0:
            data.append(with_fusion)
            labels.append(f"Hop {hop}\n(fusion)")
            positions.append(base_pos + 0.5)
            colors.append("#c44e52")
        base_pos += spacing

    if not data:
        print("No data available for box plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    box = ax.boxplot(data, positions=positions, widths=0.35, patch_artist=True, showfliers=False)

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for component in ['medians', 'means']:
        if component in box:
            for line in box[component]:
                line.set_color('black')
                line.set_linewidth(1.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel('Accuracy (1 = correct, 0 = incorrect)', fontsize=12)
    ax.set_title('Fusion / skip effectiveness by question complexity', fontsize=16, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()

    # Summary stats
    print("\nFusion effectiveness summary:")
    for hop in hop_groups:
        sub = df[df["number_of_hops"] == hop]
        no_fusion = sub[~sub["has_fusion"]]["accuracy"]
        with_fusion = sub[sub["has_fusion"]]["accuracy"]
        if len(no_fusion) == 0 and len(with_fusion) == 0:
            continue
        print(f"\nHop {hop}:")
        if len(no_fusion) > 0:
            print(f"  No fusion: mean={no_fusion.mean():.3f}, n={len(no_fusion)}")
        if len(with_fusion) > 0:
            print(f"  Fusion:    mean={with_fusion.mean():.3f}, n={len(with_fusion)}")


def main():
    _step_df, run_df, _output_dir, _csv_dir = load_quality_data()
    plot_dir = Path(__file__).resolve().parent
    output_path = plot_dir / "fusion_skip_effectiveness.png"
    build_plot(run_df, output_path)


if __name__ == "__main__":
    main()
