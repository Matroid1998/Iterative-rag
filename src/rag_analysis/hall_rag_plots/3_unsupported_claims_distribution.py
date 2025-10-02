"""
Plot 3: Unsupported Claims Distribution by Model.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt

from hall_plot_utils import load_hallucination_data


def build_plot(df, output_path: Path) -> None:
    if df.empty:
        print("No hallucination data found.")
        return

    systems = sorted(df["system"].unique())
    cols = 2
    rows = math.ceil(len(systems) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = axes.flatten()

    max_count = int(df["unsupported_claims_count"].max()) if not df["unsupported_claims_count"].isna().all() else 0
    bins = range(0, max_count + 2)

    for ax, system in zip(axes, systems):
        subset = df[df["system"] == system]["unsupported_claims_count"].fillna(0)
        if subset.empty:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            ax.axis('off')
            continue
        ax.hist(subset, bins=bins, color="#4c72b0", alpha=0.75, edgecolor='black')
        ax.set_title(system, fontsize=13)
        ax.set_xlabel("Unsupported claims count")
        ax.set_ylabel("Run count")
        ax.grid(True, alpha=0.3)

    # Hide leftover axes
    for ax in axes[len(systems):]:
        ax.axis('off')

    fig.suptitle("Unsupported claim distribution per model", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()


def main():
    df, _output_dir, _csv_dir = load_hallucination_data()
    plot_dir = Path(__file__).resolve().parent
    output_path = plot_dir / "unsupported_claims_distribution.png"
    build_plot(df, output_path)


if __name__ == "__main__":
    main()
