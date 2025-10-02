"""
Plot 7: Miscalibration Direction Mix per Model.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hall_plot_utils import PALETTE, load_hallucination_data


def build_plot(df, output_path: Path) -> None:
    if df.empty:
        print("No hallucination data.")
        return

    counts = df.groupby(["system", "direction"]).size().unstack(fill_value=0)
    systems = counts.index.tolist()
    x = np.arange(len(systems))
    cumulative = np.zeros(len(systems))

    fig, ax = plt.subplots(figsize=(10, 6))

    for direction, color in PALETTE.items():
        if direction not in counts.columns:
            continue
        values = counts[direction].values
        ax.bar(x, values, bottom=cumulative, color=color, label=direction, alpha=0.85)
        cumulative += values

    miscal = df.groupby("system")["is_miscalibrated"].mean()
    for xpos, system, total in zip(x, systems, cumulative):
        rate = miscal.get(system, float('nan'))
        if not np.isnan(rate):
            ax.text(xpos, total + max(cumulative) * 0.02, f"miscal={rate:.2f}", ha='center', va='bottom', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=20, ha='right')
    ax.set_ylabel('Run count')
    ax.set_title('Confidence direction mix per model', fontsize=16, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(title='Direction', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()


def main():
    df, _output_dir, _csv_dir = load_hallucination_data()
    plot_dir = Path(__file__).resolve().parent
    output_path = plot_dir / "miscalibration_mix.png"
    build_plot(df, output_path)


if __name__ == "__main__":
    main()
