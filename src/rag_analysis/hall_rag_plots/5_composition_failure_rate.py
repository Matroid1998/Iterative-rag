"""
Plot 5: Composition Failure Rate per Model.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from hall_plot_utils import load_hallucination_data


def build_plot(df, output_path: Path) -> None:
    if df.empty:
        print("No hallucination data.")
        return
    grouped = df.groupby("system")["composition_failure"].mean().sort_values(ascending=False)
    if grouped.empty:
        print("No composition failure labels present.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(grouped.index, grouped.values, color="#c44e52", alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Proportion of runs', fontsize=12)
    ax.set_title('Composition failure rate per model', fontsize=16, fontweight='bold')
    ax.set_xticklabels(grouped.index, rotation=20, ha='right')
    ax.grid(True, axis='y', alpha=0.3)

    for x, y in zip(range(len(grouped)), grouped.values):
        ax.text(x, y + 0.01, f"{y:.2f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()


def main():
    df, _output_dir, _csv_dir = load_hallucination_data()
    plot_dir = Path(__file__).resolve().parent
    output_path = plot_dir / "composition_failure_rate.png"
    build_plot(df, output_path)


if __name__ == "__main__":
    main()
