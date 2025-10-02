"""
Plot 4: Composition Failure Root Causes
Grouped bar showing overlap of composition failures with retrieval/query issues.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hall_plot_utils import load_hallucination_data

CAUSES = [
    ("coverage_gap", "Coverage gap"),
    ("carry_drop", "Carry drop"),
    ("late_hit", "Late hit"),
    ("poor_query_quality", "Poor query"),
]


def build_plot(df, output_path: Path) -> None:
    failures = df[df["composition_failure"]]
    if failures.empty:
        print("No composition failures detected.")
        return

    systems = sorted(failures["system"].unique())
    x = np.arange(len(CAUSES))
    width = 0.8 / max(len(systems), 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, system in enumerate(systems):
        subset = failures[failures["system"] == system]
        if subset.empty:
            continue
        values = []
        for col, _ in CAUSES:
            series = subset[col].fillna(False).astype(bool)
            values.append(series.mean() if not series.empty else 0.0)
        offsets = x - 0.4 + idx * width + width / 2
        ax.bar(offsets, values, width=width, label=system)
        for offset, val in zip(offsets, values):
            ax.text(offset, val + 0.01, f"{val:.2f}", ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in CAUSES])
    ax.set_ylabel('Share of composition failures')
    ax.set_ylim(0, 1)
    ax.set_title('Root causes associated with composition failures', fontsize=16, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()


def main():
    df, _output_dir, _csv_dir = load_hallucination_data()
    plot_dir = Path(__file__).resolve().parent
    output_path = plot_dir / "composition_failure_root_causes.png"
    build_plot(df, output_path)


if __name__ == "__main__":
    main()
