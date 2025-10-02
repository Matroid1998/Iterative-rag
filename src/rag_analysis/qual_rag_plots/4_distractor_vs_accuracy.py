"""
Plot 4: Distractor Latch vs Model Performance
Bars for distractor latch rate by system with accuracy overlay line.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from quality_plot_utils import load_quality_data, load_model_accuracy, match_accuracy


def build_plot(run_df, csv_dir: Path, output_path: Path) -> None:
    if run_df.empty:
        print("No run-level data available.")
        return

    grouped = run_df.groupby("system")
    systems = []
    distractor_rates = []
    accuracies = []

    accuracy_table = load_model_accuracy(csv_dir)

    for system, group in grouped:
        systems.append(system)
        distractors = group["distractor_latch"].dropna()
        rate = distractors.mean() if not distractors.empty else 0.0
        distractor_rates.append(rate)

        acc = match_accuracy(system, accuracy_table)
        if acc is None:
            # fall back to observed accuracy if available
            if group["is_correct"].notna().any():
                acc = group["is_correct"].dropna().mean() * 100
            else:
                acc = 0.0
        accuracies.append(acc)

    if not systems:
        print("No systems discovered for plot.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x_positions = range(len(systems))

    bars = ax1.bar(x_positions, distractor_rates, color="#4c72b0", alpha=0.75)
    ax1.set_ylabel("Distractor latch rate", fontsize=12)
    ax1.set_ylim(0, max(distractor_rates) * 1.2 if distractor_rates else 1)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(systems, rotation=20, ha="right")
    ax1.set_title("Distractor latch vs model accuracy", fontsize=16, fontweight="bold")
    ax1.grid(True, axis='y', alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x_positions, accuracies, marker="o", color="#c44e52", linewidth=2, label="Accuracy (%)")
    ax2.set_ylabel("Reported accuracy (%)", fontsize=12, color="#c44e52")
    ax2.set_ylim(0, max(accuracies) * 1.1 if accuracies else 100)

    for idx, (bar, rate, acc) in enumerate(zip(bars, distractor_rates, accuracies)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{rate:.2f}",
                 ha='center', va='bottom', fontsize=10)
        ax2.text(idx, acc + 1, f"{acc:.1f}%", color="#c44e52", fontsize=10, ha='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()


def main():
    _step_df, run_df, _output_dir, csv_dir = load_quality_data()
    plot_dir = Path(__file__).resolve().parent
    output_path = plot_dir / "distractor_vs_accuracy.png"
    build_plot(run_df, csv_dir, output_path)


if __name__ == "__main__":
    main()
