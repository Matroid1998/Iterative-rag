"""
Plot 2: Model comparison radar chart.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adv_plot_utils import load_all_data
from quality_plot_utils import load_model_accuracy, match_accuracy

METRICS = [
    ("accuracy", "Accuracy"),
    ("avg_steps", "Avg steps"),
    ("specificity", "Specificity"),
    ("on_topic", "On-topic"),
    ("sufficiency", "Sufficiency"),
    ("coverage", "Coverage"),
    ("miscalibration", "(1 - miscal)")
]


def gather_metrics(run_df, quality_step_df, hall_df, csv_dir: Path):
    systems = sorted(set(run_df["system"].dropna()) | set(quality_step_df["system"].dropna()) | set(hall_df["system"].dropna()))
    accuracy_table = load_model_accuracy(csv_dir)

    results = {}
    # Accuracy
    for system in systems:
        acc = match_accuracy(system, accuracy_table)
        if acc is None and "is_correct" in run_df.columns:
            subset = run_df[run_df["system"] == system]
            if not subset.empty and subset["is_correct"].notna().any():
                acc = subset["is_correct"].dropna().mean() * 100
        results.setdefault(system, {})["accuracy"] = (acc or 0.0) / 100.0

    # Average steps and query scores
    if not quality_step_df.empty:
        per_run = quality_step_df.groupby(["system", "question"]).agg({"step": "max"}).reset_index()
        for system in systems:
            subset_run = per_run[per_run["system"] == system]
            if not subset_run.empty:
                results.setdefault(system, {})["avg_steps"] = subset_run["step"].mean()
            subset_step = quality_step_df[quality_step_df["system"] == system]
            if not subset_step.empty:
                results[system]["specificity"] = subset_step["specificity_score"].mean()
                results[system]["on_topic"] = subset_step["on_topic_score"].mean()

    # Sufficiency & miscalibration
    if not hall_df.empty:
        for system in systems:
            subset = hall_df[hall_df["system"] == system]
            if subset.empty:
                continue
            results.setdefault(system, {})["sufficiency"] = subset["sufficiency_score_est"].dropna().mean()
            miscal = subset["is_miscalibrated"].dropna()
            if not miscal.empty:
                results[system]["miscalibration"] = 1 - miscal.mean()

    # Coverage
    if "has_gap" in run_df.columns:
        for system in systems:
            subset = run_df[run_df["system"] == system]
            if subset.empty:
                continue
            results.setdefault(system, {})["coverage"] = 1 - subset["has_gap"].fillna(False).astype(bool).mean()

    # Normalise avg_steps to 0..1 scale (higher better). Use min/max.
    values = [metrics.get("avg_steps") for metrics in results.values() if metrics.get("avg_steps") is not None]
    if values:
        max_steps = max(values)
        min_steps = min(values)
        for system in systems:
            if "avg_steps" in results.get(system, {}):
                val = results[system]["avg_steps"]
                if max_steps == min_steps:
                    norm = 1.0
                else:
                    norm = 1 - (val - min_steps) / (max_steps - min_steps)
                results[system]["avg_steps"] = norm

    # Ensure defaults
    for system in systems:
        metrics = results.setdefault(system, {})
        for key, _label in METRICS:
            metrics.setdefault(key, 0.0)
    return results


def plot_radar(results: dict, output_path: Path) -> None:
    if not results:
        print("No metrics for radar chart.")
        return
    systems = sorted(results)
    angles = np.linspace(0, 2 * math.pi, len(METRICS), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for system in systems:
        values = [results[system][key] for key, _label in METRICS]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=system)
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([label for _key, label in METRICS], fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison radar", fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved radar chart to {output_path}")
    plt.close()


def main():
    run_df, _coverage_df, _coverage_step_df, quality_step_df, hall_df, _late_hit_df = load_all_data()
    csv_dir = Path(__file__).resolve().parents[2] / "results" / "new_results_csv"
    metrics = gather_metrics(run_df, quality_step_df, hall_df, csv_dir)
    plot_dir = Path(__file__).resolve().parent
    output_path = plot_dir / "model_radar_profile.png"
    plot_radar(metrics, output_path)


if __name__ == "__main__":
    main()
