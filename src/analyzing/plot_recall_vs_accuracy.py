#!/usr/bin/env python3
"""
Recall vs. Accuracy Visualization

Consumes the aggregate metrics emitted by `calc_recall_metrics.py` and
plots per-step recall curves alongside the relationship between recall
and answer accuracy. This helps determine whether additional retrieval
actually unlocks new evidence or whether models are simply reasoning
longer without hitting the supporting passages.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from config import (
    PLOTS_DIR,
    get_iterative_model_entries,
    get_model_color,
)

# Default location of the summary json produced by calc_recall_metrics.py
SUMMARY_PATH_DEFAULT = Path(__file__).resolve().parent / "recall_metrics_summary.json"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_records(path: Path) -> List[dict]:
    """Load JSONL records for a model."""
    records: List[dict] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                records.append(json.loads(raw_line))
            except json.JSONDecodeError:
                continue
    return records


def extract_max_source_step(record: dict) -> int:
    """
    Best-effort extraction of the final retrieval step used by a record.
    Falls back to 1 when the metadata is missing.
    """
    steps: List[int] = []
    for key in ("raw_response", "raw"):
        payload = record.get(key)
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = None
        if not isinstance(payload, dict):
            continue
        evidence = payload.get("evidence")
        if not isinstance(evidence, list):
            continue
        for item in evidence:
            if not isinstance(item, dict):
                continue
            step_value = item.get("source_step")
            if isinstance(step_value, str) and step_value.isdigit():
                steps.append(int(step_value))
            elif isinstance(step_value, (int, float)):
                step_int = int(step_value)
                if step_int > 0:
                    steps.append(step_int)
    if steps:
        return max(steps)
    return 1


def compute_cumulative_accuracy_by_step(records: Iterable[dict]) -> Dict[int, float]:
    """
    Compute cumulative accuracy (in percentage) by retrieval step.
    Steps are determined by the maximum source_step observed in evidence.
    """
    questions_by_step: Dict[int, List[bool]] = defaultdict(list)

    for record in records:
        is_correct = bool(record.get("is_correct", False))
        final_step = extract_max_source_step(record)
        questions_by_step[final_step].append(is_correct)

    if not questions_by_step:
        return {}

    cumulative: Dict[int, float] = {}
    running: List[bool] = []
    for step in sorted(questions_by_step.keys()):
        running.extend(questions_by_step[step])
        if not running:
            cumulative[step] = 0.0
        else:
            cumulative[step] = 100.0 * sum(running) / len(running)
    return cumulative


def build_recall_dataset(
    summary: dict,
    k: int,
) -> Dict[str, Dict[int, float]]:
    """
    Extract cumulative Recall@K by step for each model (values in %).
    """
    recall_by_model: Dict[str, Dict[int, float]] = {}
    for file_stem, payload in summary.items():
        step_by_k = payload.get("step_by_k", {})
        model_steps: Dict[int, float] = {}
        for step_str, values in step_by_k.items():
            step = int(step_str)
            value = values.get(str(k))
            if value is None:
                continue
            model_steps[step] = value * 100.0
        if model_steps:
            recall_by_model[file_stem] = dict(sorted(model_steps.items()))
    return recall_by_model


def align_recall_accuracy(
    model_entries: List[Tuple[Path, str]],
    recall_summary: Dict[str, Dict[int, float]],
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, float]]]:
    """
    Build aligned recall and accuracy dictionaries keyed by display name.
    Only steps present in both datasets are preserved.
    """
    recall_output: Dict[str, Dict[int, float]] = {}
    accuracy_output: Dict[str, Dict[int, float]] = {}

    for jsonl_path, display_name in model_entries:
        model_key = jsonl_path.stem
        recall_steps = recall_summary.get(model_key)
        if not recall_steps:
            continue

        records = load_records(jsonl_path)
        accuracy_steps = compute_cumulative_accuracy_by_step(records)
        if not accuracy_steps:
            continue

        common_steps = sorted(set(recall_steps.keys()) & set(accuracy_steps.keys()))
        if not common_steps:
            continue

        recall_output[display_name] = {step: recall_steps[step] for step in common_steps}
        accuracy_output[display_name] = {step: accuracy_steps[step] for step in common_steps}

    return recall_output, accuracy_output


def plot_recall_and_effect(
    recall_by_model: Dict[str, Dict[int, float]],
    accuracy_by_model: Dict[str, Dict[int, float]],
    k: int,
    output_path: Path,
) -> None:
    if not recall_by_model:
        print("No recall data available to plot.")
        return

    # Sort models to keep lines ordered deterministically
    ordered_models = sorted(recall_by_model.keys())

    max_step = max(max(steps.keys()) for steps in recall_by_model.values())

    fig, (ax_recall, ax_scatter) = plt.subplots(
        2, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [2.2, 1.2]}
    )

    # --- Top plot: cumulative Recall@K by step ---
    for model in ordered_models:
        recall_steps = recall_by_model[model]
        steps_sorted = sorted(recall_steps.keys())
        recalls = [recall_steps[s] for s in steps_sorted]
        color = get_model_color(model)
        ax_recall.plot(
            steps_sorted,
            recalls,
            marker="o",
            linewidth=2.5,
            markersize=7,
            alpha=0.9,
            label=model,
            color=color,
        )

    ax_recall.set_title(
        f"Cumulative Recall@{k} by Retrieval Step",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )
    ax_recall.set_xlabel("Retrieval Step", fontsize=13, fontweight="bold")
    ax_recall.set_ylabel(f"Recall@{k} (%)", fontsize=13, fontweight="bold")
    ax_recall.set_xticks(range(1, max_step + 1))
    ax_recall.set_xlim(0.8, max_step + 0.2)
    ax_recall.set_ylim(0, 100)
    ax_recall.grid(True, alpha=0.3, linestyle="--")
    ax_recall.legend(loc="upper left", fontsize=9, ncol=2)

    # --- Bottom plot: recall vs accuracy relationship (per step) ---
    xs: List[float] = []
    ys: List[float] = []
    colors: List[str] = []
    labels: List[str] = []

    unique_steps = sorted(
        set(step for steps in recall_by_model.values() for step in steps.keys())
    )
    cmap = plt.cm.viridis
    step_colors = {
        step: cmap((step - 1) / max(1, unique_steps[-1] - 1 if len(unique_steps) > 1 else 1))
        for step in unique_steps
    }

    for model in ordered_models:
        recall_steps = recall_by_model[model]
        accuracy_steps = accuracy_by_model.get(model, {})
        for step, recall_value in recall_steps.items():
            if step not in accuracy_steps:
                continue
            accuracy_value = accuracy_steps[step]
            xs.append(recall_value)
            ys.append(accuracy_value)
            colors.append(step_colors[step])
            labels.append(f"{model} · step {step}")
            ax_scatter.scatter(
                recall_value,
                accuracy_value,
                color=step_colors[step],
                edgecolor="black",
                linewidth=0.4,
                s=60,
                alpha=0.8,
            )

    if xs and ys:
        corr = np.corrcoef(xs, ys)[0, 1]
        ax_scatter.text(
            0.02,
            0.94,
            f"Pearson r = {corr:.2f}",
            transform=ax_scatter.transAxes,
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.6),
        )

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label=f"Step {step}", markerfacecolor=step_colors[step], markersize=8)
        for step in unique_steps
    ]
    ax_scatter.legend(
        handles=legend_handles,
        title="First supporting passage at step",
        loc="lower right",
        fontsize=9,
        title_fontsize=10,
    )

    ax_scatter.set_title(
        f"Accuracy vs. Recall@{k} by Retrieval Step",
        fontsize=16,
        fontweight="bold",
        pad=16,
    )
    ax_scatter.set_xlabel(f"Recall@{k} (%)", fontsize=13, fontweight="bold")
    ax_scatter.set_ylabel("Cumulative Accuracy (%)", fontsize=13, fontweight="bold")
    ax_scatter.set_xlim(0, 100)
    ax_scatter.set_ylim(0, 100)
    ax_scatter.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved recall vs accuracy plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-step Recall@K curves and their relationship to accuracy."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=SUMMARY_PATH_DEFAULT,
        help="Path to recall_metrics_summary.json generated by calc_recall_metrics.py",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Which Recall@K to visualise (must exist in the summary file).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the plot (defaults to src/plots/recall_vs_accuracy_k{K}.png).",
    )
    args = parser.parse_args()

    summary = load_json(args.summary)
    recall_summary = build_recall_dataset(summary, k=args.k)

    model_entries = get_iterative_model_entries(existing_only=True)
    recall_by_model, accuracy_by_model = align_recall_accuracy(model_entries, recall_summary)

    if not recall_by_model:
        print("No overlapping recall/accuracy data found for the available models.")
        return

    output_path = (
        args.output
        if args.output is not None
        else PLOTS_DIR / f"recall_vs_accuracy_k{args.k}.png"
    )

    plot_recall_and_effect(recall_by_model, accuracy_by_model, args.k, output_path)


if __name__ == "__main__":
    main()

