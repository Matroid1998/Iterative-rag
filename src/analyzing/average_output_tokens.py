#!/usr/bin/env python3
"""Plot average output tokens for correct vs wrong answers across model groups."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

import matplotlib.pyplot as plt

from config import (
    get_responses_dir,
    PLOTS_DIR,
    get_display_name,
    is_reasoning_model,
    discover_jsonl_files,
    ITERATIVE_MODEL_ENTRIES,
)


def accumulate_output_stats(path: Path) -> Dict[bool, List[float]]:
    """Return list of output tokens keyed by correctness."""
    values: Dict[bool, List[float]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"Skipping {path.name}:{line_number} (invalid JSON: {error})")
                continue

            is_correct = bool(record.get("is_correct"))
            
            # Get output tokens from root level (already includes reasoning for reasoning models)
            raw_value = record.get("output_tokens")
            
            try:
                output_tokens = float(raw_value)
            except (TypeError, ValueError):
                print(f"Skipping {path.name}:{line_number} (invalid output_tokens: {raw_value!r})")
                continue

            values[is_correct].append(output_tokens)
    return values


def compute_average_and_std(values: List[float]) -> Tuple[float, float]:
    """Compute average and standard deviation from list of values."""
    if not values:
        return 0.0, 0.0
    arr = np.array(values)
    return float(np.mean(arr)), float(np.std(arr))


def main() -> None:
    responses_dir = get_responses_dir()
    
    # Use ITERATIVE_MODEL_ENTRIES for consistent ordering and exclusion
    model_entries: List[Tuple[str, str, float, float, float, float]] = []

    for filename, display_name in ITERATIVE_MODEL_ENTRIES:
        path = responses_dir / filename
        
        if not path.exists():
            print(f"Warning: File not found: {path}")
            continue
        
        model_key = path.stem

        stats = accumulate_output_stats(path)
        correct_values = stats.get(True, [])
        wrong_values = stats.get(False, [])
        
        correct_avg, correct_std = compute_average_and_std(correct_values)
        wrong_avg, wrong_std = compute_average_and_std(wrong_values)

        model_entries.append(
            (
                model_key,
                display_name,
                correct_avg,
                wrong_avg,
                correct_std,
                wrong_std,
            )
        )

    reasoning_entries = [entry for entry in model_entries if is_reasoning_model(entry[0])]
    non_reasoning_entries = [entry for entry in model_entries if not is_reasoning_model(entry[0])]

    if model_entries:
        plot_average_tokens(
            model_entries,
            PLOTS_DIR / "average_output_tokens.png",
            "Average output tokens by correctness",
        )

    if reasoning_entries:
        plot_average_tokens(
            reasoning_entries,
            PLOTS_DIR / "average_output_tokens_reasoning.png",
            "Average output tokens (reasoning models)",
        )
    else:
        print("No reasoning models found for plotting average output tokens.")

    if non_reasoning_entries:
        plot_average_tokens(
            non_reasoning_entries,
            PLOTS_DIR / "average_output_tokens_non_reasoning.png",
            "Average output tokens (non-reasoning models)",
        )
    else:
        print("No non-reasoning models found for plotting average output tokens.")


def plot_average_tokens(
    entries: List[Tuple[str, str, float, float, float, float]],
    output_path: Path,
    title: str,
) -> None:
    labels = [display_name for _, display_name, _, _, _, _ in entries]
    correct_avgs = [correct_avg for _, _, correct_avg, _, _, _ in entries]
    wrong_avgs = [wrong_avg for _, _, _, wrong_avg, _, _ in entries]
    correct_stds = [correct_std for _, _, _, _, correct_std, _ in entries]
    wrong_stds = [wrong_std for _, _, _, _, _, wrong_std in entries]

    x_positions = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    ax.bar([x - width / 2 for x in x_positions], correct_avgs, width=width, label="Correct", 
           color="#55a868", yerr=correct_stds, capsize=5)
    ax.bar([x + width / 2 for x in x_positions], wrong_avgs, width=width, label="Wrong", 
           color="#c44e52", yerr=wrong_stds, capsize=5)

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Average output tokens (log scale)")
    ax.set_yscale('log')
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4, which='both')

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
