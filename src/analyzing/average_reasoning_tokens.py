#!/usr/bin/env python3
"""Plot average reasoning tokens for correct vs wrong answers across reasoning models."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt

from config import (
    get_responses_dir,
    PLOTS_DIR,
    get_display_name,
    discover_reasoning_jsonl_files,
)


def accumulate_reasoning_stats(path: Path) -> Dict[bool, Tuple[float, int]]:
    """Return total reasoning tokens and counts keyed by correctness."""
    totals: Dict[bool, Tuple[float, int]] = defaultdict(lambda: (0.0, 0))
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
            
            # Try to get reasoning tokens from usage dict first, then fallback to direct field
            usage = record.get("usage", {})
            raw_value = usage.get("reasoning_tokens") if usage else None
            if raw_value is None:
                raw_value = record.get("reasoning_tokens")
            
            try:
                reasoning_tokens = float(raw_value)
            except (TypeError, ValueError):
                print(f"Skipping {path.name}:{line_number} (invalid reasoning_tokens: {raw_value!r})")
                continue

            total, count = totals[is_correct]
            totals[is_correct] = (total + reasoning_tokens, count + 1)
    return totals


def compute_average(total: float, count: int) -> float:
    return total / count if count else 0.0


def main() -> None:
    responses_dir = get_responses_dir()
    reasoning_files = discover_reasoning_jsonl_files(responses_dir)
    
    if not reasoning_files:
        raise RuntimeError(f"No reasoning model files found in {responses_dir}")

    labels = []
    correct_avgs = []
    wrong_avgs = []

    for path in reasoning_files:
        display_name = get_display_name(path.stem)

        stats = accumulate_reasoning_stats(path)
        correct_total, correct_count = stats.get(True, (0.0, 0))
        wrong_total, wrong_count = stats.get(False, (0.0, 0))

        labels.append(display_name)
        correct_avgs.append(compute_average(correct_total, correct_count))
        wrong_avgs.append(compute_average(wrong_total, wrong_count))

    if not labels:
        raise RuntimeError("No reasoning models found to plot.")

    x_positions = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    ax.bar([x - width / 2 for x in x_positions], correct_avgs, width=width, label="Correct", color="#55a868")
    ax.bar([x + width / 2 for x in x_positions], wrong_avgs, width=width, label="Wrong", color="#c44e52")

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels([label.replace(" (reasoning)", "") for label in labels], rotation=20, ha="right")
    ax.set_ylabel("Average reasoning tokens")
    ax.set_title("Average reasoning tokens by correctness")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    output_path = PLOTS_DIR / "average_reasoning_tokens.png"
    fig.savefig(output_path, dpi=300)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
