#!/usr/bin/env python3
"""Generate bar charts of wrong answers per hop count for each response file."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt


def load_wrong_answer_counts(path: Path) -> Counter:
    """Return counts of wrong answers keyed by number_of_hops for a JSONL file."""
    counts: Counter = Counter()
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

            if record.get("is_correct") is True:
                continue

            hops: Union[int, float, str, None] = record.get("number_of_hops")
            if hops is None:
                hops = "unknown"
            counts[hops] += 1
    return counts


def determine_layout(n_items: int) -> tuple[int, int]:
    """Choose a subplot grid layout for the given number of items."""
    if n_items <= 3:
        return 1, n_items
    if n_items == 4:
        return 2, 2
    return 2, 3  # Works well for 5-6 items; add rows if needed later.


def sortable_hop_key(value: Union[int, float, str]) -> tuple[int, Union[float, str]]:
    """Sort hops numerically when possible, otherwise fall back to string ordering."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return (1, str(value))
    else:
        return (0, numeric)


def flatten_axes(axes) -> list:
    """Return a flat list of axes regardless of the input structure."""
    if hasattr(axes, "flat"):
        return list(axes.flat)
    if isinstance(axes, (list, tuple)):
        flat = []
        for item in axes:
            flat.extend(flatten_axes(item))
        return flat
    return [axes]


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    responses_dir = repo_root / "src" / "responses_reverified"
    if not responses_dir.exists():
        responses_dir = repo_root / "src" / "responses"
    output_path = script_dir / "wrong_answers_per_hop.png"

    jsonl_files = sorted(responses_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise RuntimeError(f"No JSONL files found in {responses_dir}")

    n_rows, n_cols = determine_layout(len(jsonl_files))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True)
    axes_flat = flatten_axes(axes)

    for ax, path in zip(axes_flat, jsonl_files):
        counts = load_wrong_answer_counts(path)
        if counts:
            sorted_hops = sorted(counts.keys(), key=sortable_hop_key)
            labels = [str(hop) for hop in sorted_hops]
            values = [counts[hop] for hop in sorted_hops]
            ax.bar(labels, values, color="#4c72b0")
        else:
            ax.bar([], [])
            ax.text(0.5, 0.5, "No wrong answers", ha="center", va="center")

        ax.set_title(path.stem, fontsize=10)
        ax.set_xlabel("Number of hops")

    for ax in axes_flat[len(jsonl_files):]:
        ax.axis("off")

    used_axes = axes_flat[:len(jsonl_files)]
    for idx, ax in enumerate(used_axes):
        if n_cols == 0 or idx % n_cols == 0:
            ax.set_ylabel("Wrong answers")

    fig.suptitle("Wrong answers per hop count", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=300)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
