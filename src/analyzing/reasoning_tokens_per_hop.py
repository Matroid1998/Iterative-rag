#!/usr/bin/env python3
"""Plot average reasoning tokens grouped by hop count for reasoning-capable models."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

MODEL_NAME_MAP: Dict[str, str] = {
    "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning": "Claude 3.7 Sonnet",
    "responses_bedrock_us.deepseek.r1-v1:0-reasoning": "DeepSeek R1",
    "responses_openai_gpt-5": "GPT-5",
}

# Restrict to models that actually supply reasoning tokens.
REASONING_MODEL_KEYS = set(MODEL_NAME_MAP.keys())


def determine_layout(n_items: int) -> tuple[int, int]:
    if n_items <= 1:
        return 1, 1
    if n_items == 2:
        return 1, 2
    if n_items == 3:
        return 1, 3
    if n_items == 4:
        return 2, 2
    if n_items <= 6:
        return 2, 3
    return 3, 3


def flatten_axes(axes) -> List:
    if hasattr(axes, "flat"):
        return list(axes.flat)
    if isinstance(axes, (list, tuple)):
        flat: List = []
        for entry in axes:
            flat.extend(flatten_axes(entry))
        return flat
    return [axes]


def sortable_hop_key(hop) -> tuple[int, str]:
    try:
        numeric = float(hop)
    except (TypeError, ValueError):
        return (1, str(hop))
    return (0, numeric)


def average_reasoning_tokens(path: Path) -> Dict[str, float]:
    totals: Dict[str, List[float]] = defaultdict(list)
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

            hops = record.get("number_of_hops")
            if hops is None:
                continue

            value = record.get("reasoning_tokens")
            try:
                tokens = float(value)
            except (TypeError, ValueError):
                continue

            totals[str(hops)].append(tokens)
    return {hop: (sum(values) / len(values)) for hop, values in totals.items() if values}


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    responses_dir = repo_root / "src" / "responses"
    plots_dir = repo_root / "src" / "plots"
    output_path = plots_dir / "reasoning_tokens_per_hop.png"

    all_jsonl_files = sorted(responses_dir.glob("*.jsonl"))
    reasoning_files = [path for path in all_jsonl_files if path.stem in REASONING_MODEL_KEYS]

    if not reasoning_files:
        raise RuntimeError("No reasoning-capable response files found.")

    n_rows, n_cols = determine_layout(len(reasoning_files))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=False)
    axes_list = flatten_axes(axes)

    for ax, path in zip(axes_list, reasoning_files):
        averages = average_reasoning_tokens(path)
        if averages:
            sorted_items = sorted(averages.items(), key=lambda item: sortable_hop_key(item[0]))
            labels = [str(hop) for hop, _ in sorted_items]
            values = [avg for _, avg in sorted_items]
            ax.bar(labels, values, color="#4c72b0")
            ax.set_ylabel("Avg reasoning tokens")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")

        display_name = MODEL_NAME_MAP.get(path.stem, path.stem)
        ax.set_title(display_name, fontsize=10)
        ax.set_xlabel("Number of hops")

    for ax in axes_list[len(reasoning_files):]:
        ax.axis("off")

    fig.suptitle("Average reasoning tokens per hop", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=300)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
