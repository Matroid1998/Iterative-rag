#!/usr/bin/env python3
"""Plot average output tokens grouped by hop count for each response file."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt

MODEL_NAME_MAP: Dict[str, str] = {
    "responses_bedrock_mistral.mistral-large-2402-v1:0": "Mistral Large",
    "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0": "Claude 3.7 Sonnet",
    "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning": "Claude 3.7 Sonnet (reasoning)",
    "responses_bedrock_us.deepseek.r1-v1:0": "DeepSeek R1",
    "responses_bedrock_us.deepseek.r1-v1:0-reasoning": "DeepSeek R1 (reasoning)",
    "responses_openai_gpt-4o_reverified": "GPT-4o (reverified)",
    "responses_openai_gpt-5": "GPT-5",
}


def normalize_model_key(stem: str) -> str:
    if stem.endswith("_reverified"):
        stem = stem[: -len("_reverified")]
    return stem


def determine_layout(n_items: int) -> tuple[int, int]:
    if n_items <= 2:
        return 1, n_items or 1
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


def average_output_tokens(path: Path) -> Dict[str, float]:
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
            value = record.get("output_tokens")
            try:
                tokens = float(value)
            except (TypeError, ValueError):
                continue
            totals[str(hops)].append(tokens)
    return {hop: (sum(values) / len(values)) for hop, values in totals.items() if values}


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    responses_dir = repo_root / "src" / "responses_reverified"
    if not responses_dir.exists():
        responses_dir = repo_root / "src" / "responses"
    plots_dir = repo_root / "src" / "plots"
    output_path = plots_dir / "output_tokens_per_hop.png"

    jsonl_files = sorted(responses_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise RuntimeError(f"No JSONL files found in {responses_dir}")

    n_rows, n_cols = determine_layout(len(jsonl_files))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=False)
    axes_list = flatten_axes(axes)

    for ax, path in zip(axes_list, jsonl_files):
        averages = average_output_tokens(path)
        if averages:
            sorted_items = sorted(averages.items(), key=lambda item: sortable_hop_key(item[0]))
            labels = [str(hop) for hop, _ in sorted_items]
            values = [avg for _, avg in sorted_items]
            ax.bar(labels, values, color="#55a868")
            ax.set_ylabel("Avg output tokens")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")

        model_key = normalize_model_key(path.stem)
        display_name = MODEL_NAME_MAP.get(model_key, path.stem)
        ax.set_title(display_name, fontsize=10)
        ax.set_xlabel("Number of hops")

    for ax in axes_list[len(jsonl_files):]:
        ax.axis("off")

    fig.suptitle("Average output tokens per hop", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=300)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
