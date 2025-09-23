#!/usr/bin/env python3
"""Plot average reasoning tokens for correct vs wrong answers across reasoning models."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt

# Mapping from stored file stem to human-friendly model display name.
MODEL_NAME_MAP: Dict[str, str] = {
    "responses_bedrock_mistral.mistral-large-2402-v1:0": "Mistral Large",
    "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0": "Claude 3.7 Sonnet",
    "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning": "Claude 3.7 Sonnet (reasoning)",
    "responses_bedrock_us.deepseek.r1-v1:0": "DeepSeek R1",
    "responses_bedrock_us.deepseek.r1-v1:0-reasoning": "DeepSeek R1 (reasoning)",
    "responses_openai_gpt-4o_reverified": "GPT-4o (reverified)",
    "responses_openai_gpt-5": "GPT-5",
}

# Only these models supply reasoning tokens that we want to compare.
REASONING_MODEL_KEYS = {
    "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning",
    "responses_bedrock_us.deepseek.r1-v1:0-reasoning",
    "responses_openai_gpt-5",
}


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
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    responses_dir = repo_root / "src" / "responses"
    plots_dir = repo_root / "src" / "plots"
    output_path = plots_dir / "average_reasoning_tokens.png"

    jsonl_files = sorted(responses_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise RuntimeError(f"No JSONL files found in {responses_dir}")

    labels = []
    correct_avgs = []
    wrong_avgs = []

    for path in jsonl_files:
        model_key = path.stem
        if model_key not in REASONING_MODEL_KEYS:
            continue
        display_name = MODEL_NAME_MAP.get(model_key, model_key)

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
    fig.savefig(output_path, dpi=300)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
