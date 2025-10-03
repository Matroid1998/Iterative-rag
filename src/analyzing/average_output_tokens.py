#!/usr/bin/env python3
"""Plot average output tokens for correct vs wrong answers across model groups."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

import matplotlib.pyplot as plt

MODEL_NAME_MAP: Dict[str, str] = {
    "responses_bedrock_mistral.mistral-large-2402-v1:0": "Mistral Large",
    "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0": "Claude 3.7 Sonnet",
    "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning": "Claude 3.7 Sonnet (reasoning)",
    "responses_bedrock_us.deepseek.r1-v1:0": "DeepSeek R1",
    "responses_bedrock_us.deepseek.r1-v1:0-reasoning": "DeepSeek R1 (reasoning)",
    "responses_openai_gpt-4o": "GPT-4o",
    "responses_openai_gpt-5": "GPT-5",
}

REASONING_MODEL_KEYS = {
    "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning",
    "responses_bedrock_us.deepseek.r1-v1:0-reasoning",
    "responses_openai_gpt-5",
}


def normalize_model_key(stem: str) -> str:
    if stem.endswith("_reverified"):
        stem = stem[: -len("_reverified")]
    return stem


def accumulate_output_stats(path: Path, adjust_reasoning: bool) -> Dict[bool, List[float]]:
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
            raw_value = record.get("output_tokens")
            try:
                output_tokens = float(raw_value)
            except (TypeError, ValueError):
                print(f"Skipping {path.name}:{line_number} (invalid output_tokens: {raw_value!r})")
                continue

            if adjust_reasoning:
                raw_reasoning = record.get("reasoning_tokens")
                try:
                    reasoning_tokens = float(raw_reasoning)
                except (TypeError, ValueError):
                    reasoning_tokens = None
                if reasoning_tokens is not None:
                    output_tokens = max(0.0, output_tokens - reasoning_tokens)

            values[is_correct].append(output_tokens)
    return values


def compute_average_and_std(values: List[float]) -> Tuple[float, float]:
    """Compute average and standard deviation from list of values."""
    if not values:
        return 0.0, 0.0
    arr = np.array(values)
    return float(np.mean(arr)), float(np.std(arr))


def resolve_label(model_key: str) -> str:
    return MODEL_NAME_MAP.get(model_key, model_key)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    responses_dir = repo_root / "src" / "responses_reverified"
    if not responses_dir.exists():
        responses_dir = repo_root / "src" / "responses"
    plots_dir = repo_root / "src" / "plots"
    jsonl_files = sorted(responses_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise RuntimeError(f"No JSONL files found in {responses_dir}")

    model_entries: List[Tuple[str, str, float, float, float, float]] = []

    plots_dir.mkdir(parents=True, exist_ok=True)

    for path in jsonl_files:
        model_key = normalize_model_key(path.stem)
        display_name = resolve_label(model_key)

        stats = accumulate_output_stats(path, adjust_reasoning=model_key in REASONING_MODEL_KEYS)
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

    reasoning_entries = [entry for entry in model_entries if entry[0] in REASONING_MODEL_KEYS]
    non_reasoning_entries = [entry for entry in model_entries if entry[0] not in REASONING_MODEL_KEYS]

    if model_entries:
        plot_average_tokens(
            model_entries,
            plots_dir / "average_output_tokens.png",
            "Average output tokens by correctness",
        )

    if reasoning_entries:
        plot_average_tokens(
            reasoning_entries,
            plots_dir / "average_output_tokens_reasoning.png",
            "Average output tokens (reasoning models)",
        )
    else:
        print("No reasoning models found for plotting average output tokens.")

    if non_reasoning_entries:
        plot_average_tokens(
            non_reasoning_entries,
            plots_dir / "average_output_tokens_non_reasoning.png",
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
    ax.set_ylabel("Average output tokens")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
