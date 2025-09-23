#!/usr/bin/env python3
"""Plot token usage comparisons for easy vs. hard questions."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt

from easy_hard_question_analysis import (
    MODEL_NAME_MAP,
    load_question_map_jsonl,
    summarise_token_usage,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "data"
REASONING_MODEL_KEYS = {
    "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning",
    "responses_bedrock_us.deepseek.r1-v1:0-reasoning",
    "responses_openai_gpt-5",
}


def model_display_name(model_key: str) -> str:
    name = MODEL_NAME_MAP.get(model_key, model_key)
    return name.replace(" (reasoning)", "")


def collect_models(*question_maps: Dict[str, list]) -> List[str]:
    models = set()
    for question_map in question_maps:
        for records in question_map.values():
            for record in records:
                models.add(record.model)
    return sorted(models)


def prepare_series(summary: Dict[str, Dict[str, float]], metric_key: str) -> Dict[str, float]:
    series: Dict[str, float] = {}
    for model, stats in summary.items():
        value = stats.get(metric_key)
        if value is None:
            continue
        series[model] = value
    return series


def plot_metric(
    models: List[str],
    easy_vals: Dict[str, float],
    hard_vals: Dict[str, float],
    title: str,
    ylabel: str,
    output_path: Path,
    *,
    allowed_models: Optional[Iterable[str]] = None,
) -> None:
    allowed_set = set(allowed_models) if allowed_models is not None else None

    def include_model(model: str) -> bool:
        if allowed_set is not None and model not in allowed_set:
            return False
        easy_val = easy_vals.get(model)
        hard_val = hard_vals.get(model)
        easy_valid = easy_val is not None and easy_val == easy_val
        hard_valid = hard_val is not None and hard_val == hard_val
        return easy_valid or hard_valid

    filtered_models = [model for model in models if include_model(model)]
    if not filtered_models:
        print(f"No data available for {title}; skipping plot.")
        return

    x_positions = range(len(filtered_models))
    width = 0.35

    easy_values = [easy_vals.get(model, float("nan")) for model in filtered_models]
    hard_values = [hard_vals.get(model, float("nan")) for model in filtered_models]

    fig, ax = plt.subplots(figsize=(max(6, len(filtered_models) * 1.8), 5))
    ax.bar([x - width / 2 for x in x_positions], easy_values, width=width, label="Easy", color="#55a868")
    ax.bar([x + width / 2 for x in x_positions], hard_values, width=width, label="Hard (wrong)", color="#c44e52")

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels([model_display_name(model) for model in filtered_models], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_hop_bar_chart(easy_questions, hard_questions, output_path: Path) -> None:
    def count_hops(question_map) -> Counter:
        counter: Counter = Counter()
        for records in question_map.values():
            for record in records:
                if record.number_of_hops is not None:
                    counter[int(record.number_of_hops)] += 1
        return counter

    easy_counts = count_hops(easy_questions)
    hard_counts = count_hops(hard_questions)
    hop_values = sorted(set(easy_counts) | set(hard_counts))

    if not hop_values:
        print("No hop data available for hop distribution plot; skipping.")
        return

    x_positions = range(len(hop_values))
    width = 0.35

    easy_values = [easy_counts.get(hop, 0) for hop in hop_values]
    hard_values = [hard_counts.get(hop, 0) for hop in hop_values]

    fig, ax = plt.subplots(figsize=(max(6, len(hop_values) * 1.6), 5))
    ax.bar([x - width / 2 for x in x_positions], easy_values, width=width, label="Easy", color="#55a868")
    ax.bar([x + width / 2 for x in x_positions], hard_values, width=width, label="Hard (>=4 wrong)", color="#c44e52")

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels([str(hop) for hop in hop_values])
    ax.set_xlabel("Hop count")
    ax.set_ylabel("Questions")
    ax.set_title("Question counts by hop count")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    easy_path = DATASET_DIR / "easy_questions.jsonl"
    hard_path = DATASET_DIR / "hard_questions_min4.jsonl"

    if not easy_path.exists() or not hard_path.exists():
        raise FileNotFoundError("Run export_question_difficulty.py first to generate JSONL datasets.")

    easy_questions = load_question_map_jsonl(easy_path)
    hard_questions = load_question_map_jsonl(hard_path)

    easy_summary = summarise_token_usage(easy_questions, only_wrong=False)
    hard_summary = summarise_token_usage(hard_questions, only_wrong=True)

    all_models = collect_models(easy_questions, hard_questions)

    reasoning_easy = prepare_series(easy_summary, "reasoning_avg")
    reasoning_hard = prepare_series(hard_summary, "reasoning_avg")
    output_easy = prepare_series(easy_summary, "output_avg")
    output_hard = prepare_series(hard_summary, "output_avg")

    repo_root = SCRIPT_DIR.parents[1]
    plots_dir = repo_root / "src" / "plots"

    plot_metric(
        all_models,
        reasoning_easy,
        reasoning_hard,
        "Average reasoning tokens (easy vs hard questions)",
        "Average reasoning tokens",
        plots_dir / "question_difficulty_reasoning_tokens_min4.png",
        allowed_models=REASONING_MODEL_KEYS,
    )

    plot_metric(
        all_models,
        output_easy,
        output_hard,
        "Average output tokens (easy vs hard questions)",
        "Average output tokens",
        plots_dir / "question_difficulty_output_tokens_min4.png",
    )

    plot_metric(
        all_models,
        output_easy,
        output_hard,
        "Average output tokens (reasoning models)",
        "Average output tokens",
        plots_dir / "question_difficulty_output_tokens_reasoning_min4.png",
        allowed_models=REASONING_MODEL_KEYS,
    )

    plot_hop_bar_chart(
        easy_questions,
        hard_questions,
        plots_dir / "question_difficulty_hop_distribution_min4.png",
    )


if __name__ == "__main__":
    main()
