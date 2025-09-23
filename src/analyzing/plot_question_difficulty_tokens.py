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


def plot_two_series(
    models: List[str],
    series_a: Dict[str, float],
    series_b: Dict[str, float],
    label_a: str,
    label_b: str,
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
        val_a = series_a.get(model)
        val_b = series_b.get(model)
        valid_a = val_a is not None and val_a == val_a
        valid_b = val_b is not None and val_b == val_b
        return valid_a or valid_b

    filtered_models = [model for model in models if include_model(model)]
    if not filtered_models:
        print(f"No data available for {title}; skipping plot.")
        return

    x_positions = range(len(filtered_models))
    width = 0.35

    values_a = [series_a.get(model, float("nan")) for model in filtered_models]
    values_b = [series_b.get(model, float("nan")) for model in filtered_models]

    fig, ax = plt.subplots(figsize=(max(6, len(filtered_models) * 1.8), 5))
    ax.bar([x - width / 2 for x in x_positions], values_a, width=width, label=label_a, color="#4c72b0")
    ax.bar([x + width / 2 for x in x_positions], values_b, width=width, label=label_b, color="#c44e52")

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
    ax.bar([x + width / 2 for x in x_positions], hard_values, width=width, label="Hard (≥4 wrong)", color="#c44e52")

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


def filter_hard_questions_by_wrong_count(
    question_map: Dict[str, List],
    *,
    min_wrong: Optional[int] = None,
    exact_wrong: Optional[int] = None,
) -> Dict[str, List]:
    filtered: Dict[str, List] = {}
    for question, records in question_map.items():
        wrong_count = sum(1 for record in records if not record.is_correct)
        if min_wrong is not None and wrong_count < min_wrong:
            continue
        if exact_wrong is not None and wrong_count != exact_wrong:
            continue
        filtered[question] = records
    return filtered


def main() -> None:
    easy_path = DATASET_DIR / "easy_questions.jsonl"
    hard_path = DATASET_DIR / "hard_questions_min4.jsonl"

    if not easy_path.exists() or not hard_path.exists():
        raise FileNotFoundError("Run export_question_difficulty.py first to generate JSONL datasets.")

    easy_questions = load_question_map_jsonl(easy_path)
    hard_questions_min4 = load_question_map_jsonl(hard_path)
    hard_questions_4 = filter_hard_questions_by_wrong_count(hard_questions_min4, exact_wrong=4)
    hard_questions_6 = filter_hard_questions_by_wrong_count(hard_questions_min4, exact_wrong=6)

    easy_summary = summarise_token_usage(easy_questions, only_wrong=False)
    hard_summary_min4 = summarise_token_usage(hard_questions_min4, only_wrong=True)
    hard_summary_4 = summarise_token_usage(hard_questions_4, only_wrong=True)
    hard_summary_6 = summarise_token_usage(hard_questions_6, only_wrong=True)

    all_models = collect_models(easy_questions, hard_questions_min4, hard_questions_4, hard_questions_6)

    reasoning_easy = prepare_series(easy_summary, "reasoning_avg")
    reasoning_hard_min4 = prepare_series(hard_summary_min4, "reasoning_avg")
    reasoning_hard_4 = prepare_series(hard_summary_4, "reasoning_avg")
    reasoning_hard_6 = prepare_series(hard_summary_6, "reasoning_avg")

    output_easy = prepare_series(easy_summary, "output_avg")
    output_hard_min4 = prepare_series(hard_summary_min4, "output_avg")
    output_hard_4 = prepare_series(hard_summary_4, "output_avg")
    output_hard_6 = prepare_series(hard_summary_6, "output_avg")

    repo_root = SCRIPT_DIR.parents[1]
    plots_dir = repo_root / "src" / "plots"

    # Existing comparisons
    plot_two_series(
        all_models,
        reasoning_easy,
        reasoning_hard_min4,
        "Easy",
        "≥4 wrong",
        "Average reasoning tokens (easy vs ≥4 wrong questions)",
        "Average reasoning tokens",
        plots_dir / "question_difficulty_reasoning_tokens_min4.png",
        allowed_models=REASONING_MODEL_KEYS,
    )

    plot_two_series(
        all_models,
        output_easy,
        output_hard_min4,
        "Easy",
        "≥4 wrong",
        "Average output tokens (easy vs ≥4 wrong questions)",
        "Average output tokens",
        plots_dir / "question_difficulty_output_tokens_min4.png",
    )

    plot_two_series(
        all_models,
        output_easy,
        output_hard_min4,
        "Easy",
        "≥4 wrong",
        "Average output tokens (reasoning models, ≥4 wrong)",
        "Average output tokens",
        plots_dir / "question_difficulty_output_tokens_reasoning_min4.png",
        allowed_models=REASONING_MODEL_KEYS,
    )

    # New 4-wrong vs 6-wrong comparisons
    plot_two_series(
        all_models,
        reasoning_hard_4,
        reasoning_hard_6,
        "4 wrong",
        "6 wrong",
        "Average reasoning tokens (4 wrong vs 6 wrong questions)",
        "Average reasoning tokens",
        plots_dir / "question_difficulty_reasoning_tokens_4_vs_6.png",
        allowed_models=REASONING_MODEL_KEYS,
    )

    plot_two_series(
        all_models,
        output_hard_4,
        output_hard_6,
        "4 wrong",
        "6 wrong",
        "Average output tokens (4 wrong vs 6 wrong questions)",
        "Average output tokens",
        plots_dir / "question_difficulty_output_tokens_4_vs_6.png",
    )

    plot_hop_bar_chart(
        easy_questions,
        hard_questions_min4,
        plots_dir / "question_difficulty_hop_distribution_min4.png",
    )


if __name__ == "__main__":
    main()
