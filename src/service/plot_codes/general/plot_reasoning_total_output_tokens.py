#!/usr/bin/env python3
"""Generate reasoning-model plots of total output tokens on hard questions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from config import get_model_color
from plot_hard_question_grouped import (
    compute_hard_question_data,
    plot_grouped_bar_with_std,
    compute_average_map,
    compute_std_map,
)

CATEGORY_GROUPS = {
    "Easy": (0, 1, 2),
    "Medium": (5, 6, 7),
    "Hard": (9, 10, 11),
}


def _aggregate_by_group(
    source_map: Dict[int, Dict[str, int]],
    model_names: List[str],
    groups: Dict[str, Iterable[int]],
) -> Dict[str, Dict[str, int]]:
    aggregated: Dict[str, Dict[str, int]] = {
        label: {model: 0 for model in model_names} for label in groups
    }
    for label, categories in groups.items():
        for category in categories:
            for model, value in source_map.get(category, {}).items():
                aggregated[label][model] += value
    return aggregated


def _aggregate_token_values(
    source_map: Dict[int, Dict[str, List[int]]],
    model_names: List[str],
    groups: Dict[str, Iterable[int]],
) -> Dict[str, Dict[str, List[int]]]:
    aggregated: Dict[str, Dict[str, List[int]]] = {
        label: {model: [] for model in model_names} for label in groups
    }
    for label, categories in groups.items():
        for category in categories:
            for model, values in source_map.get(category, {}).items():
                aggregated[label][model].extend(values)
    return aggregated


def main() -> None:
    base = Path(__file__).resolve().parents[3]
    responses_dir = base / "responses_reverified"

    # Use ITERATIVE_MODEL_ENTRIES from config to include all models
    from config import ITERATIVE_MODEL_ENTRIES
    
    model_entries = list(ITERATIVE_MODEL_ENTRIES)

    (
        _category_questions,
        raw_correct_counts,
        raw_incorrect_counts,
        raw_correct_tokens,
        raw_incorrect_tokens,
        raw_correct_token_values,
        raw_incorrect_token_values,
        model_names,
    ) = compute_hard_question_data(
        responses_dir,
        model_entries,
        subtract_reasoning=False,  # Use full output_tokens (not subtracting reasoning)
        categories=sorted({cat for cats in CATEGORY_GROUPS.values() for cat in cats}),
    )

    # Aggregate raw category statistics into easy/medium/hard groups
    correct_counts = _aggregate_by_group(raw_correct_counts, model_names, CATEGORY_GROUPS)
    incorrect_counts = _aggregate_by_group(raw_incorrect_counts, model_names, CATEGORY_GROUPS)
    correct_tokens = _aggregate_by_group(raw_correct_tokens, model_names, CATEGORY_GROUPS)
    incorrect_tokens = _aggregate_by_group(raw_incorrect_tokens, model_names, CATEGORY_GROUPS)
    correct_token_values = _aggregate_token_values(
        raw_correct_token_values, model_names, CATEGORY_GROUPS
    )
    incorrect_token_values = _aggregate_token_values(
        raw_incorrect_token_values, model_names, CATEGORY_GROUPS
    )

    # Convert totals to averages
    correct_avg_tokens = compute_average_map(correct_tokens, correct_counts)
    incorrect_avg_tokens = compute_average_map(incorrect_tokens, incorrect_counts)
    
    # Compute standard deviations from individual token values
    correct_std_tokens = compute_std_map(correct_token_values)
    incorrect_std_tokens = compute_std_map(incorrect_token_values)

    # model_names already contains all models from compute_hard_question_data

    categories = list(CATEGORY_GROUPS.keys())
    plots_dir = base.parent / "data" / "plots" / "general"

    model_colors = {model: get_model_color(model) for model in model_names}

    plot_grouped_bar_with_std(
        categories,
        correct_avg_tokens,
        correct_std_tokens,
        model_names,
        model_colors,
        ylabel="Average Output Tokens (log scale)",
        title="Average Output Tokens on Questions Answered Correctly (All Models)",
        output_path=plots_dir / "hard_questions_reasoning_correct_total_tokens.png",
        use_log_scale=True,
    )

    plot_grouped_bar_with_std(
        categories,
        incorrect_avg_tokens,
        incorrect_std_tokens,
        model_names,
        model_colors,
        ylabel="Average Output Tokens (log scale)",
        title="Average Output Tokens on Questions Missed (All Models)",
        output_path=plots_dir / "hard_questions_reasoning_incorrect_total_tokens.png",
        use_log_scale=True,
    )

    print("Generated all-model average output token plots (with log scale) in", plots_dir)


if __name__ == "__main__":
    main()
