#!/usr/bin/env python3
"""Generate reasoning-model plots of total output tokens on hard questions."""

from __future__ import annotations

from pathlib import Path

from config import get_model_color
from plot_hard_question_grouped import (
    CATEGORIES,
    compute_hard_question_data,
    plot_grouped_bar_with_std,
    compute_average_map,
    compute_std_map,
)


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    responses_dir = base / "responses_reverified"

    # Use ITERATIVE_MODEL_ENTRIES from config to include all models
    from config import ITERATIVE_MODEL_ENTRIES
    
    model_entries = list(ITERATIVE_MODEL_ENTRIES)

    (
        _category_questions,
        correct_counts,
        incorrect_counts,
        correct_tokens,
        incorrect_tokens,
        correct_token_values,
        incorrect_token_values,
        model_names,
    ) = compute_hard_question_data(
        responses_dir,
        model_entries,
        subtract_reasoning=False,  # Use full output_tokens (not subtracting reasoning)
    )

    # No filtering - use all models
    # correct_counts, incorrect_counts, etc. already contain all models

    # Convert totals to averages
    correct_avg_tokens = compute_average_map(correct_tokens, correct_counts)
    incorrect_avg_tokens = compute_average_map(incorrect_tokens, incorrect_counts)
    
    # Compute standard deviations from individual token values
    correct_std_tokens = compute_std_map(correct_token_values)
    incorrect_std_tokens = compute_std_map(incorrect_token_values)

    # model_names already contains all models from compute_hard_question_data

    categories = list(CATEGORIES)
    plots_dir = base / "plots"

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
