#!/usr/bin/env python3
"""Generate reasoning-model plots of total output tokens on hard questions."""

from __future__ import annotations

from pathlib import Path

from plot_hard_question_grouped import (
    compute_hard_question_data, 
    plot_grouped_bar_with_std,
    compute_average_map,
    compute_std_map,
)


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    responses_dir = base / "responses_reverified"

    model_entries = [
        (
            "responses_bedrock_mistral.mistral-large-2402-v1:0_reverified.jsonl",
            "Mistral Large 2402",
        ),
        (
            "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning_reverified.jsonl",
            "Claude 3.7 Sonnet Thinking",
        ),
        (
            "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0_reverified.jsonl",
            "Claude 3.7 Sonnet",
        ),
        (
            "responses_bedrock_us.deepseek.r1-v1:0-reasoning_reverified.jsonl",
            "DeepSeek R1",
        ),
        (
            "responses_openai_gpt-4o_reverified.jsonl",
            "GPT-4o",
        ),
        (
            "responses_openai_gpt-5_reverified.jsonl",
            "GPT-5",
        ),
    ]

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
        subtract_reasoning=False,
    )

    reasoning_models = [
        "Claude 3.7 Sonnet Thinking",
        "DeepSeek R1",
        "GPT-5",
    ]

    def filter_models(mapping: dict[int, dict[str, float | int]]) -> dict[int, dict[str, float | int]]:
        return {
            category: {
                model: value
                for model, value in model_map.items()
                if model in reasoning_models
            }
            for category, model_map in mapping.items()
        }

    def filter_token_values(mapping: dict[int, dict[str, list[int]]]) -> dict[int, dict[str, list[int]]]:
        return {
            category: {
                model: values
                for model, values in model_map.items()
                if model in reasoning_models
            }
            for category, model_map in mapping.items()
        }

    correct_counts = filter_models(correct_counts)
    incorrect_counts = filter_models(incorrect_counts)
    correct_tokens = filter_models(correct_tokens)
    incorrect_tokens = filter_models(incorrect_tokens)
    correct_token_values = filter_token_values(correct_token_values)
    incorrect_token_values = filter_token_values(incorrect_token_values)

    # Convert totals to averages
    correct_avg_tokens = compute_average_map(correct_tokens, correct_counts)
    incorrect_avg_tokens = compute_average_map(incorrect_tokens, incorrect_counts)
    
    # Compute standard deviations from individual token values
    correct_std_tokens = compute_std_map(correct_token_values)
    incorrect_std_tokens = compute_std_map(incorrect_token_values)

    model_names = reasoning_models

    categories = [4, 5, 6]
    plots_dir = base / "plots"

    model_colors = {
        "Claude 3.7 Sonnet Thinking": "#2ca02c",
        "DeepSeek R1": "#ff9896",
        "GPT-5": "#c7c7c7",
    }

    plot_grouped_bar_with_std(
        categories,
        correct_avg_tokens,
        correct_std_tokens,
        model_names,
        model_colors,
        ylabel="Average output tokens",
        title="Average output tokens on questions answered correctly (reasoning models)",
        output_path=plots_dir / "hard_questions_reasoning_correct_total_tokens.png",
    )

    plot_grouped_bar_with_std(
        categories,
        incorrect_avg_tokens,
        incorrect_std_tokens,
        model_names,
        model_colors,
        ylabel="Average output tokens",
        title="Average output tokens on questions missed (reasoning models)",
        output_path=plots_dir / "hard_questions_reasoning_incorrect_total_tokens.png",
    )

    print("Generated reasoning-model average token plots in", plots_dir)


if __name__ == "__main__":
    main()
