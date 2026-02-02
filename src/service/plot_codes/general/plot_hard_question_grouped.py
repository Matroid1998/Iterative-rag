#!/usr/bin/env python3
"""Summarise hard-question performance per model with grouped bar charts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from collections import defaultdict
from itertools import cycle
from numbers import Number

import matplotlib.pyplot as plt
import numpy as np

from config import get_model_color

CATEGORIES = (9, 10, 11)


def iter_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError:
                continue


def extract_question(record: dict) -> str | None:
    question = record.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()
    for key in ("raw", "raw_response"):
        raw = record.get(key)
        if isinstance(raw, dict):
            q = raw.get("question")
            if isinstance(q, str) and q.strip():
                return q.strip()
    return None


def load_model_metrics(path: Path) -> Dict[str, Dict[str, object]]:
    metrics: Dict[str, Dict[str, object]] = {}
    for record in iter_records(path):
        question = extract_question(record)
        if not question:
            continue
        output_tokens = record.get("output_tokens")
        reasoning_tokens = record.get("reasoning_tokens")
        metrics[question] = {
            "is_correct": bool(record.get("is_correct")),
            "output_tokens": int(output_tokens) if isinstance(output_tokens, Number) else None,
            "reasoning_tokens": int(reasoning_tokens) if isinstance(reasoning_tokens, Number) else None,
        }
    return metrics


def compute_hard_question_data(
    responses_dir: Path,
    model_entries: List[Tuple[str, str]],
    subtract_reasoning: bool = True,
    categories: Iterable[int] | None = None,
) -> Tuple[
    Dict[int, List[dict]],
    Dict[int, Dict[str, int]],
    Dict[int, Dict[str, int]],
    Dict[int, Dict[str, int]],
    Dict[int, Dict[str, int]],
    Dict[int, Dict[str, List[int]]],  # Added: token values for std calculation
    Dict[int, Dict[str, List[int]]],  # Added: token values for std calculation
    List[str],
]:
    model_metrics: Dict[str, Dict[str, Dict[str, object]]] = {}

    if categories is None:
        category_sequence = list(CATEGORIES)
    else:
        category_sequence = list(dict.fromkeys(categories))
    if not category_sequence:
        raise ValueError("At least one category is required to compute hard question data.")
    category_set = set(category_sequence)

    for filename, display_name in model_entries:
        path = responses_dir / filename
        if not path.exists():
            continue
        metrics = load_model_metrics(path)
        if metrics:
            model_metrics[display_name] = metrics

    if not model_metrics:
        raise SystemExit("No reverified response files found for plotting")

    question_sets = [set(results.keys()) for results in model_metrics.values() if results]
    if not question_sets:
        raise SystemExit("No questions available after loading model responses")
    common_questions = set.intersection(*question_sets)

    category_questions: Dict[int, List[dict]] = {cat: [] for cat in category_sequence}
    incorrect_counts: Dict[int, Dict[str, int]] = {
        cat: {model: 0 for model in model_metrics.keys()} for cat in category_sequence
    }
    correct_counts: Dict[int, Dict[str, int]] = {
        cat: {model: 0 for model in model_metrics.keys()} for cat in category_sequence
    }
    incorrect_tokens: Dict[int, Dict[str, int]] = {
        cat: {model: 0 for model in model_metrics.keys()} for cat in category_sequence
    }
    correct_tokens: Dict[int, Dict[str, int]] = {
        cat: {model: 0 for model in model_metrics.keys()} for cat in category_sequence
    }
    # Add lists to store individual token values for std calculation
    incorrect_token_values: Dict[int, Dict[str, List[int]]] = {
        cat: {model: [] for model in model_metrics.keys()} for cat in category_sequence
    }
    correct_token_values: Dict[int, Dict[str, List[int]]] = {
        cat: {model: [] for model in model_metrics.keys()} for cat in category_sequence
    }

    for question in common_questions:
        wrong_models = []
        correct_models = []
        for model, metrics in model_metrics.items():
            record = metrics.get(question)
            if not record:
                continue
            if record.get("is_correct"):
                correct_models.append(model)
            else:
                wrong_models.append(model)
        wrong_count = len(wrong_models)
        if wrong_count not in category_set:
            continue
        category_questions[wrong_count].append(
            {
                "question": question,
                "models_wrong": wrong_models,
                "models_correct": correct_models,
            }
        )
        for model in wrong_models:
            incorrect_counts[wrong_count][model] += 1
            record = model_metrics[model][question]
            token_value = adjusted_output_tokens(
                model,
                record,
                subtract_reasoning=subtract_reasoning,
            )
            incorrect_tokens[wrong_count][model] += token_value
            incorrect_token_values[wrong_count][model].append(token_value)
        for model in correct_models:
            correct_counts[wrong_count][model] += 1
            record = model_metrics[model][question]
            token_value = adjusted_output_tokens(
                model,
                record,
                subtract_reasoning=subtract_reasoning,
            )
            correct_tokens[wrong_count][model] += token_value
            correct_token_values[wrong_count][model].append(token_value)

    model_names = list(model_metrics.keys())
    if not model_names:
        raise SystemExit("No model metrics loaded. Ensure reverified JSONL files are available.")
    numeric_categories = [cat for cat in category_sequence if isinstance(cat, int)]
    if numeric_categories:
        max_required = max(numeric_categories)
        if len(model_names) < max_required:
            raise SystemExit(
                f"Loaded {len(model_names)} models, but hard question categories require at least {max_required}. "
                "Verify that all reverified JSONL files are present (git lfs pull) or adjust requested categories."
            )
    return (
        category_questions,
        correct_counts,
        incorrect_counts,
        correct_tokens,
        incorrect_tokens,
        correct_token_values,
        incorrect_token_values,
        model_names,
    )


REASONING_MODELS = {
    "Claude 3.7 Sonnet Thinking",
    "DeepSeek R1",
    "GPT-5",
    "Claude Sonnet 4.5",
    "Gemini 2.5 Pro",
    "Grok 4 Fast",
    "GLM 4.6",
}


def adjusted_output_tokens(
    model: str,
    record: Dict[str, object],
    subtract_reasoning: bool = True,
) -> int:
    output_tokens = record.get("output_tokens")
    if not isinstance(output_tokens, int):
        return 0
    value = output_tokens
    if subtract_reasoning and model in REASONING_MODELS:
        reasoning_tokens = record.get("reasoning_tokens")
        if isinstance(reasoning_tokens, int):
            value = max(0, value - reasoning_tokens)
    return value


def compute_average_map(
    token_sums: Dict[int, Dict[str, int]],
    count_map: Dict[int, Dict[str, int]],
) -> Dict[int, Dict[str, float]]:
    averages: Dict[int, Dict[str, float]] = {}
    for category, model_totals in token_sums.items():
        averages[category] = {}
        for model, total in model_totals.items():
            count = count_map.get(category, {}).get(model, 0)
            if count:
                averages[category][model] = total / count
            else:
                averages[category][model] = 0.0
    return averages


def compute_std_map(
    token_values: Dict[int, Dict[str, List[int]]],
) -> Dict[int, Dict[str, float]]:
    """Compute standard deviation from individual token values."""
    import numpy as np
    std_devs: Dict[int, Dict[str, float]] = {}
    for category, model_values in token_values.items():
        std_devs[category] = {}
        for model, values in model_values.items():
            if len(values) > 1:
                std_devs[category][model] = float(np.std(values))
            else:
                std_devs[category][model] = 0.0
    return std_devs


def format_bar_label(value: float) -> str:
    if abs(value) < 1e-9:
        return ""
    if abs(value - round(value)) < 1e-6:
        return f"{int(round(value))}"
    return f"{value:.1f}"


def save_question_categories(path: Path, category_questions: Dict[int, List[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(category_questions, handle, ensure_ascii=False, indent=2)


def plot_segmented_bar(
    categories: List[int],
    counts_map: Dict[int, Dict[str, int]],
    model_names: List[str],
    model_colors: Dict[str, str],
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    value_models: Dict[int, List[str]] = defaultdict(list)
    for cat in categories:
        for model in model_names:
            value = counts_map[cat].get(model, 0)
            if value > 0:
                value_models[value].append(model)

    base_palette = [
        "#d62728",
        "#2ca02c",
        "#7f7f7f",
        "#ff9896",
        "#98df8a",
        "#c7c7c7",
    ]
    multi_palette = cycle([
        "#8c564b",
        "#e377c2",
        "#bcbd22",
        "#17becf",
    ])
    value_colors: Dict[int, str] = {}
    value_labels: Dict[int, str] = {}
    for value in sorted(value_models.keys()):
        models = sorted(set(value_models[value]))
        if len(models) == 1:
            model = models[0]
            value_colors[value] = model_colors.get(model, base_palette[0])
            value_labels[value] = f"{value}: {model}"
        else:
            color = next(multi_palette)
            value_colors[value] = color
            value_labels[value] = f"{value}: {', '.join(models)}"

    x_positions = range(len(categories))
    category_labels = [f"{cat} models wrong" for cat in categories]

    for idx, cat in enumerate(categories):
        bottom = 0
        pairs = sorted(
            [
                (counts_map[cat].get(model, 0), model)
                for model in model_names
                if counts_map[cat].get(model, 0) > 0
            ]
        )

        for value, model in pairs:
            color = value_colors.get(value, "#7f7f7f")
            ax.bar(
                x_positions[idx],
                value,
                width=0.6,
                bottom=bottom,
                color=color,
                edgecolor="#ffffff",
            )
            ax.text(
                x_positions[idx],
                bottom + value / 2,
                str(value),
                ha="center",
                va="center",
                fontsize=10,
                color="black",
            )
            bottom += value

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(category_labels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Hard questions category")
    ax.set_title(title)

    if value_models:
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color=value_colors.get(value, "#7f7f7f"))
            for value in sorted(value_labels.keys())
        ]
        legend_labels = [value_labels[value] for value in sorted(value_labels.keys())]
        ax.legend(
            handles=legend_handles,
            labels=legend_labels,
            title="Question count → models",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_bar_with_std(
    categories: List[object],
    counts_map: Dict[object, Dict[str, float]],
    std_map: Dict[object, Dict[str, float]],
    model_names: List[str],
    model_colors: Dict[str, str],
    ylabel: str,
    title: str,
    output_path: Path,
    use_log_scale: bool = False,
) -> None:
    x_positions = np.arange(len(categories))
    num_models = len(model_names)
    if num_models == 0:
        raise ValueError("At least one model is required to plot grouped bars")

    width = min(0.8 / num_models, 0.18)
    offsets = (np.arange(num_models) - (num_models - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(12, 7))

    for idx, model in enumerate(model_names):
        heights = [counts_map[cat].get(model, 0) for cat in categories]
        stds = [std_map[cat].get(model, 0) for cat in categories]
        bars = ax.bar(
            x_positions + offsets[idx],
            heights,
            width=width,
            label=model,
            color=model_colors.get(model, "#7f7f7f"),
            edgecolor="#ffffff",
            yerr=stds,
            capsize=3,
        )
        labels = [format_bar_label(float(height)) for height in heights]
        ax.bar_label(bars, labels=labels, padding=3, fontsize=7)

    def format_category_label(cat: object) -> str:
        if isinstance(cat, (int, float)):
            return f"{int(cat)} models wrong" if float(cat).is_integer() else f"{cat} models wrong"
        return str(cat)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([format_category_label(cat) for cat in categories])
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_xlabel("Question category", fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)
    
    # Add log scale if requested
    if use_log_scale:
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, linestyle='--', which='both')
    else:
        ax.grid(True, alpha=0.3, linestyle='--')
    
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, ncol=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_bar(
    categories: List[int],
    counts_map: Dict[int, Dict[str, int]],
    model_names: List[str],
    model_colors: Dict[str, str],
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    x_positions = np.arange(len(categories))
    num_models = len(model_names)
    if num_models == 0:
        raise ValueError("At least one model is required to plot grouped bars")

    width = min(0.8 / num_models, 0.18)
    offsets = (np.arange(num_models) - (num_models - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, model in enumerate(model_names):
        heights = [counts_map[cat].get(model, 0) for cat in categories]
        bars = ax.bar(
            x_positions + offsets[idx],
            heights,
            width=width,
            label=model,
            color=model_colors.get(model, "#7f7f7f"),
            edgecolor="#ffffff",
        )
        labels = [format_bar_label(float(height)) for height in heights]
        ax.bar_label(bars, labels=labels, padding=3)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{cat} models wrong" for cat in categories])
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Hard questions category")
    ax.set_title(title)
    ax.legend(loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    base = Path(__file__).resolve().parents[3]
    responses_dir = base / "responses_reverified"

    model_entries: List[Tuple[str, str]] = [
        ("responses_bedrock_mistral.mistral-large-2402-v1:0_reverified.jsonl", "Mistral Large 2402"),
        ("responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning_reverified.jsonl", "Claude 3.7 Sonnet Thinking"),
        ("responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0_reverified.jsonl", "Claude 3.7 Sonnet"),
        ("responses_bedrock_us.deepseek.r1-v1:0-reasoning_reverified.jsonl", "DeepSeek R1"),
        ("responses_bedrock_us.meta.llama3-3-70b-instruct-v1:0_reverified.jsonl", "Llama 3.3 70B Instruct"),
        ("responses_openai_gpt-4o_reverified.jsonl", "GPT-4o"),
        ("responses_openai_gpt-5_reverified.jsonl", "GPT-5"),
        ("responses_openrouter_anthropic__claude-sonnet-4.5_reverified.jsonl", "Claude Sonnet 4.5"),
        ("responses_openrouter_google__gemini-2.5-pro_reverified.jsonl", "Gemini 2.5 Pro"),
        ("responses_openrouter_x-ai__grok-4-fast_reverified.jsonl", "Grok 4 Fast"),
        ("responses_openrouter_z-ai__glm-4.6_reverified.jsonl", "GLM 4.6"),
    ]

    (
        category_questions,
        correct_counts,
        incorrect_counts,
        correct_tokens,
        incorrect_tokens,
        _correct_token_values,
        _incorrect_token_values,
        model_names,
    ) = compute_hard_question_data(responses_dir, model_entries)

    categories = list(CATEGORIES)
    category_file = base / "results" / "unanswered_questions" / "hard_question_categories.json"
    save_question_categories(category_file, category_questions)

    model_colors = {model: get_model_color(model) for model in model_names}

    correct_token_avgs = compute_average_map(correct_tokens, correct_counts)
    incorrect_token_avgs = compute_average_map(incorrect_tokens, incorrect_counts)

    plots_dir = base / "plots"
    plot_grouped_bar(
        categories,
        correct_counts,
        model_names,
        model_colors,
        ylabel="Questions answered correctly",
        title="Hard questions answered by the models",
        output_path=plots_dir / "hard_questions_correct_grouped.png",
    )

    plot_grouped_bar(
        categories,
        incorrect_counts,
        model_names,
        model_colors,
        ylabel="Questions answered incorrectly",
        title="Hard questions missed by the models",
        output_path=plots_dir / "hard_questions_incorrect_grouped.png",
    )

    plot_grouped_bar(
        categories,
        correct_token_avgs,
        model_names,
        model_colors,
        ylabel="Average output tokens",
        title="Output tokens on questions answered correctly",
        output_path=plots_dir / "hard_questions_correct_grouped_tokens.png",
    )

    plot_grouped_bar(
        categories,
        incorrect_token_avgs,
        model_names,
        model_colors,
        ylabel="Average output tokens",
        title="Output tokens on questions missed by the models",
        output_path=plots_dir / "hard_questions_incorrect_grouped_tokens.png",
    )

    plot_segmented_bar(
        categories,
        correct_counts,
        model_names,
        model_colors,
        ylabel="Questions answered correctly",
        title="Hard questions answered by the models",
        output_path=plots_dir / "hard_questions_correct_segments.png",
    )

    plot_segmented_bar(
        categories,
        incorrect_counts,
        model_names,
        model_colors,
        ylabel="Questions answered incorrectly",
        title="Hard questions missed by the models",
        output_path=plots_dir / "hard_questions_incorrect_segments.png",
    )

    print(f"Stored hard-question categories in {category_file}")
    print("Generated grouped bar plots in", plots_dir)


if __name__ == "__main__":
    main()
