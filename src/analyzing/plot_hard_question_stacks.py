#!/usr/bin/env python3
"""Plot stacked bar chart of hard-question accuracy across models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


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


def load_model_answers(path: Path) -> Dict[str, bool]:
    results: Dict[str, bool] = {}
    for record in iter_records(path):
        question = extract_question(record)
        if not question:
            continue
        results[question] = bool(record.get("is_correct"))
    return results


def plot_hard_question_stack(
    categories: List[int],
    model_names: List[str],
    model_colors: Dict[str, str],
    category_counts: Dict[int, Dict[str, int]],
    output_path: Path,
) -> None:
    x_positions = np.arange(len(categories))
    bottoms = np.zeros(len(categories), dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))

    for model in model_names:
        heights = [category_counts[cat].get(model, 0) for cat in categories]
        bars = ax.bar(
            x_positions,
            heights,
            bottom=bottoms,
            color=model_colors.get(model, "#7f7f7f"),
            label=model,
        )
        ax.bar_label(bars, padding=2)
        bottoms += np.array(heights)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{cat} models wrong" for cat in categories])
    ax.set_ylabel("Questions answered correctly")
    ax.set_xlabel("Hard questions category")
    ax.set_title("Hard questions answered by the models")
    ax.legend(loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    responses_dir = base / "responses_reverified"

    model_entries: List[Tuple[str, str]] = [
        ("responses_bedrock_mistral.mistral-large-2402-v1:0_reverified.jsonl", "Mistral Large 2402"),
        ("responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning_reverified.jsonl", "Claude 3.7 Sonnet Thinking"),
        ("responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0_reverified.jsonl", "Claude 3.7 Sonnet"),
        ("responses_bedrock_us.deepseek.r1-v1:0-reasoning_reverified.jsonl", "DeepSeek R1"),
        ("responses_openai_gpt-4o_reverified.jsonl", "GPT-4o"),
        ("responses_openai_gpt-5_reverified.jsonl", "GPT-5"),
    ]

    model_answers: Dict[str, Dict[str, bool]] = {}
    for filename, display_name in model_entries:
        path = responses_dir / filename
        if not path.exists():
            continue
        model_answers[display_name] = load_model_answers(path)

    if not model_answers:
        raise SystemExit("No reverified response files found for plotting")

    # Focus on questions present in all models considered
    question_sets = [set(results.keys()) for results in model_answers.values() if results]
    if not question_sets:
        raise SystemExit("No questions available after loading model responses")
    common_questions = set.intersection(*question_sets)

    categories = [4, 5, 6]
    model_names = list(model_answers.keys())
    category_counts: Dict[int, Dict[str, int]] = {
        cat: {model: 0 for model in model_names} for cat in categories
    }

    for question in common_questions:
        wrong_count = sum(
            1 for model in model_names if not model_answers[model].get(question, False)
        )
        if wrong_count in category_counts:
            for model in model_names:
                if model_answers[model].get(question, False):
                    category_counts[wrong_count][model] += 1

    model_colors = {
        "Mistral Large 2402": "#d62728",
        "Claude 3.7 Sonnet Thinking": "#2ca02c",
        "Claude 3.7 Sonnet": "#7f7f7f",
        "DeepSeek R1": "#ff9896",
        "GPT-4o": "#98df8a",
        "GPT-5": "#c7c7c7",
    }

    output_path = base / "plots" / "hard_questions_correct_stacked.png"
    plot_hard_question_stack(categories, model_names, model_colors, category_counts, output_path)
    print(f"Generated {output_path}")


if __name__ == "__main__":
    main()
