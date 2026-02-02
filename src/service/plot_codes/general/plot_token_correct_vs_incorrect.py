#!/usr/bin/env python3
"""
Generate side-by-side plots showing token usage for correct vs incorrect answers.

Shows how token usage differs when models answer correctly vs incorrectly,
across different difficulty levels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Set
from numbers import Number

import matplotlib.pyplot as plt
import numpy as np

from config import ITERATIVE_MODEL_ENTRIES, get_model_color


# Define category groupings
CATEGORY_GROUPS = {
    "Easy": [0, 1, 2],
    "Medium": [5, 6, 7],
    "Hard": [9, 10, 11]
}

# Define reasoning models
REASONING_MODELS = {
    "Claude 3.7 Sonnet Thinking",
    "DeepSeek R1",
    "GPT-o1",
    "GPT-o3"
}


def sort_models_non_reasoning_first(model_names: List[str]) -> List[str]:
    """Sort models so non-reasoning models appear first, then reasoning models."""
    non_reasoning = []
    reasoning = []
    
    for model in model_names:
        if model in REASONING_MODELS:
            reasoning.append(model)
        else:
            non_reasoning.append(model)
    
    return non_reasoning + reasoning


def iter_records(path: Path) -> Iterable[dict]:
    """Iterate over JSONL records."""
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
    """Extract question text from a record."""
    question = record.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()
    
    for key in ("raw_response", "raw"):
        raw = record.get(key)
        if isinstance(raw, dict):
            q = raw.get("question")
            if isinstance(q, str) and q.strip():
                return q.strip()
    return None


def load_hard_question_categories(hard_questions_path: Path) -> Dict[int, Set[str]]:
    """Load hard question categories."""
    if not hard_questions_path.exists():
        print(f"Warning: Hard questions file not found: {hard_questions_path}")
        return {}
    
    categories: Dict[int, Set[str]] = {}
    with hard_questions_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    
    for category_str, questions in data.items():
        category = int(category_str)
        categories[category] = set()
        for q_data in questions:
            question = q_data.get("question")
            if isinstance(question, str) and question.strip():
                categories[category].add(question.strip())
    
    return categories


def load_model_data(
    responses_dir: Path,
    model_entries: List[Tuple[str, str]],
    question_categories: Dict[int, Set[str]]
) -> Dict[str, Dict[str, dict]]:
    """Load model response data for all questions."""
    # Create reverse mapping: question -> category
    question_to_category: Dict[str, int] = {}
    for category, questions in question_categories.items():
        for question in questions:
            question_to_category[question] = category
    
    model_data: Dict[str, Dict[str, dict]] = {}
    
    for filename, display_name in model_entries:
        path = responses_dir / filename
        if not path.exists():
            print(f"Warning: File not found: {path}")
            continue
        
        model_data[display_name] = {}
        
        for record in iter_records(path):
            question = extract_question(record)
            if not question or question not in question_to_category:
                continue
            
            category = question_to_category[question]
            is_correct = bool(record.get("is_correct", False))
            output_tokens = record.get("output_tokens")
            
            model_data[display_name][question] = {
                "is_correct": is_correct,
                "output_tokens": int(output_tokens) if isinstance(output_tokens, Number) else None,
                "category": category
            }
    
    return model_data


def compute_average_tokens_by_correctness(
    model_data: Dict[str, Dict[str, dict]],
    model_name: str,
    categories: List[int],
    is_correct: bool
) -> float:
    """Compute average token usage for correct or incorrect answers."""
    if model_name not in model_data:
        return 0.0
    
    tokens = []
    questions = model_data[model_name]
    
    for question, data in questions.items():
        if data["category"] in categories and data["is_correct"] == is_correct:
            if data["output_tokens"] is not None:
                tokens.append(data["output_tokens"])
    
    return np.mean(tokens) if tokens else 0.0


def plot_correct_vs_incorrect_tokens(
    model_data: Dict[str, Dict[str, dict]],
    output_path: Path
) -> None:
    """
    Create side-by-side bar charts for correct vs incorrect token usage.
    """
    model_names = [display_name for _, display_name in ITERATIVE_MODEL_ENTRIES]
    available_models = [m for m in model_names if m in model_data]
    available_models = sort_models_non_reasoning_first(available_models)
    
    group_names = ["Easy", "Medium", "Hard"]
    n_models = len(available_models)
    
    # Compute averages for correct and incorrect
    correct_data = []
    incorrect_data = []
    
    for model_name in available_models:
        correct_row = []
        incorrect_row = []
        for group_name in group_names:
            categories = CATEGORY_GROUPS[group_name]
            correct_avg = compute_average_tokens_by_correctness(model_data, model_name, categories, True)
            incorrect_avg = compute_average_tokens_by_correctness(model_data, model_name, categories, False)
            correct_row.append(correct_avg)
            incorrect_row.append(incorrect_avg)
        correct_data.append(correct_row)
        incorrect_data.append(incorrect_row)
    
    correct_data = np.array(correct_data)
    incorrect_data = np.array(incorrect_data)
    
    # Create side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    
    x = np.arange(n_models)
    width = 0.25
    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
    
    # Left plot: Correct answers
    for i, group_name in enumerate(group_names):
        offset = (i - 1) * width
        bars = ax1.bar(x + offset, correct_data[:, i], width, 
                      label=group_name, color=colors[i], alpha=0.8, edgecolor='black')
        
        # Add value labels
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=7, rotation=0)
    
    ax1.set_ylabel('Average Output Tokens (log scale)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax1.set_title('Token Usage for CORRECT Answers', fontsize=13, fontweight='bold', 
                 color='green')
    ax1.set_xticks(x)
    ax1.set_xticklabels(available_models, rotation=45, ha='right')
    ax1.set_yscale('log')
    ax1.set_ylim(100, 25000)
    ax1.legend(title='Question Difficulty', fontsize=10, title_fontsize=11)
    ax1.grid(axis='y', alpha=0.3, which='both')
    
    # Add separator line between GPT-4o and GPT-5
    try:
        gpt4o_idx = available_models.index("GPT-4o")
        separator_idx = gpt4o_idx + 1
        if 0 < separator_idx < n_models:
            ax1.axvline(x=separator_idx - 0.5, color='black', linestyle='--', linewidth=2, alpha=0.5)
    except ValueError:
        pass  # GPT-4o not in list
    
    # Right plot: Incorrect answers
    for i, group_name in enumerate(group_names):
        offset = (i - 1) * width
        bars = ax2.bar(x + offset, incorrect_data[:, i], width, 
                      label=group_name, color=colors[i], alpha=0.8, edgecolor='black')
        
        # Add value labels
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=7, rotation=0)
    
    ax2.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax2.set_title('Token Usage for INCORRECT Answers', fontsize=13, fontweight='bold',
                 color='red')
    ax2.set_xticks(x)
    ax2.set_xticklabels(available_models, rotation=45, ha='right')
    ax2.set_yscale('log')
    ax2.legend(title='Question Difficulty', fontsize=10, title_fontsize=11)
    ax2.grid(axis='y', alpha=0.3, which='both')
    
    # Add separator line between GPT-4o and GPT-5
    try:
        gpt4o_idx = available_models.index("GPT-4o")
        separator_idx = gpt4o_idx + 1
        if 0 < separator_idx < n_models:
            ax2.axvline(x=separator_idx - 0.5, color='black', linestyle='--', linewidth=2, alpha=0.5)
    except ValueError:
        pass  # GPT-4o not in list
    
    plt.suptitle('Token Usage Comparison: Correct vs Incorrect Answers', 
                fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {output_path}")


def main():
    """Main execution function."""
    project_root = Path(__file__).resolve().parents[2]
    responses_dir = project_root / "src" / "responses_reverified"
    hard_questions_path = project_root / "src" / "results" / "unanswered_questions" / "hard_question_categories.json"
    output_path = project_root / "data" / "plots" / "general" / "token_usage_correct_vs_incorrect.png"
    
    print("Loading hard question categories...")
    question_categories = load_hard_question_categories(hard_questions_path)
    
    if not question_categories:
        print("Error: No question categories loaded!")
        return
    
    print("\nLoading model data...")
    model_data = load_model_data(responses_dir, ITERATIVE_MODEL_ENTRIES, question_categories)
    
    if not model_data:
        print("Error: No model data loaded!")
        return
    
    print(f"Loaded data for {len(model_data)} models")
    
    print("\nGenerating correct vs incorrect comparison...")
    plot_correct_vs_incorrect_tokens(model_data, output_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
