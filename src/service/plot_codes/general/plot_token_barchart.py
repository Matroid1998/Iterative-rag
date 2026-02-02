#!/usr/bin/env python3
"""
Generate bar chart showing average token usage per category for each model.

Shows side-by-side bars (Easy, Medium, Hard) for each model, making it easy
to compare how models scale their token usage with question difficulty.
"""

from __future__ import annotations

import json
from collections import defaultdict
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


def compute_average_tokens(
    model_data: Dict[str, Dict[str, dict]],
    model_name: str,
    categories: List[int]
) -> float:
    """Compute average token usage for a model in specific categories."""
    if model_name not in model_data:
        return 0.0
    
    tokens = []
    questions = model_data[model_name]
    
    for question, data in questions.items():
        if data["category"] not in categories and data["output_tokens"] is not None:
            continue
        if data["output_tokens"] is not None:
            tokens.append(data["output_tokens"])
    
    return np.mean(tokens) if tokens else 0.0


def plot_bar_chart_average_tokens(
    model_data: Dict[str, Dict[str, dict]],
    output_path: Path
) -> None:
    """
    Create bar chart showing average tokens per category for each model.
    """
    model_names = [display_name for _, display_name in ITERATIVE_MODEL_ENTRIES]
    available_models = [m for m in model_names if m in model_data]
    available_models = sort_models_non_reasoning_first(available_models)
    
    group_names = ["Easy", "Medium", "Hard"]
    n_models = len(available_models)
    
    # Compute averages
    data_matrix = []
    for model_name in available_models:
        model_avgs = []
        for group_name in group_names:
            categories = CATEGORY_GROUPS[group_name]
            avg = compute_average_tokens(model_data, model_name, categories)
            model_avgs.append(avg)
        data_matrix.append(model_avgs)
    
    data_matrix = np.array(data_matrix)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(n_models)
    width = 0.25
    
    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
    
    for i, group_name in enumerate(group_names):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, data_matrix[:, i], width, 
                     label=group_name, color=colors[i], alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=8, rotation=0)
    
    ax.set_ylabel('Average Output Tokens (log scale)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_title('Average Token Usage by Question Difficulty', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(available_models, rotation=45, ha='right')
    ax.set_yscale('log')
    ax.set_ylim(100, 25000)
    ax.legend(title='Question Difficulty', fontsize=11, title_fontsize=12)
    ax.grid(axis='y', alpha=0.3, which='both')
    
    # Add separator line between GPT-4o and GPT-5
    try:
        gpt4o_idx = available_models.index("GPT-4o")
        separator_idx = gpt4o_idx + 1
        if 0 < separator_idx < n_models:
            ax.axvline(x=separator_idx - 0.5, color='black', linestyle='--', linewidth=2, alpha=0.5)
            ax.text(separator_idx - 0.5, ax.get_ylim()[1] * 0.5, 
                   'Non-Reasoning | Reasoning', 
                   rotation=90, va='center', ha='right', fontsize=10, fontweight='bold')
    except ValueError:
        pass  # GPT-4o not in list
    
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
    output_path = project_root / "data" / "plots" / "general" / "token_usage_barchart.png"
    
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
    
    print("\nGenerating bar chart...")
    plot_bar_chart_average_tokens(model_data, output_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
