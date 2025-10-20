#!/usr/bin/env python3
"""
Generate box plot visualization showing token distribution across difficulty categories.

Shows how each model's token usage changes from Easy → Medium → Hard questions,
with box plots revealing both central tendency and variance.
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


def collect_token_data_by_category(
    model_data: Dict[str, Dict[str, dict]],
    model_name: str,
    group_name: str,
    categories: List[int],
    correctness_filter: str = "all"  # "all", "correct", "incorrect"
) -> List[int]:
    """
    Collect all token values for a model in a category group.
    
    Args:
        correctness_filter: "all", "correct", or "incorrect"
    """
    tokens = []
    
    if model_name not in model_data:
        return tokens
    
    questions = model_data[model_name]
    
    for question, data in questions.items():
        if data["category"] not in categories:
            continue
        if data["output_tokens"] is None:
            continue
        
        # Apply correctness filter
        if correctness_filter == "correct" and not data["is_correct"]:
            continue
        elif correctness_filter == "incorrect" and data["is_correct"]:
            continue
        
        tokens.append(data["output_tokens"])
    
    return tokens


def plot_box_plot_token_distribution(
    model_data: Dict[str, Dict[str, dict]],
    output_path: Path
) -> None:
    """
    Create box plot showing token distribution across difficulty categories.
    
    Layout: One subplot per model, 3 box plots per subplot (Easy, Medium, Hard)
    """
    model_names = [display_name for _, display_name in ITERATIVE_MODEL_ENTRIES]
    available_models = [m for m in model_names if m in model_data]
    available_models = sort_models_non_reasoning_first(available_models)
    
    n_models = len(available_models)
    n_cols = 3  # 3 columns for better layout
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    
    # Flatten axes array for easier iteration
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    group_names = ["Easy", "Medium", "Hard"]
    positions = [1, 2, 3]
    
    for idx, model_name in enumerate(available_models):
        ax = axes[idx]
        model_color = get_model_color(model_name)
        
        data_to_plot = []
        labels = []
        
        for group_name in group_names:
            categories = CATEGORY_GROUPS[group_name]
            tokens = collect_token_data_by_category(model_data, model_name, group_name, categories)
            
            if tokens:
                data_to_plot.append(tokens)
                labels.append(group_name)
            else:
                data_to_plot.append([0])  # Placeholder for empty data
                labels.append(group_name)
        
        # Create box plot
        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                        patch_artist=True, showfliers=True,
                        boxprops=dict(facecolor=model_color, alpha=0.7, edgecolor='black'),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(color='black'),
                        capprops=dict(color='black'),
                        flierprops=dict(marker='o', markersize=3, alpha=0.3))
        
        ax.set_yscale('log')
        ax.set_ylim(100, 25000)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Output Tokens (log scale)', fontsize=10)
        ax.set_title(model_name, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, which='both')
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Token Usage Distribution by Question Difficulty', 
                 fontsize=14, fontweight='bold', y=0.995)
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
    output_path = project_root / "src" / "plots" / "token_distribution_boxplot.png"
    
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
    
    print("\nGenerating box plot...")
    plot_box_plot_token_distribution(model_data, output_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
