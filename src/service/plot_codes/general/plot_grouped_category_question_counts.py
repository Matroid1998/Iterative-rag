#!/usr/bin/env python3
"""
Generate comprehensive analysis plot for Easy/Medium/Hard question categories.
Shows count of correct vs incorrect questions per retrieval step.

This script creates a multi-panel plot with:
- 3 columns: Easy (0,1,2 models wrong), Medium (5,6,7 models wrong), Hard (9,10,11 models wrong)
- First row: Distribution of questions by number of hops (retrieval steps)
- Subsequent rows: Count of correct vs incorrect questions per retrieval step per model
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

# Define reasoning models (models that use chain-of-thought reasoning)
REASONING_MODELS = {
    "Claude 3.7 Sonnet Thinking",
    "DeepSeek R1",
    "GPT-o1",
    "GPT-o3"
}


def sort_models_non_reasoning_first(model_names: List[str]) -> List[str]:
    """
    Sort models so non-reasoning models appear first, then reasoning models.
    Within each group, maintain original order.
    """
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


def extract_max_source_step(record: dict) -> int | None:
    """Return the maximum retrieval step (source_step) found in a record."""
    steps: List[int] = []
    for key in ("raw_response", "raw"):
        raw = record.get(key)
        if not isinstance(raw, dict):
            continue
        evidence = raw.get("evidence")
        if not isinstance(evidence, list):
            continue
        for item in evidence:
            if not isinstance(item, dict):
                continue
            step = item.get("source_step")
            if isinstance(step, (int, float)):
                step_int = int(round(step))
                if step_int > 0:
                    steps.append(step_int)
    if steps:
        return max(steps)
    return None


def extract_hop_count(record: dict) -> int | None:
    """Extract the number of hops from a record."""
    raw_hops = record.get("number_of_hops")
    if isinstance(raw_hops, (int, float)):
        return int(round(raw_hops))
    return None


def load_hard_question_categories(hard_questions_path: Path) -> Dict[int, Set[str]]:
    """
    Load hard question categories and return mapping of category -> set of questions.
    
    Returns:
        Dict mapping category number (0-11) to set of question strings
    """
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
    """
    Load model response data for all questions.
    
    Returns:
        Dict[model_name][question] = {
            "is_correct": bool,
            "output_tokens": int,
            "max_source_step": int,
            "hop_count": int,
            "category": int (0-11)
        }
    """
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
            max_source_step = extract_max_source_step(record)
            hop_count = extract_hop_count(record)
            
            model_data[display_name][question] = {
                "is_correct": is_correct,
                "output_tokens": int(output_tokens) if isinstance(output_tokens, Number) else None,
                "max_source_step": max_source_step,
                "hop_count": hop_count,
                "category": category
            }
    
    return model_data


def compute_hop_distribution(
    model_data: Dict[str, Dict[str, dict]],
    group_name: str,
    categories: List[int]
) -> Dict[int, int]:
    """
    Compute hop distribution for a category group across all models.
    
    Returns:
        Dict[hop_count] = question_count
    """
    hop_counter: Dict[int, Set[str]] = defaultdict(set)
    
    for model_name, questions in model_data.items():
        for question, data in questions.items():
            if data["category"] in categories and data["hop_count"] is not None:
                hop_counter[data["hop_count"]].add(question)
    
    # Convert sets to counts
    return {hop: len(questions) for hop, questions in hop_counter.items()}


def compute_model_question_counts(
    model_data: Dict[str, Dict[str, dict]],
    model_name: str,
    categories: List[int]
) -> Dict[str, Dict[int, int]]:
    """
    Compute question counts for a model across retrieval steps.
    
    Returns:
        {
            "correct": {step: count},
            "incorrect": {step: count}
        }
    """
    correct_counts: Dict[int, int] = defaultdict(int)
    incorrect_counts: Dict[int, int] = defaultdict(int)
    
    if model_name not in model_data:
        return {"correct": {}, "incorrect": {}}
    
    questions = model_data[model_name]
    
    for question, data in questions.items():
        if data["category"] not in categories:
            continue
        if data["max_source_step"] is None:
            continue
        
        step = data["max_source_step"]
        
        if data["is_correct"]:
            correct_counts[step] += 1
        else:
            incorrect_counts[step] += 1
    
    return {"correct": dict(correct_counts), "incorrect": dict(incorrect_counts)}


def plot_grouped_category_question_counts(
    model_data: Dict[str, Dict[str, dict]],
    output_path: Path
) -> None:
    """
    Create comprehensive plot with Easy/Medium/Hard columns and model rows.
    Shows count of correct vs incorrect questions.
    
    Layout:
    - Row 0: Hop distributions for each category group
    - Rows 1+: Question counts per model (correct vs incorrect)
    """
    group_names = ["Easy", "Medium", "Hard"]
    model_names = [display_name for _, display_name in ITERATIVE_MODEL_ENTRIES]
    
    # Filter to models that have data
    available_models = [m for m in model_names if m in model_data]
    
    # Sort models: non-reasoning first, then reasoning
    available_models = sort_models_non_reasoning_first(available_models)
    
    n_rows = len(available_models) + 1  # +1 for hop distribution row
    n_cols = len(group_names)
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(16, 3 + 2 * len(available_models)),
        gridspec_kw={'height_ratios': [1.5] + [1] * len(available_models)}
    )
    
    # Ensure axes is 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Row 0: Hop distributions
    for col_idx, group_name in enumerate(group_names):
        ax = axes[0, col_idx]
        categories = CATEGORY_GROUPS[group_name]
        hop_dist = compute_hop_distribution(model_data, group_name, categories)
        
        if hop_dist:
            steps = sorted(hop_dist.keys())
            counts = [hop_dist[s] for s in steps]
            
            bars = ax.bar(steps, counts, color='steelblue', edgecolor='black', alpha=0.7)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontsize=9)
            
            ax.set_xlabel('Number of Hops', fontsize=10)
            ax.set_ylabel('Number of Questions', fontsize=10)
            ax.set_title(f'{group_name} Questions\n(Categories: {", ".join(map(str, categories))})',
                        fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.set_xticks(steps)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{group_name} Questions', fontsize=11, fontweight='bold')
    
    # Rows 1+: Model question counts
    max_step = 5  # Maximum retrieval steps to show
    all_steps = list(range(1, max_step + 1))
    
    # Find y-axis range for Easy questions (column 0) - dynamic
    max_count_easy = 0
    for model_name in available_models:
        categories = CATEGORY_GROUPS["Easy"]
        counts = compute_model_question_counts(model_data, model_name, categories)
        for step_dict in [counts["correct"], counts["incorrect"]]:
            if step_dict:
                max_count_easy = max(max_count_easy, max(step_dict.values()))
    
    y_max_easy = max(max_count_easy * 1.15, 10)  # At least 10 for scale
    
    # Fixed y-axis for Medium and Hard questions (columns 1 and 2)
    y_max_medium_hard = 50
    
    for row_idx, model_name in enumerate(available_models, start=1):
        model_color = get_model_color(model_name)
        
        for col_idx, group_name in enumerate(group_names):
            ax = axes[row_idx, col_idx]
            categories = CATEGORY_GROUPS[group_name]
            counts = compute_model_question_counts(model_data, model_name, categories)
            
            correct_counts = counts["correct"]
            incorrect_counts = counts["incorrect"]
            
            # Get counts per step
            correct_vals = [correct_counts.get(step, 0) for step in all_steps]
            incorrect_vals = [incorrect_counts.get(step, 0) for step in all_steps]
            
            # Plot grouped bars
            x = np.arange(len(all_steps))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, correct_vals, width,
                          label='Correct', color=model_color, alpha=0.8, edgecolor='black')
            bars2 = ax.bar(x + width/2, incorrect_vals, width,
                          label='Incorrect', color=model_color, alpha=0.4, edgecolor='black',
                          hatch='///')
            
            # Add value labels on bars (only if non-zero)
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}',
                               ha='center', va='bottom', fontsize=7)
            
            ax.set_xlabel('Retrieval Step', fontsize=9)
            ax.set_ylabel('Number of Questions', fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(all_steps)
            # Set y-axis limit based on column
            if col_idx == 0:  # Easy questions
                ax.set_ylim(0, y_max_easy)
            else:  # Medium and Hard questions
                ax.set_ylim(0, y_max_medium_hard)
            ax.grid(axis='y', alpha=0.3)
            
            # Only show legend on first column
            if col_idx == 0:
                ax.set_ylabel(f'{model_name}\nNumber of Questions', fontsize=9)
                if row_idx == 1:
                    ax.legend(loc='upper right', fontsize=8)
            
            # Only show model name on leftmost column
            if col_idx > 0:
                ax.set_ylabel('')
    
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
    output_path = project_root / "data" / "plots" / "general" / "grouped_category_question_counts.png"
    
    print("Loading hard question categories...")
    question_categories = load_hard_question_categories(hard_questions_path)
    
    if not question_categories:
        print("Error: No question categories loaded!")
        return
    
    # Print category statistics
    print("\nCategory statistics:")
    for group_name, categories in CATEGORY_GROUPS.items():
        total = sum(len(question_categories.get(cat, set())) for cat in categories)
        print(f"  {group_name}: {total} questions (categories {categories})")
    
    print("\nLoading model data...")
    model_data = load_model_data(responses_dir, ITERATIVE_MODEL_ENTRIES, question_categories)
    
    if not model_data:
        print("Error: No model data loaded!")
        return
    
    print(f"Loaded data for {len(model_data)} models")
    
    print("\nGenerating plot...")
    plot_grouped_category_question_counts(model_data, output_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
