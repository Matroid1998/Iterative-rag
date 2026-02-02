#!/usr/bin/env python3
"""
Generate scatter plot showing Accuracy vs Token Usage.

Creates a 2D scatter plot to identify model efficiency patterns:
- Bottom-right quadrant: Efficient (low tokens, high accuracy) - IDEAL
- Top-right quadrant: Thorough (high tokens, high accuracy)
- Top-left quadrant: Wasteful (high tokens, low accuracy) - WORST
- Bottom-left quadrant: Quick but wrong (low tokens, low accuracy)
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
    question_to_category: Dict[str, int] = {}
    for category, questions in question_categories.items():
        for question in questions:
            question_to_category[question] = category
    
    model_data: Dict[str, Dict[str, dict]] = {}
    
    for filename, display_name in model_entries:
        path = responses_dir / filename
        if not path.exists():
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


def compute_model_stats(
    model_data: Dict[str, Dict[str, dict]],
    model_name: str
) -> Tuple[float, float, int]:
    """
    Compute overall accuracy and average tokens for a model.
    Returns: (accuracy, avg_tokens, question_count)
    """
    if model_name not in model_data:
        return 0.0, 0.0, 0
    
    questions = model_data[model_name]
    correct_count = 0
    total_count = 0
    tokens = []
    
    for question, data in questions.items():
        if data["output_tokens"] is not None:
            total_count += 1
            tokens.append(data["output_tokens"])
            if data["is_correct"]:
                correct_count += 1
    
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    avg_tokens = np.mean(tokens) if tokens else 0
    
    return accuracy, avg_tokens, total_count


def plot_accuracy_vs_tokens_scatter(
    model_data: Dict[str, Dict[str, dict]],
    output_path: Path
) -> None:
    """Create scatter plot of accuracy vs token usage."""
    model_names = [display_name for _, display_name in ITERATIVE_MODEL_ENTRIES]
    available_models = [m for m in model_names if m in model_data]
    
    # Compute stats for each model
    accuracies = []
    avg_tokens = []
    question_counts = []
    colors = []
    
    for model_name in available_models:
        accuracy, tokens, count = compute_model_stats(model_data, model_name)
        accuracies.append(accuracy)
        avg_tokens.append(tokens)
        question_counts.append(count)
        colors.append(get_model_color(model_name))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Scale point sizes by question count
    sizes = [count * 0.5 for count in question_counts]
    
    # Plot points
    scatter = ax.scatter(accuracies, avg_tokens, s=sizes, c=colors, 
                        alpha=0.7, edgecolors='black', linewidth=2)
    
    # Add model labels
    for i, model_name in enumerate(available_models):
        ax.annotate(model_name, (accuracies[i], avg_tokens[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Add quadrant lines
    median_accuracy = np.median(accuracies)
    median_tokens = np.median(avg_tokens)
    
    ax.axvline(x=median_accuracy, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.axhline(y=median_tokens, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Label quadrants
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    ax.text(xlim[0] + (median_accuracy - xlim[0]) / 2, ylim[1] * 0.95,
           'Wasteful\n(High tokens, Low accuracy)',
           ha='center', va='top', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.7))
    
    ax.text(xlim[1] - (xlim[1] - median_accuracy) / 2, ylim[1] * 0.95,
           'Thorough\n(High tokens, High accuracy)',
           ha='center', va='top', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.7))
    
    ax.text(xlim[0] + (median_accuracy - xlim[0]) / 2, ylim[0] + (median_tokens - ylim[0]) * 0.1,
           'Quick but Wrong\n(Low tokens, Low accuracy)',
           ha='center', va='bottom', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#ffddcc', alpha=0.7))
    
    ax.text(xlim[1] - (xlim[1] - median_accuracy) / 2, ylim[0] + (median_tokens - ylim[0]) * 0.1,
           'EFFICIENT\n(Low tokens, High accuracy)',
           ha='center', va='bottom', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.7))
    
    ax.set_xlabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Output Tokens (log scale)', fontsize=13, fontweight='bold')
    ax.set_title('Model Efficiency: Accuracy vs Token Usage\n(Bottom-right is ideal: High accuracy, Low tokens)', 
                fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    
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
    output_path = project_root / "data" / "plots" / "general" / "accuracy_vs_tokens_scatter.png"
    
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
    
    print("\nGenerating accuracy vs tokens scatter plot...")
    plot_accuracy_vs_tokens_scatter(model_data, output_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
