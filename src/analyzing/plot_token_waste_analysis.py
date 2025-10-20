#!/usr/bin/env python3
"""
Generate Token Waste Analysis plot showing tokens wasted on incorrect answers.

Shows what percentage of total tokens were spent on questions that were
answered incorrectly. Lower is better (less computational waste).
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


def compute_waste_percentage(
    model_data: Dict[str, Dict[str, dict]],
    model_name: str,
    categories: List[int]
) -> Tuple[float, int, int]:
    """
    Compute percentage of tokens wasted on incorrect answers.
    Returns: (waste_percentage, wasted_tokens, total_tokens)
    """
    if model_name not in model_data:
        return 0.0, 0, 0
    
    total_tokens = 0
    wasted_tokens = 0
    
    questions = model_data[model_name]
    
    for question, data in questions.items():
        if data["category"] in categories and data["output_tokens"] is not None:
            total_tokens += data["output_tokens"]
            if not data["is_correct"]:
                wasted_tokens += data["output_tokens"]
    
    waste_pct = (wasted_tokens / total_tokens * 100) if total_tokens > 0 else 0
    
    return waste_pct, wasted_tokens, total_tokens


def plot_token_waste_analysis(
    model_data: Dict[str, Dict[str, dict]],
    output_path: Path
) -> None:
    """Create plot showing token waste percentage."""
    model_names = [display_name for _, display_name in ITERATIVE_MODEL_ENTRIES]
    available_models = [m for m in model_names if m in model_data]
    available_models = sort_models_non_reasoning_first(available_models)
    
    group_names = ["Easy", "Medium", "Hard"]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    # Compute waste percentages
    waste_data = []
    for model_name in available_models:
        model_wastes = []
        for group_name in group_names:
            categories = CATEGORY_GROUPS[group_name]
            waste_pct, _, _ = compute_waste_percentage(model_data, model_name, categories)
            model_wastes.append(waste_pct)
        waste_data.append(model_wastes)
    
    waste_data = np.array(waste_data)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(available_models))
    width = 0.25
    
    for i, (group_name, color) in enumerate(zip(group_names, colors)):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, waste_data[:, i], width,
                     label=group_name, color=color, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=8, rotation=0)
    
    ax.set_ylabel('Wasted Tokens (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_title('Token Waste Analysis: Percentage of Tokens Spent on Incorrect Answers\n(Lower is Better - Less Waste)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(available_models, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.legend(title='Question Difficulty', fontsize=11, title_fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add reference lines
    ax.axhline(y=50, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='50% waste')
    ax.axhline(y=25, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='25% waste')
    
    # Add separator line between GPT-4o and GPT-5
    try:
        gpt4o_idx = available_models.index("GPT-4o")
        separator_idx = gpt4o_idx + 1
        if 0 < separator_idx < len(available_models):
            ax.axvline(x=separator_idx - 0.5, color='black', linestyle='--', linewidth=2, alpha=0.5)
            ax.text(separator_idx - 0.5, ax.get_ylim()[1] * 0.9,
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
    output_path = project_root / "src" / "plots" / "token_waste_analysis.png"
    
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
    
    print("\nGenerating token waste analysis...")
    plot_token_waste_analysis(model_data, output_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
