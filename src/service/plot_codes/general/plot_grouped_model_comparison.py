#!/usr/bin/env python3
"""
Generate Grouped Model Comparison showing explicit model groups.

Groups models into categories and compares group averages:
- Efficient Non-Reasoning (GPT-4o, Gemini)
- Standard Non-Reasoning (Claude 3.7, GLM, Grok, GPT-5)
- Struggling Non-Reasoning (Llama, Mistral)
- Reasoning Models (Claude Thinking, DeepSeek R1)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Set
from numbers import Number

import matplotlib.pyplot as plt
import numpy as np

from config import ITERATIVE_MODEL_ENTRIES


# Define category groupings
CATEGORY_GROUPS = {
    "Easy": [0, 1, 2],
    "Medium": [5, 6, 7],
    "Hard": [9, 10, 11]
}

# Define model groups
MODEL_GROUPS = {
    "Efficient Non-Reasoning": ["GPT-4o", "Gemini 2.5 Pro"],
    "Standard Non-Reasoning": ["Claude 3.7 Sonnet", "Claude Sonnet 4.5", "GLM 4.6", "Grok 4 Fast", "GPT-5"],
    "Struggling Non-Reasoning": ["Llama 3.3 70B Instruct", "Mistral Large 2402"],
    "Reasoning Models": ["Claude 3.7 Sonnet Thinking", "DeepSeek R1"]
}

GROUP_COLORS = {
    "Efficient Non-Reasoning": "#27ae60",
    "Standard Non-Reasoning": "#3498db",
    "Struggling Non-Reasoning": "#e67e22",
    "Reasoning Models": "#9b59b6"
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


def compute_group_stats(
    model_data: Dict[str, Dict[str, dict]],
    group_models: List[str],
    categories: List[int]
) -> Tuple[float, float]:
    """
    Compute average tokens and std dev for a group of models.
    Returns: (mean, std_dev)
    """
    all_tokens = []
    
    for model_name in group_models:
        if model_name not in model_data:
            continue
        
        questions = model_data[model_name]
        for question, data in questions.items():
            if data["category"] in categories and data["output_tokens"] is not None:
                all_tokens.append(data["output_tokens"])
    
    if not all_tokens:
        return 0.0, 0.0
    
    return np.mean(all_tokens), np.std(all_tokens)


def plot_grouped_model_comparison(
    model_data: Dict[str, Dict[str, dict]],
    output_path: Path
) -> None:
    """Create grouped model comparison plot."""
    group_names = list(MODEL_GROUPS.keys())
    difficulty_levels = ["Easy", "Medium", "Hard"]
    
    # Compute stats for each group and difficulty
    data_matrix = []
    error_matrix = []
    
    for group_name in group_names:
        group_models = MODEL_GROUPS[group_name]
        group_row = []
        error_row = []
        
        for difficulty in difficulty_levels:
            categories = CATEGORY_GROUPS[difficulty]
            mean, std = compute_group_stats(model_data, group_models, categories)
            group_row.append(mean)
            error_row.append(std)
        
        data_matrix.append(group_row)
        error_matrix.append(error_row)
    
    data_matrix = np.array(data_matrix)
    error_matrix = np.array(error_matrix)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(group_names))
    width = 0.25
    difficulty_colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    for i, (difficulty, color) in enumerate(zip(difficulty_levels, difficulty_colors)):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, data_matrix[:, i], width, yerr=error_matrix[:, i],
                     label=difficulty, color=color, alpha=0.8, edgecolor='black',
                     capsize=5, error_kw={'linewidth': 2, 'elinewidth': 2})
        
        # Add value labels
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Average Output Tokens (log scale)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model Groups', fontsize=12, fontweight='bold')
    ax.set_title('Grouped Model Comparison: Average Token Usage by Model Category\n(Error bars show standard deviation)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(group_names, rotation=15, ha='right')
    ax.set_yscale('log')
    ax.set_ylim(100, 30000)
    ax.legend(title='Question Difficulty', fontsize=11, title_fontsize=12)
    ax.grid(axis='y', alpha=0.3, which='both')
    
    # Add group info text
    info_text = "Groups:\n"
    for group_name, models in MODEL_GROUPS.items():
        info_text += f"• {group_name}: {len(models)} models\n"
    
    ax.text(0.02, 0.98, info_text,
           transform=ax.transAxes, fontsize=9,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
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
    output_path = project_root / "data" / "plots" / "general" / "grouped_model_comparison.png"
    
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
    
    print("\nGenerating grouped model comparison...")
    plot_grouped_model_comparison(model_data, output_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
