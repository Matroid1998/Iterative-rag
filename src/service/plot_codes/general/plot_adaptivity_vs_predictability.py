#!/usr/bin/env python3
"""
Generate Adaptivity vs. Predictability scatter plot.

Combines:
- X-axis: Scaling Factor (Hard / Easy) - Higher = More Adaptive Effort
- Y-axis: Token Usage Consistency (Average CV %) - Lower = More Predictable
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

# Define reasoning models
REASONING_MODELS = {
    "Claude 3.7 Sonnet Thinking",
    "DeepSeek R1",
    "GPT-o1",
    "GPT-o3"
}

def is_reasoning_model(model_name: str) -> bool:
    """Check if a model is a reasoning model."""
    return model_name in REASONING_MODELS

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
        if data["category"] in categories and data["output_tokens"] is not None:
            tokens.append(data["output_tokens"])
    
    return np.mean(tokens) if tokens else 0.0

def compute_coefficient_of_variation(
    model_data: Dict[str, Dict[str, dict]],
    model_name: str,
    categories: List[int]
) -> float:
    """
    Compute coefficient of variation (CV = std_dev / mean * 100).
    """
    if model_name not in model_data:
        return 0.0
    
    tokens = []
    questions = model_data[model_name]
    
    for question, data in questions.items():
        if data["category"] in categories and data["output_tokens"] is not None:
            tokens.append(data["output_tokens"])
    
    if not tokens or len(tokens) < 2:
        return 0.0
    
    mean = np.mean(tokens)
    std = np.std(tokens)
    
    if mean == 0:
        return 0.0
    
    cv = (std / mean) * 100
    return cv

def plot_adaptivity_vs_predictability(
    model_data: Dict[str, Dict[str, dict]],
    output_path: Path
) -> None:
    """Create scatter plot showing Adaptivity vs. Predictability."""
    model_names = [display_name for _, display_name in ITERATIVE_MODEL_ENTRIES]
    
    # Filter out requested models
    excluded_models = ["GPT-5.1", "Gemini 3"]
    available_models = [
        m for m in model_names 
        if m in model_data and not any(ex in m for ex in excluded_models)
    ]
    
    x_values = [] # Scaling Factor
    y_values = [] # Average CV
    colors = []
    labels = []
    sizes = []
    
    
    # Define sorted order to match cost_vs_accuracy_scatter.png colors
    # NOTE: Names must match exactly those in config.py ITERATIVE_MODEL_ENTRIES
    sorted_models = [
        "Llama 3.3 70B Instruct",
        "Mistral Large 2402",
        "Claude 3.7 Sonnet",
        "Claude 3.7 Sonnet Thinking", 
        "DeepSeek R1",
        "GPT-4o",
        "GPT-5",
        "Claude Sonnet 4.5",
        "Gemini 2.5 Pro",
        "Grok 4 Fast",
        "GLM 4.6",
    ]
    
    # Generate tab10 colors
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))
    color_map = {}
    for i, model in enumerate(sorted_models):
        color_map[model] = cmap[i % 10]

    for model_name in available_models:
        # Calculate Scaling Factor (Hard / Easy)
        easy_tokens = compute_average_tokens(model_data, model_name, CATEGORY_GROUPS["Easy"])
        hard_tokens = compute_average_tokens(model_data, model_name, CATEGORY_GROUPS["Hard"])
        
        scaling_factor = 0.0
        if easy_tokens > 0:
            scaling_factor = hard_tokens / easy_tokens
            
        # Calculate Average CV (across Easy, Medium, Hard)
        cvs = []
        for group_name in ["Easy", "Medium", "Hard"]:
            cv = compute_coefficient_of_variation(model_data, model_name, CATEGORY_GROUPS[group_name])
            if cv > 0:
                cvs.append(cv)
        
        avg_cv = np.mean(cvs) if cvs else 0.0
        
        if scaling_factor > 0 and avg_cv > 0:
            x_values.append(scaling_factor)
            y_values.append(avg_cv)
            labels.append(model_name)
            
            # Use specific color for model
            color = color_map.get(model_name, '#808080')
            colors.append(color) 
            sizes.append(200) # Increased size
    
    # Create plot - slightly larger figsize to make text look smaller relatively
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Plot points
    scatter = ax.scatter(x_values, y_values, c=colors, s=sizes, alpha=0.8, edgecolors='black', linewidth=1.5)
    
    # Add labels
    for i, label in enumerate(labels):
        # Default position
        xytext = (8, 0)
        ha = 'left'
        va = 'center'
        
        # Specific overrides
        # Ensure "Thinking" variant stays on the right (default)
        if "Claude 3.7 Sonnet" in label and "Reasoning" not in label and "Thinking" not in label:
             xytext = (-5, -5)
             ha = 'right'
             va = 'top'
        
        # Removed fontweight='bold' and reduced fontsize
        ax.annotate(label, (x_values[i], y_values[i]), 
                   xytext=xytext, textcoords='offset points',
                   fontsize=10, ha=ha, va=va)
    
    # Axis labels and Title
    ax.set_xlabel('Token Scaling Factor (Hard/Easy Multiplier)\n(Higher = More Adaptive Effort)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Token Usage Consistency (Average CV %)\n(Lower = More Predictable)', fontsize=13, fontweight='bold')
    ax.set_title('Adaptivity vs. Predictability', fontsize=16, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Reference line for 1.0x scaling
    ax.axvline(x=1.0, color='#999', linestyle='--', alpha=0.5)
    
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
    output_path = project_root / "data" / "plots" / "general" / "adaptivity_vs_predictability.png"
    
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
    
    print("\nGenerating Adaptivity vs. Predictability plot...")
    plot_adaptivity_vs_predictability(model_data, output_path)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
