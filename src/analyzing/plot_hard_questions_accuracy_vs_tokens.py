#!/usr/bin/env python3
"""
Generate scatter plots showing accuracy vs average output tokens for hard questions.
Separate plots for each hard question category (9, 10, 11 models wrong).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from config import ITERATIVE_MODEL_ENTRIES, get_model_color, MODEL_COLOR_MAP


def load_hard_question_stats(responses_dir: Path) -> Tuple[Dict[str, Dict[int, Dict]], Dict[str, Dict[int, Dict]]]:
    """
    Load statistics for hard questions by model and category, separated by correct/incorrect.
    
    Returns:
        Tuple of (correct_stats, incorrect_stats) where each is:
        Dict[model_name][category] = {
            'count': int,
            'total_tokens': int,
            'avg_tokens': float,
            'token_values': List[int],
            'std_tokens': float
        }
    """
    # First, identify hard questions (questions that 9, 10, or 11 models got wrong)
    # Load all model responses
    all_model_data = {}
    
    for filename, display_name in ITERATIVE_MODEL_ENTRIES:
        file_path = responses_dir / filename
        if not file_path.exists():
            print(f"Warning: {filename} not found")
            continue
        
        model_questions = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    
                    # Extract question
                    question = record.get("question")
                    if not question:
                        raw = record.get("raw") or record.get("raw_response")
                        if isinstance(raw, dict):
                            question = raw.get("question")
                    
                    if not question or not isinstance(question, str):
                        continue
                    
                    question = question.strip()
                    
                    # Extract metrics
                    is_correct = bool(record.get("is_correct", False))
                    output_tokens = record.get("output_tokens")
                    
                    if isinstance(output_tokens, (int, float)) and output_tokens > 0:
                        model_questions[question] = {
                            'is_correct': is_correct,
                            'output_tokens': int(output_tokens)
                        }
                
                except json.JSONDecodeError:
                    continue
        
        if model_questions:
            all_model_data[display_name] = model_questions
    
    if not all_model_data:
        raise SystemExit("No model data loaded")
    
    # Find common questions
    question_sets = [set(data.keys()) for data in all_model_data.values()]
    common_questions = set.intersection(*question_sets)
    
    print(f"Total models: {len(all_model_data)}")
    print(f"Common questions: {len(common_questions)}")
    
    # Categorize questions by how many models got them wrong
    question_categories = defaultdict(list)  # category -> list of questions
    
    for question in common_questions:
        wrong_count = sum(
            1 for model_data in all_model_data.values()
            if not model_data[question]['is_correct']
        )
        if wrong_count >= 9:  # Hard questions: 9, 10, or 11 models wrong
            question_categories[wrong_count].append(question)
    
    print(f"\nHard question distribution:")
    for cat in sorted(question_categories.keys()):
        print(f"  {cat} models wrong: {len(question_categories[cat])} questions")
    
    # Compute statistics per model per category, separated by correct/incorrect
    correct_stats = defaultdict(lambda: defaultdict(lambda: {
        'count': 0,
        'total_tokens': 0,
        'token_values': []
    }))
    
    incorrect_stats = defaultdict(lambda: defaultdict(lambda: {
        'count': 0,
        'total_tokens': 0,
        'token_values': []
    }))
    
    for category, questions in question_categories.items():
        for model_name, model_data in all_model_data.items():
            for question in questions:
                if question not in model_data:
                    continue
                
                record = model_data[question]
                tokens = record['output_tokens']
                
                if record['is_correct']:
                    correct_stats[model_name][category]['count'] += 1
                    correct_stats[model_name][category]['total_tokens'] += tokens
                    correct_stats[model_name][category]['token_values'].append(tokens)
                else:
                    incorrect_stats[model_name][category]['count'] += 1
                    incorrect_stats[model_name][category]['total_tokens'] += tokens
                    incorrect_stats[model_name][category]['token_values'].append(tokens)
    
    # Calculate derived metrics
    for stats_dict in [correct_stats, incorrect_stats]:
        for model_name in stats_dict:
            for category in stats_dict[model_name]:
                s = stats_dict[model_name][category]
                if s['count'] > 0:
                    s['avg_tokens'] = s['total_tokens'] / s['count']
                    s['std_tokens'] = np.std(s['token_values']) if len(s['token_values']) > 1 else 0
                else:
                    s['avg_tokens'] = 0
                    s['std_tokens'] = 0
    
    return dict(correct_stats), dict(incorrect_stats)


def plot_correct_vs_incorrect_by_category(
    correct_stats: Dict[str, Dict[int, Dict]],
    incorrect_stats: Dict[str, Dict[int, Dict]],
    output_dir: Path,
) -> None:
    """Create side-by-side scatter plots showing correct vs incorrect for each category."""
    
    # Get all categories
    all_cats = set()
    for model_stats in correct_stats.values():
        all_cats.update(model_stats.keys())
    for model_stats in incorrect_stats.values():
        all_cats.update(model_stats.keys())
    categories = sorted(all_cats)
    
    if not categories:
        print("No categories to plot")
        return
    
    # Calculate uniform y-axis range (log scale)
    all_tokens = []
    for stats_dict in [correct_stats, incorrect_stats]:
        for model_stats in stats_dict.values():
            for cat_stats in model_stats.values():
                if cat_stats['count'] > 0:
                    all_tokens.append(cat_stats['avg_tokens'])
    
    if all_tokens:
        token_min = min(all_tokens)
        token_max = max(all_tokens)
        # Add padding in log space
        token_range = np.log10(token_max) - np.log10(token_min)
        y_lim = (10 ** (np.log10(token_min) - token_range * 0.1), 
                 10 ** (np.log10(token_max) + token_range * 0.1))
    else:
        y_lim = (100, 10000)
    
    # Create figure: 3 categories x 2 columns (correct/incorrect)
    n_cats = len(categories)
    fig, axes = plt.subplots(n_cats, 2, figsize=(14, 5 * n_cats))
    
    if n_cats == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, category in enumerate(categories):
        # Left column: Correct answers
        ax_correct = axes[row_idx, 0]
        
        # Collect data for correct answers
        counts_correct = []
        avg_tokens_correct = []
        model_names_correct = []
        
        for model_name in correct_stats:
            if category in correct_stats[model_name]:
                s = correct_stats[model_name][category]
                if s['count'] > 0:
                    counts_correct.append(s['count'])
                    avg_tokens_correct.append(s['avg_tokens'])
                    model_names_correct.append(model_name)
        
        if counts_correct:
            colors = [MODEL_COLOR_MAP.get(model, '#808080') for model in model_names_correct]
            
            ax_correct.scatter(
                counts_correct,
                avg_tokens_correct,
                s=250,
                c=colors,
                alpha=0.7,
                edgecolors='black',
                linewidth=1.5
            )
            
            # Add model labels
            for model, count, tokens in zip(model_names_correct, counts_correct, avg_tokens_correct):
                ax_correct.annotate(
                    model,
                    (count, tokens),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=7,
                    ha='left',
                    bbox=dict(
                        boxstyle='round,pad=0.2',
                        facecolor='white',
                        alpha=0.7,
                        edgecolor='gray',
                        linewidth=0.5
                    )
                )
            
            # Calculate correlation if possible
            if len(counts_correct) > 1 and len(set(counts_correct)) > 1:
                correlation = np.corrcoef(counts_correct, avg_tokens_correct)[0, 1]
                if not np.isnan(correlation):
                    ax_correct.text(
                        0.02, 0.98,
                        f'r = {correlation:.3f}',
                        transform=ax_correct.transAxes,
                        fontsize=9,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7)
                    )
        
        ax_correct.set_xlabel('Number of Correct Answers', fontsize=10, fontweight='bold')
        ax_correct.set_ylabel('Avg Output Tokens (log)', fontsize=10, fontweight='bold')
        ax_correct.set_title(f'{category} Models Wrong - CORRECT Answers', 
                            fontsize=11, fontweight='bold', pad=10, color='darkgreen')
        ax_correct.set_yscale('log')
        ax_correct.set_ylim(y_lim)
        ax_correct.grid(True, alpha=0.3, linestyle='--', which='both')
        ax_correct.set_axisbelow(True)
        
        # Right column: Incorrect answers
        ax_incorrect = axes[row_idx, 1]
        
        # Collect data for incorrect answers
        counts_incorrect = []
        avg_tokens_incorrect = []
        model_names_incorrect = []
        
        for model_name in incorrect_stats:
            if category in incorrect_stats[model_name]:
                s = incorrect_stats[model_name][category]
                if s['count'] > 0:
                    counts_incorrect.append(s['count'])
                    avg_tokens_incorrect.append(s['avg_tokens'])
                    model_names_incorrect.append(model_name)
        
        if counts_incorrect:
            colors = [MODEL_COLOR_MAP.get(model, '#808080') for model in model_names_incorrect]
            
            ax_incorrect.scatter(
                counts_incorrect,
                avg_tokens_incorrect,
                s=250,
                c=colors,
                alpha=0.7,
                edgecolors='black',
                linewidth=1.5
            )
            
            # Add model labels
            for model, count, tokens in zip(model_names_incorrect, counts_incorrect, avg_tokens_incorrect):
                ax_incorrect.annotate(
                    model,
                    (count, tokens),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=7,
                    ha='left',
                    bbox=dict(
                        boxstyle='round,pad=0.2',
                        facecolor='white',
                        alpha=0.7,
                        edgecolor='gray',
                        linewidth=0.5
                    )
                )
            
            # Calculate correlation if possible
            if len(counts_incorrect) > 1 and len(set(counts_incorrect)) > 1:
                correlation = np.corrcoef(counts_incorrect, avg_tokens_incorrect)[0, 1]
                if not np.isnan(correlation):
                    ax_incorrect.text(
                        0.02, 0.98,
                        f'r = {correlation:.3f}',
                        transform=ax_incorrect.transAxes,
                        fontsize=9,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7)
                    )
        
        ax_incorrect.set_xlabel('Number of Incorrect Answers', fontsize=10, fontweight='bold')
        ax_incorrect.set_ylabel('Avg Output Tokens (log)', fontsize=10, fontweight='bold')
        ax_incorrect.set_title(f'{category} Models Wrong - INCORRECT Answers', 
                              fontsize=11, fontweight='bold', pad=10, color='darkred')
        ax_incorrect.set_yscale('log')
        ax_incorrect.set_ylim(y_lim)
        ax_incorrect.grid(True, alpha=0.3, linestyle='--', which='both')
        ax_incorrect.set_axisbelow(True)
    
    # Main title
    fig.suptitle(
        'Hard Questions: Correct vs Incorrect - Token Usage by Count\n(Uniform Y-axis across all subplots)',
        fontsize=14,
        fontweight='bold',
        y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    output_path = output_dir / "hard_questions_accuracy_vs_tokens_by_category.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()


def create_summary_table(
    correct_stats: Dict[str, Dict[int, Dict]],
    incorrect_stats: Dict[str, Dict[int, Dict]],
    output_dir: Path,
) -> None:
    """Create a summary table showing stats for each model and category."""
    
    print("\n" + "=" * 110)
    print("SUMMARY: Hard Question Statistics - Correct vs Incorrect")
    print("=" * 110)
    
    # Header
    print(f"\n{'Model':<30} {'Category':<10} {'Correct':<8} {'Avg Tokens':<12} {'Incorrect':<10} {'Avg Tokens':<12}")
    print("-" * 110)
    
    # Get all models and categories
    all_models = set(list(correct_stats.keys()) + list(incorrect_stats.keys()))
    
    for model_name in sorted(all_models):
        all_cats = set()
        if model_name in correct_stats:
            all_cats.update(correct_stats[model_name].keys())
        if model_name in incorrect_stats:
            all_cats.update(incorrect_stats[model_name].keys())
        
        for category in sorted(all_cats):
            correct_count = 0
            correct_tokens = 0
            incorrect_count = 0
            incorrect_tokens = 0
            
            if model_name in correct_stats and category in correct_stats[model_name]:
                s = correct_stats[model_name][category]
                correct_count = s['count']
                correct_tokens = s['avg_tokens']
            
            if model_name in incorrect_stats and category in incorrect_stats[model_name]:
                s = incorrect_stats[model_name][category]
                incorrect_count = s['count']
                incorrect_tokens = s['avg_tokens']
            
            total = correct_count + incorrect_count
            accuracy = (correct_count / total * 100) if total > 0 else 0
            
            print(f"{model_name:<30} {category:>3} wrong   {correct_count:>5}    {correct_tokens:>10.2f}    {incorrect_count:>7}    {incorrect_tokens:>10.2f}  ({accuracy:>5.1f}%)")
    
    print("-" * 110)


def main() -> None:
    """Main execution function."""
    base = Path(__file__).resolve().parents[1]
    responses_dir = base / "responses_reverified"
    plots_dir = base / "plots"
    
    print("Loading hard question statistics...")
    correct_stats, incorrect_stats = load_hard_question_stats(responses_dir)
    
    print("\nGenerating plots...")
    
    # Plot: Side-by-side correct vs incorrect for each category (6 subplots total)
    plot_correct_vs_incorrect_by_category(correct_stats, incorrect_stats, plots_dir)
    
    # Summary table
    create_summary_table(correct_stats, incorrect_stats, plots_dir)
    
    print("\n" + "=" * 100)
    print("Completed! Generated:")
    print("  1. hard_questions_accuracy_vs_tokens_by_category.png (3 rows x 2 cols)")
    print("     - Each row: one difficulty category (9, 10, 11 models wrong)")
    print("     - Left column: Correct answers")
    print("     - Right column: Incorrect answers")
    print("     - Uniform y-axis across all subplots")
    print("=" * 100)


if __name__ == "__main__":
    main()
