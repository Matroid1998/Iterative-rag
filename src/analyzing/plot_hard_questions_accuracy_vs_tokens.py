#!/usr/bin/env python3
"""
Generate scatter plots showing accuracy vs average output tokens grouped by question difficulty.
Difficulty buckets:
- Easy: questions missed by at most 2 models (0, 1, 2 models wrong)
- Medium: questions missed by 5, 6, or 7 models
- Hard: questions missed by 9, 10, or 11 models
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from config import ITERATIVE_MODEL_ENTRIES, MODEL_COLOR_MAP

CATEGORY_GROUPS = {
    "Easy": (0, 1, 2),
    "Medium": (5, 6, 7),
    "Hard": (9, 10, 11),
}


def load_hard_question_stats(
    responses_dir: Path,
    hard_question_path: Path,
) -> Tuple[Dict[str, Dict[str, Dict]], Dict[str, Dict[str, Dict]]]:
    """
    Load statistics for question difficulty groups, separated by correct/incorrect.

    Returns:
        Tuple of (correct_stats, incorrect_stats) where each is:
        Dict[model_name][difficulty_label] = {
            'count': int,
            'total_tokens': int,
            'avg_tokens': float,
            'token_values': List[int],
            'std_tokens': float
        }
    """
    all_model_data: Dict[str, Dict[str, Dict[str, object]]] = {}

    for filename, display_name in ITERATIVE_MODEL_ENTRIES:
        file_path = responses_dir / filename
        if not file_path.exists():
            print(f"Warning: {filename} not found")
            continue

        model_questions: Dict[str, Dict[str, object]] = {}
        with file_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                question = record.get("question")
                if not question:
                    raw = record.get("raw") or record.get("raw_response")
                    if isinstance(raw, dict):
                        question = raw.get("question")

                if not isinstance(question, str) or not question.strip():
                    continue
                question = question.strip()

                is_correct = bool(record.get("is_correct", False))
                output_tokens = record.get("output_tokens")
                if not isinstance(output_tokens, (int, float)) or output_tokens <= 0:
                    continue

                model_questions[question] = {
                    "is_correct": is_correct,
                    "output_tokens": int(output_tokens),
                }

        if model_questions:
            all_model_data[display_name] = model_questions

    if not all_model_data:
        raise SystemExit("No model data loaded")

    question_sets = [set(data.keys()) for data in all_model_data.values()]
    common_questions = set.intersection(*question_sets)

    print(f"Total models: {len(all_model_data)}")
    print(f"Common questions: {len(common_questions)}")

    if not hard_question_path.exists():
        raise SystemExit(f"Hard questions file not found: {hard_question_path}")

    with hard_question_path.open("r", encoding="utf-8") as handle:
        hard_question_data = json.load(handle)

    category_to_questions: Dict[int, set[str]] = {}
    for category_str, entries in hard_question_data.items():
        try:
            category = int(category_str)
        except (TypeError, ValueError):
            continue
        questions: set[str] = set()
        if isinstance(entries, list):
            for item in entries:
                if not isinstance(item, dict):
                    continue
                question = item.get("question")
                if isinstance(question, str) and question.strip():
                    questions.add(question.strip())
        if questions:
            category_to_questions[category] = questions

    group_questions: Dict[str, set[str]] = {}
    for label, categories in CATEGORY_GROUPS.items():
        merged: set[str] = set()
        for category in categories:
            merged |= category_to_questions.get(category, set())
        group_questions[label] = merged & common_questions

    print("\nQuestion distribution by difficulty:")
    for label in CATEGORY_GROUPS:
        print(f"  {label}: {len(group_questions.get(label, set()))} questions")

    correct_stats: Dict[str, Dict[str, Dict[str, object]]] = defaultdict(
        lambda: defaultdict(lambda: {"count": 0, "total_tokens": 0, "token_values": []})
    )
    incorrect_stats: Dict[str, Dict[str, Dict[str, object]]] = defaultdict(
        lambda: defaultdict(lambda: {"count": 0, "total_tokens": 0, "token_values": []})
    )

    for label, questions in group_questions.items():
        if not questions:
            continue
        for model_name, model_data in all_model_data.items():
            for question in questions:
                record = model_data.get(question)
                if not record:
                    continue
                tokens = record["output_tokens"]
                if record["is_correct"]:
                    correct_stats[model_name][label]["count"] += 1
                    correct_stats[model_name][label]["total_tokens"] += tokens
                    correct_stats[model_name][label]["token_values"].append(tokens)
                else:
                    incorrect_stats[model_name][label]["count"] += 1
                    incorrect_stats[model_name][label]["total_tokens"] += tokens
                    incorrect_stats[model_name][label]["token_values"].append(tokens)

    for stats_dict in (correct_stats, incorrect_stats):
        for model_name in stats_dict:
            for label in stats_dict[model_name]:
                stats = stats_dict[model_name][label]
                count = stats["count"]
                if count > 0:
                    stats["avg_tokens"] = stats["total_tokens"] / count
                    stats["std_tokens"] = (
                        float(np.std(stats["token_values"]))
                        if len(stats["token_values"]) > 1
                        else 0.0
                    )
                else:
                    stats["avg_tokens"] = 0.0
                    stats["std_tokens"] = 0.0

    return dict(correct_stats), dict(incorrect_stats)


def plot_correct_vs_incorrect_by_category(
    correct_stats: Dict[str, Dict[str, Dict]],
    incorrect_stats: Dict[str, Dict[str, Dict]],
    output_dir: Path,
) -> None:
    """Create side-by-side scatter plots showing correct vs incorrect for each category."""
    
    # Get all categories
    all_cats = set()
    for model_stats in correct_stats.values():
        all_cats.update(model_stats.keys())
    for model_stats in incorrect_stats.values():
        all_cats.update(model_stats.keys())
    categories = [label for label in CATEGORY_GROUPS if label in all_cats]
    if len(categories) < len(all_cats):
        remaining = sorted(all_cats - set(categories))
        categories.extend(remaining)
    
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
    fig, axes = plt.subplots(n_cats, 2, figsize=(14, 5 * n_cats), sharey=True)

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
        ax_correct.set_title(f'{category} - CORRECT Answers', 
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
        ax_incorrect.set_title(f'{category} - INCORRECT Answers', 
                              fontsize=11, fontweight='bold', pad=10, color='darkred')
        ax_incorrect.set_yscale('log')
        ax_incorrect.set_ylim(y_lim)
        ax_incorrect.grid(True, alpha=0.3, linestyle='--', which='both')
        ax_incorrect.set_axisbelow(True)
        ax_incorrect.tick_params(labelleft=True)
    
    # Main title
    fig.suptitle(
        'Difficulty Buckets: Correct vs Incorrect - Token Usage by Count\n(Uniform Y-axis across all subplots)',
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
    correct_stats: Dict[str, Dict[str, Dict]],
    incorrect_stats: Dict[str, Dict[str, Dict]],
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
        
        ordered_categories = [label for label in CATEGORY_GROUPS if label in all_cats]
        leftover = [label for label in all_cats if label not in ordered_categories]
        ordered_categories.extend(sorted(leftover))

        for category in ordered_categories:
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
            
            print(f"{model_name:<30} {category:<10} {correct_count:>5}    {correct_tokens:>10.2f}    {incorrect_count:>7}    {incorrect_tokens:>10.2f}  ({accuracy:>5.1f}%)")
    
    print("-" * 110)


def main() -> None:
    """Main execution function."""
    base = Path(__file__).resolve().parents[1]
    responses_dir = base / "responses_reverified"
    plots_dir = base / "plots"
    hard_questions_path = (
        base / "results" / "unanswered_questions" / "hard_question_categories.json"
    )
    
    print("Loading hard question statistics...")
    correct_stats, incorrect_stats = load_hard_question_stats(
        responses_dir, hard_questions_path
    )
    
    print("\nGenerating plots...")
    
    # Plot: Side-by-side correct vs incorrect for each category (6 subplots total)
    plot_correct_vs_incorrect_by_category(correct_stats, incorrect_stats, plots_dir)
    
    # Summary table
    create_summary_table(correct_stats, incorrect_stats, plots_dir)
    
    print("\n" + "=" * 100)
    print("Completed! Generated:")
    print("  1. hard_questions_accuracy_vs_tokens_by_category.png (3 rows x 2 cols)")
    print("     - Each row: one difficulty category (Easy, Medium, Hard)")
    print("     - Left column: Correct answers")
    print("     - Right column: Incorrect answers")
    print("     - Uniform y-axis across all subplots")
    print("=" * 100)


if __name__ == "__main__":
    main()
