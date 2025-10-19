#!/usr/bin/env python3
"""
Knowledge Gap Analysis: Waterfall and Matrix plots

This script analyzes the "knowledge gap" across three conditions:
1. No Context (baseline) - tests intrinsic model knowledge
2. Gold Context - tests with oracle retrieval
3. Iterative RAG - tests with iterative retrieval system

Creates two main visualizations:
1. Knowledge Gap Waterfall Plot - shows progression of question-solving capability
2. Knowledge Gap Matrix/Heatmap - shows distribution of question categories per model

Categories:
- Intrinsic Knowledge: Correct without any context
- Retrieval Dependent: Wrong without context, correct with Gold Context
- Iterative Dependent: Wrong with Gold Context, correct with Iterative RAG
- Knowledge Gap: Wrong in all three conditions
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Set

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import (
    get_iterative_model_entries,
    get_display_name,
)


# -------- JSONL helpers ---------

def _iter_jsonl(path: Path) -> Iterable[dict]:
    """Iterate over JSONL records."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                continue


def _extract_question(rec: dict) -> str | None:
    """Extract question text from a record."""
    q = rec.get("question")
    if isinstance(q, str) and q.strip():
        return q.strip()
    for key in ("raw", "raw_response"):
        raw = rec.get(key)
        if isinstance(raw, dict):
            q2 = raw.get("question")
            if isinstance(q2, str) and q2.strip():
                return q2.strip()
    return None


def _is_correct(rec: dict) -> bool:
    """Check if a record is marked as correct."""
    return bool(rec.get("is_correct", False))


def _dedup_latest(records: Iterable[dict]) -> Dict[str, dict]:
    """Deduplicate by question, keeping the last record encountered."""
    by_q: Dict[str, dict] = {}
    for rec in records:
        q = _extract_question(rec)
        if not q:
            continue
        by_q[q] = rec
    return by_q


def load_responses_by_question(path: Path) -> Dict[str, bool]:
    """Load responses and return {question: is_correct} mapping."""
    by_q = _dedup_latest(_iter_jsonl(path))
    return {q: _is_correct(rec) for q, rec in by_q.items()}


def scan_context_dir(context_dir: Path, model_display: str) -> Dict[str, bool]:
    """Load responses from a context directory for a specific model."""
    for f in sorted(context_dir.glob("*.jsonl")):
        display = get_display_name(f.stem)
        if display == model_display:
            return load_responses_by_question(f)
    return {}


def categorize_questions(
    no_ctx: Dict[str, bool],
    gold_ctx: Dict[str, bool],
    iterative: Dict[str, bool],
) -> Dict[str, Set[str]]:
    """
    Categorize questions into four groups based on correctness across conditions.
    
    Returns:
        Dict with categories:
        - 'intrinsic': Correct without context
        - 'retrieval': Wrong without context, correct with gold context
        - 'iterative': Wrong with gold context, correct with iterative RAG
        - 'gap': Wrong in all three conditions
        
    Note: Only categorizes questions that exist in ALL three conditions to ensure
    fair comparison. Questions missing from any condition are excluded from analysis.
    """
    # Only consider questions that exist in all three conditions
    common_questions = set(no_ctx.keys()) & set(gold_ctx.keys()) & set(iterative.keys())
    
    categories = {
        'intrinsic': set(),
        'retrieval': set(),
        'iterative': set(),
        'gap': set(),
    }
    
    for q in common_questions:
        no_correct = no_ctx[q]  # We know it exists
        gold_correct = gold_ctx[q]
        iter_correct = iterative[q]
        
        if no_correct:
            # If correct without context, it's intrinsic knowledge
            categories['intrinsic'].add(q)
        elif gold_correct:
            # If wrong without context but correct with gold, it's retrieval dependent
            categories['retrieval'].add(q)
        elif iter_correct:
            # If wrong with gold but correct with iterative, it's iterative dependent
            categories['iterative'].add(q)
        else:
            # Wrong in all three - knowledge gap
            categories['gap'].add(q)
    
    return categories


def plot_knowledge_gap_waterfall(
    models: List[str],
    model_categories: Dict[str, Dict[str, Set[str]]],
    out_path: Path,
) -> None:
    """
    Create a waterfall plot showing the progression of question-solving capability.
    
    For each model, shows stacked bars with:
    - Intrinsic knowledge (green)
    - + Retrieval dependent (blue)
    - + Iterative dependent (orange)
    - Knowledge gap (red)
    """
    fig, ax = plt.subplots(figsize=(max(14, len(models) * 1.2), 8))
    
    # Prepare data
    intrinsic_counts = []
    retrieval_counts = []
    iterative_counts = []
    gap_counts = []
    
    for model in models:
        cats = model_categories.get(model, {})
        intrinsic_counts.append(len(cats.get('intrinsic', set())))
        retrieval_counts.append(len(cats.get('retrieval', set())))
        iterative_counts.append(len(cats.get('iterative', set())))
        gap_counts.append(len(cats.get('gap', set())))
    
    x = np.arange(len(models))
    width = 0.65
    
    # Create stacked bars
    p1 = ax.bar(x, intrinsic_counts, width, label='Intrinsic Knowledge (No Context needed)',
                color='#2ecc71')
    p2 = ax.bar(x, retrieval_counts, width, bottom=intrinsic_counts,
                label='Retrieval Dependent (Gold Context needed)', color='#3498db')
    p3 = ax.bar(x, iterative_counts, width,
                bottom=np.array(intrinsic_counts) + np.array(retrieval_counts),
                label='Iterative Dependent (Iterative RAG needed)', color='#f39c12')
    p4 = ax.bar(x, gap_counts, width,
                bottom=np.array(intrinsic_counts) + np.array(retrieval_counts) + np.array(iterative_counts),
                label='Knowledge Gap (All methods fail)', color='#e74c3c')
    
    # Add value labels on each segment
    for i, model in enumerate(models):
        y_offset = 0
        for count, color in [
            (intrinsic_counts[i], '#2ecc71'),
            (retrieval_counts[i], '#3498db'),
            (iterative_counts[i], '#f39c12'),
            (gap_counts[i], '#e74c3c'),
        ]:
            if count > 0:
                ax.text(i, y_offset + count / 2, str(count),
                       ha='center', va='center', fontweight='bold',
                       color='white', fontsize=9)
                y_offset += count
        
        # Add total at top
        total = sum([intrinsic_counts[i], retrieval_counts[i], iterative_counts[i], gap_counts[i]])
        ax.text(i, total + 15, f'{total}', ha='center', va='bottom',
               fontweight='bold', fontsize=10)
    
    ax.set_ylabel('Number of Questions', fontsize=12, fontweight='bold')
    ax.set_title('Knowledge Gap Waterfall: Question-Solving Capability Progression',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha='right', fontsize=10)
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), framealpha=0.95, fontsize=9, ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add percentage labels
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_ylabel('Coverage (%)', fontsize=12, fontweight='bold')
    
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {out_path.name}")


def plot_knowledge_gap_matrix(
    models: List[str],
    model_categories: Dict[str, Dict[str, Set[str]]],
    out_path: Path,
) -> None:
    """
    Create a heatmap matrix showing the distribution of question categories.
    
    Rows: Models
    Columns: Categories (Intrinsic, Retrieval, Iterative, Gap)
    Color: Count or percentage
    """
    categories = ['intrinsic', 'retrieval', 'iterative', 'gap']
    category_labels = ['Intrinsic\nKnowledge', 'Retrieval\nDependent', 'Iterative\nDependent', 'Knowledge\nGap']
    
    # Build matrix
    matrix_counts = np.zeros((len(models), len(categories)))
    matrix_pcts = np.zeros((len(models), len(categories)))
    
    for i, model in enumerate(models):
        cats = model_categories.get(model, {})
        total = sum(len(cats.get(cat, set())) for cat in categories)
        
        for j, cat in enumerate(categories):
            count = len(cats.get(cat, set()))
            matrix_counts[i, j] = count
            matrix_pcts[i, j] = (count / total * 100) if total > 0 else 0
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(8, len(models) * 0.5)))
    
    # Plot 1: Absolute counts
    im1 = ax1.imshow(matrix_counts, cmap='YlOrRd', aspect='auto', vmin=0)
    ax1.set_xticks(np.arange(len(categories)))
    ax1.set_yticks(np.arange(len(models)))
    ax1.set_xticklabels(category_labels, fontsize=11)
    ax1.set_yticklabels(models, fontsize=10)
    ax1.set_title('Question Counts by Category', fontsize=12, fontweight='bold', pad=15)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(categories)):
            text = ax1.text(j, i, f'{int(matrix_counts[i, j])}',
                          ha='center', va='center', color='black' if matrix_counts[i, j] < matrix_counts.max() * 0.6 else 'white',
                          fontweight='bold', fontsize=9)
    
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Count', rotation=270, labelpad=20, fontweight='bold')
    
    # Plot 2: Percentages
    im2 = ax2.imshow(matrix_pcts, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
    ax2.set_xticks(np.arange(len(categories)))
    ax2.set_yticks(np.arange(len(models)))
    ax2.set_xticklabels(category_labels, fontsize=11)
    ax2.set_yticklabels(models, fontsize=10)
    ax2.set_title('Question Distribution by Category (%)', fontsize=12, fontweight='bold', pad=15)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(categories)):
            text = ax2.text(j, i, f'{matrix_pcts[i, j]:.1f}%',
                          ha='center', va='center', color='black' if matrix_pcts[i, j] < 60 else 'white',
                          fontweight='bold', fontsize=9)
    
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Percentage (%)', rotation=270, labelpad=20, fontweight='bold')
    
    fig.suptitle('Knowledge Gap Matrix: Distribution of Question Categories Across Models',
                fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {out_path.name}")


def plot_knowledge_gap_detailed_breakdown(
    models: List[str],
    model_data: Dict[str, Tuple[Dict[str, bool], Dict[str, bool], Dict[str, bool]]],
    out_path: Path,
) -> None:
    """
    Create a detailed breakdown showing actual accuracy across three conditions.
    
    Shows three bars per model with ACTUAL accuracies calculated as:
    - No Context: correct_count / total_questions * 100
    - Gold Context: correct_count / total_questions * 100
    - Iterative RAG: correct_count / total_questions * 100
    
    Args:
        models: List of model display names
        model_data: Dict mapping model name to (no_ctx, gold_ctx, iterative) response dicts
        out_path: Output path for the plot
    """
    fig, ax = plt.subplots(figsize=(max(14, len(models) * 1.2), 7))
    
    no_ctx_acc = []
    gold_ctx_acc = []
    iter_acc = []
    
    for model in models:
        no_ctx, gold_ctx, iterative = model_data.get(model, ({}, {}, {}))
        
        # Calculate actual accuracies: correct / total * 100
        total_questions = 1186  # Standard benchmark size
        
        no_ctx_correct = sum(1 for v in no_ctx.values() if v)
        gold_ctx_correct = sum(1 for v in gold_ctx.values() if v)
        iter_correct = sum(1 for v in iterative.values() if v)
        
        # Use actual count divided by benchmark size
        no_ctx_acc.append((no_ctx_correct / total_questions * 100) if len(no_ctx) > 0 else 0)
        gold_ctx_acc.append((gold_ctx_correct / total_questions * 100) if len(gold_ctx) > 0 else 0)
        iter_acc.append((iter_correct / total_questions * 100) if len(iterative) > 0 else 0)
    
    x = np.arange(len(models))
    width = 0.25
    
    bars1 = ax.bar(x - width, no_ctx_acc, width, label='No Context', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x, gold_ctx_acc, width, label='Gold Context', color='#3498db', alpha=0.8)
    bars3 = ax.bar(x + width, iter_acc, width, label='Iterative RAG', color='#f39c12', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Across Different Contexts', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha='right', fontsize=10)
    ax.legend(loc='lower left', framealpha=0.95, fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)
    
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {out_path.name}")


def plot_knowledge_gap_scatter(
    models: List[str],
    model_raw_data: Dict[str, Tuple[Dict[str, bool], Dict[str, bool], Dict[str, bool]]],
    out_path: Path,
) -> None:
    """
    Create a scatter plot showing relationship between no-context accuracy (x-axis)
    and both gold-context and iterative RAG accuracy (y-axis).
    
    This shows how much models improve with context/retrieval.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Prepare data
    no_ctx_accuracies = []
    gold_ctx_accuracies = []
    iterative_accuracies = []
    
    for model in models:
        no_ctx, gold_ctx, iterative = model_raw_data[model]
        
        # Calculate accuracies
        no_ctx_acc = sum(1 for v in no_ctx.values() if v) / 1186 * 100
        gold_ctx_acc = sum(1 for v in gold_ctx.values() if v) / 1186 * 100
        iter_acc = sum(1 for v in iterative.values() if v) / 1186 * 100
        
        no_ctx_accuracies.append(no_ctx_acc)
        gold_ctx_accuracies.append(gold_ctx_acc)
        iterative_accuracies.append(iter_acc)
    
    # Plot scatter points
    scatter_gold = ax.scatter(no_ctx_accuracies, gold_ctx_accuracies, 
                             s=200, alpha=0.7, color='#3498db', 
                             edgecolors='black', linewidth=1.5,
                             label='Gold Context', marker='o', zorder=3)
    
    scatter_iter = ax.scatter(no_ctx_accuracies, iterative_accuracies, 
                             s=200, alpha=0.7, color='#f39c12',
                             edgecolors='black', linewidth=1.5,
                             label='Iterative RAG', marker='s', zorder=3)
    
    # Add model labels
    for i, model in enumerate(models):
        # Label for gold context (slightly offset up)
        ax.annotate(model, 
                   (no_ctx_accuracies[i], gold_ctx_accuracies[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='#3498db', alpha=0.7))
        
        # Label for iterative (slightly offset down)
        ax.annotate(model, 
                   (no_ctx_accuracies[i], iterative_accuracies[i]),
                   xytext=(5, -5), textcoords='offset points',
                   fontsize=8, alpha=0.8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='#f39c12', alpha=0.7))
    
    # Add diagonal reference line (y=x, no improvement)
    min_acc = min(min(no_ctx_accuracies), min(gold_ctx_accuracies), min(iterative_accuracies))
    max_acc = max(max(no_ctx_accuracies), max(gold_ctx_accuracies), max(iterative_accuracies))
    ax.plot([min_acc, max_acc], [min_acc, max_acc], 'k--', alpha=0.3, 
           linewidth=2, label='No Improvement (y=x)', zorder=1)
    
    # Styling
    ax.set_xlabel('No Context Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy with Context/RAG (%)', fontsize=12, fontweight='bold')
    ax.set_title('Knowledge Gap Scatter: Impact of Context and Retrieval\n(Higher y-values show greater benefit from context/RAG)',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', framealpha=0.95, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set reasonable axis limits with padding
    x_padding = (max(no_ctx_accuracies) - min(no_ctx_accuracies)) * 0.1
    y_padding = (max_acc - min_acc) * 0.1
    ax.set_xlim(min(no_ctx_accuracies) - x_padding, max(no_ctx_accuracies) + x_padding)
    ax.set_ylim(min_acc - y_padding, max_acc + y_padding)
    
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {out_path.name}")


def print_summary_stats(
    models: List[str],
    model_categories: Dict[str, Dict[str, Set[str]]],
) -> None:
    """Print summary statistics to console."""
    print("\n" + "=" * 80)
    print("KNOWLEDGE GAP ANALYSIS SUMMARY")
    print("=" * 80)
    
    categories = ['intrinsic', 'retrieval', 'iterative', 'gap']
    category_labels = {
        'intrinsic': 'Intrinsic Knowledge',
        'retrieval': 'Retrieval Dependent',
        'iterative': 'Iterative Dependent',
        'gap': 'Knowledge Gap'
    }
    
    for model in models:
        cats = model_categories.get(model, {})
        total = sum(len(cats.get(cat, set())) for cat in categories)
        
        print(f"\n{model}:")
        print(f"  Total Questions: {total}")
        
        for cat in categories:
            count = len(cats.get(cat, set()))
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {category_labels[cat]:22s}: {count:4d} ({pct:5.1f}%)")
        
        # Calculate cumulative accuracies
        intrinsic = len(cats.get('intrinsic', set()))
        retrieval = len(cats.get('retrieval', set()))
        iterative = len(cats.get('iterative', set()))
        
        if total > 0:
            no_ctx_acc = intrinsic / total * 100
            gold_ctx_acc = (intrinsic + retrieval) / total * 100
            iter_acc = (intrinsic + retrieval + iterative) / total * 100
            
            print(f"  Cumulative Accuracy:")
            print(f"    No Context:     {no_ctx_acc:5.1f}%")
            print(f"    + Gold Context: {gold_ctx_acc:5.1f}% (gain: {gold_ctx_acc - no_ctx_acc:+.1f} pp)")
            print(f"    + Iterative:    {iter_acc:5.1f}% (gain: {iter_acc - gold_ctx_acc:+.1f} pp)")
    
    print("\n" + "=" * 80)


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    plots_dir = base / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model entries
    model_entries = get_iterative_model_entries(existing_only=True)
    model_order = [display for _, display in model_entries]
    
    # Paths
    dir_no_ctx = base / "response-jsonl-without-context"
    dir_gold_ctx = base / "response-jsonl-with-context"
    dir_iterative = base / "responses_reverified"
    
    print("Loading data...")
    
    # Load data for each model
    model_categories: Dict[str, Dict[str, Set[str]]] = {}
    model_raw_data: Dict[str, Tuple[Dict[str, bool], Dict[str, bool], Dict[str, bool]]] = {}
    
    for iterative_path, display in model_entries:
        print(f"  Processing: {display}")
        
        # Load responses from all three conditions
        no_ctx = scan_context_dir(dir_no_ctx, display)
        gold_ctx = scan_context_dir(dir_gold_ctx, display)
        iterative = load_responses_by_question(iterative_path)
        
        # Store raw data for cumulative accuracy plot
        model_raw_data[display] = (no_ctx, gold_ctx, iterative)
        
        # Debug: show counts for each condition
        no_ctx_count = sum(1 for v in no_ctx.values() if v)
        gold_ctx_count = sum(1 for v in gold_ctx.values() if v)
        iter_count = sum(1 for v in iterative.values() if v)
        
        print(f"    No Context: {no_ctx_count}/{len(no_ctx)} correct ({no_ctx_count/1186*100:.2f}%)")
        print(f"    Gold Context: {gold_ctx_count}/{len(gold_ctx)} correct ({gold_ctx_count/1186*100:.2f}%)")
        print(f"    Iterative: {iter_count}/{len(iterative)} correct ({iter_count/1186*100:.2f}%)")
        
        # Check intersection
        common = set(no_ctx.keys()) & set(gold_ctx.keys()) & set(iterative.keys())
        print(f"    Common questions: {len(common)}")
        
        # Categorize questions
        categories = categorize_questions(no_ctx, gold_ctx, iterative)
        model_categories[display] = categories
    
    print("\nGenerating plots...")
    
    # Plot 1: Waterfall
    plot_knowledge_gap_waterfall(
        model_order,
        model_categories,
        plots_dir / "knowledge_gap_waterfall.png"
    )
    
    # Plot 2: Matrix/Heatmap
    plot_knowledge_gap_matrix(
        model_order,
        model_categories,
        plots_dir / "knowledge_gap_matrix.png"
    )
    
    # Plot 3: Detailed breakdown (using raw data for actual accuracies)
    plot_knowledge_gap_detailed_breakdown(
        model_order,
        model_raw_data,
        plots_dir / "knowledge_gap_cumulative_accuracy.png"
    )
    
    # Plot 4: Scatter plot showing relationship between no-context and with-context accuracy
    plot_knowledge_gap_scatter(
        model_order,
        model_raw_data,
        plots_dir / "knowledge_gap_scatter.png"
    )
    
    # Print summary
    print_summary_stats(model_order, model_categories)
    
    print("\n✅ All plots generated successfully!")


if __name__ == "__main__":
    main()
