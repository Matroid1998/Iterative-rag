#!/usr/bin/env python3
"""
Generate two plots showing unanswered questions by model and retrieval steps:
1. No Context plot: Shows unanswered questions by max source step (from iterative RAG)
2. Gold Context plot: Shows unanswered questions by max source step

Logic matches hop_distributions_all_models.png but only shows:
- Column 1 (No Context) and Column 2 (Gold Context)  
- Rows 2-7 (individual models, not the aggregated first row)
Consolidated into 2 separate plots.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set
import numpy as np
import re

from config import (
    get_iterative_model_entries,
    get_iterative_display_names,
    get_display_name,
)


def _simplify_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _build_canonical_map(names: List[str]) -> Dict[str, str]:
    return {_simplify_name(name): name for name in names}


def _canonicalize_display_name(name: str, canonical_map: Dict[str, str]) -> str:
    simplified = _simplify_name(name)
    if simplified in canonical_map:
        return canonical_map[simplified]
    if simplified.endswith("reasoning"):
        base = simplified[: -len("reasoning")]
        if base in canonical_map:
            return canonical_map[base]
    return name


def load_records(path: Path) -> List[dict]:
    """Load JSONL records from a file."""
    if not path.exists():
        return []
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
    return records


def extract_question(record: dict) -> str | None:
    """Extract question text from a record."""
    question = record.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()
    for key in ("raw", "raw_response"):
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


def get_unanswered_questions_set(unanswered_file: Path) -> Set[str]:
    """
    Extract the set of unanswered question texts from the unanswered file.
    
    Returns:
        Set of question strings that are unanswered
    """
    if not unanswered_file.exists():
        print(f"Warning: {unanswered_file} not found")
        return set()
    
    unanswered_questions = set()
    records = load_records(unanswered_file)
    
    for record in records:
        question = record.get("question", "").strip()
        if question:
            unanswered_questions.add(question)
    
    return unanswered_questions


def load_unanswered_steps_by_model(
    iterative_dir: Path, 
    model_files: Dict[str, str], 
    unanswered_questions: Set[str]
) -> Dict[str, List[int]]:
    """
    Load max source steps for questions that were SOLVED by iterative RAG (previously unanswered).
    
    Args:
        iterative_dir: Directory containing iterative RAG result files
        model_files: Mapping of filename to display name
        unanswered_questions: Set of questions that are unanswered in baseline
    
    Returns:
        Dict mapping model display name to list of max source steps for SOLVED questions only
    """
    model_solved_steps = {}
    
    for filename, display_name in model_files.items():
        iterative_path = iterative_dir / filename
        if not iterative_path.exists():
            print(f"  {display_name}: File not found, skipping")
            model_solved_steps[display_name] = []
            continue
        
        solved_steps = []
        for record in load_records(iterative_path):
            question = extract_question(record)
            if not question or question not in unanswered_questions:
                continue
            
            # Only include questions that iterative RAG solved (is_correct = True)
            is_correct = bool(record.get("is_correct", False))
            if not is_correct:
                continue
            
            max_step = extract_max_source_step(record)
            if max_step is not None:
                solved_steps.append(max_step)
        
        model_solved_steps[display_name] = solved_steps
        print(f"  {display_name}: {len(solved_steps)} solved")
    
    return model_solved_steps


def plot_unanswered_questions(
    model_unanswered_steps: Dict[str, List[int]],
    output_path: Path,
    model_order: List[str],
    title_suffix: str
) -> None:
    """
    Create a bar chart showing SOLVED question counts by model and max source step.
    
    X-axis: Models
    Y-axis: Question count (solved questions only)
    Bars: Grouped by max source step (1-5), showing questions recovered by iterative RAG
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required. Install with 'pip install matplotlib'.") from exc

    # Filter to ordered models
    ordered_data = {model: model_unanswered_steps.get(model, []) for model in model_order}
    
    # Determine max step across all models
    max_step = 0
    for steps in ordered_data.values():
        if steps:
            max_step = max(max_step, max(steps))
    
    if max_step == 0:
        max_step = 5  # Default
    
    # Count unanswered questions by step for each model
    step_range = list(range(1, min(max_step, 5) + 1))  # Limit to steps 1-5
    model_counts = {}
    
    for model, steps in ordered_data.items():
        counter = Counter(steps)
        model_counts[model] = [counter.get(step, 0) for step in step_range]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(max(14, len(model_order) * 1.6), 8))

    x = np.arange(len(model_order))
    bar_width = 0.8 / max(1, len(step_range))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % cmap.N) for i in range(len(step_range))]
    
    # Plot bars for each step
    for i, step in enumerate(step_range):
        counts = [model_counts[model][i] for model in model_order]
        offset = (i - (len(step_range) - 1) / 2) * bar_width
        bars = ax.bar(x + offset, counts, bar_width, 
                     label=f'Step {step}', color=colors[i % len(colors)],
                     edgecolor='black', linewidth=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('Solved Questions Count', fontsize=13, fontweight='bold')
    ax.set_title(f'Questions Recovered by Iterative RAG\n({title_suffix})', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_order, rotation=15, ha='right', fontsize=11)
    ax.legend(title='Max Source Step', loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved plot to {output_path}")


def main() -> None:
    base = Path(__file__).resolve().parents[3]
    
    # Output directory
    output_dir = base.parent / "data" / "plots" / "general"
    output_dir.mkdir(exist_ok=True)
    
    # Results directory with unanswered questions
    results_dir = base / "results" / "unanswered_questions"
    
    # Iterative RAG directory
    iterative_dir = base / "responses_reverified"
    
    # Model files mapping
    model_files = {
        path.name: display_name
        for path, display_name in get_iterative_model_entries()
    }

    # Model order as specified
    model_order = get_iterative_display_names(existing_only=True)
    
    # Process No Context unanswered questions
    print("Loading unanswered questions (No Context)...")
    no_context_file = results_dir / "response-jsonl-without-context_unanswered.jsonl"
    unanswered_no_context = get_unanswered_questions_set(no_context_file)
    print(f"Found {len(unanswered_no_context)} unique unanswered questions in No Context")
    
    print("\nLoading solved questions from iterative RAG (No Context)...")
    model_steps_no_context = load_unanswered_steps_by_model(
        iterative_dir, model_files, unanswered_no_context
    )
    
    # Process Gold Context unanswered questions
    print("\nLoading unanswered questions (Gold Context)...")
    gold_context_file = results_dir / "response-jsonl-with-context_unanswered.jsonl"
    unanswered_gold_context = get_unanswered_questions_set(gold_context_file)
    print(f"Found {len(unanswered_gold_context)} unique unanswered questions in Gold Context")
    
    print("\nLoading solved questions from iterative RAG (Gold Context)...")
    model_steps_gold_context = load_unanswered_steps_by_model(
        iterative_dir, model_files, unanswered_gold_context
    )
    
    # Generate Plot 1: No Context
    print("\nGenerating Plot 1: Questions Recovered - No Context...")
    plot1_path = output_dir / "solved_questions_no_context.png"
    plot_unanswered_questions(model_steps_no_context, plot1_path, model_order, "No Context")
    plot_unanswered_questions(
        model_steps_no_context,
        output_dir / "correct_answers_no_context.png",
        model_order,
        "No Context",
    )
    
    # Generate Plot 2: Gold Context
    print("\nGenerating Plot 2: Questions Recovered - Gold Context...")
    plot2_path = output_dir / "solved_questions_gold_context.png"
    plot_unanswered_questions(model_steps_gold_context, plot2_path, model_order, "Gold Context")
    plot_unanswered_questions(
        model_steps_gold_context,
        output_dir / "correct_answers_gold_context.png",
        model_order,
        "Gold Context",
    )
    
    print("\n" + "="*80)
    print("Summary Statistics:")
    print("="*80)
    print("\nNo Context (Questions Recovered by Iterative RAG):")
    for model in model_order:
        steps = model_steps_no_context.get(model, [])
        if steps:
            print(f"  {model:30s}: {len(steps):4d} solved (avg step: {np.mean(steps):.2f})")
        else:
            print(f"  {model:30s}: No questions solved")
    
    print("\nGold Context (Questions Recovered by Iterative RAG):")
    for model in model_order:
        steps = model_steps_gold_context.get(model, [])
        if steps:
            print(f"  {model:30s}: {len(steps):4d} solved (avg step: {np.mean(steps):.2f})")
        else:
            print(f"  {model:30s}: No questions solved")


if __name__ == "__main__":
    main()
