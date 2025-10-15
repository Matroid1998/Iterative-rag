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
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from config import get_display_name, get_iterative_display_names


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


def load_qa_hops(qa_path: Path) -> Dict[str, int]:
    """Load question to hop count mapping from chemrxiv_qa.json."""
    if not qa_path.exists():
        return {}
    
    with qa_path.open("r", encoding="utf-8") as f:
        qa_data = json.load(f)
    
    qa_hops = {}
    for item in qa_data:
        question = item.get("q", "").strip()
        path = item.get("path", [])
        hops = len(path) if path else 1
        if question:
            qa_hops[question] = hops
    
    return qa_hops


def get_unanswered_questions_by_model(
    unanswered_file: Path,
    qa_hops: Dict[str, int],
    allowed_models: List[str],
) -> Dict[str, List[int]]:
    """
    For each model, get the list of hop counts for unanswered questions.
    
    The unanswered file has structure:
    {
        "question": "...",
        "number_of_hops": {... or int},
        "model_attempts": [
            {"file": "...", "answer": "...", "is_correct": false},
            ...
        ]
    }
    
    Returns:
        Dict mapping model name to list of hop counts for unanswered questions
    """
    if not unanswered_file.exists():
        print(f"Warning: {unanswered_file} not found")
        return {}
    
    canonical_map = _build_canonical_map(allowed_models)
    allowed_set = set(allowed_models)
    model_unanswered_hops = defaultdict(list)
    
    records = load_records(unanswered_file)
    for record in records:
        question = record.get("question", "").strip()
        if not question:
            continue
        
        # Get hop count - try from record first, then from qa_hops
        hop_count = record.get("number_of_hops")
        if isinstance(hop_count, dict):
            # If it's a dict, try to get actual number, otherwise use qa_hops
            hop_count = None
        
        if not hop_count or not isinstance(hop_count, int):
            hop_count = qa_hops.get(question)
        
        if not hop_count:
            continue
        
        model_attempts = record.get("model_attempts", [])
        for attempt in model_attempts:
            if attempt.get("is_correct", True):
                continue
            model_file = attempt.get("file", "")
            if not model_file:
                continue
            model_stem = Path(model_file).stem
            display_name = _canonicalize_display_name(
                get_display_name(model_stem),
                canonical_map,
            )
            if display_name in allowed_set:
                model_unanswered_hops[display_name].append(hop_count)

    for model in allowed_models:
        model_unanswered_hops.setdefault(model, [])

    return dict(model_unanswered_hops)


def plot_unanswered_questions_no_context(
    model_unanswered_hops: Dict[str, List[int]],
    output_path: Path,
    model_order: List[str]
) -> None:
    """
    Create a bar chart showing unanswered question counts by model and hop complexity.
    This represents the "No Context" scenario.
    
    X-axis: Models
    Y-axis: Question count
    Bars: Grouped by hop (1-5), showing unanswered questions only
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required. Install with 'pip install matplotlib'.") from exc

    # Filter to ordered models
    ordered_data = {model: model_unanswered_hops.get(model, []) for model in model_order}
    
    # Determine max hop across all models
    max_hop = 0
    for hops in ordered_data.values():
        if hops:
            max_hop = max(max_hop, max(hops))
    
    if max_hop == 0:
        max_hop = 5  # Default
    
    # Count unanswered questions by hop for each model
    hop_range = list(range(1, min(max_hop, 5) + 1))  # Limit to hops 1-5
    model_counts = {}
    
    for model, hops in ordered_data.items():
        counter = Counter(hops)
        model_counts[model] = [counter.get(hop, 0) for hop in hop_range]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(max(14, len(model_order) * 1.6), 8))

    x = np.arange(len(model_order))
    bar_width = 0.8 / max(1, len(hop_range))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % cmap.N) for i in range(len(hop_range))]
    
    # Plot bars for each hop
    for i, hop in enumerate(hop_range):
        counts = [model_counts[model][i] for model in model_order]
        offset = (i - (len(hop_range) - 1) / 2) * bar_width
        bars = ax.bar(x + offset, counts, bar_width, 
                     label=f'Hop {hop}', color=colors[i % len(colors)],
                     edgecolor='black', linewidth=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('Unanswered Questions Count', fontsize=13, fontweight='bold')
    ax.set_title('Unanswered Questions by Model and Hop Complexity\n(No Context Scenario)', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_order, rotation=15, ha='right', fontsize=11)
    ax.legend(title='Number of Hops', loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved plot to {output_path}")


def plot_unanswered_questions_gold_context(
    model_unanswered_hops: Dict[str, List[int]],
    output_path: Path,
    model_order: List[str]
) -> None:
    """
    Create a bar chart showing unanswered question counts by model and hop complexity.
    This represents the "Gold Context" scenario.
    
    X-axis: Models
    Y-axis: Question count
    Bars: Grouped by hop (1-5), showing unanswered questions only
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required. Install with 'pip install matplotlib'.") from exc

    # Filter to ordered models
    ordered_data = {model: model_unanswered_hops.get(model, []) for model in model_order}
    
    # Determine max hop across all models
    max_hop = 0
    for hops in ordered_data.values():
        if hops:
            max_hop = max(max_hop, max(hops))
    
    if max_hop == 0:
        max_hop = 5
    
    # Count unanswered questions by hop for each model
    hop_range = list(range(1, min(max_hop, 5) + 1))
    model_counts = {}
    
    for model, hops in ordered_data.items():
        counter = Counter(hops)
        model_counts[model] = [counter.get(hop, 0) for hop in hop_range]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(max(14, len(model_order) * 1.6), 8))

    x = np.arange(len(model_order))
    bar_width = 0.8 / max(1, len(hop_range))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % cmap.N) for i in range(len(hop_range))]
    
    # Plot bars for each hop
    for i, hop in enumerate(hop_range):
        counts = [model_counts[model][i] for model in model_order]
        offset = (i - (len(hop_range) - 1) / 2) * bar_width
        bars = ax.bar(x + offset, counts, bar_width,
                     label=f'Hop {hop}', color=colors[i % len(colors)],
                     edgecolor='black', linewidth=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('Unanswered Questions Count', fontsize=13, fontweight='bold')
    ax.set_title('Unanswered Questions by Model and Hop Complexity\n(Gold Context Scenario)', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_order, rotation=15, ha='right', fontsize=11)
    ax.legend(title='Number of Hops', loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved plot to {output_path}")


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    
    # Output directory
    output_dir = base / "plots"
    output_dir.mkdir(exist_ok=True)
    
    # Results directory with unanswered questions
    results_dir = base / "results" / "unanswered_questions"
    
    # Load QA hop data
    qa_path = base / "docs" / "chemrxiv_qa.json"
    print("Loading question hop data...")
    qa_hops = load_qa_hops(qa_path)
    print(f"Loaded hop data for {len(qa_hops)} questions")
    
    # Model order as specified
    model_order = get_iterative_display_names(existing_only=True)
    if not model_order:
        raise SystemExit("No reverified response files found when building model order")
    
    # Process No Context unanswered questions
    print("\nLoading unanswered questions (No Context)...")
    no_context_file = results_dir / "response-jsonl-without-context_unanswered.jsonl"
    model_unanswered_no_context = get_unanswered_questions_by_model(
        no_context_file, qa_hops, model_order
    )
    
    for model in model_order:
        count = len(model_unanswered_no_context.get(model, []))
        print(f"  {model}: {count} unanswered questions")
    
    # Process Gold Context unanswered questions
    print("\nLoading unanswered questions (Gold Context)...")
    gold_context_file = results_dir / "response-jsonl-with-context_unanswered.jsonl"
    model_unanswered_gold_context = get_unanswered_questions_by_model(
        gold_context_file, qa_hops, model_order
    )
    
    for model in model_order:
        count = len(model_unanswered_gold_context.get(model, []))
        print(f"  {model}: {count} unanswered questions")
    
    # Generate Plot 1: No Context
    print("\nGenerating Plot 1: Unanswered Questions - No Context...")
    plot1_path = output_dir / "unanswered_no_context.png"
    plot_unanswered_questions_no_context(model_unanswered_no_context, plot1_path, model_order)
    
    # Generate Plot 2: Gold Context
    print("\nGenerating Plot 2: Unanswered Questions - Gold Context...")
    plot2_path = output_dir / "unanswered_gold_context.png"
    plot_unanswered_questions_gold_context(model_unanswered_gold_context, plot2_path, model_order)
    
    print("\n" + "="*80)
    print("Summary Statistics:")
    print("="*80)
    print("\nNo Context:")
    for model in model_order:
        hops = model_unanswered_no_context.get(model, [])
        if hops:
            print(f"  {model:30s}: {len(hops):4d} unanswered (avg hop: {np.mean(hops):.2f})")
        else:
            print(f"  {model:30s}: No unanswered questions")
    
    print("\nGold Context:")
    for model in model_order:
        hops = model_unanswered_gold_context.get(model, [])
        if hops:
            print(f"  {model:30s}: {len(hops):4d} unanswered (avg hop: {np.mean(hops):.2f})")
        else:
            print(f"  {model:30s}: No unanswered questions")


if __name__ == "__main__":
    main()
