#!/usr/bin/env python3
"""
Scatter plot comparing single-hop (1-hop) question accuracy between gold context and iterative RAG.
X-axis: Gold context accuracy
Y-axis: Iterative RAG accuracy
Color: Average number of steps taken for 1-hop questions
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from config import get_iterative_model_entries, get_model_color


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


def get_single_hop_accuracy_and_steps(
    file_path: Path,
    qa_hops: Dict[str, int]
) -> Tuple[float, float]:
    """
    Calculate accuracy and average steps for single-hop questions.
    
    Returns:
        (accuracy, avg_steps) - both as percentages/numbers
    """
    records = load_records(file_path)
    
    single_hop_questions = []
    
    for record in records:
        question = extract_question(record)
        if not question:
            continue
        
        # Check if it's a single-hop question
        hop_count = record.get("number_of_hops")
        if not isinstance(hop_count, int) or hop_count <= 0:
            hop_count = qa_hops.get(question)
        
        if hop_count != 1:
            continue
        
        is_correct = bool(record.get("is_correct", False))
        max_step = extract_max_source_step(record)
        
        single_hop_questions.append({
            "question": question,
            "is_correct": is_correct,
            "max_step": max_step if max_step else 1
        })
    
    if not single_hop_questions:
        return 0.0, 0.0
    
    correct_count = sum(1 for q in single_hop_questions if q["is_correct"])
    accuracy = 100 * correct_count / len(single_hop_questions)
    
    avg_steps = np.mean([q["max_step"] for q in single_hop_questions])
    
    return accuracy, avg_steps


def create_scatter_plot(
    model_data: Dict[str, Dict[str, float]],
    output_path: Path
) -> None:
    """
    Create scatter plot comparing gold context vs iterative RAG accuracy for 1-hop questions.
    
    model_data: {model_name: {"gold_acc": float, "iter_acc": float, "avg_steps": float}}
    """
    if not model_data:
        print("No data to plot!")
        return
    
    # Extract data
    models = list(model_data.keys())
    gold_acc = [model_data[m]["gold_acc"] for m in models]
    iter_acc = [model_data[m]["iter_acc"] for m in models]
    avg_steps = [model_data[m]["avg_steps"] for m in models]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Normalize colors based on avg_steps
    norm = Normalize(vmin=min(avg_steps), vmax=max(avg_steps))
    cmap = plt.cm.viridis
    
    # Plot each model
    for i, model in enumerate(models):
        x = gold_acc[i]
        y = iter_acc[i]
        steps = avg_steps[i]
        
        # Plot point
        scatter = ax.scatter(x, y, s=200, c=[steps], cmap=cmap, norm=norm,
                           alpha=0.7, edgecolors='black', linewidth=2)
        
        # Add model label
        ax.annotate(model, (x, y), xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                           edgecolor='gray', alpha=0.8))
    
    # Add diagonal line (y=x) for reference
    min_val = min(min(gold_acc), min(iter_acc)) - 2
    max_val = max(max(gold_acc), max(iter_acc)) + 2
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=2,
           label='Equal Performance Line')
    
    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Average Number of Steps\n(for 1-hop questions)', 
                   fontsize=12, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Gold Context Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Iterative RAG Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Single-Hop Question Performance:\nGold Context vs Iterative RAG',
                fontsize=16, fontweight='bold', pad=20)
    
    # Set equal aspect ratio and limits
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend
    ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved scatter plot to {output_path}")


def main() -> None:
    base = Path(__file__).resolve().parents[3]
    
    # Output directory
    output_dir = base.parent / "data" / "plots" / "general"
    output_dir.mkdir(exist_ok=True)
    
    # Gold context and no context directories
    gold_context_dir = base / "response-jsonl-with-context"
    
    # Load QA hop data
    qa_path = base.parent / "data" / "corpus" / "chemrxiv_qa.json"
    print("Loading question hop data...")
    qa_hops = load_qa_hops(qa_path)
    print(f"Loaded hop data for {len(qa_hops)} questions")
    
    # Model data
    model_data: Dict[str, Dict[str, float]] = {}
    
    print("\nProcessing models...")
    
    for iterative_path, display_name in get_iterative_model_entries(existing_only=True):
        if not iterative_path.exists():
            continue
        
        # Get iterative RAG accuracy and steps for 1-hop questions
        iter_acc, iter_steps = get_single_hop_accuracy_and_steps(iterative_path, qa_hops)
        
        # Find corresponding gold context file
        # Try multiple naming patterns
        gold_filename_candidates = [
            iterative_path.name,
            iterative_path.name.replace("_reverified", ""),
            iterative_path.name.replace("_reverified", "-reasoning"),
        ]
        
        # Special mappings for some models
        special_mappings = {
            "responses_openrouter_google__gemini-2.5-pro_reverified.jsonl": "responses_openrouter_google__gemini-2.5-pro-reasoning.jsonl",
            "responses_openrouter_x-ai__grok-4-fast_reverified.jsonl": "responses_openrouter_x-ai__grok-4-fast-reasoning.jsonl",
            "responses_openrouter_z-ai__glm-4.6_reverified.jsonl": "responses_openrouter_z-ai__glm-4.6-reasoning_reverified.jsonl",
        }
        
        if iterative_path.name in special_mappings:
            gold_filename_candidates.insert(0, special_mappings[iterative_path.name])
        
        gold_path = None
        for candidate in gold_filename_candidates:
            candidate_path = gold_context_dir / candidate
            if candidate_path.exists():
                gold_path = candidate_path
                break
        
        if not gold_path:
            print(f"  Warning: No gold context file found for {display_name}")
            continue
        
        # Get gold context accuracy and steps for 1-hop questions
        gold_acc, gold_steps = get_single_hop_accuracy_and_steps(gold_path, qa_hops)
        
        model_data[display_name] = {
            "gold_acc": gold_acc,
            "iter_acc": iter_acc,
            "avg_steps": iter_steps  # Use iterative RAG steps for coloring
        }
        
        print(f"  {display_name:30s}: Gold={gold_acc:5.1f}%, Iter={iter_acc:5.1f}%, AvgSteps={iter_steps:.2f}")
    
    if not model_data:
        print("\nNo model data found!")
        return
    
    # Generate plot
    print("\nGenerating scatter plot...")
    output_path = output_dir / "single_hop_gold_vs_iterative.png"
    create_scatter_plot(model_data, output_path)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SINGLE-HOP QUESTION PERFORMANCE SUMMARY")
    print("="*80)
    
    for model in sorted(model_data.keys()):
        data = model_data[model]
        improvement = data["iter_acc"] - data["gold_acc"]
        print(f"{model:30s}: Gold={data['gold_acc']:5.1f}%, Iter={data['iter_acc']:5.1f}%, "
              f"Δ={improvement:+5.1f}%, AvgSteps={data['avg_steps']:.2f}")
    
    # Calculate overall statistics
    avg_gold = np.mean([d["gold_acc"] for d in model_data.values()])
    avg_iter = np.mean([d["iter_acc"] for d in model_data.values()])
    avg_steps_overall = np.mean([d["avg_steps"] for d in model_data.values()])
    
    print(f"\nOverall Averages:")
    print(f"  Gold Context:    {avg_gold:.1f}%")
    print(f"  Iterative RAG:   {avg_iter:.1f}%")
    print(f"  Avg Steps:       {avg_steps_overall:.2f}")
    print(f"  Improvement:     {avg_iter - avg_gold:+.1f}%")


if __name__ == "__main__":
    main()
