#!/usr/bin/env python3
"""
Regressions vs. Recoveries Bar Chart (per model)

Counts of:
- Gold correct → Iterative wrong (Regressions)
- Gold wrong → Iterative correct (Recoveries)

This quantifies iteration's net effect and flags models prone to over-search 
or premature finalization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set
import numpy as np
import matplotlib.pyplot as plt

from config import get_iterative_model_entries


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


def analyze_regressions_recoveries(
    iterative_path: Path,
    gold_path: Path
) -> Dict[str, int]:
    """
    Analyze regressions and recoveries between gold context and iterative RAG.
    
    Returns:
        {
            'recoveries': count of questions wrong in gold, correct in iterative,
            'regressions': count of questions correct in gold, wrong in iterative,
            'both_correct': count of questions correct in both,
            'both_wrong': count of questions wrong in both
        }
    """
    # Load records
    gold_records = load_records(gold_path)
    iterative_records = load_records(iterative_path)
    
    # Build question → correctness maps
    gold_correctness: Dict[str, bool] = {}
    for record in gold_records:
        question = extract_question(record)
        if question:
            gold_correctness[question] = bool(record.get("is_correct", False))
    
    iterative_correctness: Dict[str, bool] = {}
    for record in iterative_records:
        question = extract_question(record)
        if question:
            iterative_correctness[question] = bool(record.get("is_correct", False))
    
    # Find common questions
    common_questions = set(gold_correctness.keys()) & set(iterative_correctness.keys())
    
    # Categorize
    recoveries = 0  # Gold wrong → Iterative correct
    regressions = 0  # Gold correct → Iterative wrong
    both_correct = 0
    both_wrong = 0
    
    for question in common_questions:
        gold_correct = gold_correctness[question]
        iter_correct = iterative_correctness[question]
        
        if gold_correct and iter_correct:
            both_correct += 1
        elif not gold_correct and not iter_correct:
            both_wrong += 1
        elif not gold_correct and iter_correct:
            recoveries += 1
        elif gold_correct and not iter_correct:
            regressions += 1
    
    return {
        'recoveries': recoveries,
        'regressions': regressions,
        'both_correct': both_correct,
        'both_wrong': both_wrong,
        'total_questions': len(common_questions)
    }


def plot_regressions_vs_recoveries(
    model_data: Dict[str, Dict[str, int]],
    output_path: Path
) -> None:
    """
    Create a grouped bar chart showing regressions and recoveries per model.
    
    model_data: {model_name: {'recoveries': int, 'regressions': int, ...}}
    """
    if not model_data:
        print("No data to plot!")
        return
    
    # Sort models by net gain (recoveries - regressions)
    models = sorted(model_data.keys(), 
                   key=lambda m: model_data[m]['recoveries'] - model_data[m]['regressions'],
                   reverse=True)
    
    recoveries = [model_data[m]['recoveries'] for m in models]
    regressions = [model_data[m]['regressions'] for m in models]
    net_gains = [r - reg for r, reg in zip(recoveries, regressions)]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), 
                                     gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot 1: Stacked bar chart
    x = np.arange(len(models))
    width = 0.6
    
    # Plot recoveries (positive) and regressions (negative)
    bars_recoveries = ax1.bar(x, recoveries, width, label='Recoveries (Gold Wrong → Iter Correct)',
                              color='#2ecc71', edgecolor='black', linewidth=1.2, alpha=0.85)
    bars_regressions = ax1.bar(x, [-r for r in regressions], width, 
                               label='Regressions (Gold Correct → Iter Wrong)',
                               color='#e74c3c', edgecolor='black', linewidth=1.2, alpha=0.85)
    
    # Add value labels
    for i, (rec, reg) in enumerate(zip(recoveries, regressions)):
        if rec > 0:
            ax1.text(i, rec, f'{rec}', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')
        if reg > 0:
            ax1.text(i, -reg, f'{reg}', ha='center', va='top', 
                    fontsize=9, fontweight='bold')
    
    # Add net gain as a line
    ax1_twin = ax1.twinx()
    line = ax1_twin.plot(x, net_gains, 'o-', color='black', linewidth=2.5, 
                         markersize=5, label='Net Gain (Recoveries - Regressions)',
                         zorder=10)
    
    # Add zero line
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    
    # Customize plot 1
    ax1.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Number of Questions', fontsize=13, fontweight='bold')
    ax1_twin.set_ylabel('Net Gain', fontsize=13, fontweight='bold', rotation=270, labelpad=20)
    ax1.set_title('Regressions vs. Recoveries: Gold Context → Iterative RAG\n' + 
                  'Quantifying Iteration\'s Net Effect per Model',
                  fontsize=15, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    
    # Combine legends
    bars_handles, bars_labels = ax1.get_legend_handles_labels()
    line_handles, line_labels = ax1_twin.get_legend_handles_labels()
    ax1.legend(bars_handles + line_handles, bars_labels + line_labels, 
              loc='upper right', fontsize=10)
    
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Net gain bar chart
    colors = ['#2ecc71' if ng > 0 else '#e74c3c' for ng in net_gains]
    bars_net = ax2.bar(x, net_gains, width, color=colors, 
                       edgecolor='black', linewidth=1.2, alpha=0.85)
    
    # Add value labels
    for i, ng in enumerate(net_gains):
        if ng != 0:
            ax2.text(i, ng, f'{ng:+d}', ha='center', 
                    va='bottom' if ng > 0 else 'top', 
                    fontsize=9, fontweight='bold')
    
    # Add zero line
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    
    # Customize plot 2
    ax2.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Net Gain', fontsize=13, fontweight='bold')
    ax2.set_title('Net Effect (Recoveries - Regressions)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved regressions vs recoveries plot to {output_path}")


def main() -> None:
    base = Path(__file__).resolve().parents[3]
    
    # Output directory
    output_dir = base / "plots"
    output_dir.mkdir(exist_ok=True)
    
    # Gold context directory
    gold_context_dir = base / "response-jsonl-with-context"
    
    # Special mappings for gold context files
    gold_context_mapping = {
        "responses_openrouter_google__gemini-2.5-pro_reverified.jsonl": "responses_openrouter_google__gemini-2.5-pro-reasoning.jsonl",
        "responses_openrouter_x-ai__grok-4-fast_reverified.jsonl": "responses_openrouter_x-ai__grok-4-fast-reasoning.jsonl",
        "responses_openrouter_z-ai__glm-4.6_reverified.jsonl": "responses_openrouter_z-ai__glm-4.6-reasoning_reverified.jsonl",
    }
    
    model_data: Dict[str, Dict[str, int]] = {}
    
    print("Analyzing regressions and recoveries...")
    
    for iterative_path, display_name in get_iterative_model_entries(existing_only=True):
        if not iterative_path.exists():
            continue
        
        # Find corresponding gold context file
        # Try multiple naming patterns
        if iterative_path.name in gold_context_mapping:
            gold_filename = gold_context_mapping[iterative_path.name]
            gold_path = gold_context_dir / gold_filename
        else:
            # Try: same name, without _reverified, with _reverified
            candidates = [
                iterative_path.name,
                iterative_path.name.replace("_reverified", ""),
                iterative_path.name if "_reverified" in iterative_path.name else iterative_path.name.replace(".jsonl", "_reverified.jsonl"),
            ]
            gold_path = None
            for candidate in candidates:
                candidate_path = gold_context_dir / candidate
                if candidate_path.exists():
                    gold_path = candidate_path
                    break
        
        if not gold_path or not gold_path.exists():
            print(f"  Warning: No gold context file found for {display_name}")
            continue
        
        # Analyze
        results = analyze_regressions_recoveries(iterative_path, gold_path)
        model_data[display_name] = results
        
        net_gain = results['recoveries'] - results['regressions']
        print(f"  {display_name:30s}: Recoveries={results['recoveries']:4d}, "
              f"Regressions={results['regressions']:3d}, Net={net_gain:+4d}")
    
    if not model_data:
        print("\nNo model data found!")
        return
    
    # Generate plot
    print("\nGenerating plot...")
    output_path = output_dir / "regressions_vs_recoveries.png"
    plot_regressions_vs_recoveries(model_data, output_path)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("REGRESSIONS VS. RECOVERIES SUMMARY")
    print("="*80)
    
    total_recoveries = sum(d['recoveries'] for d in model_data.values())
    total_regressions = sum(d['regressions'] for d in model_data.values())
    total_net = total_recoveries - total_regressions
    
    print(f"\nOverall Totals:")
    print(f"  Total Recoveries:  {total_recoveries:5d}")
    print(f"  Total Regressions: {total_regressions:5d}")
    print(f"  Net Gain:          {total_net:+5d}")
    print(f"  Recovery/Regression Ratio: {total_recoveries/max(1, total_regressions):.2f}:1")
    
    print(f"\nPer-Model Statistics:")
    for model in sorted(model_data.keys()):
        data = model_data[model]
        ratio = data['recoveries'] / max(1, data['regressions'])
        net = data['recoveries'] - data['regressions']
        print(f"  {model:30s}: R={data['recoveries']:4d}, "
              f"Reg={data['regressions']:3d}, Net={net:+4d}, Ratio={ratio:.2f}:1")


if __name__ == "__main__":
    main()
