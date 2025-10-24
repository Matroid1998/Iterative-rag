#!/usr/bin/env python3
"""
Analyze the relationship between retrieval coverage and accuracy by step.

For each model and step, calculate:
1. Accuracy when all hops are covered vs when there are coverage gaps
2. Whether better performance at early steps correlates with better coverage
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

from config import get_iterative_model_entries


def load_jsonl(path: Path) -> List[dict]:
    """Load JSONL file."""
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
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


def analyze_coverage_by_step(
    iterative_path: Path,
    coverage_gap_path: Path
) -> Dict:
    """
    Analyze accuracy and coverage gaps by step.
    
    Returns:
        {
            'by_step': {
                step: {
                    'with_gap': {'correct': count, 'total': count},
                    'without_gap': {'correct': count, 'total': count}
                }
            },
            'questions_by_step': {
                step: {
                    'with_gap': [questions],
                    'without_gap': [questions]
                }
            }
        }
    """
    # Load iterative RAG results
    iterative_records = load_jsonl(iterative_path)
    iterative_by_question = {}
    
    for record in iterative_records:
        question = extract_question(record)
        if not question:
            continue
        
        is_correct = bool(record.get("is_correct", False))
        max_step = extract_max_source_step(record)
        if max_step is None:
            max_step = 1
        
        iterative_by_question[question] = {
            'is_correct': is_correct,
            'max_step': max_step
        }
    
    # Load coverage gap judgments
    coverage_records = load_jsonl(coverage_gap_path)
    
    # Analyze by step
    by_step = defaultdict(lambda: {
        'with_gap': {'correct': 0, 'total': 0},
        'without_gap': {'correct': 0, 'total': 0}
    })
    
    questions_by_step = defaultdict(lambda: {
        'with_gap': [],
        'without_gap': []
    })
    
    for record in coverage_records:
        question = record.get("question", "").strip()
        if not question or question not in iterative_by_question:
            continue
        
        parsed = record.get("parsed_judgment", {})
        coverage_gap = parsed.get("retrieval_coverage_gap", {})
        has_gap = coverage_gap.get("has_gap", False)
        
        iter_data = iterative_by_question[question]
        is_correct = iter_data['is_correct']
        max_step = iter_data['max_step']
        
        gap_key = 'with_gap' if has_gap else 'without_gap'
        
        by_step[max_step][gap_key]['total'] += 1
        if is_correct:
            by_step[max_step][gap_key]['correct'] += 1
        
        questions_by_step[max_step][gap_key].append({
            'question': question,
            'is_correct': is_correct,
            'has_gap': has_gap
        })
    
    return {
        'by_step': dict(by_step),
        'questions_by_step': dict(questions_by_step)
    }


def plot_coverage_gap_impact(
    model_data: Dict[str, Dict],
    output_path: Path
) -> None:
    """
    Plot accuracy by step, comparing questions with vs without coverage gaps.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Sort models by average step 1 accuracy
    model_step1_acc = []
    for model, data in model_data.items():
        by_step = data['by_step']
        if 1 in by_step:
            total_correct = by_step[1]['with_gap']['correct'] + by_step[1]['without_gap']['correct']
            total_questions = by_step[1]['with_gap']['total'] + by_step[1]['without_gap']['total']
            acc = 100 * total_correct / total_questions if total_questions > 0 else 0
            model_step1_acc.append((model, acc))
    
    model_step1_acc.sort(key=lambda x: x[1], reverse=True)
    sorted_models = [m for m, _ in model_step1_acc]
    
    # Plot 1: Accuracy with coverage gaps
    ax = axes[0]
    for model in sorted_models[:6]:  # Top 6 models
        data = model_data[model]
        by_step = data['by_step']
        
        steps = sorted([s for s in by_step.keys() if by_step[s]['with_gap']['total'] > 0])
        accuracies = []
        
        for step in steps:
            correct = by_step[step]['with_gap']['correct']
            total = by_step[step]['with_gap']['total']
            acc = 100 * correct / total if total > 0 else 0
            accuracies.append(acc)
        
        if steps and accuracies:
            ax.plot(steps, accuracies, marker='o', linewidth=2, markersize=6, label=model, alpha=0.8)
    
    ax.set_xlabel('Retrieval Step', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Accuracy by Step - Questions WITH Coverage Gaps\n(Top 6 models)', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    # Plot 2: Accuracy without coverage gaps
    ax = axes[1]
    for model in sorted_models[:6]:
        data = model_data[model]
        by_step = data['by_step']
        
        steps = sorted([s for s in by_step.keys() if by_step[s]['without_gap']['total'] > 0])
        accuracies = []
        
        for step in steps:
            correct = by_step[step]['without_gap']['correct']
            total = by_step[step]['without_gap']['total']
            acc = 100 * correct / total if total > 0 else 0
            accuracies.append(acc)
        
        if steps and accuracies:
            ax.plot(steps, accuracies, marker='s', linewidth=2, markersize=6, label=model, alpha=0.8)
    
    ax.set_xlabel('Retrieval Step', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Accuracy by Step - Questions WITHOUT Coverage Gaps\n(Top 6 models)', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    # Plot 3: Coverage gap percentage by step
    ax = axes[2]
    for model in sorted_models[:6]:
        data = model_data[model]
        by_step = data['by_step']
        
        steps = sorted(by_step.keys())
        gap_percentages = []
        
        for step in steps:
            with_gap = by_step[step]['with_gap']['total']
            without_gap = by_step[step]['without_gap']['total']
            total = with_gap + without_gap
            gap_pct = 100 * with_gap / total if total > 0 else 0
            gap_percentages.append(gap_pct)
        
        if steps and gap_percentages:
            ax.plot(steps, gap_percentages, marker='D', linewidth=2, markersize=6, label=model, alpha=0.8)
    
    ax.set_xlabel('Retrieval Step', fontsize=11, fontweight='bold')
    ax.set_ylabel('Coverage Gap Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Percentage of Questions with Coverage Gaps by Step\n(Top 6 models)', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    # Plot 4: Accuracy difference (with gap vs without gap) by step
    ax = axes[3]
    for model in sorted_models[:6]:
        data = model_data[model]
        by_step = data['by_step']
        
        steps = sorted(by_step.keys())
        diff_values = []
        
        for step in steps:
            with_gap_correct = by_step[step]['with_gap']['correct']
            with_gap_total = by_step[step]['with_gap']['total']
            without_gap_correct = by_step[step]['without_gap']['correct']
            without_gap_total = by_step[step]['without_gap']['total']
            
            acc_with = 100 * with_gap_correct / with_gap_total if with_gap_total > 0 else 0
            acc_without = 100 * without_gap_correct / without_gap_total if without_gap_total > 0 else 0
            
            diff = acc_without - acc_with  # Positive = better without gap
            diff_values.append(diff)
        
        if steps and diff_values:
            ax.plot(steps, diff_values, marker='^', linewidth=2, markersize=6, label=model, alpha=0.8)
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='No difference')
    ax.set_xlabel('Retrieval Step', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy Difference (%)', fontsize=11, fontweight='bold')
    ax.set_title('Accuracy Boost from Complete Coverage\n(Without Gap - With Gap, Top 6 models)', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved coverage vs accuracy analysis to {output_path}")


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    
    # Output directory
    output_dir = base / "plots"
    output_dir.mkdir(exist_ok=True)
    
    # Coverage gap directory
    coverage_dir = base / "rag_analysis" / "output"
    
    # Mapping from iterative file to coverage gap file
    coverage_mapping = {}
    for coverage_file in coverage_dir.glob("*coverage_gap_judgments.jsonl"):
        # Extract model identifier
        # Format: 2_responses_bedrock_us.meta.llama3-3-70b-instruct-v1:0_reverified_coverage_gap_judgments.jsonl
        name = coverage_file.name
        if name.startswith("2_"):
            name = name[2:]  # Remove "2_" prefix
        if name.endswith("_coverage_gap_judgments.jsonl"):
            name = name[:-len("_coverage_gap_judgments.jsonl")]
        
        coverage_mapping[name + ".jsonl"] = coverage_file
    
    print(f"Found {len(coverage_mapping)} coverage gap files\n")
    
    model_data = {}
    
    print("Analyzing coverage gaps and accuracy by step for each model...")
    
    for iterative_path, display_name in get_iterative_model_entries(existing_only=True):
        if not iterative_path.exists():
            continue
        
        # Find corresponding coverage gap file
        coverage_path = None
        
        # Try exact match
        if iterative_path.name in coverage_mapping:
            coverage_path = coverage_mapping[iterative_path.name]
        else:
            # Try without _reverified
            alt_name = iterative_path.name.replace("_reverified", "")
            if alt_name in coverage_mapping:
                coverage_path = coverage_mapping[alt_name]
        
        if not coverage_path or not coverage_path.exists():
            print(f"  ⚠ No coverage gap file for {display_name}")
            continue
        
        # Analyze this model
        analysis = analyze_coverage_by_step(iterative_path, coverage_path)
        model_data[display_name] = analysis
        
        # Print summary
        by_step = analysis['by_step']
        print(f"\n{display_name}:")
        for step in sorted(by_step.keys()):
            with_gap = by_step[step]['with_gap']
            without_gap = by_step[step]['without_gap']
            
            acc_with = 100 * with_gap['correct'] / with_gap['total'] if with_gap['total'] > 0 else 0
            acc_without = 100 * without_gap['correct'] / without_gap['total'] if without_gap['total'] > 0 else 0
            
            print(f"  Step {step}: With gap={acc_with:.1f}% ({with_gap['total']}q), "
                  f"Without gap={acc_without:.1f}% ({without_gap['total']}q)")
    
    if not model_data:
        print("\nNo model data found!")
        return
    
    # Generate plot
    print("\nGenerating coverage vs accuracy analysis plot...")
    output_path = output_dir / "coverage_gap_vs_accuracy_by_step.png"
    plot_coverage_gap_impact(model_data, output_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
