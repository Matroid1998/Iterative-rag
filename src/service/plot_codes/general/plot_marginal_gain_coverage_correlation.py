#!/usr/bin/env python3
"""
Correlate marginal gain patterns with retrieval coverage completeness.

This plot ties with marginal_gain_per_step_multihop.png to validate whether:
1. High step 1 accuracy is due to complete coverage
2. Later step drops are due to coverage gaps
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple
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


def load_no_context_wrong_questions(no_context_path: Path) -> set:
    """Load questions that were answered incorrectly in the no-context scenario."""
    records = load_jsonl(no_context_path)
    wrong_questions = set()
    
    for record in records:
        question = extract_question(record)
        if not question:
            continue
        
        is_correct = bool(record.get("is_correct", False))
        if not is_correct:
            wrong_questions.add(question)
    
    return wrong_questions


def load_hard_questions(hard_questions_path: Path) -> set:
    """
    Load hard questions (answered incorrectly by 9, 10, or 11 models).
    
    Returns a set of question strings.
    """
    if not hard_questions_path.exists():
        return set()
    
    with hard_questions_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    hard_questions = set()
    # Categories 9, 10, 11 are hard questions
    for category in ["9", "10", "11"]:
        if category in data:
            for item in data[category]:
                question = item.get("question", "").strip()
                if question:
                    hard_questions.add(question)
    
    return hard_questions


def load_gold_context_wrong_questions(gold_context_path: Path) -> set:
    """
    Load questions that were answered incorrectly in gold context.
    
    Returns a set of question strings.
    """
    # Load gold context results
    gold_records = load_jsonl(gold_context_path)
    gold_wrong = set()
    
    for record in gold_records:
        question = extract_question(record)
        if not question:
            continue
        
        is_correct = bool(record.get("is_correct", False))
        if not is_correct:
            gold_wrong.add(question)
    
    return gold_wrong


def analyze_coverage_and_accuracy_by_step(
    iterative_path: Path,
    coverage_gap_path: Path,
    qa_hops: Dict[str, int],
    filter_questions: set = None,
    exclude_single_hop: bool = False
) -> Dict[int, Dict]:
    """
    For each step, calculate:
    1. Overall accuracy
    2. Percentage with complete coverage
    3. Accuracy when coverage is complete
    4. Accuracy when coverage has gaps
    
    Returns:
        {
            step: {
                'total_accuracy': float,
                'coverage_complete_pct': float,
                'accuracy_complete': float,
                'accuracy_with_gaps': float,
                'total_questions': int
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
        
        # Filter: only include questions in the filter set if provided
        if filter_questions is not None and question not in filter_questions:
            continue
        
        # Filter: exclude single-hop if requested
        if exclude_single_hop:
            hop_count = record.get("number_of_hops")
            if not isinstance(hop_count, int) or hop_count <= 0:
                hop_count = qa_hops.get(question)
            if hop_count == 1:
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
        'total_correct': 0,
        'total_questions': 0,
        'complete_coverage_count': 0,
        'complete_correct': 0,
        'gap_coverage_count': 0,
        'gap_correct': 0
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
        
        by_step[max_step]['total_questions'] += 1
        if is_correct:
            by_step[max_step]['total_correct'] += 1
        
        if has_gap:
            by_step[max_step]['gap_coverage_count'] += 1
            if is_correct:
                by_step[max_step]['gap_correct'] += 1
        else:
            by_step[max_step]['complete_coverage_count'] += 1
            if is_correct:
                by_step[max_step]['complete_correct'] += 1
    
    # Calculate percentages and accuracy
    result = {}
    for step, data in by_step.items():
        total = data['total_questions']
        if total == 0:
            continue
        
        result[step] = {
            'total_accuracy': 100 * data['total_correct'] / total,
            'coverage_complete_pct': 100 * data['complete_coverage_count'] / total,
            'accuracy_complete': 100 * data['complete_correct'] / data['complete_coverage_count'] if data['complete_coverage_count'] > 0 else 0,
            'accuracy_with_gaps': 100 * data['gap_correct'] / data['gap_coverage_count'] if data['gap_coverage_count'] > 0 else 0,
            'total_questions': total,
            'complete_count': data['complete_coverage_count'],
            'gap_count': data['gap_coverage_count']
        }
    
    return result


def plot_marginal_gain_coverage_correlation(
    model_data: Dict[str, Dict[int, Dict]],
    output_path: Path,
    version_title: str = "Multi-hop Questions (2+ hops, wrong in no-context)"
) -> None:
    """
    Create a multi-panel plot showing:
    1. Accuracy by step (overall)
    2. Coverage completeness percentage by step
    3. Accuracy gap (complete vs incomplete coverage)
    """
    # Sort models by step 1 accuracy
    model_step1_acc = []
    for model, steps_data in model_data.items():
        if 1 in steps_data:
            acc = steps_data[1]['total_accuracy']
            model_step1_acc.append((model, acc))
    
    model_step1_acc.sort(key=lambda x: x[1], reverse=True)
    sorted_models = [m for m, _ in model_step1_acc]
    
    # Create figure with 3 rows
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(sorted_models)))
    
    # Plot 1: Cumulative Accuracy by Step (Multi-hop only)
    ax = axes[0]
    for idx, model in enumerate(sorted_models):
        steps_data = model_data[model]
        steps = sorted(steps_data.keys())
        accuracies = [steps_data[s]['total_accuracy'] for s in steps]
        
        ax.plot(steps, accuracies, marker='o', linewidth=2.5, markersize=8, 
                label=model, color=colors[idx], alpha=0.9)
    
    ax.set_xlabel('Retrieval Step', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Cumulative Accuracy by Step - {version_title}\n' +
                 'Does high step 1 or later drops correlate with coverage?',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)
    
    # Plot 2: Coverage Completeness Percentage by Step
    ax = axes[1]
    for idx, model in enumerate(sorted_models):
        steps_data = model_data[model]
        steps = sorted(steps_data.keys())
        coverage_complete_pct = [steps_data[s]['coverage_complete_pct'] for s in steps]
        
        ax.plot(steps, coverage_complete_pct, marker='s', linewidth=2.5, markersize=8,
                label=model, color=colors[idx], alpha=0.9)
    
    ax.set_xlabel('Retrieval Step', fontsize=13, fontweight='bold')
    ax.set_ylabel('Complete Coverage (%)', fontsize=13, fontweight='bold')
    ax.set_title('Percentage of Questions with Complete Retrieval Coverage by Step\n' +
                 'Higher % = retrieved all needed information',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)
    
    # Plot 3: Accuracy Difference (Complete Coverage - With Gaps)
    ax = axes[2]
    for idx, model in enumerate(sorted_models):
        steps_data = model_data[model]
        steps = sorted(steps_data.keys())
        accuracy_diffs = []
        
        for s in steps:
            acc_complete = steps_data[s]['accuracy_complete']
            acc_gaps = steps_data[s]['accuracy_with_gaps']
            diff = acc_complete - acc_gaps
            accuracy_diffs.append(diff)
        
        ax.plot(steps, accuracy_diffs, marker='D', linewidth=2.5, markersize=8,
                label=model, color=colors[idx], alpha=0.9)
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='No difference')
    ax.set_xlabel('Retrieval Step', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy Boost from Complete Coverage (%)', fontsize=13, fontweight='bold')
    ax.set_title('Impact of Complete Coverage on Accuracy\n' +
                 '(Accuracy with complete coverage - Accuracy with gaps)\n' +
                 'Positive values = complete coverage helps significantly',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved marginal gain vs coverage correlation plot to {output_path}")


def main() -> None:
    base = Path(__file__).resolve().parents[4]
    
    # Output directory
    output_dir = base / "data" / "plots" / "general"
    output_dir.mkdir(exist_ok=True)
    
    # Load QA hop data
    qa_path = base.parent / "data" / "corpus" / "chemrxiv_qa.json"
    print("Loading question hop data...")
    qa_hops = load_qa_hops(qa_path)
    print(f"Loaded hop data for {len(qa_hops)} questions\n")
    
    # Load hard questions
    hard_questions_path = base / "src" / "results" / "unanswered_questions" / "hard_question_categories.json"
    print("Loading hard questions (9, 10, 11 models wrong)...")
    hard_questions = load_hard_questions(hard_questions_path)
    print(f"Loaded {len(hard_questions)} hard questions\n")
    
    # Coverage gap directory
    coverage_dir = base  / "data" / "results" / "failure_modes"
    
    # Mapping from iterative file to coverage gap file
    coverage_mapping = {}
    for coverage_file in coverage_dir.glob("*coverage_gap_judgments.jsonl"):
        name = coverage_file.name
        if name.startswith("2_"):
            name = name[2:]
        if name.endswith("_coverage_gap_judgments.jsonl"):
            name = name[:-len("_coverage_gap_judgments.jsonl")]
        
        coverage_mapping[name + ".jsonl"] = coverage_file
    
    print(f"Found {len(coverage_mapping)} coverage gap files\n")
    
    # No-context directory
    no_context_dir = base / "response-jsonl-without-context"
    
    # Gold context directory
    gold_context_dir = base / "response-jsonl-with-context"
    
    # Special mappings for no-context files
    no_context_mapping = {
        "responses_openrouter_google__gemini-2.5-pro_reverified.jsonl": "responses_openrouter_google__gemini-2.5-pro-reasoning.jsonl",
        "responses_openrouter_x-ai__grok-4-fast_reverified.jsonl": "responses_openrouter_x-ai__grok-4-fast-reasoning.jsonl",
        "responses_openrouter_z-ai__glm-4.6_reverified.jsonl": "responses_openrouter_z-ai__glm-4.6-reasoning_reverified.jsonl",
    }
    
    # Special mappings for gold context files
    gold_context_mapping = {
        "responses_openrouter_google__gemini-2.5-pro_reverified.jsonl": "responses_openrouter_google__gemini-2.5-pro-reasoning_reverified.jsonl",
        "responses_openrouter_x-ai__grok-4-fast_reverified.jsonl": "responses_openrouter_x-ai__grok-4-fast-reasoning_reverified.jsonl",
        "responses_openrouter_z-ai__glm-4.6_reverified.jsonl": "responses_openrouter_z-ai__glm-4.6-reasoning_reverified.jsonl",
    }
    
    # Version configurations
    versions = [
        {
            'name': 'multihop_no_context_wrong',
            'title': 'Multi-hop Questions (2+ hops, wrong in no-context)',
            'output': 'marginal_gain_coverage_correlation_multihop.png',
            'exclude_single_hop': True,
            'use_no_context_filter': True,
            'use_gold_context_filter': False,
            'use_hard_questions': False
        },
        {
            'name': 'all_no_context_wrong',
            'title': 'All Questions (wrong in no-context)',
            'output': 'marginal_gain_coverage_correlation.png',
            'exclude_single_hop': False,
            'use_no_context_filter': True,
            'use_gold_context_filter': False,
            'use_hard_questions': False
        },
        {
            'name': 'hard_questions',
            'title': 'Hard Questions (9-11 models answered incorrectly)',
            'output': 'marginal_gain_coverage_correlation_hard.png',
            'exclude_single_hop': False,
            'use_no_context_filter': False,
            'use_gold_context_filter': False,
            'use_hard_questions': True
        },
        {
            'name': 'gold_context_wrong',
            'title': 'Questions answered incorrectly in gold context',
            'output': 'marginal_gain_coverage_correlation_gold_wrong.png',
            'exclude_single_hop': False,
            'use_no_context_filter': False,
            'use_gold_context_filter': True,
            'use_hard_questions': False
        }
    ]
    
    for version_config in versions:
        print("\n" + "="*80)
        print(f"PROCESSING: {version_config['title']}")
        print("="*80 + "\n")
        
        model_data = {}
        
        for iterative_path, display_name in get_iterative_model_entries(existing_only=True):
            if not iterative_path.exists():
                continue
            
            # Find corresponding coverage gap file
            coverage_path = None
            
            if iterative_path.name in coverage_mapping:
                coverage_path = coverage_mapping[iterative_path.name]
            else:
                alt_name = iterative_path.name.replace("_reverified", "")
                if alt_name in coverage_mapping:
                    coverage_path = coverage_mapping[alt_name]
            
            if not coverage_path or not coverage_path.exists():
                print(f"  ⚠ No coverage gap file for {display_name}")
                continue
            
            # Determine filter questions
            filter_questions = None
            
            if version_config['use_hard_questions']:
                filter_questions = hard_questions
            
            elif version_config['use_no_context_filter']:
                # Find corresponding no-context file
                if iterative_path.name in no_context_mapping:
                    no_context_filename = no_context_mapping[iterative_path.name]
                    no_context_path = no_context_dir / no_context_filename
                else:
                    candidates = [
                        iterative_path.name,
                        iterative_path.name.replace("_reverified", ""),
                        iterative_path.name if "_reverified" in iterative_path.name else iterative_path.name.replace(".jsonl", "_reverified.jsonl"),
                    ]
                    no_context_path = None
                    for candidate in candidates:
                        candidate_path = no_context_dir / candidate
                        if candidate_path.exists():
                            no_context_path = candidate_path
                            break
                
                if not no_context_path or not no_context_path.exists():
                    print(f"  ⚠ No no-context file found for {display_name}, skipping")
                    continue
                
                filter_questions = load_no_context_wrong_questions(no_context_path)
            
            elif version_config['use_gold_context_filter']:
                # Find corresponding gold context file
                if iterative_path.name in gold_context_mapping:
                    gold_context_filename = gold_context_mapping[iterative_path.name]
                    gold_context_path = gold_context_dir / gold_context_filename
                else:
                    candidates = [
                        iterative_path.name,
                        iterative_path.name.replace("_reverified", ""),
                        iterative_path.name if "_reverified" in iterative_path.name else iterative_path.name.replace(".jsonl", "_reverified.jsonl"),
                    ]
                    gold_context_path = None
                    for candidate in candidates:
                        candidate_path = gold_context_dir / candidate
                        if candidate_path.exists():
                            gold_context_path = candidate_path
                            break
                
                if not gold_context_path or not gold_context_path.exists():
                    print(f"  ⚠ No gold context file found for {display_name}, skipping")
                    continue
                
                filter_questions = load_gold_context_wrong_questions(gold_context_path)
            
            # Analyze this model
            # Analyze this model
            steps_data = analyze_coverage_and_accuracy_by_step(
                iterative_path, coverage_path, qa_hops, 
                filter_questions, 
                exclude_single_hop=version_config['exclude_single_hop']
            )
            
            if not steps_data:
                continue
            
            model_data[display_name] = steps_data
            
            # Print summary
            print(f"\n{display_name}:")
            print(f"  {'Step':<6} {'Accuracy':<10} {'Complete Cov%':<15} {'Acc(Complete)':<15} {'Acc(Gaps)':<15} {'Δ Accuracy':<12}")
            print(f"  {'-'*6} {'-'*10} {'-'*15} {'-'*15} {'-'*15} {'-'*12}")
            
            for step in sorted(steps_data.keys()):
                data = steps_data[step]
                acc = data['total_accuracy']
                cov_pct = data['coverage_complete_pct']
                acc_complete = data['accuracy_complete']
                acc_gaps = data['accuracy_with_gaps']
                delta = acc_complete - acc_gaps
                
                print(f"  {step:<6} {acc:>6.1f}%    {cov_pct:>6.1f}%         "
                      f"{acc_complete:>6.1f}%         {acc_gaps:>6.1f}%         {delta:>+6.1f}%")
                print(f"         (n={data['total_questions']}: {data['complete_count']} complete, {data['gap_count']} gaps)")
        
        if not model_data:
            print(f"\n⚠ No model data found for {version_config['name']}!")
            continue
        
        # Generate plot for this version
        print("\n" + "="*80)
        print(f"Generating plot for {version_config['name']}...")
        output_path = output_dir / version_config['output']
        plot_marginal_gain_coverage_correlation(model_data, output_path, version_config['title'])
    
    print("\n" + "="*80)
    print("✅ All versions complete!")
    print("\nGenerated plots:")
    for version_config in versions:
        print(f"  - {version_config['output']}")


if __name__ == "__main__":
    main()
