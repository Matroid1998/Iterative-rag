#!/usr/bin/env python3
"""
Marginal Gain per Step Curve

For each model, shows:
- Δ accuracy from step k-1 → k (marginal gain)
- Cumulative accuracy by step

Shows "early gain, late taper" and helps justify a default budget
(e.g., stop after step 3 if marginal gain < τ).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
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


def load_no_context_wrong_questions(no_context_path: Path) -> set:
    """Load questions that were answered incorrectly in the no-context scenario."""
    records = load_records(no_context_path)
    wrong_questions = set()
    
    for record in records:
        question = extract_question(record)
        if not question:
            continue
        
        is_correct = bool(record.get("is_correct", False))
        if not is_correct:
            wrong_questions.add(question)
    
    return wrong_questions


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
    gold_records = load_records(gold_context_path)
    gold_wrong = set()
    
    for record in gold_records:
        question = extract_question(record)
        if not question:
            continue
        
        is_correct = bool(record.get("is_correct", False))
        if not is_correct:
            gold_wrong.add(question)
    
    return gold_wrong


def calculate_accuracy_by_step(
    iterative_path: Path, 
    no_context_wrong_questions: set = None,
    qa_hops: Dict[str, int] = None,
    exclude_single_hop: bool = False,
    hard_questions_only: set = None,
    recovery_questions_only: set = None
) -> Dict[int, float]:
    """
    Calculate accuracy for questions answered at each step.
    
    If no_context_wrong_questions is provided, only count questions that were 
    wrong in the no-context scenario.
    
    If exclude_single_hop is True, exclude 1-hop questions.
    
    If hard_questions_only is provided, only count questions in that set.
    
    If recovery_questions_only is provided, only count questions in that set
    (questions wrong in gold context but correct in iterative RAG).
    
    Returns:
        {step: accuracy} where accuracy is % of questions answered correctly by that step
    """
    records = load_records(iterative_path)
    
    # Track correctness by the step at which each question was answered
    questions_by_step: Dict[int, List[bool]] = defaultdict(list)
    
    for record in records:
        question = extract_question(record)
        if not question:
            continue
        
        # Filter: only include recovery questions if specified
        if recovery_questions_only is not None and question not in recovery_questions_only:
            continue
        
        # Filter: only include hard questions if specified
        if hard_questions_only is not None and question not in hard_questions_only:
            continue
        
        # Filter: only include questions that were wrong in no-context
        if no_context_wrong_questions is not None and question not in no_context_wrong_questions:
            continue
        
        # Filter: exclude single-hop questions if requested
        if exclude_single_hop and qa_hops:
            hop_count = record.get("number_of_hops")
            if not isinstance(hop_count, int) or hop_count <= 0:
                hop_count = qa_hops.get(question)
            if hop_count == 1:
                continue
        
        is_correct = bool(record.get("is_correct", False))
        max_step = extract_max_source_step(record)
        
        if max_step is None:
            max_step = 1
        
        questions_by_step[max_step].append(is_correct)
    
    # Calculate cumulative accuracy at each step
    accuracy_by_step: Dict[int, float] = {}
    
    all_questions: List[bool] = []
    for step in sorted(questions_by_step.keys()):
        all_questions.extend(questions_by_step[step])
        correct_count = sum(all_questions)
        total_count = len(all_questions)
        accuracy_by_step[step] = 100 * correct_count / total_count if total_count > 0 else 0
    
    return accuracy_by_step


def plot_marginal_gain_curves(
    model_data: Dict[str, Dict[int, float]],
    output_path: Path,
    multihop_only: bool = False
) -> None:
    """
    Create two plots:
    1. Cumulative accuracy by step
    2. Marginal gain (Δ accuracy) from step k-1 to k
    
    model_data: {model_name: {step: cumulative_accuracy}}
    multihop_only: If True, adjust titles for multi-hop questions only
    """
    if not model_data:
        print("No data to plot!")
        return
    
    # Determine max step across all models
    max_step = max(max(steps.keys()) for steps in model_data.values())
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Color map for models
    colors = plt.cm.tab20(np.linspace(0, 1, len(model_data)))
    
    # Calculate marginal gains
    marginal_data: Dict[str, Dict[int, float]] = {}
    
    for (model, steps), color in zip(model_data.items(), colors):
        sorted_steps = sorted(steps.keys())
        
        # Plot 1: Cumulative accuracy
        accuracies = [steps[s] for s in sorted_steps]
        ax1.plot(sorted_steps, accuracies, 'o-', label=model, color=color,
                linewidth=2.5, markersize=8, alpha=0.8)
        
        # Calculate marginal gains
        marginal_gains = {}
        for i, step in enumerate(sorted_steps):
            if i == 0:
                # First step: marginal gain is the accuracy itself
                marginal_gains[step] = steps[step]
            else:
                prev_step = sorted_steps[i-1]
                marginal_gains[step] = steps[step] - steps[prev_step]
        
        marginal_data[model] = marginal_gains
        
        # Plot 2: Marginal gains
        mg_steps = sorted(marginal_gains.keys())
        mg_values = [marginal_gains[s] for s in mg_steps]
        ax2.plot(mg_steps, mg_values, 'o-', label=model, color=color,
                linewidth=2.5, markersize=8, alpha=0.8)
    
    # Customize plot 1: Cumulative Accuracy
    ax1.set_xlabel('Retrieval Step', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Cumulative Accuracy (%)', fontsize=13, fontweight='bold')
    if multihop_only:
        ax1.set_title('Cumulative Accuracy by Retrieval Step\n' + 
                      'Multi-hop questions only (2+ hops), wrong without context',
                      fontsize=15, fontweight='bold', pad=20)
    else:
        ax1.set_title('Cumulative Accuracy by Retrieval Step\n' + 
                      'For questions answered incorrectly without context',
                      fontsize=15, fontweight='bold', pad=20)
    ax1.legend(loc='best', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0.5, max_step + 0.5)
    ax1.set_xticks(range(1, max_step + 1))
    
    # Customize plot 2: Marginal Gain
    ax2.set_xlabel('Retrieval Step', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Marginal Gain in Accuracy (Δ%)', fontsize=13, fontweight='bold')
    if multihop_only:
        ax2.set_title('Marginal Gain per Step (Δ Accuracy from Step k-1 → k)\n' + 
                      'Multi-hop questions only (2+ hops), wrong without context',
                      fontsize=15, fontweight='bold', pad=20)
    else:
        ax2.set_title('Marginal Gain per Step (Δ Accuracy from Step k-1 → k)\n' + 
                      'For questions that needed retrieval (wrong without context)',
                      fontsize=15, fontweight='bold', pad=20)
    ax2.legend(loc='best', fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlim(0.5, max_step + 0.5)
    ax2.set_xticks(range(1, max_step + 1))
    
    # Add threshold lines for marginal gain (e.g., τ = 1%, 2%, 5%)
    for threshold, label in [(5, 'τ = 5%'), (2, 'τ = 2%'), (1, 'τ = 1%')]:
        ax2.axhline(y=threshold, color='red', linestyle='--', 
                   linewidth=1.5, alpha=0.4, label=label)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved marginal gain curves to {output_path}")


def plot_marginal_gain_curves_hard(
    model_data: Dict[str, Dict[int, float]],
    output_path: Path
) -> None:
    """
    Create two plots for hard questions only:
    1. Cumulative accuracy by step
    2. Marginal gain (Δ accuracy) from step k-1 to k
    
    model_data: {model_name: {step: cumulative_accuracy}}
    """
    if not model_data:
        print("No data to plot!")
        return
    
    # Determine max step across all models
    max_step = max(max(steps.keys()) for steps in model_data.values())
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Color map for models
    colors = plt.cm.tab20(np.linspace(0, 1, len(model_data)))
    
    # Calculate marginal gains
    marginal_data: Dict[str, Dict[int, float]] = {}
    
    for (model, steps), color in zip(model_data.items(), colors):
        sorted_steps = sorted(steps.keys())
        
        # Plot 1: Cumulative accuracy
        accuracies = [steps[s] for s in sorted_steps]
        ax1.plot(sorted_steps, accuracies, 'o-', label=model, color=color,
                linewidth=2.5, markersize=8, alpha=0.8)
        
        # Calculate marginal gains
        marginal_gains = {}
        for i, step in enumerate(sorted_steps):
            if i == 0:
                # First step: marginal gain is the accuracy itself
                marginal_gains[step] = steps[step]
            else:
                prev_step = sorted_steps[i-1]
                marginal_gains[step] = steps[step] - steps[prev_step]
        
        marginal_data[model] = marginal_gains
        
        # Plot 2: Marginal gains
        mg_steps = sorted(marginal_gains.keys())
        mg_values = [marginal_gains[s] for s in mg_steps]
        ax2.plot(mg_steps, mg_values, 'o-', label=model, color=color,
                linewidth=2.5, markersize=8, alpha=0.8)
    
    # Customize plot 1: Cumulative Accuracy
    ax1.set_xlabel('Retrieval Step', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Cumulative Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Cumulative Accuracy by Retrieval Step\n' + 
                  'Hard questions only (9-11 models answered incorrectly)',
                  fontsize=15, fontweight='bold', pad=20)
    ax1.legend(loc='best', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0.5, max_step + 0.5)
    ax1.set_xticks(range(1, max_step + 1))
    
    # Customize plot 2: Marginal Gain
    ax2.set_xlabel('Retrieval Step', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Marginal Gain in Accuracy (Δ%)', fontsize=13, fontweight='bold')
    ax2.set_title('Marginal Gain per Step (Δ Accuracy from Step k-1 → k)\n' + 
                  'Hard questions only (9-11 models answered incorrectly)',
                  fontsize=15, fontweight='bold', pad=20)
    ax2.legend(loc='best', fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlim(0.5, max_step + 0.5)
    ax2.set_xticks(range(1, max_step + 1))
    
    # Add threshold lines for marginal gain
    for threshold, label in [(5, 'τ = 5%'), (2, 'τ = 2%'), (1, 'τ = 1%')]:
        ax2.axhline(y=threshold, color='red', linestyle='--', 
                   linewidth=1.5, alpha=0.4, label=label)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved marginal gain curves (hard questions) to {output_path}")


def plot_average_marginal_gain_hard(
    model_data: Dict[str, Dict[int, float]],
    output_path: Path
) -> None:
    """
    Create a bar chart showing average marginal gain per step across all models for hard questions.
    """
    # Calculate marginal gains for all models
    all_marginal_gains: Dict[int, List[float]] = defaultdict(list)
    
    for model, steps in model_data.items():
        sorted_steps = sorted(steps.keys())
        
        for i, step in enumerate(sorted_steps):
            if i == 0:
                marginal_gain = steps[step]
            else:
                prev_step = sorted_steps[i-1]
                marginal_gain = steps[step] - steps[prev_step]
            
            all_marginal_gains[step].append(marginal_gain)
    
    # Calculate average and std dev per step
    steps = sorted(all_marginal_gains.keys())
    avg_gains = [np.mean(all_marginal_gains[s]) for s in steps]
    std_gains = [np.std(all_marginal_gains[s]) for s in steps]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(steps))
    bars = ax.bar(x, avg_gains, width=0.6, color='steelblue', 
                  edgecolor='black', linewidth=1.5, alpha=0.8,
                  yerr=std_gains, capsize=5)
    
    # Add value labels on bars
    for i, (bar, gain, std) in enumerate(zip(bars, avg_gains, std_gains)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{gain:.2f}%\n±{std:.2f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add threshold lines
    for threshold, label, color in [(5, 'τ = 5%', 'red'), 
                                     (2, 'τ = 2%', 'orange'), 
                                     (1, 'τ = 1%', 'yellow')]:
        ax.axhline(y=threshold, color=color, linestyle='--', 
                  linewidth=2, alpha=0.6, label=label)
    
    # Customize plot
    ax.set_xlabel('Retrieval Step', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Marginal Gain (Δ Accuracy %)', fontsize=13, fontweight='bold')
    ax.set_title('Average Marginal Gain per Step Across All Models\n' + 
                 'Hard questions only (9-11 models wrong) - Error bars show standard deviation',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Step {s}' for s in steps])
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved average marginal gain chart (hard questions) to {output_path}")


def plot_average_marginal_gain(
    model_data: Dict[str, Dict[int, float]],
    output_path: Path
) -> None:
    """
    Create a bar chart showing average marginal gain per step across all models.
    """
    # Calculate marginal gains for all models
    all_marginal_gains: Dict[int, List[float]] = defaultdict(list)
    
    for model, steps in model_data.items():
        sorted_steps = sorted(steps.keys())
        
        for i, step in enumerate(sorted_steps):
            if i == 0:
                marginal_gain = steps[step]
            else:
                prev_step = sorted_steps[i-1]
                marginal_gain = steps[step] - steps[prev_step]
            
            all_marginal_gains[step].append(marginal_gain)
    
    # Calculate average and std dev per step
    steps = sorted(all_marginal_gains.keys())
    avg_gains = [np.mean(all_marginal_gains[s]) for s in steps]
    std_gains = [np.std(all_marginal_gains[s]) for s in steps]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(steps))
    bars = ax.bar(x, avg_gains, width=0.6, color='steelblue', 
                  edgecolor='black', linewidth=1.5, alpha=0.8,
                  yerr=std_gains, capsize=5)
    
    # Add value labels on bars
    for i, (bar, gain, std) in enumerate(zip(bars, avg_gains, std_gains)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{gain:.2f}%\n±{std:.2f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add threshold lines
    for threshold, label, color in [(5, 'τ = 5%', 'red'), 
                                     (2, 'τ = 2%', 'orange'), 
                                     (1, 'τ = 1%', 'yellow')]:
        ax.axhline(y=threshold, color=color, linestyle='--', 
                  linewidth=2, alpha=0.6, label=label)
    
    # Customize plot
    ax.set_xlabel('Retrieval Step', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Marginal Gain (Δ Accuracy %)', fontsize=13, fontweight='bold')
    ax.set_title('Average Marginal Gain per Step Across All Models\n' + 
                 'For questions wrong without context (Error bars show standard deviation)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Step {s}' for s in steps])
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved average marginal gain chart to {output_path}")


def plot_marginal_gain_curves_recovery(
    model_data: Dict[str, Dict[int, float]],
    output_path: Path
) -> None:
    """
    Create two plots specifically for questions wrong in gold context:
    1. Cumulative accuracy by step
    2. Marginal gain (Δ accuracy) from step k-1 to k
    
    model_data: {model_name: {step: cumulative_accuracy}}
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(model_data)))
    
    for idx, (model, steps) in enumerate(sorted(model_data.items())):
        sorted_steps = sorted(steps.keys())
        cumulative_acc = [steps[s] for s in sorted_steps]
        
        # Plot 1: Cumulative accuracy
        ax1.plot(sorted_steps, cumulative_acc, marker='o', linewidth=2.5,
                markersize=8, label=model, color=colors[idx], alpha=0.9)
        
        # Plot 2: Marginal gain
        marginal_gains = []
        for i, step in enumerate(sorted_steps):
            if i == 0:
                marginal_gains.append(steps[step])
            else:
                prev_step = sorted_steps[i-1]
                marginal_gains.append(steps[step] - steps[prev_step])
        
        ax2.plot(sorted_steps, marginal_gains, marker='s', linewidth=2.5,
                markersize=8, label=model, color=colors[idx], alpha=0.9)
    
    # Customize Plot 1
    ax1.set_xlabel('Retrieval Step', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Cumulative Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Cumulative Accuracy by Step\nQuestions answered incorrectly in gold context',
                  fontsize=15, fontweight='bold', pad=20)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 105)
    
    # Customize Plot 2
    ax2.set_xlabel('Retrieval Step', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Marginal Gain (Δ Accuracy %)', fontsize=13, fontweight='bold')
    ax2.set_title('Marginal Gain per Step (step k vs. step k-1)\nQuestions answered incorrectly in gold context',
                  fontsize=15, fontweight='bold', pad=20)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved marginal gain curves (gold context wrong) to {output_path}")


def plot_average_marginal_gain_recovery(
    model_data: Dict[str, Dict[int, float]],
    output_path: Path
) -> None:
    """
    Create a bar chart showing average marginal gain per step across all models
    for questions wrong in gold context.
    """
    # Calculate marginal gains for all models
    all_marginal_gains: Dict[int, List[float]] = defaultdict(list)
    
    for model, steps in model_data.items():
        sorted_steps = sorted(steps.keys())
        
        for i, step in enumerate(sorted_steps):
            if i == 0:
                marginal_gain = steps[step]
            else:
                prev_step = sorted_steps[i-1]
                marginal_gain = steps[step] - steps[prev_step]
            
            all_marginal_gains[step].append(marginal_gain)
    
    # Calculate average and std dev per step
    steps = sorted(all_marginal_gains.keys())
    avg_gains = [np.mean(all_marginal_gains[s]) for s in steps]
    std_gains = [np.std(all_marginal_gains[s]) for s in steps]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(steps))
    bars = ax.bar(x, avg_gains, width=0.6, color='steelblue', 
                  edgecolor='black', linewidth=1.5, alpha=0.8,
                  yerr=std_gains, capsize=5)
    
    # Add value labels on bars
    for i, (bar, gain, std) in enumerate(zip(bars, avg_gains, std_gains)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{gain:.2f}%\n±{std:.2f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add threshold lines
    for threshold, label, color in [(5, 'τ = 5%', 'red'), 
                                     (2, 'τ = 2%', 'orange'), 
                                     (1, 'τ = 1%', 'yellow')]:
        ax.axhline(y=threshold, color=color, linestyle='--', 
                  linewidth=2, alpha=0.6, label=label)
    
    # Customize plot
    ax.set_xlabel('Retrieval Step', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Marginal Gain (Δ Accuracy %)', fontsize=13, fontweight='bold')
    ax.set_title('Average Marginal Gain per Step Across All Models\n' + 
                 'Questions answered incorrectly in gold context - Error bars show standard deviation',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Step {s}' for s in steps])
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved average marginal gain chart (gold context wrong) to {output_path}")


def main() -> None:
    base = Path(__file__).resolve().parents[3]
    
    # Output directory
    output_dir = base / "plots"
    output_dir.mkdir(exist_ok=True)
    
    # Load QA hop data
    qa_path = base.parent / "data" / "corpus" / "chemrxiv_qa.json"
    print("Loading question hop data...")
    qa_hops = load_qa_hops(qa_path)
    print(f"Loaded hop data for {len(qa_hops)} questions\n")
    
    # No-context directory
    no_context_dir = base / "response-jsonl-without-context"
    
    # Special mappings for no-context files
    no_context_mapping = {
        "responses_openrouter_google__gemini-2.5-pro_reverified.jsonl": "responses_openrouter_google__gemini-2.5-pro-reasoning.jsonl",
        "responses_openrouter_x-ai__grok-4-fast_reverified.jsonl": "responses_openrouter_x-ai__grok-4-fast-reasoning.jsonl",
        "responses_openrouter_z-ai__glm-4.6_reverified.jsonl": "responses_openrouter_z-ai__glm-4.6-reasoning_reverified.jsonl",
    }
    
    # Load hard questions
    hard_questions_path = base / "results" / "unanswered_questions" / "hard_question_categories.json"
    print("Loading hard questions (9, 10, 11 models wrong)...")
    hard_questions = load_hard_questions(hard_questions_path)
    print(f"Loaded {len(hard_questions)} hard questions\n")
    
    # Gold context directory
    gold_context_dir = base / "response-jsonl-with-context"
    
    # Special mappings for gold context files
    gold_context_mapping = {
        "responses_openrouter_google__gemini-2.5-pro_reverified.jsonl": "responses_openrouter_google__gemini-2.5-pro-reasoning_reverified.jsonl",
        "responses_openrouter_x-ai__grok-4-fast_reverified.jsonl": "responses_openrouter_x-ai__grok-4-fast-reasoning_reverified.jsonl",
        "responses_openrouter_z-ai__glm-4.6_reverified.jsonl": "responses_openrouter_z-ai__glm-4.6-reasoning_reverified.jsonl",
    }
    
    model_data: Dict[str, Dict[int, float]] = {}
    model_data_multihop: Dict[str, Dict[int, float]] = {}
    model_data_hard: Dict[str, Dict[int, float]] = {}
    model_data_recovery: Dict[str, Dict[int, float]] = {}
    
    print("Calculating accuracy by step for each model (filtered by no-context wrong questions)...")
    
    for iterative_path, display_name in get_iterative_model_entries(existing_only=True):
        if not iterative_path.exists():
            continue
        
        # Find corresponding no-context file
        if iterative_path.name in no_context_mapping:
            no_context_filename = no_context_mapping[iterative_path.name]
            no_context_path = no_context_dir / no_context_filename
        else:
            # Try multiple naming patterns
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
            print(f"  Warning: No no-context file found for {display_name}, using all questions")
            no_context_wrong = None
        else:
            # Load questions that were wrong in no-context
            no_context_wrong = load_no_context_wrong_questions(no_context_path)
            print(f"  {display_name}: {len(no_context_wrong)} questions wrong in no-context")
        
        # Version 1: All questions (that were wrong in no-context)
        accuracy_by_step = calculate_accuracy_by_step(iterative_path, no_context_wrong, qa_hops, exclude_single_hop=False)
        
        if not accuracy_by_step:
            print(f"  Warning: No step data for {display_name}")
            continue
        
        model_data[display_name] = accuracy_by_step
        
        # Version 2: Multi-hop questions only (2+ hops, wrong in no-context)
        accuracy_by_step_multihop = calculate_accuracy_by_step(iterative_path, no_context_wrong, qa_hops, exclude_single_hop=True)
        
        if accuracy_by_step_multihop:
            model_data_multihop[display_name] = accuracy_by_step_multihop
        
        # Version 3: Hard questions only (9, 10, 11 models wrong)
        accuracy_by_step_hard = calculate_accuracy_by_step(iterative_path, None, qa_hops, exclude_single_hop=False, hard_questions_only=hard_questions)
        
        if accuracy_by_step_hard:
            model_data_hard[display_name] = accuracy_by_step_hard
        
        # Version 4: Questions wrong in gold context
        # Find corresponding gold context file
        if iterative_path.name in gold_context_mapping:
            gold_context_filename = gold_context_mapping[iterative_path.name]
            gold_context_path = gold_context_dir / gold_context_filename
        else:
            # Try multiple naming patterns
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
        
        if gold_context_path and gold_context_path.exists():
            gold_wrong_questions = load_gold_context_wrong_questions(gold_context_path)
            if gold_wrong_questions:
                accuracy_by_step_gold_wrong = calculate_accuracy_by_step(
                    iterative_path, gold_wrong_questions, qa_hops, 
                    exclude_single_hop=False
                )
                if accuracy_by_step_gold_wrong:
                    model_data_recovery[display_name] = accuracy_by_step_gold_wrong
        
        # Print summary
        sorted_steps = sorted(accuracy_by_step.keys())
        print(f"  {display_name:30s}: ", end="")
        for step in sorted_steps:
            print(f"Step {step}={accuracy_by_step[step]:.1f}%  ", end="")
        print()
    
    if not model_data:
        print("\nNo model data found!")
        return
    
    # Generate plots - Version 1: All questions (wrong in no-context)
    print("\nGenerating marginal gain curves (all questions wrong in no-context)...")
    output_path1 = output_dir / "marginal_gain_per_step.png"
    plot_marginal_gain_curves(model_data, output_path1, multihop_only=False)
    
    print("Generating average marginal gain chart (all questions)...")
    output_path2 = output_dir / "average_marginal_gain_per_step.png"
    plot_average_marginal_gain(model_data, output_path2)
    
    # Generate plots - Version 2: Multi-hop questions only (2+ hops, wrong in no-context)
    if model_data_multihop:
        print("\nGenerating marginal gain curves (multi-hop questions only, excluding 1-hop)...")
        output_path3 = output_dir / "marginal_gain_per_step_multihop.png"
        plot_marginal_gain_curves(model_data_multihop, output_path3, multihop_only=True)
        
        print("Generating average marginal gain chart (multi-hop only)...")
        output_path4 = output_dir / "average_marginal_gain_per_step_multihop.png"
        plot_average_marginal_gain(model_data_multihop, output_path4)
    
    # Generate plots - Version 3: Hard questions only (9, 10, 11 models wrong)
    if model_data_hard:
        print("\nGenerating marginal gain curves (hard questions only)...")
        output_path5 = output_dir / "marginal_gain_per_step_hard.png"
        plot_marginal_gain_curves_hard(model_data_hard, output_path5)
        
        print("Generating average marginal gain chart (hard questions only)...")
        output_path6 = output_dir / "average_marginal_gain_per_step_hard.png"
        plot_average_marginal_gain_hard(model_data_hard, output_path6)
    
    # Generate plots - Version 4: Questions wrong in gold context
    if model_data_recovery:
        print("\nGenerating marginal gain curves (questions wrong in gold context)...")
        output_path7 = output_dir / "marginal_gain_per_step_gold_wrong.png"
        plot_marginal_gain_curves_recovery(model_data_recovery, output_path7)
        
        print("Generating average marginal gain chart (questions wrong in gold context)...")
        output_path8 = output_dir / "average_marginal_gain_per_step_gold_wrong.png"
        plot_average_marginal_gain_recovery(model_data_recovery, output_path8)
    
    # Print detailed statistics
    print("\n" + "="*80)
    print("MARGINAL GAIN STATISTICS (Questions Wrong in No-Context Only)")
    print("="*80)
    
    for model in sorted(model_data.keys()):
        steps = model_data[model]
        sorted_steps = sorted(steps.keys())
        
        print(f"\n{model}:")
        print(f"  {'Step':<6} {'Cumulative Acc':<15} {'Marginal Gain':<15}")
        print(f"  {'-'*6} {'-'*15} {'-'*15}")
        
        for i, step in enumerate(sorted_steps):
            cum_acc = steps[step]
            if i == 0:
                marg_gain = cum_acc
            else:
                prev_step = sorted_steps[i-1]
                marg_gain = steps[step] - steps[prev_step]
            
            print(f"  {step:<6d} {cum_acc:>6.2f}%         {marg_gain:>+6.2f}%")
    
    # Print statistics for multi-hop only version
    if model_data_multihop:
        print("\n" + "="*80)
        print("MARGINAL GAIN STATISTICS - MULTI-HOP ONLY (2+ hops, Wrong in No-Context)")
        print("="*80)
        
        for model in sorted(model_data_multihop.keys()):
            steps = model_data_multihop[model]
            sorted_steps = sorted(steps.keys())
            
            print(f"\n{model}:")
            print(f"  {'Step':<6} {'Cumulative Acc':<15} {'Marginal Gain':<15}")
            print(f"  {'-'*6} {'-'*15} {'-'*15}")
            
            for i, step in enumerate(sorted_steps):
                cum_acc = steps[step]
                if i == 0:
                    marg_gain = cum_acc
                else:
                    prev_step = sorted_steps[i-1]
                    marg_gain = steps[step] - steps[prev_step]
                
                print(f"  {step:<6d} {cum_acc:>6.2f}%         {marg_gain:>+6.2f}%")
    
    # Print statistics for hard questions version
    if model_data_hard:
        print("\n" + "="*80)
        print("MARGINAL GAIN STATISTICS - HARD QUESTIONS ONLY (9-11 Models Wrong)")
        print("="*80)
        
        for model in sorted(model_data_hard.keys()):
            steps = model_data_hard[model]
            sorted_steps = sorted(steps.keys())
            
            print(f"\n{model}:")
            print(f"  {'Step':<6} {'Cumulative Acc':<15} {'Marginal Gain':<15}")
            print(f"  {'-'*6} {'-'*15} {'-'*15}")
            
            for i, step in enumerate(sorted_steps):
                cum_acc = steps[step]
                if i == 0:
                    marg_gain = cum_acc
                else:
                    prev_step = sorted_steps[i-1]
                    marg_gain = steps[step] - steps[prev_step]
                
                print(f"  {step:<6d} {cum_acc:>6.2f}%         {marg_gain:>+6.2f}%")
    
    # Print statistics for gold context wrong questions version
    if model_data_recovery:
        print("\n" + "="*80)
        print("MARGINAL GAIN STATISTICS - GOLD CONTEXT WRONG QUESTIONS")
        print("="*80)
        
        for model in sorted(model_data_recovery.keys()):
            steps = model_data_recovery[model]
            sorted_steps = sorted(steps.keys())
            
            print(f"\n{model}:")
            print(f"  {'Step':<6} {'Cumulative Acc':<15} {'Marginal Gain':<15}")
            print(f"  {'-'*6} {'-'*15} {'-'*15}")
            
            for i, step in enumerate(sorted_steps):
                cum_acc = steps[step]
                if i == 0:
                    marg_gain = cum_acc
                else:
                    prev_step = sorted_steps[i-1]
                    marg_gain = steps[step] - steps[prev_step]
                
                print(f"  {step:<6d} {cum_acc:>6.2f}%         {marg_gain:>+6.2f}%")


if __name__ == "__main__":
    main()
