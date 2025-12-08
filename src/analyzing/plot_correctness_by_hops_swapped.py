#!/usr/bin/env python3
"""
Generate correctness plot with swapped axes:
X-axis: Number of Hops (1-4)
Stacked Bars: Retrieval Steps (1-5)
"""

from __future__ import annotations

import json
import re
from math import ceil
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional, Any

import matplotlib.pyplot as plt
import numpy as np

from config import get_iterative_model_entries

# -----------------------------------------------------------------------------
# Data Loading Functions (Copied/Adapted from plot_all_questions_hop_distributions.py)
# -----------------------------------------------------------------------------

def load_records(path: Path) -> List[dict]:
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

def iter_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError:
                continue

def extract_question(record: dict) -> str | None:
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

def load_iterative_summary(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    summary: Dict[str, dict] = {}
    for record in iter_records(path):
        question = extract_question(record)
        if not question:
            continue
        is_correct = bool(record.get("is_correct", False))
        raw_hops = record.get("number_of_hops")
        max_source_step = extract_max_source_step(record)
        summary[question] = {
            "is_correct": is_correct,
            "raw_hops": raw_hops,
            "max_source_step": max_source_step,
        }
    return summary

def load_gold_context_summary(path: Path) -> Dict[str, bool]:
    if not path.exists():
        return {}
    summary: Dict[str, bool] = {}
    for record in iter_records(path):
        question = extract_question(record)
        if not question:
            continue
        is_correct = bool(record.get("is_correct", False))
        summary[question] = is_correct
    return summary

def load_no_context_wrong_questions(no_context_path: Path) -> set:
    if not no_context_path.exists():
        return set()
    wrong_questions = set()
    for record in iter_records(no_context_path):
        question = extract_question(record)
        if not question:
            continue
        is_correct = bool(record.get("is_correct", False))
        if not is_correct:
            wrong_questions.add(question)
    return wrong_questions

def prepare_all_questions_stats(
    records: List[dict],
    question_hops_map: Dict[str, int],
    iterative_summary: Dict[str, dict],
    filter_questions: set = None,
) -> Tuple[List[int], List[Tuple[int, int, str]], List[Tuple[int, int, str]]]:
    hop_values: List[int] = []
    correct_steps: List[Tuple[int, int, str]] = []
    incorrect_steps: List[Tuple[int, int, str]] = []
    seen_questions = set()
    
    for record in records:
        question = extract_question(record)
        if not question or question in seen_questions:
            continue
        if filter_questions is not None and question not in filter_questions:
            continue
        seen_questions.add(question)
        
        hop_count = question_hops_map.get(question)
        if hop_count:
            hop_values.append(hop_count)
        
        if question in iterative_summary:
            summary = iterative_summary[question]
            max_step = summary.get("max_source_step")
            is_correct = summary.get("is_correct", False)
            if max_step is not None and hop_count is not None:
                if is_correct:
                    correct_steps.append((max_step, hop_count, question))
                else:
                    incorrect_steps.append((max_step, hop_count, question))
    return hop_values, correct_steps, incorrect_steps

# -----------------------------------------------------------------------------
# Plotting Functions
# -----------------------------------------------------------------------------

def plot_single_model_correctness_swapped(
    correct_steps: List[Tuple[int, int, str]],
    incorrect_steps: List[Tuple[int, int, str]],
    model_display_name: str,
    ax,
    global_max_y: int = None,
    gold_context_summary: Dict[str, bool] = None,
) -> Optional[Any]:
    """
    Plot correctness by HOPS (x-axis) with stacked bars for STEPS.
    """
    # X-axis: Hops 1, 2, 3, 4
    hop_ticks = [1, 2, 3, 4]
    x_positions = np.arange(len(hop_ticks))
    
    if not correct_steps and not incorrect_steps:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(model_display_name)
        return None
    
    # Define step colors (distinct colors for steps 1-5)
    step_colors = {
        1: '#e74c3c',  # Red
        2: '#f39c12',  # Orange
        3: '#3498db',  # Blue
        4: '#9b59b6',  # Purple
        5: '#2ecc71',  # Green (for 5+)
    }
    
    # Organize data: (hop, step) -> count
    hop_step_correct = {}
    hop_step_incorrect = {}
    
    # Track questions per hop for gold context comparison
    hop_questions = {h: [] for h in hop_ticks}
    
    for step, hop, question in correct_steps:
        if hop in hop_ticks:
            # Cap step at 5 for coloring/stacking
            display_step = min(step, 5)
            hop_step_correct[(hop, display_step)] = hop_step_correct.get((hop, display_step), 0) + 1
            hop_questions[hop].append(question)
            
    for step, hop, question in incorrect_steps:
        if hop in hop_ticks:
            display_step = min(step, 5)
            hop_step_incorrect[(hop, display_step)] = hop_step_incorrect.get((hop, display_step), 0) + 1
            hop_questions[hop].append(question)
            
    # Calculate accuracy per hop
    accuracies = []
    gold_accuracies = []
    step_breakdown = {s: [] for s in range(1, 6)}
    
    for hop in hop_ticks:
        total_correct = sum(hop_step_correct.get((hop, s), 0) for s in range(1, 6))
        total_incorrect = sum(hop_step_incorrect.get((hop, s), 0) for s in range(1, 6))
        total = total_correct + total_incorrect
        
        # Iterative RAG accuracy
        if total > 0:
            accuracy = (total_correct / total) * 100
            accuracies.append(accuracy)
        else:
            accuracies.append(0)
            
        # Gold context accuracy
        if gold_context_summary:
            questions_at_hop = hop_questions[hop]
            gold_correct = sum(1 for q in questions_at_hop if gold_context_summary.get(q, False))
            gold_total = len(questions_at_hop)
            if gold_total > 0:
                gold_accuracy = (gold_correct / gold_total) * 100
                gold_accuracies.append(gold_accuracy)
            else:
                gold_accuracies.append(0)
        else:
            gold_accuracies.append(None)
            
        # Breakdown by step
        for step in range(1, 6):
            step_correct = hop_step_correct.get((hop, step), 0)
            step_incorrect = hop_step_incorrect.get((hop, step), 0)
            step_total = step_correct + step_incorrect
            step_breakdown[step].append(step_total)
            
    bar_width = 0.8
    bottom = np.zeros(len(hop_ticks))
    
    # Create stacked bars
    for step in range(1, 6):
        heights = step_breakdown[step]
        if sum(heights) > 0:
            label = f'Step {step}'
            ax.bar(
                x_positions,
                heights,
                bar_width,
                bottom=bottom,
                color=step_colors[step],
                alpha=0.6,
                edgecolor='white',
                linewidth=1,
                label=label
            )
            bottom += np.array(heights)
            
    # Add total count labels
    for i, hop in enumerate(hop_ticks):
        total = int(bottom[i])
        if total > 0:
            ax.text(i, total + (global_max_y * 0.02 if global_max_y else 5), str(total),
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
            
    # Secondary axis for accuracy
    ax2 = ax.twinx()
    
    # Plot iterative RAG accuracy
    ax2.plot(x_positions, accuracies, 'o-', color='#2ca02c',
             linewidth=3, markersize=8, markerfacecolor='white',
             markeredgewidth=2, markeredgecolor='#2ca02c',
             label='Accuracy in Iterative rag', zorder=10)
             
    # Plot gold context accuracy
    if gold_context_summary:
        valid_gold_x = [x for x, acc in zip(x_positions, gold_accuracies) if acc is not None]
        valid_gold_y = [acc for acc in gold_accuracies if acc is not None]
        
        if valid_gold_x:
            ax2.plot(valid_gold_x, valid_gold_y, 's--', color='#e67e22',
                     linewidth=2.5, markersize=7, markerfacecolor='white',
                     markeredgewidth=2, markeredgecolor='#e67e22',
                     label='Gold Context', zorder=9, alpha=0.9)
                     
    # Add accuracy labels
    for i, (x, acc) in enumerate(zip(x_positions, accuracies)):
        if acc > 0:
            ax2.text(x - 0.15, acc + 2, f'{acc:.1f}%', ha='center', va='bottom',
                    fontsize=7, fontweight='bold', color='#2ca02c')
                    
    if gold_context_summary:
        for i, (x, acc) in enumerate(zip(x_positions, gold_accuracies)):
            if acc is not None and acc > 0:
                ax2.text(x + 0.15, acc - 3, f'{acc:.1f}%', ha='center', va='top',
                        fontsize=7, fontweight='bold', color='#e67e22')

    # Styling
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{h} Hop{'s' if h>1 else ''}" for h in hop_ticks])
    ax.set_xlim(-0.5, len(hop_ticks) - 0.5)
    ax.set_xlabel("Number of Hops", fontsize=10, fontweight='bold')
    ax.set_ylabel("Number of Questions", fontsize=10, fontweight='bold')
    ax.set_title(model_display_name, fontsize=11, fontweight='bold', pad=10)
    
    ax2.set_ylabel("Accuracy (%)", fontsize=10, fontweight='bold', color='#2ca02c')
    ax2.tick_params(axis='y', labelcolor='#2ca02c')
    ax2.set_ylim(0, 105)
    ax2.spines['right'].set_color='#2ca02c'
    
    if global_max_y is not None:
        ax.set_ylim(0, global_max_y)
        
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    ax.set_axisbelow(True)
    return ax2

def plot_combined_model_correctness_swapped(
    model_data: Dict[str, Tuple[List[Tuple[int, int, str]], List[Tuple[int, int, str]]]],
    output_path: Path,
    gold_context_data: Dict[str, Dict[str, bool]] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    model_names = list(model_data.keys())
    if not model_names:
        return

    # Calculate global max y
    global_max_y = 0
    for model_name, (correct_steps, incorrect_steps) in model_data.items():
        # Count by hop
        hop_counts = {h: 0 for h in [1, 2, 3, 4]}
        for _, hop, _ in correct_steps + incorrect_steps:
            if hop in hop_counts:
                hop_counts[hop] += 1
        if hop_counts:
            global_max_y = 300  # Set fixed y-axis limit as requested

    cols = 4 if len(model_names) > 6 else 3
    rows = ceil(len(model_names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5))
    axes = axes.flatten()

    all_handles = []
    all_labels = []
    seen_labels = set()

    for idx, (model_name, (correct_steps, incorrect_steps)) in enumerate(model_data.items()):
        gold_summary = gold_context_data.get(model_name, {}) if gold_context_data else {}
        ax2 = plot_single_model_correctness_swapped(
            correct_steps,
            incorrect_steps,
            model_name,
            axes[idx],
            global_max_y=global_max_y,
            gold_context_summary=gold_summary,
        )

        # Collect handles for legend (including from secondary axis)
        if idx < len(model_names):
            handles, labels = axes[idx].get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label not in seen_labels:
                    all_handles.append(handle)
                    all_labels.append(label)
                    seen_labels.add(label)
            
            if ax2:
                handles2, labels2 = ax2.get_legend_handles_labels()
                for handle, label in zip(handles2, labels2):
                    if label not in seen_labels:
                        all_handles.append(handle)
                        all_labels.append(label)
                        seen_labels.add(label)

    # Legend (handles collected during plotting)
                
    desired_order = ['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', 'Accuracy in Iterative rag', 'Gold Context']
    ordered_handles = []
    ordered_labels = []
    for desired_label in desired_order:
        if desired_label in all_labels:
            idx = all_labels.index(desired_label)
            ordered_handles.append(all_handles[idx])
            ordered_labels.append(all_labels[idx])

    ncol = len(ordered_labels)
    fig.legend(ordered_handles, ordered_labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
              ncol=ncol, frameon=True, fontsize=10, fancybox=True, shadow=True)

    for idx in range(len(model_names), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        "Model Performance by Question Complexity (Hops): Volume, Accuracy & Retrieval Steps",
        fontsize=16,
        fontweight='bold',
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main():
    base = Path(__file__).resolve().parents[1]
    output_dir = base / "plots"
    output_dir.mkdir(exist_ok=True)
    
    # Load QA hop data
    qa_lookup = {}
    qa_path = base / "docs" / "chemrxiv_qa.json"
    if qa_path.exists():
        try:
            with qa_path.open("r") as f:
                entries = json.load(f)
                for entry in entries:
                    q = entry.get("q")
                    path = entry.get("path")
                    if q and path:
                        qa_lookup[q.strip()] = len(path)
        except:
            pass

    # Load data
    model_data_no_context_wrong = {}
    gold_context_data = {}
    
    gold_context_dir = base / "response-jsonl-with-context"
    no_context_dir = base / "response-jsonl-without-context"
    
    # Mappings
    gold_context_filename_mapping = {
        "responses_openrouter_google__gemini-2.5-pro_reverified.jsonl": "responses_openrouter_google__gemini-2.5-pro-reasoning.jsonl",
        "responses_openrouter_x-ai__grok-4-fast_reverified.jsonl": "responses_openrouter_x-ai__grok-4-fast-reasoning.jsonl",
        "responses_openrouter_z-ai__glm-4.6_reverified.jsonl": "responses_openrouter_z-ai__glm-4.6-reasoning_reverified.jsonl",
    }
    
    no_context_mapping = {
        "responses_openrouter_google__gemini-2.5-pro_reverified.jsonl": "responses_openrouter_google__gemini-2.5-pro-reasoning.jsonl",
        "responses_openrouter_x-ai__grok-4-fast_reverified.jsonl": "responses_openrouter_x-ai__grok-4-fast-reasoning.jsonl",
        "responses_openrouter_z-ai__glm-4.6_reverified.jsonl": "responses_openrouter_z-ai__glm-4.6-reasoning_reverified.jsonl",
    }

    for iterative_path, display_name in get_iterative_model_entries():
        if not iterative_path.exists():
            continue
            
        iterative_summary = load_iterative_summary(iterative_path)
        
        # Load Gold Context
        if iterative_path.name in gold_context_filename_mapping:
            gold_path = gold_context_dir / gold_context_filename_mapping[iterative_path.name]
        else:
            gold_path = gold_context_dir / iterative_path.name
            if not gold_path.exists():
                gold_path = gold_context_dir / iterative_path.name.replace("_reverified", "")
        
        gold_summary = load_gold_context_summary(gold_path) if gold_path.exists() else {}
        gold_context_data[display_name] = gold_summary
        
        # Load No Context
        if iterative_path.name in no_context_mapping:
            nc_path = no_context_dir / no_context_mapping[iterative_path.name]
        else:
            nc_path = no_context_dir / iterative_path.name
            if not nc_path.exists():
                nc_path = no_context_dir / iterative_path.name.replace("_reverified", "")
                
        nc_wrong_questions = load_no_context_wrong_questions(nc_path) if nc_path and nc_path.exists() else set()
        
        # Build hop map
        question_hops_map = {}
        for q, d in iterative_summary.items():
            h = d.get("raw_hops")
            if not isinstance(h, int) or h <= 0:
                h = qa_lookup.get(q)
            if isinstance(h, int) and h > 0:
                question_hops_map[q] = max(1, min(4, h))
        
        if not question_hops_map and qa_lookup:
            question_hops_map = {k: max(1, min(4, v)) for k, v in qa_lookup.items()}
            
        # Prepare stats
        all_records = load_records(iterative_path)
        if nc_wrong_questions:
            _, correct, incorrect = prepare_all_questions_stats(
                all_records,
                question_hops_map,
                iterative_summary,
                filter_questions=nc_wrong_questions
            )
            model_data_no_context_wrong[display_name] = (correct, incorrect)

    # Generate Plot
    if model_data_no_context_wrong:
        output_path = output_dir / "all_models_correctness_by_hops_no_context_wrong_no_coverage.png"
        plot_combined_model_correctness_swapped(
            model_data_no_context_wrong,
            output_path,
            gold_context_data
        )
        print(f"Generated swapped plot: {output_path}")

if __name__ == "__main__":
    main()
