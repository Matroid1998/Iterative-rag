#!/usr/bin/env python3
"""Generate hop distribution plots for ALL questions (not just unanswered)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import re
import numpy as np
from math import ceil

from config import get_iterative_model_entries


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


def load_iterative_summary(path: Path) -> Dict[str, dict]:
    """Load iterative RAG results and build question summary."""
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
    """Load gold context results - mapping question to correctness."""
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


def prepare_all_questions_stats(
    records: List[dict],
    question_hops_map: Dict[str, int],
    iterative_summary: Dict[str, dict],
    gold_context_summary: Dict[str, bool] = None,
) -> Tuple[List[int], List[Tuple[int, int, str]], List[Tuple[int, int, str]]]:
    """
    Prepare statistics for ALL questions (not just unanswered).
    
    Returns:
        hop_values: List of hop counts for all questions
        correct_steps: List of (max_source_step, hop_count, question) tuples for correctly answered questions
        incorrect_steps: List of (max_source_step, hop_count, question) tuples for incorrectly answered questions
    """
    hop_values: List[int] = []
    correct_steps: List[Tuple[int, int, str]] = []
    incorrect_steps: List[Tuple[int, int, str]] = []
    
    seen_questions = set()
    
    for record in records:
        question = extract_question(record)
        if not question or question in seen_questions:
            continue
        seen_questions.add(question)
        
        # Get hop count
        hop_count = question_hops_map.get(question)
        if hop_count:
            hop_values.append(hop_count)
        
        # Get correctness and step info from iterative summary
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


def plot_single_model_correctness_original(
    correct_steps: List[Tuple[int, int, str]],
    incorrect_steps: List[Tuple[int, int, str]],
    model_display_name: str,
    ax,
    global_max_y: int = None,
    gold_context_summary: Dict[str, bool] = None,
) -> None:
    """Plot correct vs incorrect by max source step for a single model, with hop stacking (original version)."""
    all_step_values = [step for step, _, _ in correct_steps] + [step for step, _, _ in incorrect_steps]
    max_step = max(all_step_values) if all_step_values else 0
    step_ticks = list(range(1, max_step + 1)) if max_step else [1]
    
    x_positions = np.arange(len(step_ticks))
    bar_width = 0.35
    
    if not correct_steps and not incorrect_steps:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(model_display_name)
        return
    
    # Define hop colors (pastel colors for better visibility)
    hop_colors = {
        1: '#FFB3BA',  # Pastel red
        2: '#BAFFC9',  # Pastel green
        3: '#BAE1FF',  # Pastel blue
        4: '#FFFFBA',  # Pastel yellow
    }
    
    # Count questions by (step, hop) for correct and incorrect
    correct_step_hop_counts = {}
    incorrect_step_hop_counts = {}
    
    for step, hop, _ in correct_steps:
        correct_step_hop_counts[(step, hop)] = correct_step_hop_counts.get((step, hop), 0) + 1
    
    for step, hop, _ in incorrect_steps:
        incorrect_step_hop_counts[(step, hop)] = incorrect_step_hop_counts.get((step, hop), 0) + 1
    
    # Plot correct (left bars) - stacked by hops
    for hop in [1, 2, 3, 4]:
        hop_heights = [correct_step_hop_counts.get((step, hop), 0) for step in step_ticks]
        bottom = np.zeros(len(step_ticks))
        
        # Calculate bottom position (sum of all previous hops)
        for prev_hop in range(1, hop):
            prev_heights = [correct_step_hop_counts.get((step, prev_hop), 0) for step in step_ticks]
            bottom += np.array(prev_heights)
        
        if sum(hop_heights) > 0:  # Only plot if there's data
            ax.bar(
                x_positions - bar_width / 2,
                hop_heights,
                bar_width,
                bottom=bottom,
                color=hop_colors[hop],
                edgecolor='white',
                linewidth=0.5,
                label=f'{hop} hop{"s" if hop > 1 else ""}' if hop in [1, 2, 3, 4] else None,
            )
    
    # Plot incorrect (right bars) - stacked by hops
    for hop in [1, 2, 3, 4]:
        hop_heights = [incorrect_step_hop_counts.get((step, hop), 0) for step in step_ticks]
        bottom = np.zeros(len(step_ticks))
        
        # Calculate bottom position
        for prev_hop in range(1, hop):
            prev_heights = [incorrect_step_hop_counts.get((step, prev_hop), 0) for step in step_ticks]
            bottom += np.array(prev_heights)
        
        if sum(hop_heights) > 0:
            ax.bar(
                x_positions + bar_width / 2,
                hop_heights,
                bar_width,
                bottom=bottom,
                color=hop_colors[hop],
                edgecolor='white',
                linewidth=0.5,
            )
    
    # Add total labels on top of bars
    for i, step in enumerate(step_ticks):
        # Correct totals
        correct_total = sum(correct_step_hop_counts.get((step, hop), 0) for hop in [1, 2, 3, 4])
        if correct_total > 0:
            ax.text(i - bar_width / 2, correct_total + 2, str(correct_total),
                   ha='center', va='bottom', fontsize=8, fontweight='bold', color='#2ca02c')
        
        # Incorrect totals
        incorrect_total = sum(incorrect_step_hop_counts.get((step, hop), 0) for hop in [1, 2, 3, 4])
        if incorrect_total > 0:
            ax.text(i + bar_width / 2, incorrect_total + 2, str(incorrect_total),
                   ha='center', va='bottom', fontsize=8, fontweight='bold', color='#d62728')
    
    # Add Correct/Incorrect labels
    ax.text(-0.5, -0.05, "Correct", ha='center', va='top', fontsize=9, 
           fontweight='bold', color='#2ca02c', transform=ax.get_xaxis_transform())
    ax.text(-0.5 + bar_width, -0.05, "Incorrect", ha='center', va='top', fontsize=9,
           fontweight='bold', color='#d62728', transform=ax.get_xaxis_transform())
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(step_ticks)
    ax.set_xlim(-0.5, len(step_ticks) - 0.5)
    ax.set_xlabel("Max source step", fontsize=10)
    ax.set_ylabel("Questions", fontsize=10)
    ax.set_title(model_display_name, fontsize=11, fontweight='bold')
    
    # Set uniform y-axis limit if provided
    if global_max_y is not None:
        ax.set_ylim(0, global_max_y * 1.1)
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')


def plot_single_model_correctness(
    correct_steps: List[Tuple[int, int, str]],
    incorrect_steps: List[Tuple[int, int, str]],
    model_display_name: str,
    ax,
    global_max_y: int = None,
    gold_context_summary: Dict[str, bool] = None,
) -> None:
    """Plot correctness by max source step with line plot showing accuracy rate and hop breakdown."""
    all_step_values = [step for step, _, _ in correct_steps] + [step for step, _, _ in incorrect_steps]
    max_step = max(all_step_values) if all_step_values else 0
    step_ticks = list(range(1, max_step + 1)) if max_step else [1]
    
    if not correct_steps and not incorrect_steps:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(model_display_name)
        return
    
    # Define hop colors
    hop_colors = {
        1: '#e74c3c',  # Red
        2: '#f39c12',  # Orange
        3: '#3498db',  # Blue
        4: '#9b59b6',  # Purple
    }
    
    # Count questions by (step, hop, correctness)
    step_hop_correct = {}
    step_hop_incorrect = {}
    
    # Also track questions per step for gold context comparison
    step_questions = {}  # step -> list of questions
    
    for step, hop, question in correct_steps:
        step_hop_correct[(step, hop)] = step_hop_correct.get((step, hop), 0) + 1
        if step not in step_questions:
            step_questions[step] = []
        step_questions[step].append(question)
    
    for step, hop, question in incorrect_steps:
        step_hop_incorrect[(step, hop)] = step_hop_incorrect.get((step, hop), 0) + 1
        if step not in step_questions:
            step_questions[step] = []
        step_questions[step].append(question)
    
    # Calculate accuracy per step and per hop
    accuracies = []
    gold_accuracies = []
    hop_breakdown = {1: [], 2: [], 3: [], 4: []}
    
    for step in step_ticks:
        total_correct = sum(step_hop_correct.get((step, h), 0) for h in [1, 2, 3, 4])
        total_incorrect = sum(step_hop_incorrect.get((step, h), 0) for h in [1, 2, 3, 4])
        total = total_correct + total_incorrect
        
        # Iterative RAG accuracy
        if total > 0:
            accuracy = (total_correct / total) * 100
            accuracies.append(accuracy)
        else:
            accuracies.append(0)
        
        # Gold context accuracy for the same questions
        if gold_context_summary and step in step_questions:
            questions_at_step = step_questions[step]
            gold_correct = sum(1 for q in questions_at_step if gold_context_summary.get(q, False))
            gold_total = len(questions_at_step)
            if gold_total > 0:
                gold_accuracy = (gold_correct / gold_total) * 100
                gold_accuracies.append(gold_accuracy)
            else:
                gold_accuracies.append(0)
        else:
            gold_accuracies.append(None)  # No data available
        
        # Breakdown by hop
        for hop in [1, 2, 3, 4]:
            hop_correct = step_hop_correct.get((step, hop), 0)
            hop_incorrect = step_hop_incorrect.get((step, hop), 0)
            hop_total = hop_correct + hop_incorrect
            hop_breakdown[hop].append(hop_total)
    
    x_positions = np.arange(len(step_ticks))
    bar_width = 0.8
    
    # Create stacked bars showing question volume by hop
    bottom = np.zeros(len(step_ticks))
    hop_bars = []
    
    for hop in [1, 2, 3, 4]:
        heights = hop_breakdown[hop]
        if sum(heights) > 0:
            bars = ax.bar(
                x_positions,
                heights,
                bar_width,
                bottom=bottom,
                color=hop_colors[hop],
                alpha=0.6,
                edgecolor='white',
                linewidth=1,
                label=f'{hop} hop{"s" if hop > 1 else ""}',
            )
            hop_bars.append(bars)
            bottom += np.array(heights)
    
    # Add total count labels on top
    for i, step in enumerate(step_ticks):
        total = int(bottom[i])
        if total > 0:
            ax.text(i, total + global_max_y * 0.02, str(total),
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Create secondary axis for accuracy line
    ax2 = ax.twinx()
    
    # Plot iterative RAG accuracy line
    line1 = ax2.plot(x_positions, accuracies, 'o-', color='#2ca02c', 
                    linewidth=3, markersize=8, markerfacecolor='white',
                    markeredgewidth=2, markeredgecolor='#2ca02c',
                    label='Iterative RAG', zorder=10)
    
    # Plot gold context accuracy line (for same questions)
    if gold_context_summary and any(acc is not None for acc in gold_accuracies):
        # Filter out None values for plotting
        valid_gold_x = [x for x, acc in zip(x_positions, gold_accuracies) if acc is not None]
        valid_gold_y = [acc for acc in gold_accuracies if acc is not None]
        
        if valid_gold_x and valid_gold_y:
            line2 = ax2.plot(valid_gold_x, valid_gold_y, 's--', color='#e67e22', 
                            linewidth=2.5, markersize=7, markerfacecolor='white',
                            markeredgewidth=2, markeredgecolor='#e67e22',
                            label='Gold Context (same Qs)', zorder=9, alpha=0.9)
    
    # Add accuracy percentage labels for iterative RAG
    for i, (x, acc) in enumerate(zip(x_positions, accuracies)):
        if acc > 0:
            ax2.text(x - 0.15, acc + 2, f'{acc:.1f}%', ha='center', va='bottom',
                    fontsize=7, fontweight='bold', color='#2ca02c')
    
    # Add accuracy percentage labels for gold context
    if gold_context_summary:
        for i, (x, acc) in enumerate(zip(x_positions, gold_accuracies)):
            if acc is not None and acc > 0:
                ax2.text(x + 0.15, acc - 3, f'{acc:.1f}%', ha='center', va='top',
                        fontsize=7, fontweight='bold', color='#e67e22')
    
    # Styling
    ax.set_xticks(x_positions)
    ax.set_xticklabels(step_ticks)
    ax.set_xlim(-0.5, len(step_ticks) - 0.5)
    ax.set_xlabel("Retrieval Step", fontsize=10, fontweight='bold')
    ax.set_ylabel("Number of Questions", fontsize=10, fontweight='bold')
    ax.set_title(model_display_name, fontsize=11, fontweight='bold', pad=10)
    
    ax2.set_ylabel("Accuracy (%)", fontsize=10, fontweight='bold', color='#2ca02c')
    ax2.tick_params(axis='y', labelcolor='#2ca02c')
    ax2.set_ylim(0, 105)
    ax2.spines['right'].set_color('#2ca02c')
    
    # Set uniform y-axis limit if provided
    if global_max_y is not None:
        ax.set_ylim(0, global_max_y * 1.15)
    
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    ax.set_axisbelow(True)


def plot_combined_model_correctness_original(
    model_data: Dict[str, Tuple[List[Tuple[int, int, str]], List[Tuple[int, int, str]]]],
    output_path: Path,
    gold_context_data: Dict[str, Dict[str, bool]] = None,
) -> None:
    """Create original version with side-by-side correct/incorrect bars stacked by hops."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - external dependency
        raise SystemExit(
            "matplotlib is required for plotting. Install it with 'pip install matplotlib'."
        ) from exc

    model_names = list(model_data.keys())

    if not model_names:
        print("No model data available for combined correctness plot.")
        return

    # Calculate global max y-value across all models
    global_max_y = 0
    for model_name, (correct_steps, incorrect_steps) in model_data.items():
        # Count by step
        all_steps_data = correct_steps + incorrect_steps
        if all_steps_data:
            max_step = max(step for step, _, _ in all_steps_data)
            for step in range(1, max_step + 1):
                correct_count = sum(1 for s, _, _ in correct_steps if s == step)
                incorrect_count = sum(1 for s, _, _ in incorrect_steps if s == step)
                max_at_step = max(correct_count, incorrect_count)
                global_max_y = max(global_max_y, max_at_step)

    cols = 4 if len(model_names) > 6 else 3
    rows = ceil(len(model_names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5))
    axes = axes.flatten()

    for idx, (model_name, (correct_steps, incorrect_steps)) in enumerate(model_data.items()):
        gold_summary = gold_context_data.get(model_name, {}) if gold_context_data else {}
        plot_single_model_correctness_original(
            correct_steps,
            incorrect_steps,
            model_name,
            axes[idx],
            global_max_y=global_max_y,
            gold_context_summary=gold_summary,
        )
        if idx == 0:
            axes[idx].legend(loc="upper right", fontsize=8, ncol=2)

    for idx in range(len(model_names), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        "Model Performance: Correct vs Incorrect by Max Source Step (with Question Hardness)",
        fontsize=16,
        fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_combined_model_correctness(
    model_data: Dict[str, Tuple[List[Tuple[int, int, str]], List[Tuple[int, int, str]]]],
    output_path: Path,
    gold_context_data: Dict[str, Dict[str, bool]] = None,
) -> None:
    """Create a single plot with subplots showing correctness by max source step for each model."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - external dependency
        raise SystemExit(
            "matplotlib is required for plotting. Install it with 'pip install matplotlib'."
        ) from exc

    model_names = list(model_data.keys())

    if not model_names:
        print("No model data available for combined correctness plot.")
        return

    # Calculate global max y-value across all models
    global_max_y = 0
    for model_name, (correct_steps, incorrect_steps) in model_data.items():
        # Count by step
        all_steps_data = correct_steps + incorrect_steps
        if all_steps_data:
            max_step = max(step for step, _, _ in all_steps_data)
            for step in range(1, max_step + 1):
                correct_count = sum(1 for s, _, _ in correct_steps if s == step)
                incorrect_count = sum(1 for s, _, _ in incorrect_steps if s == step)
                max_at_step = max(correct_count, incorrect_count)
                global_max_y = max(global_max_y, max_at_step)

    cols = 4 if len(model_names) > 6 else 3
    rows = ceil(len(model_names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5))
    axes = axes.flatten()

    for idx, (model_name, (correct_steps, incorrect_steps)) in enumerate(model_data.items()):
        gold_summary = gold_context_data.get(model_name, {}) if gold_context_data else {}
        plot_single_model_correctness(
            correct_steps,
            incorrect_steps,
            model_name,
            axes[idx],
            global_max_y=global_max_y,
            gold_context_summary=gold_summary,
        )

    # Add shared legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
              ncol=6, frameon=True, fontsize=10, fancybox=True, shadow=True)

    for idx in range(len(model_names), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        "Model Performance by Retrieval Step: Question Volume by Difficulty & Accuracy Rate",
        fontsize=16,
        fontweight='bold',
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "model"


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    
    # Create a dedicated output directory for all-questions plots
    output_dir = base / "plots"
    output_dir.mkdir(exist_ok=True)

    # Source datasets - we'll use the full response files instead of unanswered subsets
    iterative_dir = base / "responses_reverified"

    # Load QA hop data
    qa_lookup: Dict[str, int] = {}
    qa_path = base / "docs" / "chemrxiv_qa.json"
    if qa_path.exists():
        try:
            with qa_path.open("r", encoding="utf-8") as handle:
                entries = json.load(handle)
        except json.JSONDecodeError:
            entries = []
        for entry in entries:
            question = entry.get("q")
            path_list = entry.get("path")
            if isinstance(question, str) and isinstance(path_list, list) and path_list:
                qa_lookup[question.strip()] = len(path_list)

    # Collect data for all models
    model_data: Dict[str, Tuple[List[Tuple[int, int, str]], List[Tuple[int, int, str]]]] = {}
    gold_context_data: Dict[str, Dict[str, bool]] = {}
    
    # Directory for gold context files
    gold_context_dir = base / "response-jsonl-with-context"
    
    # Mapping for special cases where gold context filename differs
    gold_context_filename_mapping = {
        "responses_openrouter_google__gemini-2.5-pro_reverified.jsonl": "responses_openrouter_google__gemini-2.5-pro-reasoning.jsonl",
        "responses_openrouter_x-ai__grok-4-fast_reverified.jsonl": "responses_openrouter_x-ai__grok-4-fast-reasoning.jsonl",
        "responses_openrouter_z-ai__glm-4.6_reverified.jsonl": "responses_openrouter_z-ai__glm-4.6-reasoning_reverified.jsonl",
    }

    for iterative_path, display_name in get_iterative_model_entries():
        if not iterative_path.exists():
            print(f"Skipping {display_name}: {iterative_path} not found")
            continue

        iterative_summary = load_iterative_summary(iterative_path)
        
        # Load corresponding gold context file
        # First, check if there's a special mapping
        if iterative_path.name in gold_context_filename_mapping:
            gold_context_filename = gold_context_filename_mapping[iterative_path.name]
            gold_context_path = gold_context_dir / gold_context_filename
        else:
            # Try the same filename
            gold_context_path = gold_context_dir / iterative_path.name
            if not gold_context_path.exists():
                # Try without "_reverified" suffix
                gold_context_path = gold_context_dir / iterative_path.name.replace("_reverified", "")
        
        gold_summary = {}
        if gold_context_path.exists():
            gold_summary = load_gold_context_summary(gold_context_path)
            print(f"Loaded gold context for {display_name}: {len(gold_summary)} questions")
        else:
            print(f"Warning: No gold context file found for {display_name} at {gold_context_path}")
        
        gold_context_data[display_name] = gold_summary

        # Build question-to-hops mapping
        question_hops_map: Dict[str, int] = {}
        for question, data in iterative_summary.items():
            hop_value = data.get("raw_hops")
            if not isinstance(hop_value, int) or hop_value <= 0:
                hop_value = qa_lookup.get(question)
            if isinstance(hop_value, int) and hop_value > 0:
                question_hops_map[question] = max(1, min(4, hop_value))

        # Fall back entirely on QA hop data if needed
        if not question_hops_map and qa_lookup:
            question_hops_map = {k: max(1, min(4, v)) for k, v in qa_lookup.items()}

        # Load iterative RAG records
        all_records = load_records(iterative_path)
        
        # Get correctness data for this model
        hop_values, correct_steps, incorrect_steps = prepare_all_questions_stats(
            all_records,
            question_hops_map,
            iterative_summary,
            gold_summary,
        )
        
        model_data[display_name] = (correct_steps, incorrect_steps)

    # Generate both versions of the plot
    if model_data:
        # Version 1: New design with stacked bars and accuracy line (with gold context comparison)
        output_path = output_dir / "all_models_correctness_by_steps.png"
        plot_combined_model_correctness(
            model_data,
            output_path,
            gold_context_data,
        )
        print(f"Generated combined model correctness plot (new version with gold comparison): {output_path}")
        
        # Version 2: Original design with side-by-side bars
        output_path_original = output_dir / "all_models_correctness_by_steps_sidebyside.png"
        plot_combined_model_correctness_original(
            model_data,
            output_path_original,
            gold_context_data,
        )
        print(f"Generated combined model correctness plot (original version): {output_path_original}")
        
        # Remove the old individual plots
        old_patterns = [
            "all_questions_hop_distributions_*.png"
        ]
        
        import glob
        for pattern in old_patterns:
            for old_file in glob.glob(str(output_dir / pattern)):
                old_path = Path(old_file)
                if old_path.exists():
                    old_path.unlink()
                    print(f"Removed: {old_path}")
    else:
        print("No iterative response files found for plotting.")


if __name__ == "__main__":
    main()
