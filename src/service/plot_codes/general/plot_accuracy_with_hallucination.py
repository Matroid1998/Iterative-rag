#!/usr/bin/env python3
"""
Plot accuracy lines with hallucination confidence metrics for no-context wrong questions.

This shows how model accuracy relates to confidence miscalibration
(overconfident, under-confident, and well-calibrated responses).
"""

import json
from collections import defaultdict
from math import ceil
from pathlib import Path
from typing import Dict, List, Tuple, Set

import matplotlib.pyplot as plt
import numpy as np


def get_base_path() -> Path:
    """Get the base path for the project."""
    return Path(__file__).resolve().parents[2]


def get_model_entries() -> List[Tuple[Path, Path, Path, Path, Path, str]]:
    """Get list of (quality_path, reverified_path, hallucination_path, coverage_gap_path, no_context_path, display_name) tuples."""
    base = get_base_path()
    reverified_dir = base / "src" / "responses_reverified"
    hallucination_dir = base / "src" / "rag_analysis" / "output"
    no_context_dir = base / "src" / "response-jsonl-without-context"
    quality_dir = base / "src" / "rag_analysis" / "output"
    coverage_gap_dir = base / "src" / "rag_analysis" / "output"
    
    model_names = {
        "bedrock_mistral.mistral-large-2402-v1:0": "Mistral Large 2402",
        "bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning": "Claude 3.7 Sonnet Thinking",
        "bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0": "Claude 3.7 Sonnet",
        "bedrock_us.deepseek.r1-v1:0-reasoning": "DeepSeek R1",
        "bedrock_us.meta.llama3-3-70b-instruct-v1:0": "Llama 3.3 70B Instruct",
        "openai_gpt-4o": "GPT-4o",
        "openai_gpt-5": "GPT-5",
        "openrouter_anthropic__claude-sonnet-4.5": "Claude Sonnet 4.5",
        "openrouter_google__gemini-2.5-pro": "Gemini 2.5 Pro",
        "openrouter_x-ai__grok-4-fast": "Grok 4 Fast",
        "openrouter_z-ai__glm-4.6": "GLM 4.6",
    }
    
    entries = []
    for hallucination_file in sorted(hallucination_dir.glob("*hallucination_judgment.jsonl")):
        stem = hallucination_file.stem
        
        if stem.endswith("_hallucination_judgment"):
            stem = stem[:-len("_hallucination_judgment")]
        
        if stem.startswith("2_"):
            stem = stem[2:]
        
        raw_name = stem
        if stem.endswith("_reverified"):
            raw_name = stem[:-len("_reverified")]
        
        model_key = raw_name
        if model_key.startswith("responses_"):
            model_key = model_key[len("responses_"):]
        
        display_name = model_names.get(model_key, model_key)
        
        # Find corresponding reverified file
        reverified_file = None
        for rev_file in reverified_dir.glob("*.jsonl"):
            if model_key in rev_file.name:
                reverified_file = rev_file
                break
        
        if not reverified_file:
            continue
        
        # Find corresponding no-context file
        no_context_file = None
        # Try multiple naming patterns
        patterns = [
            f"responses_{model_key}.jsonl",
            f"responses_{model_key}_reverified.jsonl",
            f"responses_{model_key}-reasoning.jsonl",
            f"responses_{model_key}-reasoning_reverified.jsonl",
        ]
        for pattern in patterns:
            candidate = no_context_dir / pattern
            if candidate.exists():
                no_context_file = candidate
                break
        
        if not no_context_file:
            print(f"Warning: No no-context file found for {display_name}")
            continue
        
        # Find corresponding quality file
        quality_file = None
        for qual_file in quality_dir.glob("*quality_judement.jsonl"):
            if model_key in qual_file.name:
                quality_file = qual_file
                break
        
        if not quality_file:
            print(f"Warning: No quality file found for {display_name}")
            continue
        
        # Find corresponding coverage gap file
        coverage_gap_file = None
        for cov_file in coverage_gap_dir.glob("*coverage_gap_judgments.jsonl"):
            if model_key in cov_file.name:
                coverage_gap_file = cov_file
                break
        
        if not coverage_gap_file:
            print(f"Warning: No coverage gap file found for {display_name}")
            continue
        
        entries.append((quality_file, reverified_file, hallucination_file, coverage_gap_file, no_context_file, display_name))
    
    return entries


def load_no_context_wrong_questions(no_context_path: Path) -> Set[str]:
    """Load questions that were answered incorrectly in no-context baseline."""
    wrong_questions = set()
    with open(no_context_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
            except:
                continue
            
            # Get question from nested raw field
            question = data.get('raw', {}).get('question', '')
            if not question:
                continue
            
            # Check if answer was incorrect
            is_correct = data.get('is_correct', False)
            if not is_correct:
                wrong_questions.add(question)
    
    return wrong_questions


def load_iterative_data(quality_path: Path, reverified_path: Path, filter_questions: Set[str] = None) -> Dict[str, Tuple[int, bool]]:
    """
    Load iterative RAG data with final step correctness.
    
    Returns:
        Dict[question -> (final_step, is_correct)]
    """
    # First load is_correct from reverified
    is_correct_map = {}
    with open(reverified_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
            except:
                continue
            
            question = data.get('raw_response', {}).get('question', '')
            if not question:
                continue
            
            is_correct = data.get('is_correct', False)
            is_correct_map[question] = is_correct
    
    # Now load step data from quality file
    question_data = {}
    with open(quality_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
            except:
                continue
            
            question = data.get('question', '')
            if not question:
                continue
            
            # Apply filter if provided
            if filter_questions is not None and question not in filter_questions:
                continue
            
            # Get final step from per_step data
            parsed = data.get('parsed_judgment', {})
            per_step = parsed.get('per_step', [])
            
            if not per_step:
                continue
            
            # Get the last step number
            final_step = max(step_data.get('step', 1) for step_data in per_step)
            
            # Get correctness from reverified data
            is_correct = is_correct_map.get(question, False)
            
            question_data[question] = (final_step, is_correct)
    
    return question_data


def load_hallucination_data(hallucination_path: Path, filter_questions: Set[str] = None) -> Dict[str, Dict]:
    """
    Load hallucination judgment data.
    
    Returns:
        Dict[question -> {'is_miscalibrated': bool, 'direction': str, 'hop_coverage_est': float}]
    """
    hallucination_data = {}
    
    with open(hallucination_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
            except:
                continue
            
            question = data.get('question', '')
            if not question:
                continue
            
            # Apply filter if provided
            if filter_questions is not None and question not in filter_questions:
                continue
            
            parsed = data.get('parsed_judgment', {})
            confidence_misc = parsed.get('confidence_miscalibration', {})
            
            is_miscalibrated = confidence_misc.get('is_miscalibrated', False)
            direction = confidence_misc.get('direction', 'ok')
            hop_coverage = confidence_misc.get('hop_coverage_est', 1.0)
            
            hallucination_data[question] = {
                'is_miscalibrated': is_miscalibrated,
                'direction': direction,
                'hop_coverage_est': hop_coverage,
            }
    
    return hallucination_data


def load_coverage_gap_data(quality_path: Path, filter_questions: Set[str] = None) -> Dict[str, Dict]:
    """
    Load coverage gap data from quality judgment files.
    
    Returns:
        Dict[question -> {'has_gap': bool}]
    """
    coverage_data = {}
    
    with open(quality_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
            except:
                continue
            
            question = data.get('question', '')
            if not question:
                continue
            
            # Apply filter if provided
            if filter_questions is not None and question not in filter_questions:
                continue
            
            parsed = data.get('parsed_judgment', {})
            retrieval_coverage = parsed.get('retrieval_coverage_gap', {})
            
            has_gap = retrieval_coverage.get('has_gap', False)
            
            coverage_data[question] = {
                'has_gap': has_gap,
            }
    
    return coverage_data


def prepare_step_statistics(
    question_data: Dict[str, Tuple[int, bool]],
    hallucination_data: Dict[str, Dict],
    coverage_gap_data: Dict[str, Dict],
) -> Tuple[Dict[int, Dict], int]:
    """
    Prepare statistics per step.
    
    Returns:
        (step_stats, max_step) where step_stats[step] = {
            'total': int,
            'correct': int,
            'overconfident': int,
            'underconfident': int,
            'ok': int,
            'coverage_complete': int,  # count of questions with complete coverage (no gap)
        }
    """
    step_stats = defaultdict(lambda: {
        'total': 0,
        'correct': 0,
        'overconfident': 0,
        'underconfident': 0,
        'ok': 0,
        'coverage_complete': 0,
    })
    
    max_step = 0
    
    for question, (final_step, is_correct) in question_data.items():
        max_step = max(max_step, final_step)
        
        step_stats[final_step]['total'] += 1
        if is_correct:
            step_stats[final_step]['correct'] += 1
        
        # Add hallucination stats
        if question in hallucination_data:
            direction = hallucination_data[question]['direction']
            
            # Count by direction (use exact strings from the data)
            if direction == 'overconfident_finalize':
                step_stats[final_step]['overconfident'] += 1
            elif direction == 'underconfident_continue':
                step_stats[final_step]['underconfident'] += 1
            else:  # 'ok'
                step_stats[final_step]['ok'] += 1
        
        # Add coverage gap stats (has_gap=False means complete coverage)
        if question in coverage_gap_data:
            has_gap = coverage_gap_data[question].get('has_gap', False)
            if not has_gap:  # Complete coverage
                step_stats[final_step]['coverage_complete'] += 1
    
    return dict(step_stats), max_step


def plot_single_model_with_hallucination(
    step_stats: Dict[int, Dict],
    max_step: int,
    model_name: str,
    ax,
) -> None:
    """Plot accuracy line with hallucination confidence metrics."""
    
    if max_step == 0:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(model_name)
        return
    
    step_ticks = list(range(1, max_step + 1))
    x_positions = np.arange(len(step_ticks))
    
    # Calculate percentages per step
    accuracies = []
    overconfident_pcts = []
    underconfident_pcts = []
    ok_pcts = []
    coverage_complete_pcts = []
    
    for step in step_ticks:
        stats = step_stats.get(step, {})
        total = stats.get('total', 0)
        
        if total > 0:
            # Accuracy
            accuracy = (stats.get('correct', 0) / total) * 100
            accuracies.append(accuracy)
            
            # Hallucination percentages
            overconfident_pcts.append((stats.get('overconfident', 0) / total) * 100)
            underconfident_pcts.append((stats.get('underconfident', 0) / total) * 100)
            ok_pcts.append((stats.get('ok', 0) / total) * 100)
            
            # Coverage complete percentage (questions with no gap)
            coverage_complete_pcts.append((stats.get('coverage_complete', 0) / total) * 100)
        else:
            accuracies.append(0)
            overconfident_pcts.append(0)
            underconfident_pcts.append(0)
            ok_pcts.append(0)
            coverage_complete_pcts.append(0)
    
    # Plot accuracy line (main metric)
    ax.plot(x_positions, accuracies, 'o-', color='#2ca02c', 
            linewidth=3, markersize=10, markerfacecolor='white',
            markeredgewidth=2.5, markeredgecolor='#2ca02c',
            label='Accuracy', zorder=10)
    
    # Plot overconfident percentage
    if any(pct > 0 for pct in overconfident_pcts):
        ax.plot(x_positions, overconfident_pcts, 's--', color='#d62728', 
                linewidth=2.5, markersize=8, markerfacecolor='white',
                markeredgewidth=2.5, markeredgecolor='#d62728',
                label='Overconfident %', zorder=9, alpha=0.9)
    
    # Plot under-confident percentage
    if any(pct > 0 for pct in underconfident_pcts):
        ax.plot(x_positions, underconfident_pcts, '^-.', color='#ff7f0e', 
                linewidth=2.5, markersize=8, markerfacecolor='white',
                markeredgewidth=2.5, markeredgecolor='#ff7f0e',
                label='Under-confident %', zorder=8, alpha=0.9)
    
    # Plot OK percentage
    if any(pct > 0 for pct in ok_pcts):
        ax.plot(x_positions, ok_pcts, 'v:', color='#1f77b4', 
                linewidth=2.5, markersize=8, markerfacecolor='white',
                markeredgewidth=2.5, markeredgecolor='#1f77b4',
                label='Well-calibrated %', zorder=7, alpha=0.9)
    
    # Plot coverage complete percentage (matching the original plot)
    if any(pct > 0 for pct in coverage_complete_pcts):
        ax.plot(x_positions, coverage_complete_pcts, '^-.', color='#9467bd', 
                linewidth=2.5, markersize=9, markerfacecolor='white',
                markeredgewidth=2.5, markeredgecolor='#9467bd',
                label='Coverage Complete %', zorder=6, alpha=0.85)
    
    # Add percentage labels for accuracy
    for i, (x, acc) in enumerate(zip(x_positions, accuracies)):
        if acc > 0:
            ax.text(x, acc + 2.5, f'{acc:.1f}%', ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color='#2ca02c')
    
    # Styling
    ax.set_xticks(x_positions)
    ax.set_xticklabels(step_ticks)
    ax.set_xlim(-0.5, len(step_ticks) - 0.5)
    ax.set_xlabel("Retrieval Step", fontsize=11, fontweight='bold')
    ax.set_ylabel("Percentage (%)", fontsize=11, fontweight='bold')
    ax.set_title(model_name, fontsize=12, fontweight='bold', pad=10)
    
    ax.set_ylim(0, 105)
    ax.grid(axis='both', alpha=0.3, linestyle='--', zorder=0)
    ax.set_axisbelow(True)


def main():
    """Main execution function."""
    base = get_base_path()
    output_dir = base / "src" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Analyzing accuracy with hallucination confidence metrics...")
    print("Filtering to no-context wrong questions only...\n")
    
    model_step_stats = {}
    
    for quality_path, reverified_path, hallucination_path, coverage_gap_path, no_context_path, display_name in get_model_entries():
        # Load no-context wrong questions
        no_context_wrong = load_no_context_wrong_questions(no_context_path)
        print(f"  {display_name:30s}: {len(no_context_wrong)} no-context wrong questions")
        
        if not no_context_wrong:
            print(f"    Warning: No wrong questions found, skipping.")
            continue
        
        # Load iterative data (filtered)
        question_data = load_iterative_data(quality_path, reverified_path, no_context_wrong)
        
        # Load hallucination data (filtered)
        hallucination_data = load_hallucination_data(hallucination_path, no_context_wrong)
        
        # Load coverage gap data (filtered)
        coverage_gap_data = load_coverage_gap_data(coverage_gap_path, no_context_wrong)
        
        # Prepare statistics
        step_stats, max_step = prepare_step_statistics(question_data, hallucination_data, coverage_gap_data)
        
        if max_step > 0:
            model_step_stats[display_name] = (step_stats, max_step)
            print(f"    Steps: 1-{max_step}")
        else:
            print(f"    Warning: No step data found")
    
    print(f"\nTotal models with data: {len(model_step_stats)}")
    
    # Create plot
    if not model_step_stats:
        print("No data available for plotting.")
        return
    
    # Sort models alphabetically
    sorted_models = sorted(model_step_stats.keys())
    
    # Create subplots
    cols = 4 if len(sorted_models) > 6 else 3
    rows = ceil(len(sorted_models) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5))
    axes = axes.flatten()
    
    for idx, model_name in enumerate(sorted_models):
        step_stats, max_step = model_step_stats[model_name]
        plot_single_model_with_hallucination(
            step_stats,
            max_step,
            model_name,
            axes[idx],
        )
    
    # Hide unused subplots
    for idx in range(len(sorted_models), len(axes)):
        axes[idx].set_visible(False)
    
    # Add shared legend at the top
    all_handles = []
    all_labels = []
    seen_labels = set()
    
    for ax in axes[:len(sorted_models)]:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in seen_labels:
                all_handles.append(handle)
                all_labels.append(label)
                seen_labels.add(label)
    
    # Reorder legend
    desired_order = ['Accuracy', 'Overconfident %', 'Under-confident %', 'Well-calibrated %', 'Coverage Complete %']
    ordered_handles = []
    ordered_labels = []
    for desired_label in desired_order:
        if desired_label in all_labels:
            idx = all_labels.index(desired_label)
            ordered_handles.append(all_handles[idx])
            ordered_labels.append(all_labels[idx])
    
    fig.legend(ordered_handles, ordered_labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
              ncol=len(ordered_labels), frameon=True, fontsize=11, fancybox=True, shadow=True)
    
    plt.suptitle(
        "Model Accuracy vs Confidence Calibration (No-Context Wrong Questions)\n" +
        "Recovery performance and confidence miscalibration patterns",
        fontsize=16,
        fontweight='bold',
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = output_dir / "accuracy_with_hallucination_no_context_wrong.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✅ Plot saved: {output_path}")


if __name__ == "__main__":
    main()
