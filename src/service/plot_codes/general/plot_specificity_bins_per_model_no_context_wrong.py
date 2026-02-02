#!/usr/bin/env python3
"""
Plot specificity score bins with accuracy breakdown for each model,
but ONLY for questions that were wrong in the no-context baseline.

This helps us understand if query specificity matters more for harder questions.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Set

import matplotlib.pyplot as plt
import numpy as np


def get_base_path() -> Path:
    """Get the base path for the project."""
    return Path(__file__).resolve().parents[2]


def load_no_context_wrong_for_model(model_key: str) -> Set[str]:
    """Load the set of questions that were answered incorrectly in no-context baseline for a specific model."""
    base = get_base_path()
    
    # Try multiple possible file paths
    possible_files = [
        base / "src" / "response-jsonl-without-context" / f"responses_{model_key}.jsonl",
        base / "src" / "response-jsonl-without-context" / f"responses_{model_key}_reverified.jsonl",
        base / "src" / "response-jsonl-without-context" / f"responses_{model_key}-reasoning.jsonl",
        base / "src" / "response-jsonl-without-context" / f"responses_{model_key}-reasoning_reverified.jsonl",
    ]
    
    no_context_file = None
    for file_path in possible_files:
        if file_path.exists():
            no_context_file = file_path
            break
    
    if not no_context_file:
        print(f"  Warning: No-context file not found for {model_key}")
        return set()
    
    wrong_questions = set()
    
    with open(no_context_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Question is in raw.question
                raw = data.get('raw', {})
                question = raw.get('question', '') or data.get('question', '')
                is_correct = data.get('is_correct', False)
                
                if not is_correct and question:
                    wrong_questions.add(question)
            except:
                continue
    
    return wrong_questions


def get_quality_model_entries() -> List[Tuple[Path, Path, str, str]]:
    """Get list of (quality_file_path, reverified_file_path, display_name, model_key) tuples."""
    base = get_base_path()
    quality_dir = base / "src" / "rag_analysis" / "output"
    reverified_dir = base / "src" / "responses_reverified"
    
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
    for quality_file in sorted(quality_dir.glob("*quality_judement.jsonl")):
        stem = quality_file.stem
        
        if stem.endswith("_quality_judement"):
            stem = stem[:-len("_quality_judement")]
        
        if stem.startswith("2_"):
            stem = stem[2:]
        
        raw_name = stem
        if stem.endswith("_reverified"):
            raw_name = stem[:-len("_reverified")]
        
        reverified_file = reverified_dir / f"{stem}.jsonl"
        
        if not reverified_file.exists():
            continue
        
        model_key = raw_name
        if model_key.startswith("responses_"):
            model_key = model_key[len("responses_"):]
        
        display_name = model_names.get(model_key, model_key)
        entries.append((quality_file, reverified_file, display_name, model_key))
    
    return entries


def analyze_specificity_by_model(quality_file: Path, reverified_file: Path, 
                                 no_context_wrong: Set[str]) -> Tuple[Dict[str, Dict], float, int]:
    """
    Analyze specificity scores vs accuracy for a model, ONLY for no-context wrong questions.
    
    Returns:
        - dict with binned data
        - overall question-level accuracy (for filtered questions)
        - total number of filtered questions
    """
    # Load is_correct from reverified file
    is_correct_map = {}
    with open(reverified_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                question = data.get('raw_response', {}).get('question', '') or data.get('question', '')
                is_correct = data.get('is_correct', False)
                
                # Only include if this question was wrong in no-context
                if question in no_context_wrong:
                    is_correct_map[question] = is_correct
            except:
                continue
    
    # Define bins
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
    
    bin_stats = {label: {'correct': 0, 'incorrect': 0, 'step_count': 0} for label in bin_labels}
    
    # Calculate average specificity score per question
    question_specs = {}
    
    with open(quality_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
            except:
                continue
            
            question = data.get('question', '')
            
            if question not in is_correct_map:
                continue
            
            parsed = data.get('parsed_judgment', {})
            per_step = parsed.get('per_step', [])
            
            spec_scores = []
            for step_data in per_step:
                qc = step_data.get('query_quality', {})
                spec_score = qc.get('specificity_score')
                
                if spec_score is not None:
                    spec_scores.append(spec_score)
            
            if spec_scores:
                question_specs[question] = spec_scores
    
    # Bin questions by their average specificity
    for question, spec_scores in question_specs.items():
        avg_spec = np.mean(spec_scores)
        is_correct = is_correct_map[question]
        
        # Find which bin
        for i in range(len(bins) - 1):
            if i == len(bins) - 2:
                if bins[i] <= avg_spec <= bins[i+1]:
                    bin_label = bin_labels[i]
                    break
            else:
                if bins[i] <= avg_spec < bins[i+1]:
                    bin_label = bin_labels[i]
                    break
        else:
            continue
        
        if is_correct:
            bin_stats[bin_label]['correct'] += 1
        else:
            bin_stats[bin_label]['incorrect'] += 1
        
        bin_stats[bin_label]['step_count'] += len(spec_scores)
    
    # Calculate overall accuracy
    total_correct = len([q for q, is_corr in is_correct_map.items() if is_corr and q in question_specs])
    total_questions = len([q for q in question_specs.keys() if q in is_correct_map])
    overall_acc = (total_correct / total_questions * 100) if total_questions > 0 else 0
    
    return bin_stats, overall_acc, total_questions


def plot_specificity_by_model(model_stats: Dict[str, Dict], model_overall_accs: Dict[str, float],
                               model_question_counts: Dict[str, int], output_path: Path):
    """Plot specificity score bins with accuracy for each model."""
    
    # Sort models by overall accuracy
    sorted_models = sorted(model_overall_accs.items(), key=lambda x: x[1], reverse=True)
    model_names = [name for name, _ in sorted_models]
    
    # Create subplots
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    bin_labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
    x = np.arange(len(bin_labels))
    
    for idx, model in enumerate(model_names):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        bin_data = model_stats[model]
        
        # Calculate accuracies, question counts, and step counts
        accuracies = []
        question_counts = []
        step_counts = []
        
        for bin_label in bin_labels:
            correct = bin_data[bin_label]['correct']
            incorrect = bin_data[bin_label]['incorrect']
            total_questions = correct + incorrect
            step_count = bin_data[bin_label]['step_count']
            
            if total_questions > 0:
                acc = (correct / total_questions) * 100
            else:
                acc = 0
            
            accuracies.append(acc)
            question_counts.append(total_questions)
            step_counts.append(step_count)
        
        # Create bar plot
        colors = ['#e74c3c' if acc < 50 else '#f39c12' if acc < 70 else '#2ecc71' 
                 for acc in accuracies]
        bars = ax.bar(x, accuracies, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
        
        # Add labels
        for i, (bar, q_count, s_count, acc) in enumerate(zip(bars, question_counts, step_counts, accuracies)):
            if q_count > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                       f'{acc:.0f}%\n(n={s_count})',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Customize subplot
        total_q = model_question_counts[model]
        ax.set_title(f'{model}\n(Overall: {model_overall_accs[model]:.1f}% on {total_q} questions)',
                    fontsize=10, fontweight='bold', pad=10)
        ax.set_xlabel('Specificity Score', fontsize=9)
        ax.set_ylabel('Accuracy (%)', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=model_overall_accs[model], color='blue', linestyle='--', 
                  linewidth=1, alpha=0.5)
    
    # Hide unused subplots
    for idx in range(len(model_names), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Accuracy by Specificity Score - Each Model\'s Own No-Context Wrong Questions\n' +
                'Questions binned by avg specificity; Accuracy = % recovered with RAG; n = step count',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved specificity by model plot (no-context wrong only): {output_path}")


def main():
    """Main execution function."""
    base = get_base_path()
    output_dir = base / "src" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Analyzing specificity scores by model (each model's own no-context wrong questions)...\n")
    
    model_stats = {}
    model_overall_accs = {}
    model_question_counts = {}
    
    for quality_path, reverified_path, display_name, model_key in get_quality_model_entries():
        # Load this model's no-context wrong questions
        no_context_wrong = load_no_context_wrong_for_model(model_key)
        
        if not no_context_wrong:
            print(f"  {display_name:30s}: No no-context file found - skipping")
            continue
        
        bin_stats, overall_acc, total_q = analyze_specificity_by_model(
            quality_path, reverified_path, no_context_wrong
        )
        
        model_stats[display_name] = bin_stats
        model_overall_accs[display_name] = overall_acc
        model_question_counts[display_name] = total_q
        
        print(f"  {display_name:30s}: {overall_acc:5.1f}% on {total_q:4d}/{len(no_context_wrong):4d} wrong questions")
    
    print(f"\nTotal models analyzed: {len(model_stats)}")
    
    # Generate plot
    print("\nGenerating plot...")
    plot_specificity_by_model(
        model_stats,
        model_overall_accs,
        model_question_counts,
        output_dir / "specificity_bins_per_model_no_context_wrong.png"
    )
    
    print("\n✅ Specificity analysis (each model's no-context wrong) completed!")


if __name__ == "__main__":
    main()
