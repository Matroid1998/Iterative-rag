#!/usr/bin/env python3
"""
Plot query quality and partial contradiction analysis.

This script analyzes:
1. Partial contradiction impact on accuracy
2. Query quality metrics (vague, over_broad, compound, off_topic, anchored)
3. Specificity and on-topic scores vs accuracy
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np


def get_base_path() -> Path:
    """Get the base path for the project."""
    return Path(__file__).resolve().parents[2]


def load_quality_judgments(quality_file: Path, reverified_file: Path) -> Dict[str, Any]:
    """Load quality judgments from JSONL file and match with reverified data for is_correct."""
    judgments = {}
    
    # First load is_correct from reverified file
    is_correct_map = {}
    with open(reverified_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                question = data.get('raw_response', {}).get('question', '') or data.get('question', '')
                is_correct = data.get('is_correct', False)
                is_correct_map[question] = is_correct
            except json.JSONDecodeError:
                continue
    
    # Now load quality judgments and match with is_correct
    with open(quality_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            question = data.get('question', '')
            
            # Only include if we have matching is_correct data
            if question in is_correct_map:
                judgments[question] = {
                    'is_correct': is_correct_map[question],
                    'parsed_judgment': data.get('parsed_judgment', {}),
                }
    
    return judgments


def get_quality_model_entries() -> List[Tuple[Path, Path, str]]:
    """Get list of (quality_file_path, reverified_file_path, display_name) tuples."""
    base = get_base_path()
    quality_dir = base / "src" / "rag_analysis" / "output"
    reverified_dir = base / "src" / "responses_reverified"
    
    # Model display names mapping
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
    quality_files = list(quality_dir.glob("*quality_judement.jsonl")) + list(quality_dir.glob("*quality_judgement.jsonl"))
    for quality_file in sorted(quality_files):
        # Extract the base name from quality file
        stem = quality_file.stem
        
        # Remove quality_judement suffix
        if stem.endswith("_quality_judement"):
            stem = stem[:-len("_quality_judement")]
        elif stem.endswith("_quality_judgement"):
            stem = stem[:-len("_quality_judgement")]
        
        # Remove leading 2_ if present
        if stem.startswith("2_"):
            stem = stem[2:]
        
        # Remove _reverified suffix to get the raw model name
        raw_name = stem
        if stem.endswith("_reverified"):
            raw_name = stem[:-len("_reverified")]
        
        # Construct reverified file path
        reverified_file = reverified_dir / f"{stem}.jsonl"
        
        if not reverified_file.exists():
            # Try with _reverified suffix
            reverified_file = reverified_dir / f"{stem}_reverified.jsonl"
            
        if not reverified_file.exists():
            continue
        
        # Extract model name for display
        model_key = raw_name
        if model_key.startswith("responses_"):
            model_key = model_key[len("responses_"):]
        
        display_name = model_names.get(model_key, model_key)
        entries.append((quality_file, reverified_file, display_name))
    
    return entries


def analyze_partial_contradictions(all_judgments: Dict[str, Dict[str, Any]]) -> Dict[str, Dict]:
    """
    Analyze the impact of partial contradictions on accuracy.
    
    Returns:
        Dict with keys 'with_contradiction', 'without_contradiction'
        Each containing correct/incorrect counts and accuracy.
    """
    stats = {
        'with_contradiction': {'correct': 0, 'incorrect': 0},
        'without_contradiction': {'correct': 0, 'incorrect': 0},
        'by_step': defaultdict(lambda: {'with_contradiction': 0, 'without_contradiction': 0, 
                                        'correct_with': 0, 'correct_without': 0})
    }
    
    for model_name, judgments in all_judgments.items():
        for question, data in judgments.items():
            is_correct = data['is_correct']
            parsed = data['parsed_judgment']
            per_step = parsed.get('per_step', [])
            
            # Check if any step has partial contradiction
            has_contradiction = any(
                step.get('partial_contradiction_with_prev', False)
                for step in per_step
            )
            
            if has_contradiction:
                if is_correct:
                    stats['with_contradiction']['correct'] += 1
                else:
                    stats['with_contradiction']['incorrect'] += 1
            else:
                if is_correct:
                    stats['without_contradiction']['correct'] += 1
                else:
                    stats['without_contradiction']['incorrect'] += 1
            
            # Per-step analysis
            for step_data in per_step:
                step_num = step_data.get('step', 0)
                has_step_contradiction = step_data.get('partial_contradiction_with_prev', False)
                
                if has_step_contradiction:
                    stats['by_step'][step_num]['with_contradiction'] += 1
                    if is_correct:
                        stats['by_step'][step_num]['correct_with'] += 1
                else:
                    stats['by_step'][step_num]['without_contradiction'] += 1
                    if is_correct:
                        stats['by_step'][step_num]['correct_without'] += 1
    
    return stats


def analyze_query_quality(all_judgments: Dict[str, Dict[str, Any]]) -> Dict[str, Dict]:
    """
    Analyze the relationship between query quality metrics and accuracy.
    
    Returns dict with quality metrics and their impact on accuracy.
    """
    quality_metrics = ['vague', 'over_broad', 'off_topic', 'fusion']
    
    stats = {
        metric: {'with_flag': {'correct': 0, 'incorrect': 0},
                'without_flag': {'correct': 0, 'incorrect': 0}}
        for metric in quality_metrics
    }
    
    # Collect specificity and on-topic scores
    scores_data = {
        'specificity': {'scores': [], 'correct': []},
        'on_topic': {'scores': [], 'correct': []},
    }
    
    for model_name, judgments in all_judgments.items():
        for question, data in judgments.items():
            is_correct = data['is_correct']
            parsed = data['parsed_judgment']
            per_step = parsed.get('per_step', [])
            
            for step_data in per_step:
                qc = step_data.get('query_quality', {})
                
                # Boolean flags
                for metric in quality_metrics:
                    if metric == 'fusion':
                        has_flag = step_data.get('fusion', False)
                    else:
                        has_flag = qc.get(metric, False)
                    
                    if has_flag:
                        if is_correct:
                            stats[metric]['with_flag']['correct'] += 1
                        else:
                            stats[metric]['with_flag']['incorrect'] += 1
                    else:
                        if is_correct:
                            stats[metric]['without_flag']['correct'] += 1
                        else:
                            stats[metric]['without_flag']['incorrect'] += 1
                
                # Numeric scores
                spec_score = qc.get('specificity_score')
                if spec_score is not None:
                    scores_data['specificity']['scores'].append(spec_score)
                    scores_data['specificity']['correct'].append(1 if is_correct else 0)
                
                topic_score = qc.get('on_topic_score')
                if topic_score is not None:
                    scores_data['on_topic']['scores'].append(topic_score)
                    scores_data['on_topic']['correct'].append(1 if is_correct else 0)
    
    return stats, scores_data


def plot_partial_contradiction_impact(stats: Dict, output_path: Path):
    """Plot the impact of partial contradictions on accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Overall impact
    categories = ['Without\nContradiction', 'With\nContradiction']
    correct = [
        stats['without_contradiction']['correct'],
        stats['with_contradiction']['correct']
    ]
    incorrect = [
        stats['without_contradiction']['incorrect'],
        stats['with_contradiction']['incorrect']
    ]
    
    total_without = correct[0] + incorrect[0]
    total_with = correct[1] + incorrect[1]
    acc_without = (correct[0] / total_without * 100) if total_without > 0 else 0
    acc_with = (correct[1] / total_with * 100) if total_with > 0 else 0
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, correct, width, label='Correct', color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x + width/2, incorrect, width, label='Incorrect', color='#e74c3c', alpha=0.8)
    
    # Add percentage labels
    for i, (c, inc) in enumerate(zip(correct, incorrect)):
        total = c + inc
        if total > 0:
            acc_pct = (c / total) * 100
            ax1.text(i, max(c, inc) + max(correct + incorrect) * 0.03, 
                    f'{acc_pct:.1f}%', ha='center', fontweight='bold', fontsize=12)
    
    ax1.set_xlabel('Partial Contradiction Status', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Questions', fontsize=12, fontweight='bold')
    ax1.set_title('Impact of Partial Contradictions on Accuracy\n(All Questions)', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add count labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                        f'{int(height)}',
                        ha='center', va='center', fontweight='bold', color='white', fontsize=10)
    
    # Plot 2: Accuracy by step
    by_step = stats['by_step']
    steps = sorted(by_step.keys())
    
    acc_with = []
    acc_without = []
    
    for step in steps:
        data = by_step[step]
        total_with = data['with_contradiction']
        total_without = data['without_contradiction']
        
        acc_w = (data['correct_with'] / total_with * 100) if total_with > 0 else None
        acc_wo = (data['correct_without'] / total_without * 100) if total_without > 0 else None
        
        acc_with.append(acc_w)
        acc_without.append(acc_wo)
    
    x_steps = np.arange(len(steps))
    
    # Plot lines
    valid_without = [(x, y) for x, y in zip(x_steps, acc_without) if y is not None]
    valid_with = [(x, y) for x, y in zip(x_steps, acc_with) if y is not None]
    
    if valid_without:
        x_wo, y_wo = zip(*valid_without)
        ax2.plot(x_wo, y_wo, 'o-', color='#2ecc71', linewidth=3, markersize=10,
                markerfacecolor='white', markeredgewidth=2.5, markeredgecolor='#2ecc71',
                label='Without Contradiction', zorder=10)
    
    if valid_with:
        x_w, y_w = zip(*valid_with)
        ax2.plot(x_w, y_w, 's--', color='#e74c3c', linewidth=3, markersize=10,
                markerfacecolor='white', markeredgewidth=2.5, markeredgecolor='#e74c3c',
                label='With Contradiction', zorder=9)
    
    ax2.set_xlabel('Retrieval Step', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy by Step: Contradiction vs No Contradiction',
                  fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x_steps)
    ax2.set_xticklabels(steps)
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=11, loc='lower right')
    ax2.grid(axis='both', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved partial contradiction analysis: {output_path}")


def plot_query_quality_flags(stats: Dict, output_path: Path):
    """Plot the impact of query quality boolean flags on accuracy."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    quality_metrics = ['vague', 'over_broad', 'off_topic', 'fusion']
    metric_labels = {
        'vague': 'Vague Query',
        'over_broad': 'Over-Broad Query',
        'off_topic': 'Off-Topic Query',
        'fusion': 'Fusion',
    }
    
    for idx, metric in enumerate(quality_metrics):
        ax = axes[idx]
        data = stats[metric]
        
        with_flag_correct = data['with_flag']['correct']
        with_flag_incorrect = data['with_flag']['incorrect']
        without_flag_correct = data['without_flag']['correct']
        without_flag_incorrect = data['without_flag']['incorrect']
        
        # Calculate accuracies
        total_without = without_flag_correct + without_flag_incorrect
        total_with = with_flag_correct + with_flag_incorrect
        acc_without = (without_flag_correct / total_without * 100) if total_without > 0 else 0
        acc_with = (with_flag_correct / total_with * 100) if total_with > 0 else 0
        
        categories = [f'Without\n{metric_labels[metric]}', f'With\n{metric_labels[metric]}']
        accuracies = [acc_without, acc_with]
        counts = [total_without, total_with]
        
        x = np.arange(len(categories))
        
        # Choose colors based on which is better
        colors = ['#2ecc71' if acc_without > acc_with else '#e74c3c',
                 '#2ecc71' if acc_with > acc_without else '#e74c3c']
        
        bars = ax.bar(x, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add percentage and count labels on top of bars
        for i, (bar, count, acc) in enumerate(zip(bars, counts, accuracies)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2, 
                   f'{acc:.1f}%\n(n={count})',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_labels[metric]} Impact', fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        

    
    # Hide the 6th subplot

    
    plt.suptitle('Query Quality Flags Impact on Accuracy', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved query quality flags analysis: {output_path}")


def plot_quality_scores(scores_data: Dict, output_path: Path):
    """Plot specificity and on-topic scores vs accuracy."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Specificity score scatter
    ax1 = axes[0, 0]
    spec_scores = np.array(scores_data['specificity']['scores'])
    spec_correct = np.array(scores_data['specificity']['correct'])
    
    # Scatter plot with jitter
    jitter_amount = 0.02
    jittered_scores = spec_scores + np.random.normal(0, jitter_amount, len(spec_scores))
    
    correct_mask = spec_correct == 1
    ax1.scatter(jittered_scores[correct_mask], spec_correct[correct_mask], 
               alpha=0.3, s=20, color='#2ecc71', label='Correct')
    ax1.scatter(jittered_scores[~correct_mask], spec_correct[~correct_mask], 
               alpha=0.3, s=20, color='#e74c3c', label='Incorrect')
    
    ax1.set_xlabel('Specificity Score', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Outcome (0=Incorrect, 1=Correct)', fontsize=12, fontweight='bold')
    ax1.set_title('Specificity Score vs Accuracy', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylim(-0.1, 1.1)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3, linestyle='--')
    
    # Specificity score binned
    ax2 = axes[0, 1]
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
    
    binned_acc = []
    binned_counts = []
    
    for i in range(len(bins) - 1):
        mask = (spec_scores >= bins[i]) & (spec_scores < bins[i+1])
        if i == len(bins) - 2:  # Last bin includes 1.0
            mask = (spec_scores >= bins[i]) & (spec_scores <= bins[i+1])
        
        if mask.sum() > 0:
            acc = spec_correct[mask].mean() * 100
            binned_acc.append(acc)
            binned_counts.append(mask.sum())
        else:
            binned_acc.append(0)
            binned_counts.append(0)
    
    x = np.arange(len(bin_labels))
    bars = ax2.bar(x, binned_acc, color='#3498db', alpha=0.8)
    
    # Add percentage and count labels
    for i, (bar, count) in enumerate(zip(bars, binned_counts)):
        height = bar.get_height()
        if count > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.1f}%\n(n={count})',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax2.set_xlabel('Specificity Score Range', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy by Specificity Score Bin', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bin_labels, rotation=45)
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # On-topic score scatter
    ax3 = axes[1, 0]
    topic_scores = np.array(scores_data['on_topic']['scores'])
    topic_correct = np.array(scores_data['on_topic']['correct'])
    
    jittered_topic = topic_scores + np.random.normal(0, jitter_amount, len(topic_scores))
    
    correct_mask_topic = topic_correct == 1
    ax3.scatter(jittered_topic[correct_mask_topic], topic_correct[correct_mask_topic],
               alpha=0.3, s=20, color='#2ecc71', label='Correct')
    ax3.scatter(jittered_topic[~correct_mask_topic], topic_correct[~correct_mask_topic],
               alpha=0.3, s=20, color='#e74c3c', label='Incorrect')
    
    ax3.set_xlabel('On-Topic Score', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Outcome (0=Incorrect, 1=Correct)', fontsize=12, fontweight='bold')
    ax3.set_title('On-Topic Score vs Accuracy', fontsize=14, fontweight='bold', pad=15)
    ax3.set_ylim(-0.1, 1.1)
    ax3.legend(fontsize=11)
    ax3.grid(alpha=0.3, linestyle='--')
    
    # On-topic score binned
    ax4 = axes[1, 1]
    binned_acc_topic = []
    binned_counts_topic = []
    
    for i in range(len(bins) - 1):
        mask = (topic_scores >= bins[i]) & (topic_scores < bins[i+1])
        if i == len(bins) - 2:
            mask = (topic_scores >= bins[i]) & (topic_scores <= bins[i+1])
        
        if mask.sum() > 0:
            acc = topic_correct[mask].mean() * 100
            binned_acc_topic.append(acc)
            binned_counts_topic.append(mask.sum())
        else:
            binned_acc_topic.append(0)
            binned_counts_topic.append(0)
    
    bars2 = ax4.bar(x, binned_acc_topic, color='#9b59b6', alpha=0.8)
    
    for i, (bar, count) in enumerate(zip(bars2, binned_counts_topic)):
        height = bar.get_height()
        if count > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.1f}%\n(n={count})',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax4.set_xlabel('On-Topic Score Range', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Accuracy by On-Topic Score Bin', fontsize=14, fontweight='bold', pad=15)
    ax4.set_xticks(x)
    ax4.set_xticklabels(bin_labels, rotation=45)
    ax4.set_ylim(0, 105)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved quality scores analysis: {output_path}")


def main():
    """Main execution function."""
    base = get_base_path()
    output_dir = base / "src" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading quality judgments from all models...")
    
    all_judgments = {}
    for quality_path, reverified_path, display_name in get_quality_model_entries():
        judgments = load_quality_judgments(quality_path, reverified_path)
        all_judgments[display_name] = judgments
        print(f"Loaded {len(judgments)} judgments for {display_name}")
    
    print(f"\nTotal models: {len(all_judgments)}")
    
    # Analyze partial contradictions
    print("\nAnalyzing partial contradictions...")
    contradiction_stats = analyze_partial_contradictions(all_judgments)
    
    print(f"\nPartial Contradiction Summary:")
    print(f"  Without contradiction: {contradiction_stats['without_contradiction']['correct']} correct, "
          f"{contradiction_stats['without_contradiction']['incorrect']} incorrect")
    print(f"  With contradiction: {contradiction_stats['with_contradiction']['correct']} correct, "
          f"{contradiction_stats['with_contradiction']['incorrect']} incorrect")
    
    # Analyze query quality
    print("\nAnalyzing query quality metrics...")
    quality_stats, scores_data = analyze_query_quality(all_judgments)
    
    # Generate plots
    print("\nGenerating plots...")
    
    plot_partial_contradiction_impact(
        contradiction_stats,
        output_dir / "partial_contradiction_impact.png"
    )
    
    plot_query_quality_flags(
        quality_stats,
        output_dir / "query_quality_flags_impact.png"
    )
    
    plot_quality_scores(
        scores_data,
        output_dir / "query_quality_scores.png"
    )
    
    print("\n✅ All query quality analysis plots generated successfully!")


if __name__ == "__main__":
    main()
