#!/usr/bin/env python3
"""
Plot per-model query quality comparison.

Shows which models produce better quality queries.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np


def get_base_path() -> Path:
    """Get the base path for the project."""
    return Path(__file__).resolve().parents[4]


def get_quality_model_entries() -> List[Tuple[Path, str]]:
    """Get list of (quality_file_path, display_name) tuples."""
    base = get_base_path()
    quality_dir = base  / "data" / "results" / "failure_modes"
    
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
        name = quality_file.stem.replace("_quality_judement", "")
        if name.startswith("2_"):
            name = name[2:]
        if name.startswith("responses_"):
            name = name[len("responses_"):]
        if name.endswith("_reverified"):
            name = name[:-len("_reverified")]
        
        display_name = model_names.get(name, name)
        entries.append((quality_file, display_name))
    
    return entries


def analyze_per_model_quality(quality_path: Path) -> Dict[str, Any]:
    """Analyze query quality metrics for a single model."""
    stats = {
        'total_steps': 0,
        'vague_count': 0,
        'over_broad_count': 0,
        'compound_count': 0,
        'off_topic_count': 0,
        'anchored_count': 0,
        'specificity_scores': [],
        'on_topic_scores': [],
        'contradiction_count': 0,
        'correct_count': 0,
    }
    
    with open(quality_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            is_correct = data.get('is_correct', False)
            if is_correct:
                stats['correct_count'] += 1
            
            parsed = data.get('parsed_judgment', {})
            per_step = parsed.get('per_step', [])
            
            for step_data in per_step:
                stats['total_steps'] += 1
                
                # Boolean flags
                qc = step_data.get('query_quality', {})
                if qc.get('vague', False):
                    stats['vague_count'] += 1
                if qc.get('over_broad', False):
                    stats['over_broad_count'] += 1
                if qc.get('compound', False):
                    stats['compound_count'] += 1
                if qc.get('off_topic', False):
                    stats['off_topic_count'] += 1
                if qc.get('anchored', False):
                    stats['anchored_count'] += 1
                
                # Scores
                spec_score = qc.get('specificity_score')
                if spec_score is not None:
                    stats['specificity_scores'].append(spec_score)
                
                topic_score = qc.get('on_topic_score')
                if topic_score is not None:
                    stats['on_topic_scores'].append(topic_score)
                
                # Contradictions
                if step_data.get('partial_contradiction_with_prev', False):
                    stats['contradiction_count'] += 1
    
    return stats


def plot_model_query_quality_comparison(model_stats: Dict[str, Dict], output_path: Path):
    """Plot comprehensive model comparison of query quality metrics."""
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    models = sorted(model_stats.keys())
    
    # Plot 1: Quality flag percentages
    ax1 = fig.add_subplot(gs[0, :])
    
    flags = ['vague', 'over_broad', 'compound', 'off_topic', 'anchored', 'contradiction']
    flag_labels = ['Vague', 'Over-Broad', 'Compound', 'Off-Topic', 'Anchored', 'Contradiction']
    
    data_matrix = []
    for model in models:
        stats = model_stats[model]
        total = stats['total_steps']
        if total == 0:
            data_matrix.append([0] * len(flags))
            continue
        
        row = [
            (stats['vague_count'] / total) * 100,
            (stats['over_broad_count'] / total) * 100,
            (stats['compound_count'] / total) * 100,
            (stats['off_topic_count'] / total) * 100,
            (stats['anchored_count'] / total) * 100,
            (stats['contradiction_count'] / total) * 100,
        ]
        data_matrix.append(row)
    
    x = np.arange(len(models))
    width = 0.13
    
    colors = ['#e74c3c', '#e67e22', '#f39c12', '#d35400', '#27ae60', '#95a5a6']
    
    for i, (flag, label, color) in enumerate(zip(flags, flag_labels, colors)):
        values = [row[i] for row in data_matrix]
        offset = (i - len(flags)/2 + 0.5) * width
        ax1.bar(x + offset, values, width, label=label, color=color, alpha=0.8)
    
    ax1.set_ylabel('Percentage of Steps (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Query Quality Flags by Model', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend(ncol=6, loc='upper right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max([max(row) for row in data_matrix]) * 1.15)
    
    # Plot 2: Average specificity scores
    ax2 = fig.add_subplot(gs[1, 0])
    
    avg_specificity = []
    for model in models:
        scores = model_stats[model]['specificity_scores']
        avg_specificity.append(np.mean(scores) if scores else 0)
    
    bars = ax2.barh(models, avg_specificity, color='#3498db', alpha=0.8)
    
    for i, (bar, val) in enumerate(zip(bars, avg_specificity)):
        ax2.text(val + 0.01, i, f'{val:.3f}', va='center', fontweight='bold', fontsize=10)
    
    ax2.set_xlabel('Average Specificity Score', fontsize=12, fontweight='bold')
    ax2.set_title('Query Specificity by Model', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlim(0, 1.05)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, linewidth=2, label='High Quality (≥0.8)')
    ax2.legend(fontsize=10)
    
    # Plot 3: Average on-topic scores
    ax3 = fig.add_subplot(gs[1, 1])
    
    avg_on_topic = []
    for model in models:
        scores = model_stats[model]['on_topic_scores']
        avg_on_topic.append(np.mean(scores) if scores else 0)
    
    bars = ax3.barh(models, avg_on_topic, color='#9b59b6', alpha=0.8)
    
    for i, (bar, val) in enumerate(zip(bars, avg_on_topic)):
        ax3.text(val + 0.01, i, f'{val:.3f}', va='center', fontweight='bold', fontsize=10)
    
    ax3.set_xlabel('Average On-Topic Score', fontsize=12, fontweight='bold')
    ax3.set_title('Query On-Topic Alignment by Model', fontsize=14, fontweight='bold', pad=15)
    ax3.set_xlim(0, 1.05)
    ax3.grid(axis='x', alpha=0.3, linestyle='--')
    ax3.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, linewidth=2, label='High Quality (≥0.8)')
    ax3.legend(fontsize=10)
    
    # Plot 4: Correlation between query quality and accuracy
    ax4 = fig.add_subplot(gs[2, 0])
    
    accuracies = []
    for model in models:
        stats = model_stats[model]
        # Get accuracy from the loaded data
        total_questions = stats['correct_count'] + (1186 - stats['correct_count'])  # Approximate
        acc = (stats['correct_count'] / 1186) * 100 if stats['correct_count'] > 0 else 0
        accuracies.append(acc)
    
    # Scatter: specificity vs accuracy
    ax4.scatter(avg_specificity, accuracies, s=200, alpha=0.7, c=range(len(models)), cmap='viridis')
    
    for i, (x, y, model) in enumerate(zip(avg_specificity, accuracies, models)):
        ax4.annotate(model, (x, y), xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.8)
    
    ax4.set_xlabel('Average Specificity Score', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Model Accuracy (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Query Specificity vs Model Accuracy', fontsize=14, fontweight='bold', pad=15)
    ax4.grid(alpha=0.3, linestyle='--')
    
    # Add correlation line
    if len(avg_specificity) > 1:
        z = np.polyfit(avg_specificity, accuracies, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(avg_specificity), max(avg_specificity), 100)
        ax4.plot(x_line, p(x_line), 'r--', alpha=0.5, linewidth=2, label=f'Trend (slope={z[0]:.1f})')
        ax4.legend(fontsize=10)
    
    # Plot 5: Bad query percentage vs accuracy
    ax5 = fig.add_subplot(gs[2, 1])
    
    bad_query_pct = []
    for model in models:
        stats = model_stats[model]
        total = stats['total_steps']
        if total == 0:
            bad_query_pct.append(0)
            continue
        
        bad_count = stats['vague_count'] + stats['off_topic_count'] + stats['over_broad_count']
        bad_query_pct.append((bad_count / total) * 100)
    
    ax5.scatter(bad_query_pct, accuracies, s=200, alpha=0.7, c=range(len(models)), cmap='plasma')
    
    for i, (x, y, model) in enumerate(zip(bad_query_pct, accuracies, models)):
        ax5.annotate(model, (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8)
    
    ax5.set_xlabel('Bad Query Percentage (Vague + Off-Topic + Over-Broad) %', 
                  fontsize=12, fontweight='bold')
    ax5.set_ylabel('Model Accuracy (%)', fontsize=12, fontweight='bold')
    ax5.set_title('Bad Queries vs Model Accuracy (Negative Correlation)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax5.grid(alpha=0.3, linestyle='--')
    
    # Add correlation line
    if len(bad_query_pct) > 1:
        z2 = np.polyfit(bad_query_pct, accuracies, 1)
        p2 = np.poly1d(z2)
        x_line2 = np.linspace(min(bad_query_pct), max(bad_query_pct), 100)
        ax5.plot(x_line2, p2(x_line2), 'r--', alpha=0.5, linewidth=2, 
                label=f'Trend (slope={z2[0]:.1f})')
        ax5.legend(fontsize=10)
    
    plt.suptitle('Model Query Quality Comparison: Comprehensive Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved model query quality comparison: {output_path}")


def main():
    """Main execution function."""
    base = get_base_path()
    output_dir = base / "data" / "plots" / "general"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Analyzing query quality per model...")
    
    model_stats = {}
    for quality_path, display_name in get_quality_model_entries():
        stats = analyze_per_model_quality(quality_path)
        model_stats[display_name] = stats
        
        avg_spec = np.mean(stats['specificity_scores']) if stats['specificity_scores'] else 0
        avg_topic = np.mean(stats['on_topic_scores']) if stats['on_topic_scores'] else 0
        
        print(f"{display_name}:")
        print(f"  Avg Specificity: {avg_spec:.3f}")
        print(f"  Avg On-Topic: {avg_topic:.3f}")
        print(f"  Vague: {stats['vague_count']/stats['total_steps']*100:.1f}%")
        print(f"  Off-Topic: {stats['off_topic_count']/stats['total_steps']*100:.1f}%")
        print()
    
    print("Generating model comparison plot...")
    plot_model_query_quality_comparison(
        model_stats,
        output_dir / "model_query_quality_comparison.png"
    )
    
    print("\n✅ Model query quality comparison plot generated!")


if __name__ == "__main__":
    main()
