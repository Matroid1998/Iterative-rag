#!/usr/bin/env python3
"""
Plot combined analysis of partial contradictions and distractor latch effects on accuracy.

This script analyzes:
1. Individual effects of contradictions and distractor latch
2. Combined effects (both present, one present, neither present)
3. Per-model breakdown
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


def get_quality_model_entries() -> List[Tuple[Path, Path, str]]:
    """Get list of (quality_file_path, reverified_file_path, display_name) tuples."""
    base = get_base_path()
    quality_dir = base  / "data" / "results" / "failure_modes"
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
        # Extract the base name from quality file
        # Format: responses_MODEL_reverified_quality_judement.jsonl
        stem = quality_file.stem
        
        # Remove quality_judement suffix
        if stem.endswith("_quality_judement"):
            stem = stem[:-len("_quality_judement")]
        
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
            print(f"Warning: No reverified file found for {stem}")
            print(f"  Tried: {reverified_file}")
            continue
        
        # Extract model name for display
        model_key = raw_name
        if model_key.startswith("responses_"):
            model_key = model_key[len("responses_"):]
        
        display_name = model_names.get(model_key, model_key)
        entries.append((quality_file, reverified_file, display_name))
    
    return entries


def analyze_contradiction_distractor_effects(quality_file: Path, reverified_file: Path) -> Dict[str, Any]:
    """
    Analyze the combined effects of contradictions and distractor latch.
    
    Returns stats for 4 categories:
    - Neither present
    - Only contradiction
    - Only distractor latch
    - Both present
    """
    stats = {
        'neither': {'correct': 0, 'incorrect': 0},
        'only_contradiction': {'correct': 0, 'incorrect': 0},
        'only_distractor': {'correct': 0, 'incorrect': 0},
        'both': {'correct': 0, 'incorrect': 0},
    }
    
    # First, load is_correct from reverified file
    is_correct_map = {}
    with open(reverified_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                # Question is in raw_response.question
                question = data.get('raw_response', {}).get('question', '') or data.get('question', '')
                is_correct = data.get('is_correct', False)
                is_correct_map[question] = is_correct
            except json.JSONDecodeError as e:
                print(f"  Warning: JSON error in {reverified_file.name} line {line_num}: {e}")
                continue
    
    # Now process quality judgments
    matched = 0
    unmatched = 0
    with open(quality_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Warning: JSON error in {quality_file.name} line {line_num}: {e}")
                continue
                
            question = data.get('question', '')
            
            # Get is_correct from the map
            if question not in is_correct_map:
                unmatched += 1
                continue
            
            matched += 1
            is_correct = is_correct_map[question]
            
            parsed = data.get('parsed_judgment', {})
            
            # Check for contradiction (any step)
            per_step = parsed.get('per_step', [])
            has_contradiction = any(
                step.get('partial_contradiction_with_prev', False)
                for step in per_step
            )
            
            # Check for distractor latch (run level)
            run_level = parsed.get('run_level', {})
            has_distractor = run_level.get('distractor_latch', False)
            
            # Categorize
            if has_contradiction and has_distractor:
                category = 'both'
            elif has_contradiction:
                category = 'only_contradiction'
            elif has_distractor:
                category = 'only_distractor'
            else:
                category = 'neither'
            
            if is_correct:
                stats[category]['correct'] += 1
            else:
                stats[category]['incorrect'] += 1
    
    if unmatched > 0:
        print(f"  Warning: {unmatched} questions could not be matched (matched: {matched})")
    
    return stats


def plot_combined_effects(all_stats: Dict[str, Dict], output_path: Path):
    """Plot the combined effects of contradictions and distractor latch."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Aggregate across all models
    aggregated = {
        'neither': {'correct': 0, 'incorrect': 0},
        'only_contradiction': {'correct': 0, 'incorrect': 0},
        'only_distractor': {'correct': 0, 'incorrect': 0},
        'both': {'correct': 0, 'incorrect': 0},
    }
    
    for model_stats in all_stats.values():
        for category in aggregated.keys():
            aggregated[category]['correct'] += model_stats[category]['correct']
            aggregated[category]['incorrect'] += model_stats[category]['incorrect']
    
    # Plot 1: Overall combined effects (top left)
    ax1 = axes[0, 0]
    
    categories = ['Neither', 'Only\nContradiction', 'Only\nDistractor\nLatch', 'Both']
    category_keys = ['neither', 'only_contradiction', 'only_distractor', 'both']
    
    correct_counts = [aggregated[key]['correct'] for key in category_keys]
    incorrect_counts = [aggregated[key]['incorrect'] for key in category_keys]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, correct_counts, width, label='Correct', color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x + width/2, incorrect_counts, width, label='Incorrect', color='#e74c3c', alpha=0.8)
    
    # Add percentage labels
    for i, (c, inc) in enumerate(zip(correct_counts, incorrect_counts)):
        total = c + inc
        if total > 0:
            acc_pct = (c / total) * 100
            ax1.text(i, max(c, inc) + max(correct_counts + incorrect_counts) * 0.05, 
                    f'{acc_pct:.1f}%', ha='center', fontweight='bold', fontsize=12)
    
    ax1.set_ylabel('Number of Questions', fontsize=12, fontweight='bold')
    ax1.set_title('Combined Effects: Partial Contradiction & Distractor Latch\n(All Models)', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add count labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 50:
                ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                        f'{int(height)}',
                        ha='center', va='center', fontweight='bold', color='white', fontsize=10)
    
    # Plot 2: Accuracy comparison (top right)
    ax2 = axes[0, 1]
    
    accuracies = []
    for key in category_keys:
        c = aggregated[key]['correct']
        total = c + aggregated[key]['incorrect']
        acc = (c / total * 100) if total > 0 else 0
        accuracies.append(acc)
    
    colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
    bars = ax2.barh(categories, accuracies, color=colors, alpha=0.8)
    
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        total = correct_counts[i] + incorrect_counts[i]
        ax2.text(acc + 1, i, f'{acc:.1f}% (n={total})', 
                va='center', fontweight='bold', fontsize=11)
    
    ax2.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy by Category', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlim(0, 100)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.axvline(x=44.7, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='Baseline (44.7%)')
    ax2.legend(fontsize=10)
    
    # Plot 3: Per-model comparison (bottom left)
    ax3 = axes[1, 0]
    
    models = sorted(all_stats.keys())
    
    # Calculate accuracy for "neither" vs "any problem" for each model
    neither_accs = []
    any_problem_accs = []
    
    for model in models:
        stats = all_stats[model]
        
        # Neither
        c_neither = stats['neither']['correct']
        total_neither = c_neither + stats['neither']['incorrect']
        acc_neither = (c_neither / total_neither * 100) if total_neither > 0 else 0
        neither_accs.append(acc_neither)
        
        # Any problem (contradiction or distractor or both)
        c_problem = (stats['only_contradiction']['correct'] + 
                     stats['only_distractor']['correct'] + 
                     stats['both']['correct'])
        total_problem = (stats['only_contradiction']['correct'] + stats['only_contradiction']['incorrect'] +
                        stats['only_distractor']['correct'] + stats['only_distractor']['incorrect'] +
                        stats['both']['correct'] + stats['both']['incorrect'])
        acc_problem = (c_problem / total_problem * 100) if total_problem > 0 else 0
        any_problem_accs.append(acc_problem)
    
    x_models = np.arange(len(models))
    width = 0.35
    
    bars1 = ax3.bar(x_models - width/2, neither_accs, width, label='No Issues', color='#2ecc71', alpha=0.8)
    bars2 = ax3.bar(x_models + width/2, any_problem_accs, width, label='Has Issues', color='#e74c3c', alpha=0.8)
    
    ax3.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Per-Model: Clean vs Problematic Questions', fontsize=14, fontweight='bold', pad=15)
    ax3.set_xticks(x_models)
    ax3.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax3.legend(fontsize=11)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_ylim(0, 100)
    
    # Plot 4: Issue prevalence by model (bottom right)
    ax4 = axes[1, 1]
    
    contradiction_pcts = []
    distractor_pcts = []
    both_pcts = []
    
    for model in models:
        stats = all_stats[model]
        total = sum(stats[cat]['correct'] + stats[cat]['incorrect'] 
                   for cat in ['neither', 'only_contradiction', 'only_distractor', 'both'])
        
        if total > 0:
            contradiction_pcts.append((stats['only_contradiction']['correct'] + 
                                      stats['only_contradiction']['incorrect']) / total * 100)
            distractor_pcts.append((stats['only_distractor']['correct'] + 
                                   stats['only_distractor']['incorrect']) / total * 100)
            both_pcts.append((stats['both']['correct'] + stats['both']['incorrect']) / total * 100)
        else:
            contradiction_pcts.append(0)
            distractor_pcts.append(0)
            both_pcts.append(0)
    
    width = 0.6
    
    bars1 = ax4.bar(x_models, contradiction_pcts, width, label='Only Contradiction', 
                   color='#f39c12', alpha=0.8)
    bars2 = ax4.bar(x_models, distractor_pcts, width, bottom=contradiction_pcts,
                   label='Only Distractor', color='#e67e22', alpha=0.8)
    bars3 = ax4.bar(x_models, both_pcts, width, 
                   bottom=np.array(contradiction_pcts) + np.array(distractor_pcts),
                   label='Both', color='#c0392b', alpha=0.8)
    
    ax4.set_ylabel('Percentage of Questions (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Issue Prevalence by Model', fontsize=14, fontweight='bold', pad=15)
    ax4.set_xticks(x_models)
    ax4.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax4.legend(fontsize=10)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.set_ylim(0, max([c + d + b for c, d, b in zip(contradiction_pcts, distractor_pcts, both_pcts)]) * 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved contradiction & distractor analysis: {output_path}")


def plot_detailed_breakdown(all_stats: Dict[str, Dict], output_path: Path):
    """Plot detailed breakdown showing all 4 categories separately."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Aggregate stats
    aggregated = {
        'neither': {'correct': 0, 'incorrect': 0},
        'only_contradiction': {'correct': 0, 'incorrect': 0},
        'only_distractor': {'correct': 0, 'incorrect': 0},
        'both': {'correct': 0, 'incorrect': 0},
    }
    
    for model_stats in all_stats.values():
        for category in aggregated.keys():
            aggregated[category]['correct'] += model_stats[category]['correct']
            aggregated[category]['incorrect'] += model_stats[category]['incorrect']
    
    # Plot 1: Detailed counts with accuracy
    categories = ['Neither\nIssue', 'Only\nContradiction', 'Only\nDistractor', 'Both\nIssues']
    category_keys = ['neither', 'only_contradiction', 'only_distractor', 'both']
    
    correct = [aggregated[key]['correct'] for key in category_keys]
    incorrect = [aggregated[key]['incorrect'] for key in category_keys]
    totals = [c + i for c, i in zip(correct, incorrect)]
    accuracies = [(c / t * 100) if t > 0 else 0 for c, t in zip(correct, totals)]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, correct, width, label='Correct', color='#27ae60', alpha=0.85)
    bars2 = ax1.bar(x + width/2, incorrect, width, label='Incorrect', color='#c0392b', alpha=0.85)
    
    # Add labels
    for i, (c, inc, acc, total) in enumerate(zip(correct, incorrect, accuracies, totals)):
        # Accuracy on top
        ax1.text(i, max(c, inc) + max(correct + incorrect) * 0.08, 
                f'{acc:.1f}%\n(n={total})', 
                ha='center', va='bottom', fontweight='bold', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
        
        # Counts on bars
        if c > 50:
            ax1.text(i - width/2, c/2, f'{c}', ha='center', va='center', 
                    fontweight='bold', color='white', fontsize=11)
        if inc > 50:
            ax1.text(i + width/2, inc/2, f'{inc}', ha='center', va='center', 
                    fontweight='bold', color='white', fontsize=11)
    
    ax1.set_ylabel('Number of Questions', fontsize=13, fontweight='bold')
    ax1.set_title('Detailed Breakdown: Contradiction & Distractor Latch Effects\n' +
                  'Impact on Question Accuracy', fontsize=15, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=12)
    ax1.legend(fontsize=12, loc='upper right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Plot 2: Relative accuracy comparison
    baseline_acc = accuracies[0]  # "Neither" is the baseline
    
    colors = ['#27ae60', '#f39c12', '#e67e22', '#c0392b']
    bars = ax2.barh(categories, accuracies, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    for i, (bar, acc, total) in enumerate(zip(bars, accuracies, totals)):
        diff = acc - baseline_acc
        sign = '+' if diff > 0 else ''
        ax2.text(acc + 2, i, f'{acc:.1f}% ({sign}{diff:.1f}pp)\nn={total}', 
                va='center', fontweight='bold', fontsize=11)
    
    ax2.axvline(x=baseline_acc, color='green', linestyle='--', alpha=0.7, linewidth=2.5, 
               label=f'Baseline (Neither): {baseline_acc:.1f}%')
    ax2.axvline(x=44.7, color='gray', linestyle=':', alpha=0.5, linewidth=2, 
               label='Overall Avg: 44.7%')
    
    ax2.set_xlabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Accuracy Comparison\n(pp = percentage points difference from baseline)', 
                  fontsize=15, fontweight='bold', pad=20)
    ax2.set_xlim(0, 100)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.legend(fontsize=11, loc='lower right')
    
    # Color-code the y-axis labels
    for i, (label, color) in enumerate(zip(ax2.get_yticklabels(), colors)):
        label.set_color(color)
        label.set_fontweight('bold')
        label.set_fontsize(12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved detailed breakdown: {output_path}")


def main():
    """Main execution function."""
    base = get_base_path()
    output_dir = base / "data" / "plots" / "general"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Analyzing contradiction & distractor latch effects...")
    
    all_stats = {}
    for quality_path, reverified_path, display_name in get_quality_model_entries():
        stats = analyze_contradiction_distractor_effects(quality_path, reverified_path)
        all_stats[display_name] = stats
        print(f"Loaded {display_name}")
    
    print(f"\nTotal models analyzed: {len(all_stats)}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("OVERALL STATISTICS")
    print("="*70)
    
    aggregated = {
        'neither': {'correct': 0, 'incorrect': 0},
        'only_contradiction': {'correct': 0, 'incorrect': 0},
        'only_distractor': {'correct': 0, 'incorrect': 0},
        'both': {'correct': 0, 'incorrect': 0},
    }
    
    for model_stats in all_stats.values():
        for category in aggregated.keys():
            aggregated[category]['correct'] += model_stats[category]['correct']
            aggregated[category]['incorrect'] += model_stats[category]['incorrect']
    
    print("\nCategory Breakdown:")
    for category, label in [('neither', 'Neither Issue'), 
                           ('only_contradiction', 'Only Contradiction'),
                           ('only_distractor', 'Only Distractor'),
                           ('both', 'Both Issues')]:
        c = aggregated[category]['correct']
        inc = aggregated[category]['incorrect']
        total = c + inc
        acc = (c / total * 100) if total > 0 else 0
        print(f"  {label:20s}: {acc:5.1f}% accuracy ({c:4d} correct / {total:4d} total)")
    
    # Generate plots
    print("\nGenerating plots...")
    
    plot_combined_effects(
        all_stats,
        output_dir / "contradiction_distractor_combined_effects.png"
    )
    
    plot_detailed_breakdown(
        all_stats,
        output_dir / "contradiction_distractor_detailed_breakdown.png"
    )
    
    print("\n✅ All contradiction & distractor analysis plots generated successfully!")


if __name__ == "__main__":
    main()
