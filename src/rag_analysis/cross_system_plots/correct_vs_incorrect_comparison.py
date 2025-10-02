"""
Correct vs Incorrect Answers: Multi-Metric Comparison

Compare various metrics between correct and incorrect answers across all models.
Shows 6 metrics in subplots to understand what differentiates successful from failed runs.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_system_plots.cross_system_utils import (
    load_all_judgments, create_merged_dataset, normalize_model_name,
    has_composition_failure, is_miscalibrated, has_coverage_gap, has_late_hit
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'rag_analysis' / 'output'
PLOT_DIR = Path(__file__).resolve().parent


def get_sufficiency_score(hallucination_judgment):
    """Get sufficiency score from hallucination judgment."""
    if not hallucination_judgment:
        return None
    
    comp_faith = hallucination_judgment.get('composition_and_faithfulness', {})
    score = comp_faith.get('sufficiency_score_est')
    
    return float(score) if score is not None else None


def get_hop_coverage(hallucination_judgment):
    """Get hop coverage from hallucination judgment."""
    if not hallucination_judgment:
        return None
    
    conf_misc = hallucination_judgment.get('confidence_miscalibration', {})
    coverage = conf_misc.get('hop_coverage_est')
    
    return float(coverage) if coverage is not None else None


def main():
    """Generate correct vs incorrect comparison plots."""
    cov_records, qual_records, hall_records = load_all_judgments(OUTPUT_DIR)
    merged = create_merged_dataset(cov_records, qual_records, hall_records)
    
    # Filter to complete records
    complete = [r for r in merged if 'quality' in r and 'hallucination' in r]
    
    # Group by model and correctness
    model_metrics = defaultdict(lambda: {
        'correct': defaultdict(list),
        'incorrect': defaultdict(list)
    })
    
    for rec in complete:
        model = normalize_model_name(rec['model'])
        is_correct = rec.get('is_correct', False)
        category = 'correct' if is_correct else 'incorrect'
        
        # Composition failure
        has_cf = has_composition_failure(rec.get('hallucination', {}))
        model_metrics[model][category]['composition_failure'].append(1 if has_cf else 0)
        
        # Sufficiency score
        suff_score = get_sufficiency_score(rec.get('hallucination', {}))
        if suff_score is not None:
            model_metrics[model][category]['sufficiency'].append(suff_score)
        
        # Hop coverage
        hop_cov = get_hop_coverage(rec.get('hallucination', {}))
        if hop_cov is not None:
            model_metrics[model][category]['hop_coverage'].append(hop_cov)
        
        # Miscalibration
        has_misc = is_miscalibrated(rec.get('hallucination', {}))
        model_metrics[model][category]['miscalibration'].append(1 if has_misc else 0)
        
        # Coverage gap
        has_gap = has_coverage_gap(rec.get('coverage', {}))
        model_metrics[model][category]['coverage_gap'].append(1 if has_gap else 0)
        
        # Late hit
        has_late = has_late_hit(rec.get('coverage', {}))
        model_metrics[model][category]['late_hit'].append(1 if has_late else 0)
    
    # Create figure with 6 subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    axes = axes.flatten()
    
    models = sorted(model_metrics.keys())
    x = np.arange(len(models))
    width = 0.35
    
    # Define metrics and their properties
    metrics = [
        ('composition_failure', 'Composition Failure Rate by Correctness', '% with Composition Failure', 100, True),
        ('sufficiency', 'Average Sufficiency Score by Correctness', 'Average Sufficiency Score', 1, False),
        ('hop_coverage', 'Average Hop Coverage by Correctness', 'Average Hop Coverage', 1, False),
        ('miscalibration', 'Miscalibration Rate by Correctness', 'Miscalibration Rate (%)', 100, True),
        ('coverage_gap', 'Coverage Gap Rate by Correctness', 'Coverage Gap Rate (%)', 100, True),
        ('late_hit', 'Late Hit Rate by Correctness', 'Late Hit Rate (%)', 100, True),
    ]
    
    for idx, (metric_key, title, ylabel, scale, is_rate) in enumerate(metrics):
        ax = axes[idx]
        
        correct_vals = []
        incorrect_vals = []
        
        for model in models:
            # Correct values
            correct_data = model_metrics[model]['correct'][metric_key]
            if correct_data and len(correct_data) > 0:
                correct_vals.append(scale * np.mean(correct_data))
            else:
                # If no data, use 0 but this shouldn't happen
                correct_vals.append(0)
            
            # Incorrect values
            incorrect_data = model_metrics[model]['incorrect'][metric_key]
            if incorrect_data and len(incorrect_data) > 0:
                incorrect_vals.append(scale * np.mean(incorrect_data))
            else:
                # If no data, use 0 but this shouldn't happen
                incorrect_vals.append(0)
        
        # Plot bars - always plot both sets
        bars1 = ax.bar(x - width/2, correct_vals, width, label='Correct', 
                       color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, incorrect_vals, width, label='Incorrect',
                       color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, val in zip(bars1, correct_vals):
            if val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}' if is_rate else f'{val:.2f}',
                       ha='center', va='bottom', fontsize=8)
        
        for bar, val in zip(bars2, incorrect_vals):
            if val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}' if is_rate else f'{val:.2f}',
                       ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        ax.set_xlabel('Model', fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha='right', fontsize=9)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Set appropriate y-limits
        max_val = max(max(correct_vals), max(incorrect_vals))
        if max_val > 0:
            ax.set_ylim(0, max_val * 1.15)
    
    fig.suptitle('Correct vs Incorrect Answers: Multi-Metric Comparison', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    output_path = PLOT_DIR / 'correct_vs_incorrect_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n=== Correct vs Incorrect Comparison Statistics ===\n")
    for model in models:
        correct_count = len(model_metrics[model]['correct']['composition_failure'])
        incorrect_count = len(model_metrics[model]['incorrect']['composition_failure'])
        total = correct_count + incorrect_count
        accuracy = 100 * correct_count / total if total > 0 else 0
        
        print(f"{model}:")
        print(f"  Total: {total} ({correct_count} correct, {incorrect_count} incorrect)")
        print(f"  Accuracy: {accuracy:.1f}%")
        
        for metric_key, _, _, scale, is_rate in metrics:
            correct_data = model_metrics[model]['correct'][metric_key]
            incorrect_data = model_metrics[model]['incorrect'][metric_key]
            
            if correct_data and incorrect_data:
                correct_avg = scale * np.mean(correct_data)
                incorrect_avg = scale * np.mean(incorrect_data)
                
                label = metric_key.replace('_', ' ').title()
                if is_rate:
                    print(f"  {label}: {correct_avg:.1f}% (correct) vs {incorrect_avg:.1f}% (incorrect)")
                else:
                    print(f"  {label}: {correct_avg:.3f} (correct) vs {incorrect_avg:.3f} (incorrect)")
        print()


if __name__ == '__main__':
    main()
