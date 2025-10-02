"""
Plot 3: Efficiency-Quality Tradeoff

Scatter plot: X=avg steps per run, Y=accuracy, color=model, size=avg_specificity_score.

Insight: Models taking more steps don't necessarily get better results.
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
    get_avg_steps, get_avg_specificity
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'rag_analysis' / 'output'
PLOT_DIR = Path(__file__).resolve().parent


def main():
    """Generate efficiency-quality tradeoff scatter plot."""
    cov_records, qual_records, hall_records = load_all_judgments(OUTPUT_DIR)
    merged = create_merged_dataset(cov_records, qual_records, hall_records)
    
    # Calculate metrics per model
    model_metrics = defaultdict(lambda: {
        'total': 0, 
        'correct': 0, 
        'steps': [], 
        'specificity': []
    })
    
    for rec in merged:
        if 'quality' not in rec:
            continue
            
        model = normalize_model_name(rec['model'])
        is_correct = rec.get('is_correct', False)
        
        model_metrics[model]['total'] += 1
        if is_correct:
            model_metrics[model]['correct'] += 1
        
        steps = get_avg_steps(rec.get('quality', {}))
        model_metrics[model]['steps'].append(steps)
        
        specificity = get_avg_specificity(rec.get('quality', {}))
        if specificity > 0:
            model_metrics[model]['specificity'].append(specificity)
    
    # Prepare scatter data
    models = []
    accuracies = []
    avg_steps_list = []
    avg_specificity_list = []
    
    for model, metrics in model_metrics.items():
        if metrics['total'] == 0:
            continue
        
        models.append(model)
        accuracies.append(100 * metrics['correct'] / metrics['total'])
        avg_steps_list.append(np.mean(metrics['steps']))
        avg_spec = np.mean(metrics['specificity']) if metrics['specificity'] else 0.5
        avg_specificity_list.append(avg_spec)
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color map for models
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    # Normalize specificity to emphasize differences
    # Map the range [min, max] to a more visible size range
    min_spec = min(avg_specificity_list)
    max_spec = max(avg_specificity_list)
    spec_range = max_spec - min_spec
    
    # Scale sizes more aggressively to emphasize differences
    # Use exponential scaling to make differences more visible
    if spec_range > 0:
        normalized_specs = [(spec - min_spec) / spec_range for spec in avg_specificity_list]
        # Scale from 200 (smallest) to 2000 (largest) with exponential curve
        sizes = [200 + 1800 * (norm_spec ** 2) for norm_spec in normalized_specs]
    else:
        sizes = [1000] * len(avg_specificity_list)
    
    for i, (model, acc, steps, spec, color) in enumerate(
        zip(models, accuracies, avg_steps_list, avg_specificity_list, colors)):
        
        ax.scatter(steps, acc, s=sizes[i], alpha=0.6, 
                  color=color, edgecolors='black', linewidth=2,
                  label=f'{model}', zorder=3)
        
        # Add text label with specificity
        ax.text(steps, acc + 1.5, f'{model}\n{spec:.3f}',
               ha='center', va='bottom', fontsize=8, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_xlabel('Average Steps per Run', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Efficiency-Quality Tradeoff\n(bubble size = avg specificity score)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(left=0)
    ax.set_ylim(0, 105)
    
    # Add legend with specificity info and range
    min_spec = min(avg_specificity_list)
    max_spec = max(avg_specificity_list)
    legend_text = (f"Bubble size represents average query specificity\n"
                   f"(larger = more specific queries)\n"
                   f"Range: {min_spec:.3f} to {max_spec:.3f}")
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    output_path = PLOT_DIR / '3_efficiency_quality_tradeoff.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n=== Efficiency-Quality Tradeoff Statistics ===")
    for model, acc, steps, spec in zip(models, accuracies, avg_steps_list, avg_specificity_list):
        print(f"\n{model}:")
        print(f"  Accuracy: {acc:.1f}%")
        print(f"  Avg Steps: {steps:.2f}")
        print(f"  Avg Specificity: {spec:.3f}")
        print(f"  Efficiency Ratio (Accuracy/Steps): {acc/steps:.2f}")


if __name__ == '__main__':
    main()
