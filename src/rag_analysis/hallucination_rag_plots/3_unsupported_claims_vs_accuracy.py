"""
Plot 3: Unsupported Claims Impact on Accuracy

Shows how the number of unsupported claims affects accuracy.
X-axis: Number of unsupported claims
Y-axis: Accuracy percentage
Only includes runs with >= 2 steps.

Insight: How much do unsupported claims hurt performance?
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
    load_all_judgments, create_merged_dataset, normalize_model_name
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'rag_analysis' / 'output'
PLOT_DIR = Path(__file__).resolve().parent


def count_unsupported_claims(hallucination_judgment):
    """Count unsupported claims in the judgment."""
    if not hallucination_judgment:
        return 0
    
    comp_faith = hallucination_judgment.get('composition_and_faithfulness', {})
    unsupported = comp_faith.get('unsupported_claims', [])
    
    count = 0
    for claim in unsupported:
        if not claim.get('is_supported', True):
            count += 1
    
    return count


def get_num_steps(quality_judgment):
    """Get number of steps from quality judgment."""
    if not quality_judgment:
        return 0
    
    per_step = quality_judgment.get('per_step', [])
    return len(per_step)


def main():
    """Generate unsupported claims vs accuracy plot."""
    # Load and merge all judgment types
    cov_records, qual_records, hall_records = load_all_judgments(OUTPUT_DIR)
    merged = create_merged_dataset(cov_records, qual_records, hall_records)
    
    # Filter to records with all data and >= 2 steps
    complete = [r for r in merged 
                if 'hallucination' in r and 'quality' in r 
                and get_num_steps(r.get('quality', {})) >= 2]
    
    print(f"Total records with >= 2 steps: {len(complete)}")
    
    # Group by model and unsupported claims count
    # Structure: {model: {unsupported_count: {'correct': count, 'total': count}}}
    model_unsupported_accuracy = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
    
    for rec in complete:
        model = normalize_model_name(rec.get('model', ''))
        is_correct = rec.get('is_correct', False)
        unsupported = count_unsupported_claims(rec.get('hallucination', {}))
        
        model_unsupported_accuracy[model][unsupported]['total'] += 1
        if is_correct:
            model_unsupported_accuracy[model][unsupported]['correct'] += 1
    
    # Sort models
    models = sorted(model_unsupported_accuracy.keys())
    
    if len(models) == 0:
        print("No model data found!")
        return
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.arange(len(models)))
    
    # Plot each model
    for idx, model in enumerate(models):
        if idx >= 6:  # Only show first 6 models
            break
        
        ax = axes[idx]
        unsupported_accuracy = model_unsupported_accuracy[model]
        
        # Get sorted unsupported claim counts
        unsupported_counts = sorted(unsupported_accuracy.keys())
        
        if not unsupported_counts:
            ax.text(0.5, 0.5, f'No data for {model}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model, fontsize=12, fontweight='bold')
            continue
        
        # Calculate accuracy for each count
        x_values = []
        y_values = []
        sizes = []  # Size based on number of samples
        
        for count in unsupported_counts:
            data = unsupported_accuracy[count]
            total = data['total']
            correct = data['correct']
            accuracy = 100 * correct / total if total > 0 else 0
            
            x_values.append(count)
            y_values.append(accuracy)
            sizes.append(total)
        
        # Create scatter plot with line
        # Scale point sizes with more dramatic differences
        min_size = 30
        max_size = 800
        if len(sizes) > 1 and max(sizes) > min(sizes):
            # Use exponential scaling for more dramatic size differences
            normalized = [(s - min(sizes)) / (max(sizes) - min(sizes)) for s in sizes]
            scaled_sizes = [min_size + (norm ** 1.5) * (max_size - min_size) 
                           for norm in normalized]
        else:
            scaled_sizes = [300] * len(sizes)
        
        ax.scatter(x_values, y_values, s=scaled_sizes, alpha=0.6, 
                  color=colors[idx], edgecolors='black', linewidth=1, zorder=3)
        ax.plot(x_values, y_values, color=colors[idx], linewidth=2, 
               alpha=0.7, zorder=2)
        
        # Add count labels on points
        for x, y, n in zip(x_values, y_values, sizes):
            ax.text(x, y + 2, f'n={n}', ha='center', va='bottom', 
                   fontsize=8, style='italic')
        
        # Calculate correlation if enough data points
        if len(x_values) > 2:
            corr = np.corrcoef(x_values, y_values)[0, 1]
            ax.text(0.95, 0.05, f'Correlation: {corr:.3f}', 
                   transform=ax.transAxes, ha='right', va='bottom',
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Formatting
        ax.set_xlabel('Number of Unsupported Claims', fontsize=10, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
        ax.set_title(model, fontsize=12, fontweight='bold', pad=10)
        ax.set_ylim(-5, 105)
        ax.grid(alpha=0.3, linestyle='--')
        
        # Add reference line at 50% accuracy
        ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # Hide unused subplots
    for idx in range(len(models), 6):
        axes[idx].axis('off')
    
    # Overall title
    fig.suptitle('Impact of Unsupported Claims on Accuracy (Runs with ≥2 Steps)\n(point size indicates sample size)',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.985])
    output_path = PLOT_DIR / '3_unsupported_claims_vs_accuracy.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n=== Unsupported Claims vs Accuracy Statistics ===")
    for model in sorted(models):
        print(f"\n{model}:")
        unsupported_accuracy = model_unsupported_accuracy[model]
        unsupported_counts = sorted(unsupported_accuracy.keys())
        
        for count in unsupported_counts:
            data = unsupported_accuracy[count]
            total = data['total']
            correct = data['correct']
            accuracy = 100 * correct / total if total > 0 else 0
            print(f"  {count} unsupported claims (n={total}): Accuracy={accuracy:.1f}%")
        
        # Calculate overall stats
        total_correct = sum(d['correct'] for d in unsupported_accuracy.values())
        total_runs = sum(d['total'] for d in unsupported_accuracy.values())
        overall_accuracy = 100 * total_correct / total_runs if total_runs > 0 else 0
        print(f"  Overall (n={total_runs}): Accuracy={overall_accuracy:.1f}%")


if __name__ == '__main__':
    main()
