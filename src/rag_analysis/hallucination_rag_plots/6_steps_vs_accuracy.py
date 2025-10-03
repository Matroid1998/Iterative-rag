"""
Plot 6: Steps vs Accuracy

Shows the relationship between number of hops and accuracy.
Includes scatter plot with bubble sizes for sample counts and colors for average steps.

Insight: How does question complexity (hops) affect accuracy? Do models take more steps for complex questions?
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hallucination_rag_plots.hall_plot_utils import (
    load_hallucination_judgments, load_coverage_judgments,
    load_quality_judgments, create_merged_dataset, normalize_model_name
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'rag_analysis' / 'output'
PLOT_DIR = Path(__file__).resolve().parent


def main():
    """Generate steps vs accuracy plot."""
    # Load all judgments
    hall_records = load_hallucination_judgments(OUTPUT_DIR)
    cov_records = load_coverage_judgments(OUTPUT_DIR)
    quality_records = load_quality_judgments(OUTPUT_DIR)
    
    # Merge datasets
    merged = create_merged_dataset(hall_records, cov_records, quality_records)
    
    # Group by model and number of hops
    model_hop_stats = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0, 'step_counts': []}))
    
    for rec in merged:
        model = normalize_model_name(rec.get('model', ''))
        is_correct = rec.get('is_correct', False)
        quality = rec.get('quality', {})
        per_step = quality.get('per_step', [])
        num_steps = len(per_step)
        num_hops = rec.get('number_of_hops', 0)
        
        if num_steps > 0 and num_hops > 0:  # Only count runs with steps and hops
            model_hop_stats[model][num_hops]['total'] += 1
            model_hop_stats[model][num_hops]['step_counts'].append(num_steps)
            if is_correct:
                model_hop_stats[model][num_hops]['correct'] += 1
    
    # Create plot with subplots for each model
    models = sorted(model_hop_stats.keys())
    n_models = len(models)
    
    # Use 2 rows of 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        hop_stats = model_hop_stats[model]
        
        hops = []
        accuracies = []
        sizes = []
        avg_steps = []
        
        for hop_count in sorted(hop_stats.keys()):
            stats = hop_stats[hop_count]
            accuracy = 100 * stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            step_counts = stats['step_counts']
            avg_step = np.mean(step_counts) if step_counts else 0
            
            hops.append(hop_count)
            accuracies.append(accuracy)
            sizes.append(stats['total'])
            avg_steps.append(avg_step)
        
        if not hops:
            continue
        
        # Normalize sizes for bubble plot (exponential scaling)
        if len(sizes) > 0:
            size_array = np.array(sizes)
            size_norm = (size_array - size_array.min()) / (size_array.max() - size_array.min() + 1e-10)
            bubble_sizes = 50 + 400 * (size_norm ** 1.5)
        else:
            bubble_sizes = [100] * len(sizes)
        
        # Color by average steps
        step_colors = plt.cm.plasma(np.array(avg_steps) / 6.0)  # Normalize by max ~6 steps
        
        # Scatter plot with bubble sizes and colors
        for i, (hop, acc, bsize, scolor) in enumerate(zip(hops, accuracies, bubble_sizes, step_colors)):
            ax.scatter([hop], [acc], s=bsize, alpha=0.7, 
                      color=scolor, edgecolors='black', linewidth=1.5)
        
        # Add trend line if enough data points
        if len(hops) > 1:
            z = np.polyfit(hops, accuracies, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(hops), max(hops), 100)
            ax.plot(x_line, p(x_line), '--', color='gray', alpha=0.5, linewidth=2)
        
        # Add labels for each point showing average steps
        for hop, acc, size, avg_step in zip(hops, accuracies, sizes, avg_steps):
            ax.annotate(f'n={size}\n{avg_step:.1f} steps', (hop, acc), 
                       textcoords="offset points", xytext=(0, 10),
                       ha='center', fontsize=7, alpha=0.8)
        
        ax.set_xlabel('Number of Hops', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{model}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0, 100)
        
        # Set x-axis to show integer hops
        if hops:
            ax.set_xlim(min(hops) - 0.5, max(hops) + 0.5)
            ax.set_xticks(range(min(hops), max(hops) + 1))
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    # Add colorbar for step counts - outside the plots
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=1, vmax=6))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Average Number of Steps', fontsize=12, fontweight='bold')
    
    plt.suptitle('Relationship Between Question Hops and Accuracy\n(Bubble size = sample count, Color = avg steps)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.subplots_adjust(left=0.05, right=0.90, top=0.96, bottom=0.05, wspace=0.25, hspace=0.25)
    
    output_path = PLOT_DIR / '6_steps_vs_accuracy.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n=== Hops vs Accuracy by Model ===")
    for model in models:
        print(f"\n{model}:")
        hop_stats = model_hop_stats[model]
        for hop_count in sorted(hop_stats.keys()):
            stats = hop_stats[hop_count]
            accuracy = 100 * stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            avg_step = np.mean(stats['step_counts']) if stats['step_counts'] else 0
            print(f"  {hop_count} hops: {accuracy:.1f}% accuracy ({stats['correct']}/{stats['total']}) - Avg {avg_step:.1f} steps")


if __name__ == '__main__':
    main()
