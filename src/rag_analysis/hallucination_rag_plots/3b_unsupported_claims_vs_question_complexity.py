"""
Plot 3b: Unsupported Claims by Question Complexity

Shows the distribution of question complexity (number of hops) for different 
numbers of unsupported claims.
X-axis: Number of unsupported claims
Y-axis: Average number of hops in original questions
Only includes runs with >= 2 steps.

Insight: Do more complex questions lead to more unsupported claims?
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
    """Generate unsupported claims vs question complexity plot."""
    # Load and merge all judgment types
    cov_records, qual_records, hall_records = load_all_judgments(OUTPUT_DIR)
    merged = create_merged_dataset(cov_records, qual_records, hall_records)
    
    # Filter to records with all data and >= 2 steps
    complete = [r for r in merged 
                if 'hallucination' in r and 'quality' in r and 'number_of_hops' in r
                and get_num_steps(r.get('quality', {})) >= 2]
    
    print(f"Total records with >= 2 steps: {len(complete)}")
    
    # Group by model and unsupported claims count
    # Structure: {model: {unsupported_count: [list of hop counts]}}
    model_unsupported_hops = defaultdict(lambda: defaultdict(list))
    
    for rec in complete:
        model = normalize_model_name(rec.get('model', ''))
        unsupported = count_unsupported_claims(rec.get('hallucination', {}))
        num_hops = rec.get('number_of_hops', 0)
        
        model_unsupported_hops[model][unsupported].append(num_hops)
    
    # Sort models
    models = sorted(model_unsupported_hops.keys())
    
    if len(models) == 0:
        print("No model data found!")
        return
    
    # Create figure with 2x3 subplots
    # Calculate grid size (3 columns, enough rows to fit all models)
    num_models = len(models)
    ncols = 3
    nrows = (num_models + ncols - 1) // ncols  # Ceiling division
    
    # Create figure with calculated subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)  # Ensure 2D array
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.arange(len(models)))
    
    # Plot each model
    for idx, model in enumerate(models):
        ax = axes[idx]
        unsupported_hops = model_unsupported_hops[model]
        
        # Get sorted unsupported claim counts
        unsupported_counts = sorted(unsupported_hops.keys())
        
        if not unsupported_counts:
            ax.text(0.5, 0.5, f'No data for {model}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model, fontsize=12, fontweight='bold')
            continue
        
        # Calculate statistics for each count
        x_values = []
        y_mean = []
        y_std = []
        sample_sizes = []
        
        for count in unsupported_counts:
            hop_list = unsupported_hops[count]
            x_values.append(count)
            y_mean.append(np.mean(hop_list))
            y_std.append(np.std(hop_list))
            sample_sizes.append(len(hop_list))
        
        # Create violin plot data for each unsupported count
        violin_data = [unsupported_hops[count] for count in unsupported_counts]
        
        # Create violin plot
        parts = ax.violinplot(violin_data, positions=x_values, widths=0.6,
                             showmeans=True, showmedians=True, showextrema=True)
        
        # Color the violins
        for pc in parts['bodies']:
            pc.set_facecolor(colors[idx])
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(1)
        
        # Style the mean, median, and extrema lines
        for partname in ['cmeans', 'cmedians', 'cmaxes', 'cmins', 'cbars']:
            if partname in parts:
                parts[partname].set_edgecolor('black')
                parts[partname].set_linewidth(1.5)
        
        # Add sample size labels
        for x, n in zip(x_values, sample_sizes):
            ax.text(x, 0.2, f'n={n}', ha='center', va='bottom', 
                   fontsize=8, style='italic')
        
        # Add mean line
        ax.plot(x_values, y_mean, color='red', linewidth=2, 
               marker='D', markersize=6, label='Mean', zorder=10)
        
        # Calculate correlation if enough data points
        if len(x_values) > 2:
            corr = np.corrcoef(x_values, y_mean)[0, 1]
            ax.text(0.95, 0.95, f'Correlation: {corr:.3f}', 
                   transform=ax.transAxes, ha='right', va='top',
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Formatting
        ax.set_xlabel('Number of Unsupported Claims', fontsize=10, fontweight='bold')
        ax.set_ylabel('Number of Hops (Question Complexity)', fontsize=10, fontweight='bold')
        ax.set_title(model, fontsize=12, fontweight='bold', pad=10)
        ax.set_ylim(0.5, 4.5)
        ax.set_yticks([1, 2, 3, 4])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    # Hide unused subplots
    for idx in range(len(models), len(axes)):
        axes[idx].axis('off')
    
    # Overall title
    fig.suptitle('Question Complexity (Number of Hops) by Unsupported Claims (Runs with ≥2 Steps)\n(violin plot shows distribution, red line shows mean)',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.985])
    output_path = PLOT_DIR / '3b_unsupported_claims_vs_question_complexity.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n=== Unsupported Claims vs Question Complexity Statistics ===")
    for model in sorted(models):
        print(f"\n{model}:")
        unsupported_hops = model_unsupported_hops[model]
        unsupported_counts = sorted(unsupported_hops.keys())
        
        for count in unsupported_counts:
            hop_list = unsupported_hops[count]
            mean_hops = np.mean(hop_list)
            median_hops = np.median(hop_list)
            std_hops = np.std(hop_list)
            
            # Count distribution
            hop_dist = {}
            for h in hop_list:
                hop_dist[h] = hop_dist.get(h, 0) + 1
            
            print(f"  {count} unsupported claims (n={len(hop_list)}):")
            print(f"    Mean hops: {mean_hops:.2f}")
            print(f"    Median hops: {median_hops:.1f}")
            print(f"    Std hops: {std_hops:.2f}")
            print(f"    Distribution: {dict(sorted(hop_dist.items()))}")


if __name__ == '__main__':
    main()
