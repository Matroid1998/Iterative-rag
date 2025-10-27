"""
Plot 1: Miscalibration Direction by Hop Count (Per Model)

Stacked bar chart showing miscalibration direction (overconfident/underconfident/ok)
by number of hops in the question, with 6 subplots (one per model).

Insight: Are models overconfident on simple questions and underconfident on complex ones?
How do different models compare?
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hallucination_rag_plots.hall_plot_utils import load_hallucination_judgments, normalize_model_name

OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'rag_analysis' / 'output'
PLOT_DIR = Path(__file__).resolve().parent


def main():
    """Generate miscalibration direction by hop count plot with 6 subplots."""
    records = load_hallucination_judgments(OUTPUT_DIR)
    
    # Group by model, hop count, and direction
    model_hop_direction = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for rec in records:
        model = normalize_model_name(rec.get('model', ''))
        hops = rec.get('number_of_hops', 0)
        if hops == 0:
            continue
            
        cm = rec.get('parsed_judgment', {}).get('confidence_miscalibration', {})
        direction = cm.get('direction', 'ok')
        
        model_hop_direction[model][hops][direction] += 1
    
    # Sort models
    models = sorted(model_hop_direction.keys())
    
    if len(models) == 0:
        print("No model data found!")
        return
    
    # Prepare plot layout
    directions = ['ok', 'underconfident_continue', 'overconfident_finalize']
    direction_labels = ['OK', 'Underconfident', 'Overconfident']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    # Calculate grid size (3 columns, enough rows to fit all models)
    num_models = len(models)
    ncols = 3
    nrows = (num_models + ncols - 1) // ncols  # Ceiling division
    
    # Create figure with calculated subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)  # Ensure 2D array
    axes = axes.flatten()
    
    # Plot each model
    for idx, model in enumerate(models):
        
        ax = axes[idx]
        hop_direction_counts = model_hop_direction[model]
        
        # Get hop counts for this model
        hop_counts = sorted(hop_direction_counts.keys())
        
        if not hop_counts:
            ax.text(0.5, 0.5, f'No data for {model}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model, fontsize=12, fontweight='bold')
            continue
        
        # Prepare data for this model
        data = {d: [] for d in directions}
        for hop in hop_counts:
            for d in directions:
                count = hop_direction_counts[hop].get(d, 0)
                data[d].append(count)
        
        # Create stacked bar chart
        x = np.arange(len(hop_counts))
        width = 0.6
        
        bottom = np.zeros(len(hop_counts))
        
        for i, (direction, label, color) in enumerate(zip(directions, direction_labels, colors)):
            values = data[direction]
            bars = ax.bar(x, values, width, label=label, bottom=bottom, 
                         color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
            
            # Add percentage labels on bars
            for j, (val, bot) in enumerate(zip(values, bottom)):
                if val > 0:
                    total = sum(data[d][j] for d in directions)
                    pct = 100 * val / total
                    if pct > 8:  # Only show label if segment is large enough
                        ax.text(x[j], bot + val/2, f'{pct:.0f}%', 
                               ha='center', va='center', fontsize=8, fontweight='bold')
            
            bottom += values
        
        # Formatting
        ax.set_xlabel('Number of Hops', fontsize=10, fontweight='bold')
        ax.set_ylabel('Count', fontsize=10, fontweight='bold')
        ax.set_title(model, fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{h}' for h in hop_counts])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Hide unused subplots
    for idx in range(len(models), len(axes)):
        axes[idx].axis('off')
    
    # Create a single shared legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, framealpha=0.95, 
              fontsize=11, bbox_to_anchor=(0.5, -0.02))
    
    # Overall title
    fig.suptitle('Miscalibration Direction by Question Complexity (Per Model)\nHow does calibration quality change with task difficulty?',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.985])
    output_path = PLOT_DIR / '1_miscalibration_by_hop.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n=== Miscalibration by Hop Count (Per Model) ===")
    for model in sorted(models):
        print(f"\n{model}:")
        hop_direction_counts = model_hop_direction[model]
        hop_counts = sorted(hop_direction_counts.keys())
        
        for hop in hop_counts:
            total = sum(hop_direction_counts[hop].values())
            print(f"  {hop}-hop (n={total}):")
            for direction, label in zip(directions, direction_labels):
                count = hop_direction_counts[hop].get(direction, 0)
                pct = 100 * count / total if total > 0 else 0
                print(f"    {label}: {count} ({pct:.1f}%)")


if __name__ == '__main__':
    main()
