"""
Plot 6: Evidence Faithfulness Distribution (Per Model)

Violin/histogram plot of faithfulness scores.
6 subplots, one for each model.

Insight: How well does the evidence support the answers?
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hallucination.hall_plot_utils import (
    load_hallucination_judgments, normalize_model_name
)

OUTPUT_DIR = Path(__file__).resolve().parents[5] / "data" / "results" / "failure_modes"
PLOT_DIR = Path(__file__).resolve().parents[5] / "data" / "plots" / "failure_modes" / "hallucination"


def main():
    """Generate faithfulness score distribution plot with 6 subplots (one per model)."""
    records = load_hallucination_judgments(OUTPUT_DIR)
    
    # Group faithfulness scores by model
    model_scores = defaultdict(list)
    
    for rec in records:
        model = normalize_model_name(rec.get('model', ''))
        cf = rec.get('parsed_judgment', {}).get('composition_and_faithfulness', {})
        suff = cf.get('sufficiency_score_est')
        if suff is not None:
            model_scores[model].append(float(suff))
    
    # Sort models
    models = sorted(model_scores.keys())
    
    if len(models) == 0:
        print("No faithfulness scores found!")
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
    
    # Plot each model
    for idx, model in enumerate(models):
        ax = axes[idx]
        scores = model_scores[model]
        
        if not scores:
            ax.text(0.5, 0.5, f'No data for {model}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model, fontsize=12, fontweight='bold')
            continue
        
        # Create violin plot
        parts = ax.violinplot([scores], positions=[0], vert=False, widths=0.7,
                             showmeans=True, showmedians=True)
        
        # Color the violin
        mean_val = np.mean(scores)
        color = '#3498db'  # Blue
        
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1)
        
        # Customize other elements
        parts['cmeans'].set_color('blue')
        parts['cmeans'].set_linewidth(2)
        parts['cmedians'].set_color('red')
        parts['cmedians'].set_linewidth(2)
        
        # Calculate statistics
        median_val = np.median(scores)
        
        # Add text box with statistics
        stats_text = f'n = {len(scores)}\n'
        stats_text += f'Mean: {mean_val:.3f}\n'
        stats_text += f'Median: {median_val:.3f}'
        
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Formatting
        ax.set_xlabel('Faithfulness Score', fontsize=10, fontweight='bold')
        ax.set_title(model, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_yticks([])
        ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Hide unused subplots
    for idx in range(len(models), len(axes)):
        axes[idx].axis('off')
    
    # Overall title
    fig.suptitle('Evidence Faithfulness Score Distribution (Per Model)\nViolet plots showing distribution of faithfulness scores',
                fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    output_path = PLOT_DIR / '6_sufficiency_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n=== Faithfulness Score Statistics (Per Model) ===")
    for model in sorted(models):
        scores = model_scores[model]
        mean_val = np.mean(scores)
        median_val = np.median(scores)
        
        print(f"\n{model}:")
        print(f"  Total runs: {len(scores)}")
        print(f"  Mean: {mean_val:.3f}")
        print(f"  Median: {median_val:.3f}")
        print(f"  Std: {np.std(scores):.3f}")
        print(f"  Min: {min(scores):.3f}")
        print(f"  Max: {max(scores):.3f}")


if __name__ == '__main__':
    main()
