"""
Plot 3: Unsupported Claims Distribution

Histogram showing the distribution of unsupported claims per run, faceted by model.

Insight: Which models make more unsupported claims?
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
    load_hallucination_judgments, count_unsupported_claims, normalize_model_name
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'rag_analysis' / 'output'
PLOT_DIR = Path(__file__).resolve().parent


def main():
    """Generate unsupported claims distribution plot."""
    records = load_hallucination_judgments(OUTPUT_DIR)
    
    # Group by model
    model_unsupported = defaultdict(list)
    
    for rec in records:
        model = normalize_model_name(rec.get('model', ''))
        judgment = rec.get('parsed_judgment', {})
        unsupported = count_unsupported_claims(judgment)
        model_unsupported[model].append(unsupported)
    
    models = sorted(model_unsupported.keys())
    n_models = len(models)
    
    # Create subplots
    fig, axes = plt.subplots(n_models, 1, figsize=(12, 3 * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]
    
    max_unsupported = max(max(vals) for vals in model_unsupported.values())
    bins = range(0, max_unsupported + 2)
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_models))
    
    for i, (model, ax) in enumerate(zip(models, axes)):
        unsupported = model_unsupported[model]
        
        ax.hist(unsupported, bins=bins, alpha=0.75, color=colors[i], 
               edgecolor='black', linewidth=0.5)
        
        # Add statistics
        mean_val = np.mean(unsupported)
        median_val = np.median(unsupported)
        zero_pct = 100 * sum(1 for u in unsupported if u == 0) / len(unsupported)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='blue', linestyle=':', linewidth=2,
                  label=f'Median: {median_val:.0f}')
        
        # Add text box with statistics (positioned on the left)
        stats_text = f'n = {len(unsupported)}\n'
        stats_text += f'Zero claims: {zero_pct:.1f}%\n'
        stats_text += f'Mean: {mean_val:.2f}\n'
        stats_text += f'Max: {max(unsupported)}'
        
        ax.text(0.02, 0.97, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(model, fontsize=12, fontweight='bold', pad=10)
        # Position legend on upper right to avoid overlap with text box
        ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    axes[-1].set_xlabel('Number of Unsupported Claims per Run', 
                        fontsize=12, fontweight='bold')
    
    fig.suptitle('Distribution of Unsupported Claims by Model', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    output_path = PLOT_DIR / '3_unsupported_claims_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n=== Unsupported Claims Statistics ===")
    for model in models:
        unsupported = model_unsupported[model]
        zero_pct = 100 * sum(1 for u in unsupported if u == 0) / len(unsupported)
        print(f"\n{model}:")
        print(f"  Total runs: {len(unsupported)}")
        print(f"  Runs with 0 unsupported: {zero_pct:.1f}%")
        print(f"  Mean unsupported: {np.mean(unsupported):.2f}")
        print(f"  Median unsupported: {np.median(unsupported):.0f}")
        print(f"  Max unsupported: {max(unsupported)}")


if __name__ == '__main__':
    main()
