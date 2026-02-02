"""
Plot 8: Coverage vs Confidence Scatter (Per Model)

Scatter plot of hop_coverage_est vs sufficiency_score_est colored by 
miscalibration direction to visualize regimes that drive miscalibration.
6 subplots, one for each model.

Insight: What combinations of coverage and confidence lead to miscalibration?
"""
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hallucination.hall_plot_utils import (
    load_hallucination_judgments, normalize_model_name
)

OUTPUT_DIR = Path(__file__).resolve().parents[5] / "data" / "results" / "failure_modes"
PLOT_DIR = Path(__file__).resolve().parents[5] / "data" / "plots" / "failure_modes" / "hallucination"


def main():
    """Generate coverage vs confidence scatter plot with 6 subplots (one per model)."""
    records = load_hallucination_judgments(OUTPUT_DIR)
    
    # Group data by model
    model_data = {}
    
    for rec in records:
        model = normalize_model_name(rec.get('model', ''))
        judgment = rec.get('parsed_judgment', {})
        cf = judgment.get('composition_and_faithfulness', {})
        cm = judgment.get('confidence_miscalibration', {})
        
        suff = cf.get('sufficiency_score_est')
        cov = cm.get('hop_coverage_est')
        direction = cm.get('direction', 'ok')
        
        if suff is not None and cov is not None:
            if model not in model_data:
                model_data[model] = {
                    'ok': {'cov': [], 'suff': []},
                    'underconfident_continue': {'cov': [], 'suff': []},
                    'overconfident_finalize': {'cov': [], 'suff': []}
                }
            
            model_data[model][direction]['cov'].append(float(cov))
            model_data[model][direction]['suff'].append(float(suff))
    
    # Sort models
    models = sorted(model_data.keys())
    
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
    
    colors = {
        'ok': '#2ecc71',
        'underconfident_continue': '#3498db',
        'overconfident_finalize': '#e74c3c'
    }
    
    labels = {
        'ok': 'OK',
        'underconfident_continue': 'Underconfident',
        'overconfident_finalize': 'Overconfident'
    }
    
    markers = {
        'ok': 'o',
        'underconfident_continue': '^',
        'overconfident_finalize': 's'
    }
    
    # Plot each model
    for idx, model in enumerate(models):
        ax = axes[idx]
        data_by_direction = model_data[model]
        
        # Plot each direction
        for direction in ['ok', 'underconfident_continue', 'overconfident_finalize']:
            data = data_by_direction[direction]
            if not data['cov']:
                continue
            
            ax.scatter(data['cov'], data['suff'],
                      s=20, alpha=0.5,
                      color=colors[direction],
                      marker=markers[direction],
                      label=f"{labels[direction]} ({len(data['cov'])})",
                      edgecolors='white', linewidth=0.2)
        
        # Add threshold lines
        ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.4, linewidth=1)
        ax.axhline(y=0.6, color='gray', linestyle=':', alpha=0.4, linewidth=1)
        
        # Add subtle quadrant shading
        rect1 = Rectangle((0.8, 0.6), 0.2, 0.4, alpha=0.05, facecolor='green')
        ax.add_patch(rect1)
        
        # Formatting
        ax.set_xlabel('Coverage', fontsize=9, fontweight='bold')
        ax.set_ylabel('Sufficiency', fontsize=9, fontweight='bold')
        ax.set_title(model, fontsize=11, fontweight='bold', pad=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3, linestyle='--')
        
        # Add small legend
        if idx == 0:
            ax.legend(loc='lower left', framealpha=0.9, fontsize=7, ncol=1)
    
    # Hide unused subplots
    for idx in range(len(models), len(axes)):
        axes[idx].axis('off')
    
    # Create a single shared legend at the bottom
    handles, labels_list = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_list, loc='lower center', ncol=3, framealpha=0.95, 
                  fontsize=10, bbox_to_anchor=(0.5, -0.02))
    
    # Overall title
    fig.suptitle('Coverage vs Confidence by Miscalibration Direction (Per Model)\nIdentifying Miscalibration Regimes',
                fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.985])
    output_path = PLOT_DIR / '8_coverage_vs_confidence.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n=== Coverage vs Confidence Statistics (Per Model) ===")
    for model in sorted(models):
        print(f"\n{model}:")
        data_by_direction = model_data[model]
        
        for direction, label in labels.items():
            data = data_by_direction[direction]
            if data['cov']:
                print(f"  {label} (n={len(data['cov'])}):")
                print(f"    Avg Coverage: {np.mean(data['cov']):.3f}")
                print(f"    Avg Sufficiency: {np.mean(data['suff']):.3f}")
                
                # Count quadrants
                high_cov_high_suff = sum(1 for c, s in zip(data['cov'], data['suff']) 
                                         if c >= 0.8 and s >= 0.6)
                low_cov_high_suff = sum(1 for c, s in zip(data['cov'], data['suff']) 
                                        if c < 0.8 and s >= 0.6)
                high_cov_low_suff = sum(1 for c, s in zip(data['cov'], data['suff']) 
                                        if c >= 0.8 and s < 0.6)
                low_cov_low_suff = sum(1 for c, s in zip(data['cov'], data['suff']) 
                                       if c < 0.8 and s < 0.6)
                
                total = len(data['cov'])
                print(f"    High Cov & High Suff: {100*high_cov_high_suff/total:.1f}%")
                print(f"    Low Cov & High Suff: {100*low_cov_high_suff/total:.1f}%")
                print(f"    High Cov & Low Suff: {100*high_cov_low_suff/total:.1f}%")
                print(f"    Low Cov & Low Suff: {100*low_cov_low_suff/total:.1f}%")


if __name__ == '__main__':
    main()
