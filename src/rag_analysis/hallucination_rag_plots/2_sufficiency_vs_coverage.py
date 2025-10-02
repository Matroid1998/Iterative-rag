"""
Plot 2: Sufficiency vs Coverage Scatter with Miscalibration (Per Model)

Scatter plot showing the relationship between sufficiency score and hop coverage,
colored by miscalibration direction, with point size indicating unsupported claims.
6 subplots, one for each model.

Insight: Can we predict miscalibration from sufficiency and coverage scores?
"""
import json
import sys
from pathlib import Path
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
    """Generate sufficiency vs coverage scatter plot with 6 subplots (one per model)."""
    records = load_hallucination_judgments(OUTPUT_DIR)
    
    # Extract data points by model
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
                    'ok': {'suff': [], 'cov': [], 'unsup': []},
                    'underconfident_continue': {'suff': [], 'cov': [], 'unsup': []},
                    'overconfident_finalize': {'suff': [], 'cov': [], 'unsup': []}
                }
            
            unsupported = count_unsupported_claims(judgment)
            model_data[model][direction]['suff'].append(float(suff))
            model_data[model][direction]['cov'].append(float(cov))
            model_data[model][direction]['unsup'].append(unsupported)
    
    # Sort models
    models = sorted(model_data.keys())
    
    if len(models) == 0:
        print("No model data found!")
        return
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
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
    
    # Plot each model
    for idx, model in enumerate(models):
        if idx >= 6:  # Only show first 6 models
            break
        
        ax = axes[idx]
        data_by_direction = model_data[model]
        
        # Plot each direction
        for direction in ['ok', 'underconfident_continue', 'overconfident_finalize']:
            data = data_by_direction[direction]
            if not data['suff']:
                continue
            
            # Scale point sizes (unsupported claims)
            sizes = [max(10, min(100, 10 + u * 15)) for u in data['unsup']]
            
            ax.scatter(data['suff'], data['cov'], 
                      s=sizes, alpha=0.6, 
                      color=colors[direction],
                      label=f"{labels[direction]} (n={len(data['suff'])})",
                      edgecolors='white', linewidth=0.3)
        
        # Add reference lines
        ax.axvline(x=0.6, color='gray', linestyle='--', alpha=0.4, linewidth=1)
        ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.4, linewidth=1)
        
        # Formatting
        ax.set_xlabel('Sufficiency Score', fontsize=9, fontweight='bold')
        ax.set_ylabel('Hop Coverage', fontsize=9, fontweight='bold')
        ax.set_title(model, fontsize=11, fontweight='bold', pad=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3, linestyle='--')
    
    # Hide unused subplots
    for idx in range(len(models), 6):
        axes[idx].axis('off')
    
    # Create a single shared legend at the bottom
    handles, labels_list = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_list, loc='lower center', ncol=3, framealpha=0.95, 
                  fontsize=10, bbox_to_anchor=(0.5, -0.02))
    
    # Overall title
    fig.suptitle('Sufficiency vs Coverage by Miscalibration Direction (Per Model)\n(point size indicates unsupported claims)',
                fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.985])
    output_path = PLOT_DIR / '2_sufficiency_vs_coverage.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n=== Sufficiency vs Coverage Statistics (Per Model) ===")
    for model in sorted(models):
        print(f"\n{model}:")
        data_by_direction = model_data[model]
        for direction, label in labels.items():
            data = data_by_direction[direction]
            if data['suff']:
                print(f"  {label} (n={len(data['suff'])}):")
                print(f"    Avg Sufficiency: {np.mean(data['suff']):.3f}")
                print(f"    Avg Coverage: {np.mean(data['cov']):.3f}")
                print(f"    Avg Unsupported: {np.mean(data['unsup']):.2f}")


if __name__ == '__main__':
    main()
