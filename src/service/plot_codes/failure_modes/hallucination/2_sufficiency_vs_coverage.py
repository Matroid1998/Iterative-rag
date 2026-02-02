"""
Plot 2: Sufficiency vs Coverage Scatter by Correctness (Per Model)

Scatter plot showing the relationship between sufficiency score and hop coverage,
colored by answer correctness (correct/incorrect), with point size indicating unsupported claims.
6 subplots, one for each model.

Insight: How do sufficiency and coverage scores differ between correct and incorrect answers?
"""
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_system.cross_system_utils import (
    load_all_judgments, create_merged_dataset, normalize_model_name
)

OUTPUT_DIR = Path(__file__).resolve().parents[5] / "data" / "results" / "failure_modes"
PLOT_DIR = Path(__file__).resolve().parents[5] / "data" / "plots" / "failure_modes" / "hallucination"


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


def main():
    """Generate sufficiency vs coverage scatter plot with 6 subplots (one per model)."""
    # Load and merge all judgment types
    cov_records, qual_records, hall_records = load_all_judgments(OUTPUT_DIR)
    merged = create_merged_dataset(cov_records, qual_records, hall_records)
    
    # Filter to records with hallucination data
    complete = [r for r in merged if 'hallucination' in r]
    
    # Extract data points by model
    model_data = {}
    
    for rec in complete:
        model = normalize_model_name(rec.get('model', ''))
        judgment = rec.get('hallucination', {})
        cf = judgment.get('composition_and_faithfulness', {})
        cm = judgment.get('confidence_miscalibration', {})
        
        suff = cf.get('sufficiency_score_est')
        cov = cm.get('hop_coverage_est')
        is_correct = rec.get('is_correct', False)
        
        if suff is not None and cov is not None:
            if model not in model_data:
                model_data[model] = {
                    'correct': {'suff': [], 'cov': [], 'unsup': []},
                    'incorrect': {'suff': [], 'cov': [], 'unsup': []}
                }
            
            category = 'correct' if is_correct else 'incorrect'
            unsupported = count_unsupported_claims(judgment)
            model_data[model][category]['suff'].append(float(suff))
            model_data[model][category]['cov'].append(float(cov))
            model_data[model][category]['unsup'].append(unsupported)
    
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
        'correct': '#2ecc71',      # Green for correct
        'incorrect': '#e74c3c'      # Red for incorrect
    }
    
    labels = {
        'correct': 'Correct',
        'incorrect': 'Incorrect'
    }
    
    # Plot each model
    for idx, model in enumerate(models):
        ax = axes[idx]
        data_by_category = model_data[model]
        
        # Plot each category (correct/incorrect)
        for category in ['correct', 'incorrect']:
            data = data_by_category[category]
            if not data['suff']:
                continue
            
            # Scale point sizes (unsupported claims)
            sizes = [max(10, min(100, 10 + u * 15)) for u in data['unsup']]
            
            ax.scatter(data['suff'], data['cov'], 
                      s=sizes, alpha=0.6, 
                      color=colors[category],
                      label=labels[category],
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
    for idx in range(len(models), len(axes)):
        axes[idx].axis('off')
    
    # Create a single shared legend at the bottom
    handles, labels_list = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_list, loc='lower center', ncol=2, framealpha=0.95, 
                  fontsize=11, bbox_to_anchor=(0.5, -0.02))
    
    # Overall title
    fig.suptitle('Sufficiency vs Coverage by Answer Correctness (Per Model)\n(point size indicates unsupported claims)',
                fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.985])
    output_path = PLOT_DIR / '2_sufficiency_vs_coverage.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n=== Sufficiency vs Coverage Statistics by Correctness (Per Model) ===")
    for model in sorted(models):
        print(f"\n{model}:")
        data_by_category = model_data[model]
        for category, label in labels.items():
            data = data_by_category[category]
            if data['suff']:
                print(f"  {label} (n={len(data['suff'])}):")
                print(f"    Avg Sufficiency: {np.mean(data['suff']):.3f}")
                print(f"    Avg Coverage: {np.mean(data['cov']):.3f}")
                print(f"    Avg Unsupported: {np.mean(data['unsup']):.2f}")


if __name__ == '__main__':
    main()
