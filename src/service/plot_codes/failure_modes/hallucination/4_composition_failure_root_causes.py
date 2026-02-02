"""
Plot 4: Composition Failure Root Causes (Per Model)

Grouped bar chart showing percentage of composition failures that also have:
- coverage_gap
- carry_drop
- late_hit
- poor_query_quality

6 subplots, one for each model.

Insight: What leads to composition failure?
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
    load_hallucination_judgments, load_coverage_judgments, 
    load_quality_judgments, create_merged_dataset, has_poor_query_quality,
    normalize_model_name
)

OUTPUT_DIR = Path(__file__).resolve().parents[5] / "data" / "results" / "failure_modes"
PLOT_DIR = Path(__file__).resolve().parents[5] / "data" / "plots" / "failure_modes" / "hallucination"


def main():
    """Generate composition failure root causes plot with 6 subplots (one per model)."""
    # Load all judgment types
    hall_records = load_hallucination_judgments(OUTPUT_DIR)
    cov_records = load_coverage_judgments(OUTPUT_DIR)
    qual_records = load_quality_judgments(OUTPUT_DIR)
    
    # Merge datasets
    merged = create_merged_dataset(hall_records, cov_records, qual_records)
    
    # Group by model
    model_data = {}
    
    for rec in merged:
        model = normalize_model_name(rec.get('model', ''))
        
        # Check if it's a composition failure
        if not rec['hallucination'].get('composition_and_faithfulness', {}).get('composition_failure', False):
            continue
        
        if model not in model_data:
            model_data[model] = {
                'total': 0,
                'Coverage Gap': 0,
                'Anchor Carry-Drop': 0,
                'Late Hit': 0,
                'Poor Query Quality': 0
            }
        
        model_data[model]['total'] += 1
        
        cov = rec.get('coverage', {})
        qual = rec.get('quality', {})
        
        if cov:
            if cov.get('retrieval_coverage_gap', {}).get('has_gap', False):
                model_data[model]['Coverage Gap'] += 1
            
            if cov.get('anchor_carry_drop', {}).get('any_carry_drop', False):
                model_data[model]['Anchor Carry-Drop'] += 1
            
            if cov.get('late_hit_per_hop', {}).get('any_late_hit', False):
                model_data[model]['Late Hit'] += 1
        
        if qual and has_poor_query_quality(qual):
            model_data[model]['Poor Query Quality'] += 1
    
    # Sort models
    models = sorted(model_data.keys())
    
    if len(models) == 0:
        print("No composition failures found!")
        return
    
    print(f"\nTotal composition failures: {sum(model_data[m]['total'] for m in models)}")
    
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
    
    causes = ['Coverage Gap', 'Anchor Carry-Drop', 'Late Hit', 'Poor Query Quality']
    colors = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6']
    
    # Plot each model
    for idx, model in enumerate(models):
        ax = axes[idx]
        data = model_data[model]
        total = data['total']
        
        if total == 0:
            ax.text(0.5, 0.5, f'No composition failures\nfor {model}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model, fontsize=12, fontweight='bold')
            continue
        
        # Calculate percentages
        percentages = [100 * data[cause] / total for cause in causes]
        counts = [data[cause] for cause in causes]
        
        # Create bars
        bars = ax.bar(range(len(causes)), percentages, color=colors, alpha=0.8, 
                     edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for i, (bar, val, count) in enumerate(zip(bars, percentages, counts)):
            height = bar.get_height()
            if height > 5:  # Only show label if bar is tall enough
                ax.text(bar.get_x() + bar.get_width()/2., height/2,
                       f'{val:.0f}%\n(n={count})',
                       ha='center', va='center', fontsize=8, fontweight='bold',
                       color='white' if height > 20 else 'black')
        
        # Add reference line at 50%
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.4, linewidth=1)
        
        # Formatting
        ax.set_ylabel('% of Failures', fontsize=9, fontweight='bold')
        ax.set_title(f'{model}\n(Total Failures: n={total})', fontsize=11, fontweight='bold', pad=10)
        ax.set_xticks(range(len(causes)))
        ax.set_xticklabels(causes, rotation=15, ha='right', fontsize=8)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Hide unused subplots
    for idx in range(len(models), len(axes)):
        axes[idx].axis('off')
    
    # Overall title
    fig.suptitle('Composition Failure Root Causes (Per Model)\nWhat leads to composition failure?',
                fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    output_path = PLOT_DIR / '4_composition_failure_root_causes.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n=== Composition Failure Root Causes (Per Model) ===")
    for model in sorted(models):
        data = model_data[model]
        total = data['total']
        print(f"\n{model} (Total failures: {total}):")
        for cause in causes:
            count = data[cause]
            pct = 100 * count / total if total > 0 else 0
            print(f"  {cause}: {count} ({pct:.1f}%)")


if __name__ == '__main__':
    main()
