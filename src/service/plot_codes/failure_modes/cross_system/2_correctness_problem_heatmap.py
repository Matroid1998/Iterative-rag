"""
Plot 2: Correctness vs Problem Type Heatmap

Heatmap showing percentage of incorrect answers that have each problem type,
by model.

Insight: Which models struggle with which specific failure modes?
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_system.cross_system_utils import (
    load_all_judgments, create_merged_dataset, normalize_model_name,
    has_coverage_gap, has_carry_drop, has_late_hit, 
    has_composition_failure, is_miscalibrated
)

OUTPUT_DIR = Path(__file__).resolve().parents[5] / "data" / "results" / "failure_modes"
PLOT_DIR = Path(__file__).resolve().parents[5] / "data" / "plots" / "failure_modes" / "cross_system"


def main():
    """Generate correctness vs problem type heatmap with subplots for each model."""
    cov_records, qual_records, hall_records = load_all_judgments(OUTPUT_DIR)
    merged = create_merged_dataset(cov_records, qual_records, hall_records)
    
    # Group by model
    model_problems = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'count': 0}))
    
    for rec in merged:
        model = normalize_model_name(rec['model'])
        is_correct = rec.get('is_correct', False)
        
        if not is_correct:  # Only analyze incorrect answers
            model_problems[model]['incorrect']['total'] += 1
            
            # Check each problem type
            if has_coverage_gap(rec.get('coverage', {})):
                model_problems[model]['has_gap']['count'] += 1
            
            if has_carry_drop(rec.get('coverage', {})):
                model_problems[model]['carry_drop']['count'] += 1
            
            if has_late_hit(rec.get('coverage', {})):
                model_problems[model]['late_hit']['count'] += 1
            
            if has_composition_failure(rec.get('hallucination', {})):
                model_problems[model]['composition_failure']['count'] += 1
            
            if is_miscalibrated(rec.get('hallucination', {})):
                model_problems[model]['miscalibration']['count'] += 1
    
    # Prepare data for heatmap
    models = sorted(model_problems.keys())
    problem_types = ['has_gap', 'carry_drop', 'late_hit', 'composition_failure', 'miscalibration']
    problem_labels = ['Coverage\nGap', 'Anchor\nCarry-Drop', 'Late\nHit', 'Composition\nFailure', 'Mis-\ncalibration']
    
    # Create figure with dynamic subplots (3 columns)
    n_models = len(models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Create a subplot for each model
    for idx, model in enumerate(models):
        ax = axes[idx]
        
        # Calculate percentages for this model
        total_incorrect = model_problems[model]['incorrect']['total']
        data = []
        counts = []
        
        if total_incorrect == 0:
            data = [0] * len(problem_types)
            counts = [0] * len(problem_types)
        else:
            for ptype in problem_types:
                count = model_problems[model][ptype]['count']
                pct = 100 * count / total_incorrect
                data.append(pct)
                counts.append(count)
        
        # Reshape data for heatmap (1 row x N columns)
        heatmap_data = np.array([data])
        
        # Create heatmap for this model
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
        
        # Set ticks
        ax.set_xticks(np.arange(len(problem_labels)))
        ax.set_yticks([0])
        ax.set_xticklabels(problem_labels, fontsize=10)
        ax.set_yticklabels([model], fontsize=11, fontweight='bold')
        
        # Add values in cells
        for j in range(len(problem_types)):
            value = data[j]
            count = counts[j]
            text_color = 'white' if value > 50 else 'black'
            ax.text(j, 0, f'{value:.1f}%\n(n={count})',
                   ha='center', va='center', color=text_color, 
                   fontsize=9, fontweight='bold')
        
        # Set title with total incorrect count
        ax.set_title(f'{model}\n{total_incorrect} incorrect answers', 
                    fontsize=11, fontweight='bold', pad=10)
    
    # Hide unused subplots
    for idx in range(len(models), len(axes)):
        axes[idx].axis('off')
    
    # Add overall title
    fig.suptitle('Failure Mode Prevalence Among Incorrect Answers by Model\n(% of incorrect answers that exhibit each problem)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add a single colorbar for all subplots
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(axes[0].images[0], cax=cbar_ax)
    cbar.set_label('% of Incorrect Answers with Problem', 
                   fontsize=12, fontweight='bold', rotation=270, labelpad=25)
    
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    output_path = PLOT_DIR / '2_correctness_problem_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n=== Problem Type Prevalence in Incorrect Answers ===")
    for model in models:
        total_incorrect = model_problems[model]['incorrect']['total']
        print(f"\n{model} (n={total_incorrect} incorrect):")
        for ptype, label in zip(problem_types, problem_labels):
            count = model_problems[model][ptype]['count']
            pct = 100 * count / total_incorrect if total_incorrect > 0 else 0
            print(f"  {label.replace(chr(10), ' ')}: {count} ({pct:.1f}%)")


if __name__ == '__main__':
    main()
