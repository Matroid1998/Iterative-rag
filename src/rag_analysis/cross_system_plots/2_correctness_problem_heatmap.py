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
from cross_system_plots.cross_system_utils import (
    load_all_judgments, create_merged_dataset, normalize_model_name,
    has_coverage_gap, has_carry_drop, has_late_hit, 
    has_composition_failure, is_miscalibrated
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'rag_analysis' / 'output'
PLOT_DIR = Path(__file__).resolve().parent


def main():
    """Generate correctness vs problem type heatmap."""
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
    
    # Calculate percentages
    heatmap_data = []
    for model in models:
        row = []
        total_incorrect = model_problems[model]['incorrect']['total']
        if total_incorrect == 0:
            row = [0] * len(problem_types)
        else:
            for ptype in problem_types:
                count = model_problems[model][ptype]['count']
                pct = 100 * count / total_incorrect
                row.append(pct)
        heatmap_data.append(row)
    
    heatmap_data = np.array(heatmap_data)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
    
    # Set ticks
    ax.set_xticks(np.arange(len(problem_labels)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(problem_labels, fontsize=11)
    ax.set_yticklabels(models, fontsize=11)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    
    # Add values in cells
    for i in range(len(models)):
        for j in range(len(problem_types)):
            value = heatmap_data[i, j]
            total_incorrect = model_problems[models[i]]['incorrect']['total']
            count = model_problems[models[i]][problem_types[j]]['count']
            
            text_color = 'white' if value > 50 else 'black'
            ax.text(j, i, f'{value:.1f}%\n(n={count})',
                   ha='center', va='center', color=text_color, 
                   fontsize=9, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('% of Incorrect Answers with Problem', 
                   fontsize=12, fontweight='bold', rotation=270, labelpad=25)
    
    ax.set_xlabel('Problem Type', fontsize=13, fontweight='bold')
    ax.set_ylabel('Model', fontsize=13, fontweight='bold')
    ax.set_title('Failure Mode Prevalence Among Incorrect Answers\n(% of incorrect answers that exhibit each problem)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
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
