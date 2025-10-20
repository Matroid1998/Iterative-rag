"""
Plot: Miscalibration Direction by Correctness (Per Model)

Two-column layout showing miscalibration direction (overconfident/underconfident/ok)
split by answer correctness - LEFT: Correct answers, RIGHT: Incorrect answers.

Insight: Are models more miscalibrated when they're wrong? 
Do correct answers show better calibration than incorrect ones?
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
    load_hallucination_judgments, load_coverage_judgments, normalize_model_name
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'rag_analysis' / 'output'
PLOT_DIR = Path(__file__).resolve().parent


def main():
    """Generate miscalibration direction by correctness plot."""
    # Load hallucination and coverage judgments separately
    hall_records = load_hallucination_judgments(OUTPUT_DIR)
    cov_records = load_coverage_judgments(OUTPUT_DIR)
    
    # Index coverage by (model, question) for fast lookup
    cov_index = {}
    for rec in cov_records:
        key = (rec.get('model', ''), rec.get('question', ''))
        cov_index[key] = rec
    
    # Merge and group by model, correctness, and direction
    model_correct_direction = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    model_incorrect_direction = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for h_rec in hall_records:
        model = normalize_model_name(h_rec.get('model', ''))
        hops = h_rec.get('number_of_hops', 0)
        if hops == 0:
            continue
        
        # Get correctness from coverage
        key = (h_rec.get('model', ''), h_rec.get('question', ''))
        is_correct = False
        if key in cov_index:
            is_correct = cov_index[key].get('is_correct', False)
        
        # Get miscalibration direction
        cm = h_rec.get('parsed_judgment', {}).get('confidence_miscalibration', {})
        direction = cm.get('direction', 'ok')
        
        # Categorize by correctness
        if is_correct:
            model_correct_direction[model][hops][direction] += 1
        else:
            model_incorrect_direction[model][hops][direction] += 1
    
    # Sort models
    models = sorted(set(model_correct_direction.keys()) | set(model_incorrect_direction.keys()))
    
    if len(models) == 0:
        print("No model data found!")
        return
    
    # Prepare plot layout
    directions = ['ok', 'underconfident_continue', 'overconfident_finalize']
    direction_labels = ['OK', 'Underconfident', 'Overconfident']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    # Calculate grid size (2 columns per model, enough rows)
    num_models = len(models)
    nrows = num_models
    ncols = 2  # Left: Correct, Right: Incorrect
    
    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)  # Ensure 2D array
    
    # First pass: calculate max y value across all models for uniform y-axis
    max_y_value = 0
    for model in models:
        all_hops = set(model_correct_direction[model].keys()) | set(model_incorrect_direction[model].keys())
        hop_counts_temp = sorted(all_hops)
        
        for hop in hop_counts_temp:
            # Check correct totals
            correct_total = sum(model_correct_direction[model][hop].values())
            max_y_value = max(max_y_value, correct_total)
            
            # Check incorrect totals
            incorrect_total = sum(model_incorrect_direction[model][hop].values())
            max_y_value = max(max_y_value, incorrect_total)
    
    # Add 10% padding for n= labels
    y_limit = max_y_value * 1.15
    
    # Plot each model
    for row_idx, model in enumerate(models):
        # LEFT COLUMN: Correct Answers
        ax_correct = axes[row_idx, 0]
        hop_direction_correct = model_correct_direction[model]
        
        # Get all hop counts (combine both correct and incorrect to get full range)
        all_hops = set(model_correct_direction[model].keys()) | set(model_incorrect_direction[model].keys())
        hop_counts = sorted(all_hops)
        
        if not hop_counts:
            ax_correct.text(0.5, 0.5, f'No data for {model}', 
                           ha='center', va='center', transform=ax_correct.transAxes)
            axes[row_idx, 1].axis('off')
            continue
        
        # Prepare data for correct answers
        data_correct = {d: [] for d in directions}
        for hop in hop_counts:
            for d in directions:
                count = hop_direction_correct[hop].get(d, 0)
                data_correct[d].append(count)
        
        # Create stacked bar chart for correct answers
        x = np.arange(len(hop_counts))
        width = 0.6
        
        bottom = np.zeros(len(hop_counts))
        
        for i, (direction, label, color) in enumerate(zip(directions, direction_labels, colors)):
            values = data_correct[direction]
            bars = ax_correct.bar(x, values, width, label=label, bottom=bottom,
                                 color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
            
            # Add percentage labels on bars
            for j, (val, bot) in enumerate(zip(values, bottom)):
                if val > 0:
                    total = sum(data_correct[d][j] for d in directions)
                    pct = 100 * val / total if total > 0 else 0
                    if pct > 8:  # Only show label if segment is large enough
                        ax_correct.text(x[j], bot + val/2, f'{pct:.0f}%',
                                       ha='center', va='center', fontsize=8, fontweight='bold')
            
            bottom += values
        
        # Formatting for correct answers
        ax_correct.set_xlabel('Number of Hops', fontsize=10, fontweight='bold')
        ax_correct.set_ylabel('Count', fontsize=10, fontweight='bold')
        ax_correct.set_title(f'{model} - CORRECT Answers', fontsize=11, fontweight='bold',
                            pad=10, color='green')
        ax_correct.set_xticks(x)
        ax_correct.set_xticklabels([f'{h}' for h in hop_counts])
        ax_correct.set_ylim(0, y_limit)
        ax_correct.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add total counts on top for correct answers
        for i, hop in enumerate(hop_counts):
            total = sum(data_correct[d][i] for d in directions)
            ax_correct.text(i, total + y_limit*0.015, f'n={total}',
                           ha='center', va='bottom', fontsize=8, style='italic')
        
        # RIGHT COLUMN: Incorrect Answers
        ax_incorrect = axes[row_idx, 1]
        hop_direction_incorrect = model_incorrect_direction[model]
        
        # Prepare data for incorrect answers
        data_incorrect = {d: [] for d in directions}
        for hop in hop_counts:
            for d in directions:
                count = hop_direction_incorrect[hop].get(d, 0)
                data_incorrect[d].append(count)
        
        # Create stacked bar chart for incorrect answers
        bottom = np.zeros(len(hop_counts))
        
        for i, (direction, label, color) in enumerate(zip(directions, direction_labels, colors)):
            values = data_incorrect[direction]
            bars = ax_incorrect.bar(x, values, width, label=label, bottom=bottom,
                                   color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
            
            # Add percentage labels on bars
            for j, (val, bot) in enumerate(zip(values, bottom)):
                if val > 0:
                    total = sum(data_incorrect[d][j] for d in directions)
                    pct = 100 * val / total if total > 0 else 0
                    if pct > 8:  # Only show label if segment is large enough
                        ax_incorrect.text(x[j], bot + val/2, f'{pct:.0f}%',
                                         ha='center', va='center', fontsize=8, fontweight='bold')
            
            bottom += values
        
        # Formatting for incorrect answers
        ax_incorrect.set_xlabel('Number of Hops', fontsize=10, fontweight='bold')
        ax_incorrect.set_ylabel('Count', fontsize=10, fontweight='bold')
        ax_incorrect.set_title(f'{model} - INCORRECT Answers', fontsize=11, fontweight='bold',
                              pad=10, color='red')
        ax_incorrect.set_xticks(x)
        ax_incorrect.set_xticklabels([f'{h}' for h in hop_counts])
        ax_incorrect.set_ylim(0, y_limit)
        ax_incorrect.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add total counts on top for incorrect answers
        for i, hop in enumerate(hop_counts):
            total = sum(data_incorrect[d][i] for d in directions)
            ax_incorrect.text(i, total + y_limit*0.015, f'n={total}',
                             ha='center', va='bottom', fontsize=8, style='italic')
    
    # Create a single shared legend at the bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, framealpha=0.95,
              fontsize=11, bbox_to_anchor=(0.5, -0.01))
    
    # Overall title
    fig.suptitle('Miscalibration by Answer Correctness (Per Model)\nDo models show different calibration patterns for correct vs incorrect answers?',
                fontsize=16, fontweight='bold', y=0.998)
    
    plt.tight_layout(rect=[0, 0.015, 1, 0.995])
    output_path = PLOT_DIR / '1_miscalibration_by_correctness.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n=== Miscalibration by Correctness (Per Model) ===")
    for model in sorted(models):
        print(f"\n{model}:")
        
        # Get all hop counts
        all_hops = set(model_correct_direction[model].keys()) | set(model_incorrect_direction[model].keys())
        hop_counts = sorted(all_hops)
        
        for hop in hop_counts:
            # Correct answers
            correct_total = sum(model_correct_direction[model][hop].values())
            incorrect_total = sum(model_incorrect_direction[model][hop].values())
            
            print(f"  {hop}-hop:")
            print(f"    CORRECT (n={correct_total}):")
            for direction, label in zip(directions, direction_labels):
                count = model_correct_direction[model][hop].get(direction, 0)
                pct = 100 * count / correct_total if correct_total > 0 else 0
                print(f"      {label}: {count} ({pct:.1f}%)")
            
            print(f"    INCORRECT (n={incorrect_total}):")
            for direction, label in zip(directions, direction_labels):
                count = model_incorrect_direction[model][hop].get(direction, 0)
                pct = 100 * count / incorrect_total if incorrect_total > 0 else 0
                print(f"      {label}: {count} ({pct:.1f}%)")


if __name__ == '__main__':
    main()
