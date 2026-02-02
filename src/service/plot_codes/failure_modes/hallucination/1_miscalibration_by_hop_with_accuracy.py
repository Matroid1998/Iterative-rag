"""
Plot 1: Miscalibration Direction by Hop Count with Accuracy (Per Model)

Stacked bar chart showing miscalibration direction (overconfident/underconfident/ok)
by number of hops in the question, with accuracy line overlay.
6 subplots (one per model).

Insight: How does miscalibration relate to actual performance?
Do overconfident models have lower accuracy? Do underconfident models have higher accuracy?
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
    load_hallucination_judgments, load_coverage_judgments, normalize_model_name
)

OUTPUT_DIR = Path(__file__).resolve().parents[5] / "data" / "results" / "failure_modes"
PLOT_DIR = Path(__file__).resolve().parents[5] / "data" / "plots" / "failure_modes" / "hallucination"


def main():
    """Generate miscalibration direction by hop count plot with accuracy overlay."""
    # Load hallucination and coverage judgments separately
    hall_records = load_hallucination_judgments(OUTPUT_DIR)
    cov_records = load_coverage_judgments(OUTPUT_DIR)
    
    # Index coverage by (model, question) for fast lookup
    cov_index = {}
    for rec in cov_records:
        key = (rec.get('model', ''), rec.get('question', ''))
        cov_index[key] = rec
    
    # Merge coverage data into hallucination records (use hallucination as base)
    complete = []
    for h_rec in hall_records:
        if h_rec.get('number_of_hops', 0) == 0:
            continue
        
        key = (h_rec.get('model', ''), h_rec.get('question', ''))
        entry = {
            'model': h_rec.get('model', ''),
            'question': h_rec.get('question', ''),
            'number_of_hops': h_rec.get('number_of_hops', 0),
            'hallucination': h_rec.get('parsed_judgment', {}),
            'is_correct': False  # Default
        }
        
        # Add is_correct from coverage if available
        if key in cov_index:
            entry['is_correct'] = cov_index[key].get('is_correct', False)
        
        complete.append(entry)
    
    # Group by model, hop count, and direction + track correctness
    model_hop_direction = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    model_hop_correct = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
    
    for rec in complete:
        model = normalize_model_name(rec.get('model', ''))
        hops = rec.get('number_of_hops', 0)
        if hops == 0:
            continue
            
        cm = rec.get('hallucination', {}).get('confidence_miscalibration', {})
        direction = cm.get('direction', 'ok')
        
        model_hop_direction[model][hops][direction] += 1
        
        # Use the is_correct field from coverage records
        is_correct = rec.get('is_correct', False)
        
        model_hop_correct[model][hops]['total'] += 1
        if is_correct:
            model_hop_correct[model][hops]['correct'] += 1
    
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
        ax2 = ax.twinx()  # Secondary y-axis for accuracy
        
        hop_direction_counts = model_hop_direction[model]
        hop_accuracy = model_hop_correct[model]
        
        # Get hop counts for this model
        hop_counts = sorted(hop_direction_counts.keys())
        
        if not hop_counts:
            ax.text(0.5, 0.5, f'No data for {model}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model, fontsize=12, fontweight='bold')
            ax2.axis('off')
            continue
        
        # Prepare data for this model
        data = {d: [] for d in directions}
        accuracies = []
        
        for hop in hop_counts:
            for d in directions:
                count = hop_direction_counts[hop].get(d, 0)
                data[d].append(count)
            
            # Calculate accuracy for this hop count
            correct = hop_accuracy[hop]['correct']
            total = hop_accuracy[hop]['total']
            accuracy = 100 * correct / total if total > 0 else 0
            accuracies.append(accuracy)
        
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
        
        # Plot accuracy line on secondary axis
        line = ax2.plot(x, accuracies, color='#8e44ad', linewidth=3, 
                       marker='o', markersize=8, markeredgecolor='white', 
                       markeredgewidth=2, label='Accuracy', zorder=10)
        
        # Add accuracy value labels on the line
        for i, (xi, acc) in enumerate(zip(x, accuracies)):
            ax2.text(xi, acc + 2, f'{acc:.1f}%', 
                    ha='center', va='bottom', fontsize=9, 
                    fontweight='bold', color='#8e44ad',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor='#8e44ad', alpha=0.8))
        
        # Formatting
        ax.set_xlabel('Number of Hops', fontsize=10, fontweight='bold')
        ax.set_ylabel('Count', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold', color='#8e44ad')
        ax2.tick_params(axis='y', labelcolor='#8e44ad')
        ax2.set_ylim(0, 105)  # Set y-axis range for accuracy
        
        ax.set_title(model, fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{h}' for h in hop_counts])
        ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
        
        # Add total counts on top
        max_y = max(bottom)
        for i, hop in enumerate(hop_counts):
            total = sum(data[d][i] for d in directions)
            ax.text(i, total + max_y*0.02, f'n={total}', 
                   ha='center', va='bottom', fontsize=8, style='italic')
    
    # Hide unused subplots
    for idx in range(len(models), len(axes)):
        axes[idx].axis('off')
    
    # Create a shared legend at the bottom
    # Get handles from first subplot
    handles1, labels1 = axes[0].get_legend_handles_labels()
    
    # Create accuracy line handle for legend
    from matplotlib.lines import Line2D
    accuracy_handle = Line2D([0], [0], color='#8e44ad', linewidth=3, 
                            marker='o', markersize=8, markeredgecolor='white',
                            markeredgewidth=2, label='Accuracy')
    
    all_handles = handles1 + [accuracy_handle]
    all_labels = labels1 + ['Accuracy']
    
    fig.legend(all_handles, all_labels, loc='lower center', ncol=4, framealpha=0.95, 
              fontsize=11, bbox_to_anchor=(0.5, -0.02))
    
    # Overall title
    fig.suptitle('Miscalibration vs. Accuracy by Question Complexity (Per Model)\nHow does confidence calibration relate to actual performance?',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.985])
    output_path = PLOT_DIR / '1_miscalibration_by_hop_with_accuracy.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n=== Miscalibration & Accuracy by Hop Count (Per Model) ===")
    for model in sorted(models):
        print(f"\n{model}:")
        hop_direction_counts = model_hop_direction[model]
        hop_counts = sorted(hop_direction_counts.keys())
        
        for hop in hop_counts:
            total = sum(hop_direction_counts[hop].values())
            correct = model_hop_correct[model][hop]['correct']
            accuracy = 100 * correct / total if total > 0 else 0
            
            print(f"  {hop}-hop (n={total}, Accuracy={accuracy:.1f}%):")
            for direction, label in zip(directions, direction_labels):
                count = hop_direction_counts[hop].get(direction, 0)
                pct = 100 * count / total if total > 0 else 0
                print(f"    {label}: {count} ({pct:.1f}%)")


if __name__ == '__main__':
    main()
