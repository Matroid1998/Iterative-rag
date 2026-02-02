"""
Plot 7: Miscalibration Mix per Model

Stacked bar chart showing confidence miscalibration directions per model,
with overall miscalibration rate annotated.

Insight: Which models are overconfident vs underconfident?
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
    """Generate miscalibration mix per model plot."""
    records = load_hallucination_judgments(OUTPUT_DIR)
    
    # Group by model and direction
    model_direction_counts = defaultdict(lambda: defaultdict(int))
    model_miscalibrated = defaultdict(int)
    model_total = defaultdict(int)
    
    for rec in records:
        model = normalize_model_name(rec.get('model', ''))
        cm = rec.get('parsed_judgment', {}).get('confidence_miscalibration', {})
        
        direction = cm.get('direction', 'ok')
        is_miscalibrated = cm.get('is_miscalibrated', False)
        
        model_direction_counts[model][direction] += 1
        model_total[model] += 1
        if is_miscalibrated:
            model_miscalibrated[model] += 1
    
    # Prepare data for plotting
    models = sorted(model_total.keys())
    directions = ['ok', 'underconfident_continue', 'overconfident_finalize']
    direction_labels = ['OK', 'Underconfident', 'Overconfident']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    # Calculate percentages
    data = {d: [] for d in directions}
    miscal_rates = []
    
    for model in models:
        total = model_total[model]
        miscal_rate = 100 * model_miscalibrated[model] / total
        miscal_rates.append(miscal_rate)
        
        for d in directions:
            count = model_direction_counts[model].get(d, 0)
            pct = 100 * count / total
            data[d].append(pct)
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(models))
    width = 0.6
    
    bottom = np.zeros(len(models))
    
    for direction, label, color in zip(directions, direction_labels, colors):
        values = data[direction]
        ax.bar(x, values, width, label=label, bottom=bottom, 
              color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
        
        # Add percentage labels on bars (only if segment is large enough)
        for i, (val, bot) in enumerate(zip(values, bottom)):
            if val > 5:  # Only show label if segment > 5%
                ax.text(x[i], bot + val/2, f'{val:.1f}%',
                       ha='center', va='center', fontsize=9, fontweight='bold',
                       color='white' if val > 15 else 'black')
        
        bottom += values
    
    # Add overall miscalibration rate above each bar
    for i, (model, rate) in enumerate(zip(models, miscal_rates)):
        ax.text(i, 102, f'Miscal: {rate:.1f}%',
               ha='center', va='bottom', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Add average miscalibration line
    avg_miscal = np.mean(miscal_rates)
    
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title(f'Confidence Miscalibration Mix by Model\n(Avg Miscalibration: {avg_miscal:.1f}%)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.set_ylim(0, 115)
    ax.legend(loc='upper left', framealpha=0.95, fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = PLOT_DIR / '7_miscalibration_mix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n=== Miscalibration Mix by Model ===")
    for model in models:
        total = model_total[model]
        miscal_rate = 100 * model_miscalibrated[model] / total
        print(f"\n{model} (n={total}):")
        print(f"  Overall miscalibration: {miscal_rate:.1f}%")
        for direction, label in zip(directions, direction_labels):
            count = model_direction_counts[model].get(direction, 0)
            pct = 100 * count / total
            print(f"  {label}: {count} ({pct:.1f}%)")


if __name__ == '__main__':
    main()
