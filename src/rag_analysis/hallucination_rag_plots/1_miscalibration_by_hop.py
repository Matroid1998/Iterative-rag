"""
Plot 1: Miscalibration Direction by Hop Count

Stacked bar chart showing miscalibration direction (overconfident/underconfident/ok)
by number of hops in the question.

Insight: Are models overconfident on simple questions and underconfident on complex ones?
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hallucination_rag_plots.hall_plot_utils import load_hallucination_judgments

OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'rag_analysis' / 'output'
PLOT_DIR = Path(__file__).resolve().parent


def main():
    """Generate miscalibration direction by hop count plot."""
    records = load_hallucination_judgments(OUTPUT_DIR)
    
    # Group by hop count and direction
    hop_direction_counts = defaultdict(lambda: defaultdict(int))
    
    for rec in records:
        hops = rec.get('number_of_hops', 0)
        if hops == 0:
            continue
            
        cm = rec.get('parsed_judgment', {}).get('confidence_miscalibration', {})
        direction = cm.get('direction', 'ok')
        
        hop_direction_counts[hops][direction] += 1
    
    # Prepare data for plotting
    hop_counts = sorted(hop_direction_counts.keys())
    directions = ['ok', 'underconfident_continue', 'overconfident_finalize']
    direction_labels = ['OK', 'Underconfident', 'Overconfident']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    data = {d: [] for d in directions}
    for hop in hop_counts:
        total = sum(hop_direction_counts[hop].values())
        for d in directions:
            count = hop_direction_counts[hop].get(d, 0)
            data[d].append(count)
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(hop_counts))
    width = 0.6
    
    bottom = np.zeros(len(hop_counts))
    
    for i, (direction, label, color) in enumerate(zip(directions, direction_labels, colors)):
        values = data[direction]
        ax.bar(x, values, width, label=label, bottom=bottom, color=color, alpha=0.85)
        
        # Add percentage labels on bars
        for j, (val, bot) in enumerate(zip(values, bottom)):
            if val > 0:
                total = sum(data[d][j] for d in directions)
                pct = 100 * val / total
                if pct > 5:  # Only show label if segment is large enough
                    ax.text(x[j], bot + val/2, f'{pct:.1f}%', 
                           ha='center', va='center', fontsize=9, fontweight='bold')
        
        bottom += values
    
    ax.set_xlabel('Number of Hops', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Questions', fontsize=12, fontweight='bold')
    ax.set_title('Miscalibration Direction by Question Complexity', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h}-hop' for h in hop_counts])
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add total counts on top
    for i, hop in enumerate(hop_counts):
        total = sum(data[d][i] for d in directions)
        ax.text(i, total + total*0.02, f'n={total}', 
               ha='center', va='bottom', fontsize=9, style='italic')
    
    plt.tight_layout()
    output_path = PLOT_DIR / '1_miscalibration_by_hop.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n=== Miscalibration by Hop Count ===")
    for hop in hop_counts:
        total = sum(hop_direction_counts[hop].values())
        print(f"\n{hop}-hop questions (n={total}):")
        for direction, label in zip(directions, direction_labels):
            count = hop_direction_counts[hop].get(direction, 0)
            pct = 100 * count / total
            print(f"  {label}: {count} ({pct:.1f}%)")


if __name__ == '__main__':
    main()
