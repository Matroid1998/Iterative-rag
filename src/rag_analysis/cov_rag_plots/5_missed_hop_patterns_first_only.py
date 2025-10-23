"""
Plot 5b: Missed Hop Patterns - First Hop Miss Only
Stacked bar chart: X=number of hops in question, Y=% with first hop missed, stacked by which hop was missed.
Only counts the FIRST hop that was missed (earliest in the retrieval sequence).
For a 4-hop question, Hop 4 is retrieved first, then Hop 3, then Hop 2, then Hop 1.
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def load_missed_hop_patterns_first_only(output_dir):
    """Load only the FIRST missed hop per question, organized by num_hops and correctness."""
    # Structure: {is_correct: {num_hops: {first_missed_hop_index: count, 'total': count}}}
    hop_patterns = {
        True: defaultdict(lambda: defaultdict(int)),
        False: defaultdict(lambda: defaultdict(int))
    }
    
    for file_path in glob.glob(str(output_dir / '*coverage_gap_judgments.jsonl')):
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    
                    # Get correctness
                    is_correct = data.get('is_correct', False)
                    
                    # Get number of hops (from ground truth)
                    parsed = data.get('parsed_judgment', {})
                    coverage = parsed.get('retrieval_coverage_gap', {})
                    
                    # Determine number of hops from late_hit_per_hop data
                    late_hit = parsed.get('late_hit_per_hop', {})
                    per_hop = late_hit.get('per_hop', [])
                    
                    if per_hop:
                        num_hops = max(h.get('hop_index', 0) for h in per_hop)
                    else:
                        num_hops = 1  # Default assumption
                    
                    hop_patterns[is_correct][num_hops]['total'] += 1
                    
                    # Check for missed hops - only count the FIRST one missed
                    if coverage.get('has_gap'):
                        missed_hops = coverage.get('missed_hops', [])
                        
                        # Sort in descending order to get the first hop that was missed
                        # (In a 4-hop question: hop 4 is retrieved first, then 3, 2, 1)
                        if missed_hops:
                            first_missed = max(missed_hops)  # Highest hop number = first retrieved
                            hop_patterns[is_correct][num_hops][first_missed] += 1
                
                except json.JSONDecodeError:
                    continue
    
    return hop_patterns


def create_stacked_bar_chart_first_only(hop_patterns, output_path):
    """Create stacked bar chart showing only first missed hop patterns."""
    # Get all unique num_hops across both correct and incorrect
    all_num_hops = set()
    for is_correct in [True, False]:
        all_num_hops.update(hop_patterns[is_correct].keys())
    num_hops_list = sorted(all_num_hops)
    
    # Determine max hop index that was missed
    all_missed_hops = set()
    for is_correct in [True, False]:
        for patterns in hop_patterns[is_correct].values():
            all_missed_hops.update(k for k in patterns.keys() if k != 'total')
    max_hop = max(all_missed_hops) if all_missed_hops else 3
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(num_hops_list))
    width = 0.35  # Width for each bar group
    
    # Color palette - distinctive colors for each hop
    colors = plt.cm.Set3(np.linspace(0, 1, max_hop))
    
    # Process correct and incorrect separately
    for offset, is_correct, label_prefix in [(-width/2, True, 'Correct'), (width/2, False, 'Incorrect')]:
        patterns_subset = hop_patterns[is_correct]
        
        # Build data matrix for stacking - REVERSE ORDER so Hop 1 is on top
        # We'll stack from highest hop to lowest, then reverse the drawing order
        data_matrix = []
        hop_indices = list(range(max_hop, 0, -1))  # From max_hop down to 1
        
        for hop_idx in hop_indices:
            row = []
            for num_hops in num_hops_list:
                total = patterns_subset[num_hops]['total']
                missed_count = patterns_subset[num_hops].get(hop_idx, 0)
                percentage = 100 * missed_count / total if total > 0 else 0
                row.append(percentage)
            data_matrix.append((hop_idx, row))
        
        bottom = np.zeros(len(num_hops_list))
        
        # Draw from bottom to top (so Hop 1 ends up on top)
        for hop_idx, percentages in data_matrix:
            # Only add label for the first bar (to avoid duplicate legends)
            if is_correct:
                bar_label = f'First Miss: Hop {hop_idx}'
            else:
                bar_label = None
            
            bars_layer = ax.bar(x + offset, percentages, width, label=bar_label,
                               bottom=bottom, color=colors[hop_idx - 1],
                               edgecolor='black', linewidth=1.2, alpha=0.85)
            
            # Add value labels for significant values
            for i, (bar, pct) in enumerate(zip(bars_layer, percentages)):
                if pct > 2:  # Only label if > 2%
                    height = bar.get_height()
                    y_pos = bottom[i] + height / 2
                    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                           f'{pct:.1f}%',
                           ha='center', va='center', fontsize=8, fontweight='bold')
            
            bottom += percentages
        
        # Add total percentage labels on top with correctness indicator
        for i, x_pos in enumerate(x):
            total_pct = bottom[i]
            if total_pct > 0:
                ax.text(x_pos + offset, total_pct + 0.5, f'{total_pct:.1f}%',
                       ha='center', va='bottom', fontsize=9, fontweight='bold',
                       color='green' if is_correct else 'red')
    
    # Add custom legend entries for correct/incorrect
    from matplotlib.patches import Patch
    legend_elements = ax.get_legend_handles_labels()[0]
    legend_labels = ax.get_legend_handles_labels()[1]
    
    # Add separator and correct/incorrect indicators
    legend_elements.extend([
        Patch(facecolor='white', edgecolor='white'),
        Patch(facecolor='green', edgecolor='black', alpha=0.3),
        Patch(facecolor='red', edgecolor='black', alpha=0.3)
    ])
    legend_labels.extend(['', 'Correct Answers', 'Incorrect Answers'])
    
    # Customize plot
    ax.set_xlabel('Number of Hops in Question', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage with First Hop Missed (%)', fontsize=14, fontweight='bold')
    ax.set_title('First Missed Hop Patterns: Correct vs Incorrect Answers\n(Which hop gets missed FIRST in multi-hop questions?)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}-hop' for n in num_hops_list])
    ax.legend(legend_elements, legend_labels, loc='upper left', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved stacked bar chart (first miss only) to {output_path}")
    plt.close()
    
    # Print detailed statistics
    print("\n" + "="*80)
    print("FIRST MISSED HOP PATTERNS BY QUESTION COMPLEXITY AND CORRECTNESS")
    print("="*80)
    
    for num_hops in num_hops_list:
        print(f"\n{num_hops}-hop questions:")
        
        for is_correct in [True, False]:
            label = "CORRECT" if is_correct else "INCORRECT"
            patterns = hop_patterns[is_correct][num_hops]
            total = patterns['total']
            
            print(f"  {label} (n={total}):")
            
            total_with_first_miss = 0
            for hop_idx in range(max_hop, 0, -1):  # Show from highest to lowest
                missed_count = patterns.get(hop_idx, 0)
                if missed_count > 0:
                    percentage = 100 * missed_count / total
                    print(f"    First miss at hop {hop_idx}: {missed_count} ({percentage:.1f}%)")
                    total_with_first_miss += missed_count
            
            if total_with_first_miss > 0:
                overall_gap_rate = 100 * total_with_first_miss / total
                print(f"    Overall first-miss rate: {overall_gap_rate:.1f}%")
            else:
                print(f"    No first missed hops detected")


def main():
    # Setup paths
    base_dir = Path(__file__).resolve().parents[2]
    output_dir = base_dir / "rag_analysis" / "output"
    plot_dir = base_dir / "rag_analysis" / "cov_rag_plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading first missed hop patterns...")
    hop_patterns = load_missed_hop_patterns_first_only(output_dir)
    
    if not hop_patterns:
        print("No missed hop data found!")
        return
    
    # Create plot
    output_path = plot_dir / "missed_hop_patterns_first_only.png"
    create_stacked_bar_chart_first_only(hop_patterns, output_path)


if __name__ == "__main__":
    main()
