"""
Plot 5b (Difference Version): Missed Hop Patterns - Difference Between Incorrect and Correct
Shows the difference (Incorrect % - Correct %) for each hop miss pattern.
Positive values = more common in incorrect answers
Negative values = more common in correct answers
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


def create_difference_chart(hop_patterns, output_path):
    """Create an elegant heatmap showing difference (Incorrect % - Correct %) for each hop miss pattern."""
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
    
    # Build difference matrix: rows = hop indices (1 to max_hop), cols = num_hops
    hop_indices = list(range(1, max_hop + 1))
    diff_matrix = np.zeros((len(hop_indices), len(num_hops_list)))
    
    for i, hop_idx in enumerate(hop_indices):
        for j, num_hops in enumerate(num_hops_list):
            correct_patterns = hop_patterns[True][num_hops]
            incorrect_patterns = hop_patterns[False][num_hops]
            
            correct_total = correct_patterns['total']
            incorrect_total = incorrect_patterns['total']
            
            correct_pct = 100 * correct_patterns.get(hop_idx, 0) / correct_total if correct_total > 0 else 0
            incorrect_pct = 100 * incorrect_patterns.get(hop_idx, 0) / incorrect_total if incorrect_total > 0 else 0
            
            diff_matrix[i, j] = incorrect_pct - correct_pct
    
    # Create figure with better styling
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap with diverging colormap (red=more incorrect, blue=more correct)
    from matplotlib.colors import TwoSlopeNorm
    
    # Set the center at 0 for diverging colors
    vmax = max(abs(diff_matrix.min()), abs(diff_matrix.max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    im = ax.imshow(diff_matrix, cmap='RdBu_r', aspect='auto', norm=norm)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(num_hops_list)))
    ax.set_yticks(np.arange(len(hop_indices)))
    ax.set_xticklabels([f'{n}-hop\nQuestions' for n in num_hops_list], fontsize=11)
    ax.set_yticklabels([f'Hop {h}\nMissed First' for h in hop_indices], fontsize=11)
    
    # Rotate the tick labels for better readability
    plt.setp(ax.get_xticklabels(), ha="center")
    
    # Add text annotations on each cell
    for i in range(len(hop_indices)):
        for j in range(len(num_hops_list)):
            value = diff_matrix[i, j]
            
            # Choose text color based on background
            text_color = 'white' if abs(value) > vmax * 0.5 else 'black'
            
            # Format the text
            if abs(value) > 0.1:
                text = f'{value:+.1f}%'
                ax.text(j, i, text, ha='center', va='center',
                       color=text_color, fontsize=11, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Difference (Incorrect % - Correct %)', 
                   rotation=270, labelpad=25, fontsize=12, fontweight='bold')
    
    # Customize plot
    ax.set_title('First Missed Hop Patterns: Incorrect vs Correct Answers\n' +
                'Red = More common in INCORRECT | Blue = More common in CORRECT',
                fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.set_xticks(np.arange(len(num_hops_list)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(hop_indices)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved difference heatmap to {output_path}")
    plt.close()
    
    # Print detailed statistics
    print("\n" + "="*80)
    print("DIFFERENCE IN FIRST MISSED HOP PATTERNS (Incorrect % - Correct %)")
    print("="*80)
    
    for num_hops in num_hops_list:
        print(f"\n{num_hops}-hop questions:")
        
        correct_patterns = hop_patterns[True][num_hops]
        incorrect_patterns = hop_patterns[False][num_hops]
        
        correct_total = correct_patterns['total']
        incorrect_total = incorrect_patterns['total']
        
        print(f"  Sample sizes: Correct={correct_total}, Incorrect={incorrect_total}")
        
        for hop_idx in range(max_hop, 0, -1):  # Show from highest to lowest
            correct_count = correct_patterns.get(hop_idx, 0)
            incorrect_count = incorrect_patterns.get(hop_idx, 0)
            
            correct_pct = 100 * correct_count / correct_total if correct_total > 0 else 0
            incorrect_pct = 100 * incorrect_count / incorrect_total if incorrect_total > 0 else 0
            
            difference = incorrect_pct - correct_pct
            
            if correct_count > 0 or incorrect_count > 0:
                print(f"    Hop {hop_idx} first miss:")
                print(f"      Correct: {correct_count}/{correct_total} ({correct_pct:.1f}%)")
                print(f"      Incorrect: {incorrect_count}/{incorrect_total} ({incorrect_pct:.1f}%)")
                print(f"      Difference: {difference:+.1f}% {'(more in INCORRECT)' if difference > 0 else '(more in CORRECT)' if difference < 0 else ''}")


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
    
    # Create difference plot
    output_path = plot_dir / "missed_hop_patterns_difference.png"
    create_difference_chart(hop_patterns, output_path)


if __name__ == "__main__":
    main()
