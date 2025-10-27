"""
Plot 5c: Sequential Pattern Analysis
Analyze the FULL SEQUENCE of missed hops, not just the first one.
Shows patterns like: "Only Hop 1 missed" vs "Hop 4 then Hop 1 missed" vs "All hops missed"
"""
import json
import glob
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np

def load_sequential_patterns(output_dir):
    """Load full sequences of missed hops, organized by num_hops and correctness."""
    # Structure: {is_correct: {num_hops: {pattern_tuple: count}}}
    patterns = {
        True: defaultdict(lambda: Counter()),
        False: defaultdict(lambda: Counter())
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
                    
                    # Get number of hops
                    parsed = data.get('parsed_judgment', {})
                    late_hit = parsed.get('late_hit_per_hop', {})
                    per_hop = late_hit.get('per_hop', [])
                    
                    if per_hop:
                        num_hops = max(h.get('hop_index', 0) for h in per_hop)
                    else:
                        num_hops = 1
                    
                    # Get all missed hops (sorted in retrieval order: descending)
                    coverage = parsed.get('retrieval_coverage_gap', {})
                    if coverage.get('has_gap'):
                        missed_hops = coverage.get('missed_hops', [])
                        if missed_hops:
                            # Sort in descending order (retrieval order: 4→3→2→1)
                            pattern = tuple(sorted(missed_hops, reverse=True))
                            patterns[is_correct][num_hops][pattern] += 1
                    else:
                        # No misses
                        patterns[is_correct][num_hops][tuple()] += 1
                
                except json.JSONDecodeError:
                    continue
    
    return patterns


def pattern_to_string(pattern):
    """Convert pattern tuple to readable string."""
    if not pattern:
        return "No Misses"
    return " → ".join([f"H{h}" for h in pattern])


def create_sequential_pattern_chart(patterns, output_path):
    """Create visualization of sequential miss patterns."""
    # Get all unique num_hops
    all_num_hops = set()
    for is_correct in [True, False]:
        all_num_hops.update(patterns[is_correct].keys())
    num_hops_list = sorted(all_num_hops)
    
    # Create subplots
    fig, axes = plt.subplots(len(num_hops_list), 1, figsize=(14, 4 * len(num_hops_list)))
    if len(num_hops_list) == 1:
        axes = [axes]
    
    for ax_idx, num_hops in enumerate(num_hops_list):
        ax = axes[ax_idx]
        
        # Get top patterns for this num_hops
        correct_patterns = patterns[True][num_hops]
        incorrect_patterns = patterns[False][num_hops]
        
        # Get all unique patterns from both
        all_patterns = set(correct_patterns.keys()) | set(incorrect_patterns.keys())
        
        # Calculate percentages
        correct_total = sum(correct_patterns.values())
        incorrect_total = sum(incorrect_patterns.values())
        
        pattern_data = []
        for pattern in all_patterns:
            correct_count = correct_patterns[pattern]
            incorrect_count = incorrect_patterns[pattern]
            
            correct_pct = 100 * correct_count / correct_total if correct_total > 0 else 0
            incorrect_pct = 100 * incorrect_count / incorrect_total if incorrect_total > 0 else 0
            
            # Only include patterns that are at least 1% in either group
            if correct_pct >= 1 or incorrect_pct >= 1:
                pattern_data.append({
                    'pattern': pattern,
                    'pattern_str': pattern_to_string(pattern),
                    'correct_pct': correct_pct,
                    'incorrect_pct': incorrect_pct,
                    'correct_count': correct_count,
                    'incorrect_count': incorrect_count,
                    'total': correct_count + incorrect_count
                })
        
        # Sort by total frequency
        pattern_data.sort(key=lambda x: x['total'], reverse=True)
        
        # Take top 15 patterns
        pattern_data = pattern_data[:15]
        
        if not pattern_data:
            ax.text(0.5, 0.5, 'No significant patterns found',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            continue
        
        # Create diverging bar chart
        labels = [p['pattern_str'] for p in pattern_data]
        correct_values = [p['correct_pct'] for p in pattern_data]
        incorrect_values = [p['incorrect_pct'] for p in pattern_data]
        
        y_pos = np.arange(len(labels))
        
        # Create bars
        ax.barh(y_pos, [-v for v in correct_values], height=0.7, 
                label='Correct Answers', color='#2E86AB', alpha=0.85, edgecolor='black', linewidth=0.8)
        ax.barh(y_pos, incorrect_values, height=0.7, 
                label='Incorrect Answers', color='#A23B72', alpha=0.85, edgecolor='black', linewidth=0.8)
        
        # Add value labels
        for i, (correct, incorrect) in enumerate(zip(correct_values, incorrect_values)):
            if correct > 0.5:
                ax.text(-correct - 0.3, i, f'{correct:.1f}%', 
                       ha='right', va='center', fontsize=9, fontweight='bold')
            if incorrect > 0.5:
                ax.text(incorrect + 0.3, i, f'{incorrect:.1f}%', 
                       ha='left', va='center', fontsize=9, fontweight='bold')
        
        # Add difference annotations for significant patterns
        for i, pd in enumerate(pattern_data):
            diff = pd['incorrect_pct'] - pd['correct_pct']
            if abs(diff) > 5:  # Show difference if > 5%
                x_pos = max(pd['correct_pct'], pd['incorrect_pct']) + 0.5
                color = 'red' if diff > 0 else 'green'
                ax.text(x_pos, i, f'Δ{diff:+.1f}%', 
                       ha='left', va='center', fontsize=8, 
                       color=color, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor=color, alpha=0.7))
        
        # Add center line
        ax.axvline(x=0, color='black', linewidth=2, linestyle='-')
        
        # Customize subplot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Pattern Frequency (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{num_hops}-Hop Questions: Sequential Miss Patterns', 
                    fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Set symmetric x-axis
        max_val = max(max(correct_values), max(incorrect_values))
        ax.set_xlim(-max_val * 1.25, max_val * 1.25)
        
        # Format x-axis to show absolute values
        xticks = ax.get_xticks()
        ax.set_xticklabels([f'{abs(x):.0f}' for x in xticks])
        
        # Add legend only on first subplot
        if ax_idx == 0:
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        # Add sample size annotation
        ax.text(0.02, 0.98, f'n_correct={correct_total}\nn_incorrect={incorrect_total}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Overall title
    fig.suptitle('Sequential Miss Patterns: Complete Hop Miss Sequences\n' +
                 'Format: H4 → H1 means "Hop 4 missed, then Hop 1 missed" (in retrieval order)',
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.97)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved sequential pattern chart to {output_path}")
    plt.close()
    
    # Print detailed statistics
    print("\n" + "="*80)
    print("SEQUENTIAL MISS PATTERNS (Top patterns by frequency)")
    print("="*80)
    
    for num_hops in num_hops_list:
        correct_patterns = patterns[True][num_hops]
        incorrect_patterns = patterns[False][num_hops]
        
        correct_total = sum(correct_patterns.values())
        incorrect_total = sum(incorrect_patterns.values())
        
        all_patterns = set(correct_patterns.keys()) | set(incorrect_patterns.keys())
        
        pattern_data = []
        for pattern in all_patterns:
            correct_count = correct_patterns[pattern]
            incorrect_count = incorrect_patterns[pattern]
            
            correct_pct = 100 * correct_count / correct_total if correct_total > 0 else 0
            incorrect_pct = 100 * incorrect_count / incorrect_total if incorrect_total > 0 else 0
            
            pattern_data.append({
                'pattern': pattern,
                'pattern_str': pattern_to_string(pattern),
                'correct_pct': correct_pct,
                'incorrect_pct': incorrect_pct,
                'correct_count': correct_count,
                'incorrect_count': incorrect_count,
                'total': correct_count + incorrect_count
            })
        
        pattern_data.sort(key=lambda x: x['total'], reverse=True)
        
        print(f"\n{num_hops}-hop questions (Correct: {correct_total}, Incorrect: {incorrect_total}):")
        print(f"{'Pattern':<25} {'Correct %':>10} {'Incorrect %':>12} {'Difference':>12}")
        print("-" * 65)
        
        for pd in pattern_data[:10]:  # Top 10
            diff = pd['incorrect_pct'] - pd['correct_pct']
            print(f"{pd['pattern_str']:<25} {pd['correct_pct']:>9.1f}% {pd['incorrect_pct']:>11.1f}% {diff:>11.1f}%")


def main():
    # Setup paths
    base_dir = Path(__file__).resolve().parents[2]
    output_dir = base_dir / "rag_analysis" / "output"
    plot_dir = base_dir / "rag_analysis" / "cov_rag_plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading sequential miss patterns...")
    patterns = load_sequential_patterns(output_dir)
    
    if not patterns:
        print("No pattern data found!")
        return
    
    # Create plot
    output_path = plot_dir / "sequential_pattern_analysis.png"
    create_sequential_pattern_chart(patterns, output_path)


if __name__ == "__main__":
    main()
