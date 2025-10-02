"""
Plot 1: Late Hit Timing Distribution
Violin plot showing distribution of (first_hit_step - hop_index) for each hop number.
Insight: How late are late hits typically? Is hop 2 consistently later than hop 1?
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def load_late_hit_data(output_dir):
    """Load late hit timing data from all coverage gap judgment files."""
    late_hit_delays = defaultdict(list)  # {hop_index: [delay1, delay2, ...]}
    
    for file_path in glob.glob(str(output_dir / '*coverage_gap_judgments.jsonl')):
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    parsed = data.get('parsed_judgment', {})
                    late_hit = parsed.get('late_hit_per_hop', {})
                    
                    for hop_data in late_hit.get('per_hop', []):
                        hop_index = hop_data.get('hop_index')
                        first_hit_step = hop_data.get('first_hit_step')
                        
                        if hop_index is not None and first_hit_step is not None:
                            delay = first_hit_step - hop_index
                            late_hit_delays[hop_index].append(delay)
                
                except json.JSONDecodeError:
                    continue
    
    return late_hit_delays


def create_violin_plot(late_hit_delays, output_path):
    """Create violin plot of late hit timing distribution."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Prepare data for violin plot
    hop_indices = sorted(late_hit_delays.keys())
    data_to_plot = [late_hit_delays[hop] for hop in hop_indices]
    positions = list(range(1, len(hop_indices) + 1))
    
    # Create violin plot
    parts = ax.violinplot(data_to_plot, positions=positions, widths=0.7,
                          showmeans=True, showmedians=True, showextrema=True)
    
    # Customize violin colors
    for pc in parts['bodies']:
        pc.set_facecolor('#4c72b0')
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    # Customize other elements
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
        if partname in parts:
            parts[partname].set_edgecolor('black')
            parts[partname].set_linewidth(1.5)
    
    # Add statistics annotations
    for i, (hop, delays) in enumerate(sorted(late_hit_delays.items()), 1):
        if delays:
            median = np.median(delays)
            mean = np.mean(delays)
            q1 = np.percentile(delays, 25)
            q3 = np.percentile(delays, 75)
            
            # Annotate with statistics
            stats_text = f"n={len(delays)}\nμ={mean:.2f}\nmed={median:.1f}"
            ax.text(i, max(delays) + 0.5, stats_text, 
                   ha='center', va='bottom', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Customize plot
    ax.set_xlabel('Hop Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('Late Hit Delay (first_hit_step - hop_index)', fontsize=14, fontweight='bold')
    ax.set_title('Late Hit Timing Distribution by Hop\n(How many steps late does each hop get retrieved?)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(positions)
    ax.set_xticklabels([f'Hop {hop}' for hop in hop_indices])
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='On-time retrieval')
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved violin plot to {output_path}")
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("LATE HIT TIMING STATISTICS")
    print("="*60)
    for hop in hop_indices:
        delays = late_hit_delays[hop]
        if delays:
            late_hits = sum(1 for d in delays if d > 0)
            print(f"\nHop {hop}:")
            print(f"  Total observations: {len(delays)}")
            print(f"  Late hits (delay > 0): {late_hits} ({100*late_hits/len(delays):.1f}%)")
            print(f"  Mean delay: {np.mean(delays):.2f} steps")
            print(f"  Median delay: {np.median(delays):.1f} steps")
            print(f"  Max delay: {max(delays)} steps")


def main():
    # Setup paths
    base_dir = Path(__file__).resolve().parents[2]
    output_dir = base_dir / "rag_analysis" / "output"
    plot_dir = base_dir / "rag_analysis" / "cov_rag_plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading late hit timing data...")
    late_hit_delays = load_late_hit_data(output_dir)
    
    if not late_hit_delays:
        print("No late hit data found!")
        return
    
    # Create plot
    output_path = plot_dir / "late_hit_timing_distribution.png"
    create_violin_plot(late_hit_delays, output_path)


if __name__ == "__main__":
    main()
