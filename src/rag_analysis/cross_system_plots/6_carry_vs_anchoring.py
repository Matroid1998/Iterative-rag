"""
Plot 6: Carry → Quality Anchoring

Correlation between step-level carry_drop (coverage) and anchored (quality) 
at the same step.

Insight: When anchors are dropped, do queries become unanchored?
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
    load_all_judgments, create_merged_dataset,
    get_step_carry_drop_flags, get_step_anchored_flags
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'rag_analysis' / 'output'
PLOT_DIR = Path(__file__).resolve().parent


def main():
    """Generate carry-drop to anchoring correlation plot."""
    cov_records, qual_records, hall_records = load_all_judgments(OUTPUT_DIR)
    merged = create_merged_dataset(cov_records, qual_records, hall_records)
    
    # Filter to records with both coverage and quality
    complete = [r for r in merged if 'coverage' in r and 'quality' in r]
    
    # Collect per-step statistics
    step_stats = defaultdict(lambda: {
        'carry_drop_count': 0,
        'anchored_count': 0,
        'both_count': 0,
        'total': 0
    })
    
    for rec in complete:
        carry_drop_flags = get_step_carry_drop_flags(rec.get('coverage', {}))
        anchored_flags = get_step_anchored_flags(rec.get('quality', {}))
        
        # Align by step index (both should be same length ideally)
        max_steps = max(len(carry_drop_flags), len(anchored_flags))
        
        for step_idx in range(max_steps):
            has_carry_drop = step_idx < len(carry_drop_flags) and carry_drop_flags[step_idx]
            is_anchored = step_idx < len(anchored_flags) and anchored_flags[step_idx]
            
            if step_idx < len(carry_drop_flags) or step_idx < len(anchored_flags):
                step_num = step_idx + 1  # 1-based
                step_stats[step_num]['total'] += 1
                
                if has_carry_drop:
                    step_stats[step_num]['carry_drop_count'] += 1
                
                if is_anchored:
                    step_stats[step_num]['anchored_count'] += 1
                
                if has_carry_drop and is_anchored:
                    step_stats[step_num]['both_count'] += 1
    
    # Calculate rates per step
    steps = sorted([s for s in step_stats.keys() if step_stats[s]['total'] >= 10])  # Min 10 samples
    carry_drop_rates = []
    anchored_rates = []
    
    for step in steps:
        stats = step_stats[step]
        total = stats['total']
        
        cd_rate = 100 * stats['carry_drop_count'] / total
        anch_rate = 100 * stats['anchored_count'] / total
        
        carry_drop_rates.append(cd_rate)
        anchored_rates.append(anch_rate)
    
    # Create dual-axis plot
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    color1 = '#e74c3c'
    color2 = '#3498db'
    
    ax1.set_xlabel('Step Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Carry-Drop Rate (%)', fontsize=12, fontweight='bold', color=color1)
    line1 = ax1.plot(steps, carry_drop_rates, color=color1, linewidth=2.5, 
                     marker='o', markersize=8, label='Carry-Drop Rate')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(alpha=0.3, linestyle='--')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Anchored Rate (%)', fontsize=12, fontweight='bold', color=color2)
    line2 = ax2.plot(steps, anchored_rates, color=color2, linewidth=2.5,
                     marker='s', markersize=8, label='Anchored Rate')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', framealpha=0.95, fontsize=11)
    
    ax1.set_title('Carry-Drop vs Query Anchoring by Step\n(Do dropped anchors lead to unanchored queries?)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(steps)
    ax1.set_ylim(0, max(carry_drop_rates) * 1.2 if carry_drop_rates else 10)
    ax2.set_ylim(0, max(anchored_rates) * 1.2 if anchored_rates else 100)
    
    plt.tight_layout()
    output_path = PLOT_DIR / '6_carry_vs_anchoring.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Calculate correlation
    if len(carry_drop_rates) > 1:
        correlation = np.corrcoef(carry_drop_rates, anchored_rates)[0, 1]
        print(f"\nCorrelation between Carry-Drop and Anchored: {correlation:.3f}")
    
    # Print statistics
    print("\n=== Carry-Drop vs Anchoring by Step ===")
    for step, cd_rate, anch_rate in zip(steps, carry_drop_rates, anchored_rates):
        total = step_stats[step]['total']
        both = step_stats[step]['both_count']
        print(f"\nStep {step} (n={total}):")
        print(f"  Carry-Drop Rate: {cd_rate:.1f}%")
        print(f"  Anchored Rate: {anch_rate:.1f}%")
        print(f"  Both (carry-drop AND anchored): {both} ({100*both/total:.1f}%)")


if __name__ == '__main__':
    main()
