"""
Plot 3: Hop Count Effects
Shows how miscalibration, late_hit, and composition_failure rates scale with number of hops
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
from advanced_utils import (
    load_all_judgments, 
    create_merged_dataset,
    normalize_model_name
)


def calculate_hop_metrics(merged_data):
    """Calculate failure rates by hop count."""
    
    hop_stats = defaultdict(lambda: {
        'total': 0,
        'miscalibrated': 0,
        'late_hit': 0,
        'composition_failure': 0,
        'coverage_gap': 0,
    })
    
    for rec in merged_data:
        num_hops = rec.get('number_of_hops', 0)
        if num_hops == 0:
            continue
        
        coverage = rec.get('coverage', {})
        hallucination = rec.get('hallucination', {})
        
        stats = hop_stats[num_hops]
        stats['total'] += 1
        
        # Miscalibration - nested in confidence_miscalibration
        conf_misc = hallucination.get('confidence_miscalibration', {})
        is_miscalibrated = conf_misc.get('is_miscalibrated', False)
        if is_miscalibrated:
            stats['miscalibrated'] += 1
        
        # Late hit
        any_late_hit = coverage.get('any_late_hit', False)
        if any_late_hit:
            stats['late_hit'] += 1
        
        # Composition failure - nested in composition_and_faithfulness
        comp_faith = hallucination.get('composition_and_faithfulness', {})
        is_failure = comp_faith.get('composition_failure', False)
        if is_failure:
            stats['composition_failure'] += 1
        
        # Coverage gap
        any_gap = coverage.get('any_coverage_gap', False)
        if any_gap:
            stats['coverage_gap'] += 1
    
    # Calculate rates
    results = {}
    for num_hops, stats in hop_stats.items():
        if stats['total'] < 10:  # Skip if too few samples
            continue
        
        results[num_hops] = {
            'total': stats['total'],
            'miscalibration_rate': (stats['miscalibrated'] / stats['total']) * 100,
            'late_hit_rate': (stats['late_hit'] / stats['total']) * 100,
            'composition_failure_rate': (stats['composition_failure'] / stats['total']) * 100,
            'coverage_gap_rate': (stats['coverage_gap'] / stats['total']) * 100,
        }
    
    return results


def create_hop_effects_plot(hop_metrics):
    """Create line plot showing how metrics scale with hop count."""
    
    if not hop_metrics:
        print("No hop metrics to plot")
        return None
    
    # Sort by hop count
    hop_counts = sorted(hop_metrics.keys())
    
    # Extract data
    miscalibration = [hop_metrics[h]['miscalibration_rate'] for h in hop_counts]
    late_hit = [hop_metrics[h]['late_hit_rate'] for h in hop_counts]
    composition_failure = [hop_metrics[h]['composition_failure_rate'] for h in hop_counts]
    coverage_gap = [hop_metrics[h]['coverage_gap_rate'] for h in hop_counts]
    totals = [hop_metrics[h]['total'] for h in hop_counts]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Main failure rates
    ax1.plot(hop_counts, miscalibration, 'o-', linewidth=2, markersize=8, 
             label='Miscalibration', color='#e74c3c')
    ax1.plot(hop_counts, late_hit, 's-', linewidth=2, markersize=8, 
             label='Late Hit', color='#f39c12')
    ax1.plot(hop_counts, composition_failure, '^-', linewidth=2, markersize=8, 
             label='Composition Failure', color='#9b59b6')
    ax1.plot(hop_counts, coverage_gap, 'd-', linewidth=2, markersize=8, 
             label='Coverage Gap', color='#e67e22')
    
    ax1.set_xlabel('Number of Hops', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Failure Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('How Task Complexity (Hop Count) Affects Failure Modes', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', frameon=True, fontsize=10)
    
    # Add value labels
    for i, hc in enumerate(hop_counts):
        ax1.annotate(f'{miscalibration[i]:.1f}%', 
                    (hc, miscalibration[i]), 
                    textcoords="offset points", 
                    xytext=(0, 8), 
                    ha='center', 
                    fontsize=8)
    
    ax1.set_xticks(hop_counts)
    ax1.set_ylim(0, max(max(miscalibration), max(late_hit), max(composition_failure), max(coverage_gap)) * 1.15)
    
    # Plot 2: Sample sizes
    colors_bars = ['#3498db' if c >= 100 else '#95a5a6' for c in totals]
    bars = ax2.bar(hop_counts, totals, color=colors_bars, edgecolor='black', linewidth=1.5, alpha=0.7)
    
    ax2.set_xlabel('Number of Hops', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Sample Size (N)', fontsize=12, fontweight='bold')
    ax2.set_title('Sample Size Distribution by Hop Count', fontsize=12, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.set_xticks(hop_counts)
    
    # Add value labels on bars
    for bar, total in zip(bars, totals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'n={total}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    return fig


def print_hop_analysis(hop_metrics):
    """Print detailed analysis of hop effects."""
    print("\n" + "="*60)
    print("HOP COUNT SCALING ANALYSIS")
    print("="*60)
    
    hop_counts = sorted(hop_metrics.keys())
    
    print("\nFailure Rate Scaling:")
    print(f"{'Hops':<6} {'N':<8} {'Miscal':<10} {'Late Hit':<10} {'Comp Fail':<12} {'Cov Gap':<10}")
    print("-" * 60)
    
    for hc in hop_counts:
        m = hop_metrics[hc]
        print(f"{hc:<6} {m['total']:<8} "
              f"{m['miscalibration_rate']:>6.1f}%    "
              f"{m['late_hit_rate']:>6.1f}%    "
              f"{m['composition_failure_rate']:>6.1f}%       "
              f"{m['coverage_gap_rate']:>6.1f}%")
    
    # Calculate trends
    if len(hop_counts) >= 2:
        print("\nKey Trends:")
        
        first_hops = hop_counts[0]
        last_hops = hop_counts[-1]
        
        misc_change = hop_metrics[last_hops]['miscalibration_rate'] - hop_metrics[first_hops]['miscalibration_rate']
        late_change = hop_metrics[last_hops]['late_hit_rate'] - hop_metrics[first_hops]['late_hit_rate']
        comp_change = hop_metrics[last_hops]['composition_failure_rate'] - hop_metrics[first_hops]['composition_failure_rate']
        gap_change = hop_metrics[last_hops]['coverage_gap_rate'] - hop_metrics[first_hops]['coverage_gap_rate']
        
        print(f"• Miscalibration: {misc_change:+.1f} percentage points ({first_hops}→{last_hops} hops)")
        print(f"• Late Hit: {late_change:+.1f} percentage points")
        print(f"• Composition Failure: {comp_change:+.1f} percentage points")
        print(f"• Coverage Gap: {gap_change:+.1f} percentage points")
        
        # Most sensitive metric
        changes = {
            'Miscalibration': abs(misc_change),
            'Late Hit': abs(late_change),
            'Composition Failure': abs(comp_change),
            'Coverage Gap': abs(gap_change)
        }
        most_sensitive = max(changes, key=changes.get)
        print(f"\n• Most hop-sensitive metric: {most_sensitive} ({changes[most_sensitive]:.1f}pp change)")


def main():
    output_dir = Path(__file__).resolve().parents[1] / 'output'
    
    print("Loading all judgments...")
    coverage, quality, hallucination = load_all_judgments(output_dir)
    
    print(f"Loaded: {len(coverage)} coverage, {len(quality)} quality, {len(hallucination)} hallucination")
    
    print("Merging datasets...")
    merged_data = create_merged_dataset(coverage, quality, hallucination)
    print(f"Merged: {len(merged_data)} records")
    
    print("Calculating hop count metrics...")
    hop_metrics = calculate_hop_metrics(merged_data)
    
    print_hop_analysis(hop_metrics)
    
    print("\nCreating hop effects plot...")
    fig = create_hop_effects_plot(hop_metrics)
    
    if fig:
        output_path = Path(__file__).parent / '3_hop_count_effects.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    else:
        print("✗ Failed to create plot")


if __name__ == '__main__':
    main()
