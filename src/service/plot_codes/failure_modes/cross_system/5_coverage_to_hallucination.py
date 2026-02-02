"""
Plot 5: Coverage → Hallucination

Bar chart showing % composition_failure conditioned on coverage issues
(has_gap, any_late_hit).

Insight: Do retrieval issues drive synthesis errors?
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_system.cross_system_utils import (
    load_all_judgments, create_merged_dataset,
    has_coverage_gap, has_late_hit, has_composition_failure
)

OUTPUT_DIR = Path(__file__).resolve().parents[5] / "data" / "results" / "failure_modes"
PLOT_DIR = Path(__file__).resolve().parents[5] / "data" / "plots" / "failure_modes" / "cross_system"


def main():
    """Generate coverage to hallucination bar chart."""
    cov_records, qual_records, hall_records = load_all_judgments(OUTPUT_DIR)
    merged = create_merged_dataset(cov_records, qual_records, hall_records)
    
    # Filter to records with hallucination judgments
    with_hall = [r for r in merged if 'hallucination' in r]
    
    # Categorize by coverage issues
    categories = {
        'No Issues': [],
        'Has Gap Only': [],
        'Late Hit Only': [],
        'Both Issues': []
    }
    
    for rec in with_hall:
        has_gap = has_coverage_gap(rec.get('coverage', {}))
        has_late = has_late_hit(rec.get('coverage', {}))
        
        if has_gap and has_late:
            categories['Both Issues'].append(rec)
        elif has_gap:
            categories['Has Gap Only'].append(rec)
        elif has_late:
            categories['Late Hit Only'].append(rec)
        else:
            categories['No Issues'].append(rec)
    
    # Calculate composition failure rates
    category_names = ['No Issues', 'Late Hit Only', 'Has Gap Only', 'Both Issues']
    failure_rates = []
    failure_counts = []
    totals = []
    
    for cat in category_names:
        records = categories[cat]
        total = len(records)
        failures = sum(1 for r in records if has_composition_failure(r.get('hallucination', {})))
        
        rate = 100 * failures / total if total > 0 else 0
        failure_rates.append(rate)
        failure_counts.append(failures)
        totals.append(total)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    
    bars = ax.bar(category_names, failure_rates, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, rate, count, total in zip(bars, failure_rates, failure_counts, totals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{rate:.1f}%\n({count}/{total})',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add reference line at overall average
    overall_failures = sum(failure_counts)
    overall_total = sum(totals)
    overall_rate = 100 * overall_failures / overall_total if overall_total > 0 else 0
    
    ax.axhline(y=overall_rate, color='gray', linestyle='--', linewidth=2,
              label=f'Overall Avg: {overall_rate:.1f}%', alpha=0.7)
    
    ax.set_ylabel('Composition Failure Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Coverage Issue Category', fontsize=12, fontweight='bold')
    ax.set_title('Impact of Coverage Issues on Composition Failures\n(Do retrieval problems drive synthesis errors?)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(failure_rates) * 1.2)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Rotate x labels if needed
    plt.xticks(rotation=15, ha='right')
    
    plt.tight_layout()
    output_path = PLOT_DIR / '5_coverage_to_hallucination.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n=== Coverage Issues → Composition Failure ===")
    print(f"\nTotal records analyzed: {overall_total}")
    print(f"Overall composition failure rate: {overall_rate:.1f}%\n")
    
    for cat, rate, count, total in zip(category_names, failure_rates, failure_counts, totals):
        print(f"{cat}:")
        print(f"  Total: {total}")
        print(f"  Failures: {count}")
        print(f"  Failure Rate: {rate:.1f}%")
        
        if overall_rate > 0:
            relative = rate / overall_rate
            print(f"  Relative Risk: {relative:.2f}x")
        print()


if __name__ == '__main__':
    main()
