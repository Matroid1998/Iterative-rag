"""
Plot 4: Composition Failure Root Causes

Grouped bar chart showing percentage of composition failures that also have:
- coverage_gap
- carry_drop
- late_hit
- poor_query_quality

Insight: What leads to composition failure?
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hallucination_rag_plots.hall_plot_utils import (
    load_hallucination_judgments, load_coverage_judgments, 
    load_quality_judgments, create_merged_dataset, has_poor_query_quality
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'rag_analysis' / 'output'
PLOT_DIR = Path(__file__).resolve().parent


def main():
    """Generate composition failure root causes plot."""
    # Load all judgment types
    hall_records = load_hallucination_judgments(OUTPUT_DIR)
    cov_records = load_coverage_judgments(OUTPUT_DIR)
    qual_records = load_quality_judgments(OUTPUT_DIR)
    
    # Merge datasets
    merged = create_merged_dataset(hall_records, cov_records, qual_records)
    
    # Analyze composition failures
    comp_failures = [rec for rec in merged 
                     if rec['hallucination'].get('composition_and_faithfulness', {}).get('composition_failure', False)]
    
    print(f"\nTotal composition failures: {len(comp_failures)}")
    
    if not comp_failures:
        print("No composition failures found!")
        return
    
    # Count co-occurrences
    root_causes = {
        'Coverage Gap': 0,
        'Anchor Carry-Drop': 0,
        'Late Hit': 0,
        'Poor Query Quality': 0
    }
    
    for rec in comp_failures:
        cov = rec.get('coverage', {})
        qual = rec.get('quality', {})
        
        if cov:
            if cov.get('retrieval_coverage_gap', {}).get('has_gap', False):
                root_causes['Coverage Gap'] += 1
            
            if cov.get('anchor_carry_drop', {}).get('any_carry_drop', False):
                root_causes['Anchor Carry-Drop'] += 1
            
            if cov.get('late_hit_per_hop', {}).get('any_late_hit', False):
                root_causes['Late Hit'] += 1
        
        if qual and has_poor_query_quality(qual):
            root_causes['Poor Query Quality'] += 1
    
    # Calculate percentages
    total = len(comp_failures)
    percentages = {k: 100 * v / total for k, v in root_causes.items()}
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    
    causes = list(root_causes.keys())
    values = [percentages[c] for c in causes]
    counts = [root_causes[c] for c in causes]
    
    colors = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6']
    
    bars = ax.bar(causes, values, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val, count in zip(bars, values, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{val:.1f}%\n(n={count})',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add reference line at 50%
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1.5,
              label='50% threshold')
    
    ax.set_ylabel('% of Composition Failures', fontsize=12, fontweight='bold')
    ax.set_xlabel('Root Cause', fontsize=12, fontweight='bold')
    ax.set_title(f'Composition Failure Root Causes\n(Total Composition Failures: n={total})', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(values) * 1.15)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=15, ha='right')
    
    plt.tight_layout()
    output_path = PLOT_DIR / '4_composition_failure_root_causes.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n=== Composition Failure Root Causes ===")
    print(f"Total composition failures: {total}")
    print("\nCo-occurrence with root causes:")
    for cause in causes:
        count = root_causes[cause]
        pct = percentages[cause]
        print(f"  {cause}: {count} ({pct:.1f}%)")


if __name__ == '__main__':
    main()
