"""
Plot 5: Composition Failure Rate by Model

Bar chart showing percentage of runs with composition failure per model.

Insight: Which models have higher composition failure rates?
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
    load_hallucination_judgments, normalize_model_name
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'rag_analysis' / 'output'
PLOT_DIR = Path(__file__).resolve().parent


def main():
    """Generate composition failure rate by model plot."""
    records = load_hallucination_judgments(OUTPUT_DIR)
    
    # Group by model
    model_stats = defaultdict(lambda: {'total': 0, 'failures': 0})
    
    for rec in records:
        model = normalize_model_name(rec.get('model', ''))
        cf = rec.get('parsed_judgment', {}).get('composition_and_faithfulness', {})
        
        model_stats[model]['total'] += 1
        if cf.get('composition_failure', False):
            model_stats[model]['failures'] += 1
    
    # Calculate percentages
    models = sorted(model_stats.keys())
    failure_rates = []
    failure_counts = []
    total_counts = []
    
    for model in models:
        stats = model_stats[model]
        rate = 100 * stats['failures'] / stats['total'] if stats['total'] > 0 else 0
        failure_rates.append(rate)
        failure_counts.append(stats['failures'])
        total_counts.append(stats['total'])
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(models))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(models)))
    
    bars = ax.bar(x, failure_rates, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, rate, fail, total) in enumerate(zip(bars, failure_rates, failure_counts, total_counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{rate:.1f}%\n({fail}/{total})',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add average line
    avg_rate = np.mean(failure_rates)
    ax.axhline(y=avg_rate, color='blue', linestyle='--', linewidth=2,
              label=f'Average: {avg_rate:.1f}%', alpha=0.7)
    
    ax.set_ylabel('Composition Failure Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Composition Failure Rate by Model', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.set_ylim(0, max(failure_rates) * 1.2)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = PLOT_DIR / '5_composition_failure_rate.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n=== Composition Failure Rate by Model ===")
    for model in models:
        stats = model_stats[model]
        rate = 100 * stats['failures'] / stats['total']
        print(f"\n{model}:")
        print(f"  Total runs: {stats['total']}")
        print(f"  Failures: {stats['failures']}")
        print(f"  Failure rate: {rate:.1f}%")


if __name__ == '__main__':
    main()
