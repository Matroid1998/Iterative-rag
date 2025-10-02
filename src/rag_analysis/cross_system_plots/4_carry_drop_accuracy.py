"""
Plot 4: Anchor Carry-Drop Impact on Accuracy

Grouped bar chart showing accuracy for runs WITH vs WITHOUT anchor carry-drop,
grouped by model.

Insight: Quantify how much anchor carry-drop hurts performance.
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
    load_all_judgments, create_merged_dataset, normalize_model_name,
    has_carry_drop
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'rag_analysis' / 'output'
PLOT_DIR = Path(__file__).resolve().parent


def main():
    """Generate carry-drop impact on accuracy plot."""
    cov_records, qual_records, hall_records = load_all_judgments(OUTPUT_DIR)
    merged = create_merged_dataset(cov_records, qual_records, hall_records)
    
    # Group by model and carry-drop status
    model_stats = defaultdict(lambda: {
        'with_carry_drop': {'total': 0, 'correct': 0},
        'without_carry_drop': {'total': 0, 'correct': 0}
    })
    
    for rec in merged:
        model = normalize_model_name(rec['model'])
        is_correct = rec.get('is_correct', False)
        has_cd = has_carry_drop(rec.get('coverage', {}))
        
        if has_cd:
            model_stats[model]['with_carry_drop']['total'] += 1
            if is_correct:
                model_stats[model]['with_carry_drop']['correct'] += 1
        else:
            model_stats[model]['without_carry_drop']['total'] += 1
            if is_correct:
                model_stats[model]['without_carry_drop']['correct'] += 1
    
    # Prepare data for plotting
    models = sorted(model_stats.keys())
    with_cd_acc = []
    without_cd_acc = []
    with_cd_counts = []
    without_cd_counts = []
    
    for model in models:
        stats = model_stats[model]
        
        with_total = stats['with_carry_drop']['total']
        with_correct = stats['with_carry_drop']['correct']
        with_acc = 100 * with_correct / with_total if with_total > 0 else 0
        with_cd_acc.append(with_acc)
        with_cd_counts.append(with_total)
        
        without_total = stats['without_carry_drop']['total']
        without_correct = stats['without_carry_drop']['correct']
        without_acc = 100 * without_correct / without_total if without_total > 0 else 0
        without_cd_acc.append(without_acc)
        without_cd_counts.append(without_total)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, without_cd_acc, width, 
                   label='Without Carry-Drop', color='#2ecc71', 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, with_cd_acc, width,
                   label='With Carry-Drop', color='#e74c3c',
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar1, bar2, without_acc, with_acc, without_count, with_count) in enumerate(
        zip(bars1, bars2, without_cd_acc, with_cd_acc, without_cd_counts, with_cd_counts)):
        
        # Without carry-drop
        height1 = bar1.get_height()
        ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 1,
               f'{without_acc:.1f}%\n(n={without_count})',
               ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # With carry-drop
        height2 = bar2.get_height()
        ax.text(bar2.get_x() + bar2.get_width()/2., height2 + 1,
               f'{with_acc:.1f}%\n(n={with_count})',
               ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Calculate and show accuracy drop
        if without_acc > 0 and with_acc > 0:
            drop = without_acc - with_acc
            drop_pct = -100 * drop / without_acc if without_acc > 0 else 0
            ax.text(i, max(height1, height2) + 8,
                   f'Δ: {drop:+.1f}pp\n({drop_pct:+.1f}%)',
                   ha='center', va='bottom', fontsize=7,
                   color='red' if drop > 0 else 'green',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Impact of Anchor Carry-Drop on Accuracy\n(accuracy drop when key entities are not carried forward)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.set_ylim(0, max(max(without_cd_acc), max(with_cd_acc)) * 1.2)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = PLOT_DIR / '4_carry_drop_accuracy.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n=== Carry-Drop Impact on Accuracy ===")
    for model, without_acc, with_acc, without_count, with_count in zip(
        models, without_cd_acc, with_cd_acc, without_cd_counts, with_cd_counts):
        
        print(f"\n{model}:")
        print(f"  Without Carry-Drop: {without_acc:.1f}% (n={without_count})")
        print(f"  With Carry-Drop: {with_acc:.1f}% (n={with_count})")
        
        if without_acc > 0:
            drop = without_acc - with_acc
            drop_pct = -100 * drop / without_acc
            print(f"  Accuracy Drop: {drop:+.1f} percentage points ({drop_pct:+.1f}%)")


if __name__ == '__main__':
    main()
