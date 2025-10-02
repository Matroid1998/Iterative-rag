"""
Plot 7: Planning → Confidence

Compare % is_next_logical_hop (quality) vs % overconfident_finalize (hallucination)
per model using small multiples.

Insight: Does poor planning lead to overconfidence?
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
    count_logical_hops, is_overconfident
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'rag_analysis' / 'output'
PLOT_DIR = Path(__file__).resolve().parent


def main():
    """Generate planning vs confidence comparison plot."""
    cov_records, qual_records, hall_records = load_all_judgments(OUTPUT_DIR)
    merged = create_merged_dataset(cov_records, qual_records, hall_records)
    
    # Calculate metrics per model
    model_metrics = defaultdict(lambda: {
        'total_runs': 0,
        'total_steps': 0,
        'logical_steps': 0,
        'overconfident_runs': 0
    })
    
    for rec in merged:
        model = normalize_model_name(rec['model'])
        
        # Count logical hops
        if 'quality' in rec:
            logical, total_steps = count_logical_hops(rec.get('quality', {}))
            model_metrics[model]['total_steps'] += total_steps
            model_metrics[model]['logical_steps'] += logical
        
        # Count overconfident
        if 'hallucination' in rec:
            model_metrics[model]['total_runs'] += 1
            if is_overconfident(rec.get('hallucination', {})):
                model_metrics[model]['overconfident_runs'] += 1
    
    # Calculate percentages
    models = sorted(model_metrics.keys())
    logical_hop_pcts = []
    overconfident_pcts = []
    
    for model in models:
        metrics = model_metrics[model]
        
        logical_pct = 100 * metrics['logical_steps'] / metrics['total_steps'] if metrics['total_steps'] > 0 else 0
        logical_hop_pcts.append(logical_pct)
        
        overconf_pct = 100 * metrics['overconfident_runs'] / metrics['total_runs'] if metrics['total_runs'] > 0 else 0
        overconfident_pcts.append(overconf_pct)
    
    # Create side-by-side bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    x = np.arange(len(models))
    colors1 = plt.cm.RdYlGn(np.linspace(0.3, 0.8, len(models)))
    colors2 = plt.cm.RdYlGn_r(np.linspace(0.3, 0.8, len(models)))
    
    # Plot 1: Logical Hop %
    bars1 = ax1.bar(x, logical_hop_pcts, color=colors1, alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    
    for bar, pct in zip(bars1, logical_hop_pcts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('% Steps that are Next Logical Hop', fontsize=11, fontweight='bold')
    ax1.set_title('Planning Quality: Logical Hop Alignment', 
                  fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=30, ha='right')
    ax1.set_ylim(0, 110)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=100*np.mean(logical_hop_pcts)/100, color='blue', 
                linestyle='--', linewidth=2, alpha=0.5)
    
    # Plot 2: Overconfident %
    bars2 = ax2.bar(x, overconfident_pcts, color=colors2, alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    
    for bar, pct in zip(bars2, overconfident_pcts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('% Runs with Overconfident Finalization', fontsize=11, fontweight='bold')
    ax2.set_title('Confidence Miscalibration: Overconfidence Rate',
                  fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=30, ha='right')
    ax2.set_ylim(0, max(overconfident_pcts) * 1.3 if overconfident_pcts else 30)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.axhline(y=np.mean(overconfident_pcts), color='red',
                linestyle='--', linewidth=2, alpha=0.5)
    
    fig.suptitle('Planning Quality vs Confidence Calibration\n(Does poor planning lead to overconfidence?)', 
                 fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    output_path = PLOT_DIR / '7_planning_vs_confidence.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n=== Planning vs Confidence Statistics ===")
    for model, logical_pct, overconf_pct in zip(models, logical_hop_pcts, overconfident_pcts):
        metrics = model_metrics[model]
        print(f"\n{model}:")
        print(f"  Logical Hop Alignment: {logical_pct:.1f}% ({metrics['logical_steps']}/{metrics['total_steps']} steps)")
        print(f"  Overconfident Rate: {overconf_pct:.1f}% ({metrics['overconfident_runs']}/{metrics['total_runs']} runs)")
        
        # Calculate inverse relationship
        planning_quality = logical_pct / 100
        overconf_risk = overconf_pct / 100
        if planning_quality > 0:
            risk_ratio = overconf_risk / planning_quality
            print(f"  Overconfidence/Planning Ratio: {risk_ratio:.3f}")
    
    # Calculate correlation
    if len(logical_hop_pcts) > 1:
        correlation = np.corrcoef(logical_hop_pcts, overconfident_pcts)[0, 1]
        print(f"\nCorrelation (Planning Quality vs Overconfidence): {correlation:.3f}")
        print("(Negative correlation suggests poor planning → overconfidence)")


if __name__ == '__main__':
    main()
