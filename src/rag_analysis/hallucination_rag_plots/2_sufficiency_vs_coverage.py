"""
Plot 2: Sufficiency vs Coverage Scatter with Miscalibration

Scatter plot showing the relationship between sufficiency score and hop coverage,
colored by miscalibration direction, with point size indicating unsupported claims.

Insight: Can we predict miscalibration from sufficiency and coverage scores?
"""
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hallucination_rag_plots.hall_plot_utils import (
    load_hallucination_judgments, count_unsupported_claims
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'rag_analysis' / 'output'
PLOT_DIR = Path(__file__).resolve().parent


def main():
    """Generate sufficiency vs coverage scatter plot."""
    records = load_hallucination_judgments(OUTPUT_DIR)
    
    # Extract data points
    data_by_direction = {
        'ok': {'suff': [], 'cov': [], 'unsup': []},
        'underconfident_continue': {'suff': [], 'cov': [], 'unsup': []},
        'overconfident_finalize': {'suff': [], 'cov': [], 'unsup': []}
    }
    
    for rec in records:
        judgment = rec.get('parsed_judgment', {})
        cf = judgment.get('composition_and_faithfulness', {})
        cm = judgment.get('confidence_miscalibration', {})
        
        suff = cf.get('sufficiency_score_est')
        cov = cm.get('hop_coverage_est')
        direction = cm.get('direction', 'ok')
        
        if suff is not None and cov is not None:
            unsupported = count_unsupported_claims(judgment)
            
            data_by_direction[direction]['suff'].append(float(suff))
            data_by_direction[direction]['cov'].append(float(cov))
            data_by_direction[direction]['unsup'].append(unsupported)
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {
        'ok': '#2ecc71',
        'underconfident_continue': '#3498db',
        'overconfident_finalize': '#e74c3c'
    }
    
    labels = {
        'ok': 'OK',
        'underconfident_continue': 'Underconfident',
        'overconfident_finalize': 'Overconfident'
    }
    
    # Plot each direction
    for direction in ['ok', 'underconfident_continue', 'overconfident_finalize']:
        data = data_by_direction[direction]
        if not data['suff']:
            continue
        
        # Scale point sizes (unsupported claims)
        sizes = [max(20, min(200, 20 + u * 30)) for u in data['unsup']]
        
        ax.scatter(data['suff'], data['cov'], 
                  s=sizes, alpha=0.6, 
                  color=colors[direction],
                  label=f"{labels[direction]} (n={len(data['suff'])})",
                  edgecolors='white', linewidth=0.5)
    
    # Add reference lines
    ax.axvline(x=0.6, color='gray', linestyle='--', alpha=0.5, linewidth=1, 
               label='Sufficiency threshold (0.6)')
    ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5, linewidth=1,
               label='Coverage threshold (0.8)')
    
    # Add quadrant annotations
    ax.text(0.3, 0.9, 'Low Suff\nHigh Cov', ha='center', va='center',
           fontsize=9, style='italic', alpha=0.5, bbox=dict(boxstyle='round', 
           facecolor='white', alpha=0.7))
    ax.text(0.85, 0.9, 'High Suff\nHigh Cov', ha='center', va='center',
           fontsize=9, style='italic', alpha=0.5, bbox=dict(boxstyle='round',
           facecolor='white', alpha=0.7))
    ax.text(0.3, 0.4, 'Low Suff\nLow Cov', ha='center', va='center',
           fontsize=9, style='italic', alpha=0.5, bbox=dict(boxstyle='round',
           facecolor='white', alpha=0.7))
    ax.text(0.85, 0.4, 'High Suff\nLow Cov', ha='center', va='center',
           fontsize=9, style='italic', alpha=0.5, bbox=dict(boxstyle='round',
           facecolor='white', alpha=0.7))
    
    ax.set_xlabel('Sufficiency Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Hop Coverage Estimate', fontsize=12, fontweight='bold')
    ax.set_title('Sufficiency vs Coverage by Miscalibration Direction\n(point size = unsupported claims)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower left', framealpha=0.95, fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = PLOT_DIR / '2_sufficiency_vs_coverage.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n=== Sufficiency vs Coverage Statistics ===")
    for direction, label in labels.items():
        data = data_by_direction[direction]
        if data['suff']:
            print(f"\n{label} (n={len(data['suff'])}):")
            print(f"  Avg Sufficiency: {np.mean(data['suff']):.3f}")
            print(f"  Avg Coverage: {np.mean(data['cov']):.3f}")
            print(f"  Avg Unsupported: {np.mean(data['unsup']):.2f}")


if __name__ == '__main__':
    main()
