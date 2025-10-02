"""
Plot 8: Coverage vs Confidence Scatter

Scatter plot of hop_coverage_est vs sufficiency_score_est colored by 
miscalibration direction to visualize regimes that drive miscalibration.

Insight: What combinations of coverage and confidence lead to miscalibration?
"""
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hallucination_rag_plots.hall_plot_utils import load_hallucination_judgments

OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'rag_analysis' / 'output'
PLOT_DIR = Path(__file__).resolve().parent


def main():
    """Generate coverage vs confidence scatter plot."""
    records = load_hallucination_judgments(OUTPUT_DIR)
    
    # Extract data by direction
    data_by_direction = {
        'ok': {'cov': [], 'suff': []},
        'underconfident_continue': {'cov': [], 'suff': []},
        'overconfident_finalize': {'cov': [], 'suff': []}
    }
    
    for rec in records:
        judgment = rec.get('parsed_judgment', {})
        cf = judgment.get('composition_and_faithfulness', {})
        cm = judgment.get('confidence_miscalibration', {})
        
        suff = cf.get('sufficiency_score_est')
        cov = cm.get('hop_coverage_est')
        direction = cm.get('direction', 'ok')
        
        if suff is not None and cov is not None:
            data_by_direction[direction]['cov'].append(float(cov))
            data_by_direction[direction]['suff'].append(float(suff))
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
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
    
    markers = {
        'ok': 'o',
        'underconfident_continue': '^',
        'overconfident_finalize': 's'
    }
    
    # Plot each direction
    for direction in ['ok', 'underconfident_continue', 'overconfident_finalize']:
        data = data_by_direction[direction]
        if not data['cov']:
            continue
        
        ax.scatter(data['cov'], data['suff'],
                  s=40, alpha=0.5,
                  color=colors[direction],
                  marker=markers[direction],
                  label=f"{labels[direction]} (n={len(data['cov'])})",
                  edgecolors='white', linewidth=0.3)
    
    # Add threshold lines
    ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, linewidth=1.5,
              label='Coverage threshold (0.8)')
    ax.axhline(y=0.6, color='gray', linestyle=':', alpha=0.5, linewidth=1.5,
              label='Sufficiency threshold (0.6)')
    
    # Add quadrant shading and labels
    # High coverage, high sufficiency (good regime)
    rect1 = Rectangle((0.8, 0.6), 0.2, 0.4, alpha=0.1, facecolor='green')
    ax.add_patch(rect1)
    ax.text(0.9, 0.8, 'GOOD\nHigh Cov\nHigh Suff', ha='center', va='center',
           fontsize=10, fontweight='bold', alpha=0.6,
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Low coverage, high sufficiency (overconfident risk)
    rect2 = Rectangle((0, 0.6), 0.8, 0.4, alpha=0.05, facecolor='red')
    ax.add_patch(rect2)
    ax.text(0.4, 0.8, 'OVERCONFIDENT RISK\nLow Cov, High Suff', 
           ha='center', va='center', fontsize=9, style='italic', alpha=0.5,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.4))
    
    # High coverage, low sufficiency (underconfident risk)
    rect3 = Rectangle((0.8, 0), 0.2, 0.6, alpha=0.05, facecolor='blue')
    ax.add_patch(rect3)
    ax.text(0.9, 0.3, 'UNDER-\nCONFIDENT\nRISK', ha='center', va='center',
           fontsize=9, style='italic', alpha=0.5,
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4))
    
    # Low coverage, low sufficiency (poor regime)
    ax.text(0.3, 0.2, 'POOR REGIME\nLow Cov\nLow Suff', ha='center', va='center',
           fontsize=9, style='italic', alpha=0.5,
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.4))
    
    ax.set_xlabel('Hop Coverage Estimate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sufficiency Score', fontsize=12, fontweight='bold')
    ax.set_title('Coverage vs Confidence by Miscalibration Direction\n(Identifying Miscalibration Regimes)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower left', framealpha=0.95, fontsize=10, ncol=2)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = PLOT_DIR / '8_coverage_vs_confidence.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n=== Coverage vs Confidence Statistics ===")
    for direction, label in labels.items():
        data = data_by_direction[direction]
        if data['cov']:
            print(f"\n{label} (n={len(data['cov'])}):")
            print(f"  Avg Coverage: {np.mean(data['cov']):.3f}")
            print(f"  Avg Sufficiency: {np.mean(data['suff']):.3f}")
            
            # Count quadrants
            high_cov_high_suff = sum(1 for c, s in zip(data['cov'], data['suff']) 
                                     if c >= 0.8 and s >= 0.6)
            low_cov_high_suff = sum(1 for c, s in zip(data['cov'], data['suff']) 
                                    if c < 0.8 and s >= 0.6)
            high_cov_low_suff = sum(1 for c, s in zip(data['cov'], data['suff']) 
                                    if c >= 0.8 and s < 0.6)
            low_cov_low_suff = sum(1 for c, s in zip(data['cov'], data['suff']) 
                                   if c < 0.8 and s < 0.6)
            
            total = len(data['cov'])
            print(f"  High Cov & High Suff: {100*high_cov_high_suff/total:.1f}%")
            print(f"  Low Cov & High Suff: {100*low_cov_high_suff/total:.1f}%")
            print(f"  High Cov & Low Suff: {100*high_cov_low_suff/total:.1f}%")
            print(f"  Low Cov & Low Suff: {100*low_cov_low_suff/total:.1f}%")


if __name__ == '__main__':
    main()
