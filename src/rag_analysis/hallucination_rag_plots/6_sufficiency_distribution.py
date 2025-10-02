"""
Plot 6: Evidence Sufficiency Distribution

Histogram/violin plot of sufficiency_score_est with threshold line at 0.6.

Insight: How well does the evidence support the answers?
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hallucination_rag_plots.hall_plot_utils import load_hallucination_judgments

OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'rag_analysis' / 'output'
PLOT_DIR = Path(__file__).resolve().parent


def main():
    """Generate sufficiency score distribution plot."""
    records = load_hallucination_judgments(OUTPUT_DIR)
    
    # Extract sufficiency scores
    sufficiency_scores = []
    
    for rec in records:
        cf = rec.get('parsed_judgment', {}).get('composition_and_faithfulness', {})
        suff = cf.get('sufficiency_score_est')
        if suff is not None:
            sufficiency_scores.append(float(suff))
    
    if not sufficiency_scores:
        print("No sufficiency scores found!")
        return
    
    # Create histogram with overlay
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                    gridspec_kw={'height_ratios': [2, 1]})
    
    # Histogram
    bins = np.linspace(0, 1, 21)
    counts, edges, patches = ax1.hist(sufficiency_scores, bins=bins, 
                                      alpha=0.7, color='#3498db',
                                      edgecolor='black', linewidth=0.5)
    
    # Color bars based on threshold
    threshold = 0.6
    for i, patch in enumerate(patches):
        if edges[i] < threshold:
            patch.set_facecolor('#e74c3c')  # Red for below threshold
        else:
            patch.set_facecolor('#2ecc71')  # Green for above threshold
    
    # Add threshold line
    ax1.axvline(x=threshold, color='black', linestyle='--', linewidth=2.5,
               label=f'Threshold: {threshold}', alpha=0.8)
    
    # Add statistics
    mean_val = np.mean(sufficiency_scores)
    median_val = np.median(sufficiency_scores)
    below_threshold = sum(1 for s in sufficiency_scores if s < threshold)
    below_pct = 100 * below_threshold / len(sufficiency_scores)
    
    ax1.axvline(x=mean_val, color='blue', linestyle=':', linewidth=2,
               label=f'Mean: {mean_val:.3f}', alpha=0.7)
    ax1.axvline(x=median_val, color='purple', linestyle='-.', linewidth=2,
               label=f'Median: {median_val:.3f}', alpha=0.7)
    
    # Add text box with statistics
    stats_text = f'n = {len(sufficiency_scores)}\n'
    stats_text += f'Below threshold: {below_pct:.1f}%\n'
    stats_text += f'Mean: {mean_val:.3f}\n'
    stats_text += f'Std: {np.std(sufficiency_scores):.3f}'
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Evidence Sufficiency Score Distribution', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Box plot
    bp = ax2.boxplot([sufficiency_scores], vert=False, widths=0.6,
                     patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=8),
                     medianprops=dict(color='black', linewidth=2),
                     boxprops=dict(facecolor='#3498db', alpha=0.7),
                     flierprops=dict(marker='o', markerfacecolor='red', 
                                    markersize=4, alpha=0.5))
    
    ax2.axvline(x=threshold, color='black', linestyle='--', linewidth=2.5, alpha=0.8)
    ax2.set_xlabel('Sufficiency Score', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_yticks([])
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = PLOT_DIR / '6_sufficiency_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n=== Sufficiency Score Statistics ===")
    print(f"Total runs: {len(sufficiency_scores)}")
    print(f"Below threshold ({threshold}): {below_pct:.1f}%")
    print(f"Mean: {mean_val:.3f}")
    print(f"Median: {median_val:.3f}")
    print(f"Std: {np.std(sufficiency_scores):.3f}")
    print(f"Min: {min(sufficiency_scores):.3f}")
    print(f"Max: {max(sufficiency_scores):.3f}")


if __name__ == '__main__':
    main()
