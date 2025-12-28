"""
Plot 2c: Sufficiency vs Coverage - Accuracy Heatmap by Quadrants

Creates a heatmap showing accuracy in different regions defined by 
sufficiency and coverage thresholds.
"""
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_system_plots.cross_system_utils import (
    load_all_judgments, create_merged_dataset, normalize_model_name
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'rag_analysis' / 'output'
PLOT_DIR = Path(__file__).resolve().parent


def main():
    """Generate accuracy heatmap based on sufficiency and coverage thresholds."""
    # Load and merge all judgment types
    cov_records, qual_records, hall_records = load_all_judgments(OUTPUT_DIR)
    merged = create_merged_dataset(cov_records, qual_records, hall_records)
    
    # Filter to records with hallucination data
    complete = [r for r in merged if 'hallucination' in r]
    
    # Extract all data points
    data_points = []
    
    for rec in complete:
        judgment = rec.get('hallucination', {})
        cf = judgment.get('composition_and_faithfulness', {})
        cm = judgment.get('confidence_miscalibration', {})
        
        suff = cf.get('sufficiency_score_est')
        cov = cm.get('hop_coverage_est')
        is_correct = rec.get('is_correct', False)
        
        if suff is not None and cov is not None:
            data_points.append({
                'suff': float(suff),
                'cov': float(cov),
                'correct': is_correct
            })
    
    print(f"Total data points: {len(data_points)}")
    
    # Define thresholds to create 6 zones (2 rows x 3 columns)
    suff_threshold_low = 0.4
    suff_threshold_high = 0.6
    cov_threshold = 0.8
    
    # Create 2x3 matrix for 6 zones
    # Rows: [High Coverage, Low Coverage]
    # Cols: [Low Sufficiency (<0.4), Mid Sufficiency (0.4-0.6), High Sufficiency (≥0.6)]
    accuracy_matrix = np.zeros((2, 3))
    count_matrix = np.zeros((2, 3))
    
    for point in data_points:
        # Determine zone
        if point['suff'] < suff_threshold_low:
            suff_idx = 0  # Low
        elif point['suff'] < suff_threshold_high:
            suff_idx = 1  # Mid
        else:
            suff_idx = 2  # High
        
        cov_idx = 0 if point['cov'] >= cov_threshold else 1
        
        count_matrix[cov_idx, suff_idx] += 1
        if point['correct']:
            accuracy_matrix[cov_idx, suff_idx] += 1
    
    # Calculate accuracy percentages (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        accuracy_matrix = np.where(count_matrix > 0, 
                                   100 * accuracy_matrix / count_matrix, 
                                   np.nan)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 8))
    
    # Create heatmap
    im = ax.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', 
                   vmin=0, vmax=100, interpolation='nearest')
    
    # Add threshold lines (vertical lines between columns, horizontal line between rows)
    ax.axvline(x=0.5, color='blue', linestyle='--', linewidth=3, alpha=0.8)
    ax.axvline(x=1.5, color='blue', linestyle='--', linewidth=3, alpha=0.8)
    ax.axhline(y=0.5, color='blue', linestyle='--', linewidth=3, alpha=0.8)
    
    # Set grid
    ax.set_xticks([0.5, 1.5], minor=True)
    ax.set_yticks([0.5], minor=True)
    ax.grid(which='minor', color='blue', linestyle='--', linewidth=3, alpha=0.8)
    
    # Set major ticks at cell centers
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1])
    
    # Labels for ticks
    ax.set_xticklabels([f'< {suff_threshold_low}', 
                        f'{suff_threshold_low}-{suff_threshold_high}', 
                        f'≥ {suff_threshold_high}'], fontsize=14, fontweight='bold')
    ax.set_yticklabels([f'≥ {cov_threshold}', f'< {cov_threshold}'], fontsize=14, fontweight='bold')
    
    # Add text annotations for each cell (6 zones)
    zone_labels = [
        ['Low Suff.\nHigh Cov.', 'Mid Suff.\nHigh Cov.', 'High Suff.\nHigh Cov.'],
        ['Low Suff.\nLow Cov.', 'Mid Suff.\nLow Cov.', 'High Suff.\nLow Cov.']
    ]
    
    for i in range(2):
        for j in range(3):
            count = count_matrix[i, j]
            acc = accuracy_matrix[i, j]
            
            if count > 0 and not np.isnan(acc):
                # Choose text color based on accuracy
                text_color = 'black'
                
                # Main text: accuracy
                txt1 = ax.text(j, i - 0.2, f'{acc:.1f}%', ha='center', va='center',
                       color=text_color, fontsize=36, fontweight='bold')
                
                # Zone label
                txt2 = ax.text(j, i + 0.05, zone_labels[i][j], ha='center', va='center',
                       color=text_color, fontsize=14, fontweight='bold', alpha=0.9)
                
                # Count
                txt3 = ax.text(j, i + 0.28, f'n = {int(count)}', ha='center', va='center',
                       color=text_color, fontsize=12, fontweight='bold')
    
    # Labels
    ax.set_xlabel('Sufficiency Score', fontsize=16, fontweight='bold')
    ax.set_ylabel('Hop Coverage', fontsize=16, fontweight='bold')
    ax.set_title('Accuracy Heatmap: Sufficiency vs Coverage\n' +
                 f'(Thresholds: Sufficiency: <{suff_threshold_low}, {suff_threshold_low}-{suff_threshold_high}, ≥{suff_threshold_high} | Coverage: ≥{cov_threshold})',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label('Accuracy (%)', rotation=270, labelpad=25, fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    output_path = PLOT_DIR / '2c_sufficiency_vs_coverage_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    # Calculate and print zone statistics
    print("\n" + "="*80)
    print("ZONE STATISTICS (6 ZONES)")
    print("="*80)
    
    zones = {
        'Low Suff. (<0.4) & High Cov. (≥0.8)': (0, suff_threshold_low, cov_threshold, 1),
        'Mid Suff. (0.4-0.6) & High Cov. (≥0.8)': (suff_threshold_low, suff_threshold_high, cov_threshold, 1),
        'High Suff. (≥0.6) & High Cov. (≥0.8)': (suff_threshold_high, 1, cov_threshold, 1),
        'Low Suff. (<0.4) & Low Cov. (<0.8)': (0, suff_threshold_low, 0, cov_threshold),
        'Mid Suff. (0.4-0.6) & Low Cov. (<0.8)': (suff_threshold_low, suff_threshold_high, 0, cov_threshold),
        'High Suff. (≥0.6) & Low Cov. (<0.8)': (suff_threshold_high, 1, 0, cov_threshold)
    }
    
    for zone_name, (suff_min, suff_max, cov_min, cov_max) in zones.items():
        zone_points = [p for p in data_points 
                      if suff_min <= p['suff'] < suff_max 
                      and cov_min <= p['cov'] < cov_max]
        
        if zone_points:
            correct_count = sum(1 for p in zone_points if p['correct'])
            total_count = len(zone_points)
            accuracy = 100 * correct_count / total_count
            
            print(f"\n{zone_name}:")
            print(f"  Total: {total_count}")
            print(f"  Correct: {correct_count}")
            print(f"  Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    main()
