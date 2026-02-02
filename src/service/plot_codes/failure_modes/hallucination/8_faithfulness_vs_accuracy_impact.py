"""
Plot 8: Faithfulness vs Accuracy Impact

Shows how faithfulness (sufficiency_score_est) relates to accuracy performance.
Includes scatter plot with trend lines and correlation analysis.

Insight: How does being unfaithful impact model accuracy?
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hallucination.hall_plot_utils import (
    load_hallucination_judgments, load_coverage_judgments,
    create_merged_dataset, normalize_model_name
)

OUTPUT_DIR = Path(__file__).resolve().parents[5] / "data" / "results" / "failure_modes"
PLOT_DIR = Path(__file__).resolve().parents[5] / "data" / "plots" / "failure_modes" / "hallucination"


def main():
    """Generate faithfulness vs accuracy impact plot."""
    # Load all judgments
    hall_records = load_hallucination_judgments(OUTPUT_DIR)
    cov_records = load_coverage_judgments(OUTPUT_DIR)
    
    # Merge datasets to get is_correct field
    merged = create_merged_dataset(hall_records, cov_records, [])
    
    # Group by model and faithfulness bins
    model_data = defaultdict(lambda: {'faithfulness': [], 'accuracy': []})
    
    for rec in merged:
        model = normalize_model_name(rec.get('model', ''))
        is_correct = rec.get('is_correct', False)
        cf = rec.get('hallucination', {}).get('composition_and_faithfulness', {})
        faithfulness = cf.get('sufficiency_score_est')
        
        if faithfulness is not None:
            model_data[model]['faithfulness'].append(float(faithfulness))
            model_data[model]['accuracy'].append(1 if is_correct else 0)
    
    # Create binned analysis for each model
    models = sorted(model_data.keys())
    n_models = len(models)
    
    # Use 2 rows of 3 columns
    # Calculate grid size (3 columns, enough rows to fit all models)
    num_models = len(models)
    ncols = 3
    nrows = (num_models + ncols - 1) // ncols  # Ceiling division
    
    # Create figure with calculated subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)  # Ensure 2D array
    axes = axes.flatten()
    
    # Color scheme for models
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        data = model_data[model]
        
        if not data['faithfulness']:
            ax.text(0.5, 0.5, f'No data for {model}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model, fontsize=12, fontweight='bold')
            continue
        
        faithfulness = np.array(data['faithfulness'])
        accuracy = np.array(data['accuracy'])
        
        # Create bins for faithfulness
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_accuracies = []
        bin_counts = []
        
        for i in range(n_bins):
            mask = (faithfulness >= bins[i]) & (faithfulness < bins[i + 1])
            if i == n_bins - 1:  # Include the last bin edge
                mask = (faithfulness >= bins[i]) & (faithfulness <= bins[i + 1])
            
            if np.sum(mask) > 0:
                bin_acc = np.mean(accuracy[mask])
                bin_accuracies.append(bin_acc * 100)  # Convert to percentage
                bin_counts.append(np.sum(mask))
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)
        
        # Plot bars with size proportional to count
        max_count = max(bin_counts) if bin_counts else 1
        alphas = [0.3 + 0.7 * (count / max_count) for count in bin_counts]
        
        bars = ax.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7,
                     color=colors[idx], edgecolor='black', linewidth=1)
        
        # Add count labels on bars
        for i, (bar, count, acc) in enumerate(zip(bars, bin_counts, bin_accuracies)):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                       f'n={count}', ha='center', va='bottom', fontsize=8, alpha=0.8)
        
        # Add trend line
        if len(bin_centers) > 1 and any(c > 0 for c in bin_counts):
            # Filter out empty bins for trend line
            valid_mask = np.array(bin_counts) > 0
            if np.sum(valid_mask) > 1:
                valid_centers = np.array(bin_centers)[valid_mask]
                valid_accs = np.array(bin_accuracies)[valid_mask]
                
                z = np.polyfit(valid_centers, valid_accs, 1)
                p = np.poly1d(z)
                x_line = np.linspace(0, 1, 100)
                ax.plot(x_line, p(x_line), '--', color='red', alpha=0.7, linewidth=2)
                
                # Calculate correlation on individual points
                correlation = np.corrcoef(faithfulness, accuracy * 100)[0, 1]
                ax.text(0.02, 0.98, f'r = {correlation:.3f}', 
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        ax.set_xlabel('Faithfulness Score', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{model}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Impact of Faithfulness on Accuracy by Model\n(Bar height = accuracy, Bar opacity = sample size)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_path = PLOT_DIR / '8_faithfulness_vs_accuracy_impact.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print overall statistics
    print("\n=== Faithfulness vs Accuracy Impact ===")
    for model in models:
        data = model_data[model]
        if data['faithfulness']:
            faithfulness = np.array(data['faithfulness'])
            accuracy = np.array(data['accuracy'])
            
            # Calculate correlation
            correlation = np.corrcoef(faithfulness, accuracy)[0, 1]
            
            # Calculate accuracy by faithfulness quartiles
            q1, q2, q3 = np.percentile(faithfulness, [25, 50, 75])
            
            low_faith = accuracy[faithfulness <= q1]
            mid_faith = accuracy[(faithfulness > q1) & (faithfulness <= q3)]
            high_faith = accuracy[faithfulness > q3]
            
            print(f"\n{model}:")
            print(f"  Overall correlation: {correlation:.3f}")
            print(f"  Low faithfulness (≤{q1:.2f}): {np.mean(low_faith)*100:.1f}% accuracy (n={len(low_faith)})")
            print(f"  Mid faithfulness ({q1:.2f}-{q3:.2f}): {np.mean(mid_faith)*100:.1f}% accuracy (n={len(mid_faith)})")
            print(f"  High faithfulness (>{q3:.2f}): {np.mean(high_faith)*100:.1f}% accuracy (n={len(high_faith)})")


if __name__ == '__main__':
    main()