"""
Plot 2b: Sufficiency vs Coverage Scatter - Averaged Across All Models

Single scatter plot showing the relationship between sufficiency score and hop coverage,
averaged across all models, colored by answer correctness (correct/incorrect), 
with point size indicating unsupported claims.

Insight: Overall relationship between sufficiency, coverage and correctness across all models.
"""
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_system.cross_system_utils import (
    load_all_judgments, create_merged_dataset, normalize_model_name
)

OUTPUT_DIR = Path(__file__).resolve().parents[5] / "data" / "results" / "failure_modes"
PLOT_DIR = Path(__file__).resolve().parents[5] / "data" / "plots" / "failure_modes" / "hallucination"


def count_unsupported_claims(hallucination_judgment):
    """Count unsupported claims in the judgment."""
    if not hallucination_judgment:
        return 0
    
    comp_faith = hallucination_judgment.get('composition_and_faithfulness', {})
    unsupported = comp_faith.get('unsupported_claims', [])
    
    count = 0
    for claim in unsupported:
        if not claim.get('is_supported', True):
            count += 1
    
    return count


def main():
    """Generate averaged sufficiency vs coverage scatter plot."""
    # Load and merge all judgment types
    cov_records, qual_records, hall_records = load_all_judgments(OUTPUT_DIR)
    merged = create_merged_dataset(cov_records, qual_records, hall_records)
    
    # Filter to records with hallucination data
    complete = [r for r in merged if 'hallucination' in r]
    
    # Extract all data points (aggregated across all models)
    all_data = {
        'correct': {'suff': [], 'cov': [], 'unsup': []},
        'incorrect': {'suff': [], 'cov': [], 'unsup': []}
    }
    
    model_counts = {}
    
    for rec in complete:
        model = normalize_model_name(rec.get('model', ''))
        judgment = rec.get('hallucination', {})
        cf = judgment.get('composition_and_faithfulness', {})
        cm = judgment.get('confidence_miscalibration', {})
        
        suff = cf.get('sufficiency_score_est')
        cov = cm.get('hop_coverage_est')
        is_correct = rec.get('is_correct', False)
        
        if suff is not None and cov is not None:
            category = 'correct' if is_correct else 'incorrect'
            unsupported = count_unsupported_claims(judgment)
            
            all_data[category]['suff'].append(float(suff))
            all_data[category]['cov'].append(float(cov))
            all_data[category]['unsup'].append(unsupported)
            
            # Count by model
            model_counts[model] = model_counts.get(model, 0) + 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {
        'correct': '#2ecc71',      # Green for correct
        'incorrect': '#e74c3c'      # Red for incorrect
    }
    
    labels = {
        'correct': 'Correct Answers',
        'incorrect': 'Incorrect Answers'
    }
    
    # Plot each category (correct/incorrect)
    # Plot incorrect first (bottom layer), then correct (top layer) for better visibility
    for category in ['incorrect', 'correct']:
        data = all_data[category]
        if not data['suff']:
            continue
        
        # Add jitter to points at coverage=1.0 to reduce overlap
        suff_values = np.array(data['suff'])
        cov_values = np.array(data['cov'])
        
        # Apply small random jitter to coverage values that are exactly 1.0
        np.random.seed(42 if category == 'correct' else 43)  # Different seeds for separation
        jitter_mask = (cov_values >= 0.999)
        if jitter_mask.any():
            # Jitter coverage slightly downward and sufficiency sideways
            cov_values = cov_values.copy()
            suff_values = suff_values.copy()
            cov_values[jitter_mask] -= np.random.uniform(0.001, 0.015, size=jitter_mask.sum())
            suff_values[jitter_mask] += np.random.uniform(-0.008, 0.008, size=jitter_mask.sum())
        
        # Scale point sizes (unsupported claims)
        sizes = [max(10, min(200, 10 + u * 20)) for u in data['unsup']]
        
        # Use higher alpha for correct (green) to make it more visible
        alpha_val = 0.6 if category == 'correct' else 0.4
        
        ax.scatter(suff_values, cov_values, 
                  s=sizes, alpha=alpha_val, 
                  color=colors[category],
                  label=labels[category],
                  edgecolors='white', linewidth=0.5)
    
    # Add reference lines with labels
    ax.axvline(x=0.6, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(0.61, 0.02, 'Sufficiency\nThreshold', fontsize=9, color='gray', 
            verticalalignment='bottom', fontweight='bold')
    
    ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.text(0.02, 0.81, 'Coverage\nThreshold', fontsize=9, color='gray', 
            verticalalignment='bottom', fontweight='bold')
    
    # Add quadrant shading (subtle)
    ax.fill_between([0.6, 1.05], 0.8, 1.05, alpha=0.05, color='green', zorder=0)
    ax.text(0.82, 0.92, 'High Suff.\n& Coverage', fontsize=10, 
            color='darkgreen', ha='center', fontweight='bold', alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Sufficiency Score', fontsize=13, fontweight='bold')
    ax.set_ylabel('Hop Coverage', fontsize=13, fontweight='bold')
    ax.set_title('Sufficiency vs Coverage: Averaged Across All Models\n' +
                 '(Point size indicates number of unsupported claims)',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Legend
    legend = ax.legend(loc='lower right', fontsize=11, framealpha=0.95, 
                      edgecolor='black', fancybox=True)
    
    # Add model count annotation
    num_models = len(model_counts)
    total_points = sum(len(all_data[cat]['suff']) for cat in ['correct', 'incorrect'])
    ax.text(0.02, 0.98, f'Aggregated from {num_models} models\nTotal: {total_points} answers',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black'))
    
    plt.tight_layout()
    output_path = PLOT_DIR / '2b_sufficiency_vs_coverage_averaged.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n" + "="*80)
    print("SUFFICIENCY VS COVERAGE STATISTICS (AVERAGED ACROSS ALL MODELS)")
    print("="*80)
    print(f"\nModels included: {num_models}")
    print(f"Model breakdown: {dict(sorted(model_counts.items()))}")
    
    for category, label in labels.items():
        data = all_data[category]
        if data['suff']:
            print(f"\n{label} (n={len(data['suff'])}):")
            print(f"  Sufficiency Score:")
            print(f"    Mean: {np.mean(data['suff']):.3f}")
            print(f"    Std:  {np.std(data['suff']):.3f}")
            print(f"    Min:  {np.min(data['suff']):.3f}")
            print(f"    Max:  {np.max(data['suff']):.3f}")
            
            print(f"  Hop Coverage:")
            print(f"    Mean: {np.mean(data['cov']):.3f}")
            print(f"    Std:  {np.std(data['cov']):.3f}")
            print(f"    Min:  {np.min(data['cov']):.3f}")
            print(f"    Max:  {np.max(data['cov']):.3f}")
            
            print(f"  Unsupported Claims:")
            print(f"    Mean: {np.mean(data['unsup']):.2f}")
            print(f"    Std:  {np.std(data['unsup']):.2f}")
            print(f"    Max:  {np.max(data['unsup'])}")
            
            # High quality zone stats
            high_quality = sum(1 for s, c in zip(data['suff'], data['cov']) 
                             if s >= 0.6 and c >= 0.8)
            high_quality_pct = 100 * high_quality / len(data['suff'])
            print(f"  High Quality (Suff≥0.6 & Cov≥0.8): {high_quality} ({high_quality_pct:.1f}%)")
    
    # Comparative statistics
    print("\n" + "="*80)
    print("COMPARATIVE STATISTICS")
    print("="*80)
    
    correct_suff = all_data['correct']['suff']
    incorrect_suff = all_data['incorrect']['suff']
    correct_cov = all_data['correct']['cov']
    incorrect_cov = all_data['incorrect']['cov']
    
    if correct_suff and incorrect_suff:
        suff_diff = np.mean(correct_suff) - np.mean(incorrect_suff)
        cov_diff = np.mean(correct_cov) - np.mean(incorrect_cov)
        
        print(f"\nCorrect vs Incorrect:")
        print(f"  Sufficiency difference: {suff_diff:+.3f} (correct - incorrect)")
        print(f"  Coverage difference:    {cov_diff:+.3f} (correct - incorrect)")
        
        # Count unsupported claims
        correct_unsup = all_data['correct']['unsup']
        incorrect_unsup = all_data['incorrect']['unsup']
        unsup_diff = np.mean(incorrect_unsup) - np.mean(correct_unsup)
        print(f"  Unsupported claims diff: {unsup_diff:+.2f} (incorrect - correct)")


if __name__ == '__main__':
    main()
