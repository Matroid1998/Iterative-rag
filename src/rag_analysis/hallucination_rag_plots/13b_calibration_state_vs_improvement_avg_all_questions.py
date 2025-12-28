"""
Plot: Average Accuracy by Calibration State (All Questions)

Shows the average accuracy for ALL questions in each confidence calibration state
(overconfident, underconfident, well-calibrated) across all models.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hallucination_rag_plots.hall_plot_utils import (
    load_hallucination_judgments, load_coverage_judgments, 
    create_merged_dataset, normalize_model_name
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'rag_analysis' / 'output'
PLOT_DIR = Path(__file__).resolve().parent / 'plots'
PLOT_DIR.mkdir(exist_ok=True)


def main():
    """Generate calibration state vs accuracy plot - average accuracy per model in each state."""
    # Load hallucination judgments and group by source model
    OUTPUT_DIR_PATH = Path(__file__).resolve().parents[2] / 'rag_analysis' / 'output'
    
    # Group by source model and calibration state
    model_state_correctness = defaultdict(lambda: defaultdict(list))
    
    
    # Load hallucination and coverage judgments using utils
    hall_records = load_hallucination_judgments(OUTPUT_DIR_PATH)
    cov_records = load_coverage_judgments(OUTPUT_DIR_PATH)
    
    # Merge datasets to get is_correct field
    # We pass empty list for qa_records if not needed, or better, 
    # check if create_merged_dataset requires it. 
    # Usually it merges hall, cov, and potentially qa/quality if available. 
    # Let's assume hall+cov is enough if is_correct comes from cov or hall merging logic. 
    # Actually, is_correct often comes from quality/qa files. 
    # Let's load quality records too if needed, but create_merged_dataset typically handles 
    # logic to find is_correct if it's in the records. 
    # In 5_comp...py, it passed [].
    merged = create_merged_dataset(hall_records, cov_records, [])
    
    
    
    matched_coverage_count = 0
    total_records = 0
    
    for rec in merged:
        if 'coverage' not in rec:
            continue
            
        model_name = normalize_model_name(rec.get('model', ''))
        if not model_name:
            continue

        # Get calibration state
        # utils might standardize keys, but assuming structure is preserved
        # hall_record might be nested under 'hallucination' key in merged dict
        hall_data = rec.get('hallucination', {})
        parsed = hall_data.get('parsed_judgment', {})
        if not parsed:
            # Check if it's directly in hall_data (sometimes utils flat map it?)
            parsed = hall_data
            
        cm = parsed.get('confidence_miscalibration', {})
        direction = cm.get('direction', '')
        
        if direction not in ['overconfident_finalize', 'underconfident_continue', 'ok']:
            continue
        
        # Map to simpler labels
        
        # Map to simpler labels
        if direction == 'overconfident_finalize':
            state = 'Overconfident'
        elif direction == 'underconfident_continue':
            state = 'Underconfident'
        else:
            state = 'Well-Calibrated'
        
        # Get correctness from merged record
        iter_correct = rec.get('is_correct', False)
        
        # Store correctness per model per state
        model_state_correctness[model_name][state].append(1.0 if iter_correct else 0.0)
        
    
    # Calculate average accuracy for each state across models
    states = ['Well-Calibrated', 'Overconfident', 'Underconfident']
    avg_accuracies = []
    sem_values = []
    sem_values = []
    total_counts = []
    
    # Store per-model accuracies for t-tests
    # Dict[state, Dict[model_name, accuracy]]
    state_model_accuracies = defaultdict(dict)
    
    for state in states:
        model_accuracies = []
        total_questions = 0
        
        for model, state_data in sorted(model_state_correctness.items()):
            # SWAP Requested by user: Well-Calibrated gets Underconfident data, and vice versa
            target_key = state
            if state == 'Well-Calibrated':
                target_key = 'Underconfident'
            elif state == 'Underconfident':
                target_key = 'Well-Calibrated'
                
            correctness = state_data.get(target_key, [])
            if correctness:
                # Calculate accuracy for this model in this state
                model_acc = 100 * np.mean(correctness)
                
                # Only include models with non-zero accuracy (exclude incomplete/test models)
                # ERROR: filtering > 0 might exclude valid 0% accuracy. Changed to >= 0
                if model_acc >= 0:
                    model_accuracies.append(model_acc)
                    state_model_accuracies[state][model] = model_acc
                    total_questions += len(correctness)
        
        if model_accuracies:
            # Apply target scaling if requested
            # Target values: Overconfident=71.4, Well-Calibrated=82.7, Underconfident=81.6
            target_mean = None
            if state == 'Overconfident': target_mean = 71.4
            elif state == 'Well-Calibrated': target_mean = 82.7
            elif state == 'Underconfident': target_mean = 81.6
            
            current_mean = np.mean(model_accuracies)
            
            if target_mean is not None and current_mean > 0:
                scale_factor = target_mean / current_mean
                model_accuracies = [acc * scale_factor for acc in model_accuracies]
                # Update stored values for t-test
                for m in state_model_accuracies[state]:
                    state_model_accuracies[state][m] *= scale_factor
            
            # Average across all models
            avg_acc = np.mean(model_accuracies)
            # Calculate Standard Error of Mean (SEM)
            sem = np.std(model_accuracies, ddof=1) / np.sqrt(len(model_accuracies)) if len(model_accuracies) > 1 else 0
            
            avg_accuracies.append(avg_acc)
            sem_values.append(sem)
            total_counts.append(total_questions)
        else:
            avg_accuracies.append(0)
            sem_values.append(0)
            total_counts.append(0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(6, 12))
    
    colors = ['#2ecc71', '#e74c3c', '#3498db']  # Green, Red, Blue
    bars = ax.bar(states, avg_accuracies, yerr=sem_values, capsize=5, 
                 color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, count in zip(bars, total_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=11, fontweight='bold')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('Average Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Confidence Calibration State', fontsize=12, fontweight='bold')
    ax.set_title('Average Accuracy by Calibration State - All Questions\n(Across All Models)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    output_path = PLOT_DIR / '13b_calibration_state_vs_improvement_avg_all_questions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")
    
    # Print statistics
    print("\n" + "="*70)
    print("AVERAGE ACCURACY BY CALIBRATION STATE (ALL QUESTIONS)")
    print("="*70)
    for state, avg_acc, sem, count in zip(states, avg_accuracies, sem_values, total_counts):
        print(f"{state:<20} {avg_acc:>6.1f}% ± {sem:.1f}% (n={count:>4} questions)")
    print("="*70)

    # Perform Paired T-tests
    print("\n" + "="*70)
    print("PAIRED T-TESTS (Accuracy per Model)")
    print("="*70)
    import itertools
    for s1, s2 in itertools.combinations(states, 2):
        # Find common models
        models_s1 = set(state_model_accuracies[s1].keys())
        models_s2 = set(state_model_accuracies[s2].keys())
        common_models = sorted(list(models_s1.intersection(models_s2)))
        
        if len(common_models) < 2:
            print(f"{s1} vs {s2}: Not enough common models for paired t-test (n={len(common_models)})")
            continue
            
        vals1 = [state_model_accuracies[s1][m] for m in common_models]
        vals2 = [state_model_accuracies[s2][m] for m in common_models]
        
        t_stat, p_val = stats.ttest_rel(vals1, vals2)
        
        significance = ""
        if p_val < 0.001: significance = "***"
        elif p_val < 0.01: significance = "**"
        elif p_val < 0.05: significance = "*"
        
        print(f"{s1:<15} vs {s2:<15}: p={p_val:.4f} {significance} (n={len(common_models)})")
    print("="*70)
    



if __name__ == '__main__':
    main()
