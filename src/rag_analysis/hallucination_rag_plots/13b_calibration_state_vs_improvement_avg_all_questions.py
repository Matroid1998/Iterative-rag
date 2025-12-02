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
    # Load hallucination and coverage judgments
    hall_records = load_hallucination_judgments(OUTPUT_DIR)
    cov_records = load_coverage_judgments(OUTPUT_DIR)
    
    # Merge datasets to get is_correct field
    merged = create_merged_dataset(hall_records, cov_records, [])
    
    # Group by source model and calibration state
    model_state_correctness = defaultdict(lambda: defaultdict(list))
    
    for rec in merged:
        model_name = normalize_model_name(rec.get('model', ''))
        
        # Get calibration state
        # Note: merged dataset puts hallucination judgment in 'hallucination' key
        hall_judgment = rec.get('hallucination', {})
        cm = hall_judgment.get('confidence_miscalibration', {})
        direction = cm.get('direction', '')
        
        if direction not in ['overconfident_finalize', 'underconfident_continue', 'ok']:
            continue
        
        # Map to simpler labels
        if direction == 'overconfident_finalize':
            state = 'Overconfident'
        elif direction == 'underconfident_continue':
            state = 'Underconfident'
        else:
            state = 'Well-Calibrated'
        
        # Get correctness
        iter_correct = rec.get('is_correct', False)
        
        # Store correctness per model per state
        model_state_correctness[model_name][state].append(1.0 if iter_correct else 0.0)
    
    # Calculate average accuracy for each state across models
    states = ['Underconfident', 'Overconfident', 'Well-Calibrated']
    avg_accuracies = []
    total_counts = []
    
    for state in states:
        model_accuracies = []
        total_questions = 0
        
        for model, state_data in sorted(model_state_correctness.items()):
            correctness = state_data.get(state, [])
            if correctness:
                # Calculate accuracy for this model in this state
                model_acc = 100 * np.mean(correctness)
                
                # Only include models with non-zero accuracy (exclude incomplete/test models)
                if model_acc > 0:
                    model_accuracies.append(model_acc)
                    total_questions += len(correctness)
        
        if model_accuracies:
            # Average across all models
            avg_acc = np.mean(model_accuracies)
            avg_accuracies.append(avg_acc)
            total_counts.append(total_questions)
        else:
            avg_accuracies.append(0)
            total_counts.append(0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green
    bars = ax.bar(states, avg_accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
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
    for state, avg_acc, count in zip(states, avg_accuracies, total_counts):
        print(f"{state:<20} {avg_acc:>6.1f}%  (n={count:>4} questions)")
    print("="*70)


if __name__ == '__main__':
    main()
