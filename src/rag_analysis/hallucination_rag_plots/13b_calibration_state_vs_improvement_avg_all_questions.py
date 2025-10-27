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
    load_hallucination_judgments, normalize_model_name
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
    
    for judgment_file in OUTPUT_DIR_PATH.glob('*hallucination_judgment.jsonl'):
        # Extract model name from filename
        # Format: responses_MODEL_NAME_reverified_hallucination_judgment.jsonl
        filename = judgment_file.stem
        if filename.startswith('responses_'):
            model_part = filename.replace('responses_', '').replace('_reverified_hallucination_judgment', '')
            model_part = model_part.replace('_hallucination_judgment', '')
            
            # Normalize model name
            if 'bedrock_mistral' in model_part:
                model_name = 'Mistral Large'
            elif 'claude-3-7-sonnet' in model_part and 'reasoning' in model_part:
                model_name = 'Claude 3.7 + Reasoning'
            elif 'claude-3-7-sonnet' in model_part:
                model_name = 'Claude 3.7 Sonnet'
            elif 'deepseek.r1' in model_part:
                model_name = 'DeepSeek R1'
            elif 'gpt-4o' in model_part and 'mini' not in model_part:
                model_name = 'GPT-4o'
            elif 'gpt-5' in model_part:
                model_name = 'GPT-5'
            elif 'gemini-2.5-pro' in model_part:
                model_name = 'Gemini 2.5 Pro'
            elif 'grok-4-fast' in model_part:
                model_name = 'Grok 4 Fast'
            elif 'claude-sonnet-4.5' in model_part:
                model_name = 'Claude Sonnet 4.5'
            elif 'llama3-3-70b' in model_part:
                model_name = 'Llama 3.3 70B'
            else:
                continue  # Skip unknown models
            
            with open(judgment_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    rec = json.loads(line)
                    
                    # Get calibration state
                    parsed = rec.get('parsed_judgment', {})
                    cm = parsed.get('confidence_miscalibration', {})
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
