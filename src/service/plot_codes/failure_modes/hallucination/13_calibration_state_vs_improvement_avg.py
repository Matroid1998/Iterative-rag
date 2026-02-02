"""
Plot: Average Improvement over No-Context by Calibration State

Shows the average improvement for questions in each confidence calibration state
(overconfident, underconfident, well-calibrated) across all models.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hallucination.hall_plot_utils import (
    load_hallucination_judgments, normalize_model_name
)

OUTPUT_DIR = Path(__file__).resolve().parents[5] / "data" / "results" / "failure_modes"
PLOT_DIR = Path(__file__).resolve().parents[5] / "data" / "plots" / "failure_modes" / "hallucination"
PLOT_DIR.mkdir(exist_ok=True)

# Response directories
ITERATIVE_DIR = Path(__file__).resolve().parents[4] / "responses_reverified"
NO_CONTEXT_DIR = Path(__file__).resolve().parents[5] / "src" / "response-jsonl-without-context"


def load_responses(directory):
    """Load all responses from a directory, keyed by question text."""
    responses = {}
    for jsonl_file in directory.glob('**/*.jsonl'):
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        # Question might be in 'raw' field or top-level
                        raw = rec.get('raw', {})
                        question = raw.get('question', '').strip()
                        if not question:
                            question = rec.get('question', '').strip()
                        
                        if question:
                            # Use question as key; last response wins if duplicates
                            responses[question] = rec
                    except json.JSONDecodeError:
                        continue
    return responses


def main():
    """Generate calibration state vs improvement plot."""
    hall_records = load_hallucination_judgments(OUTPUT_DIR)
    
    # Load responses
    iterative_responses = load_responses(ITERATIVE_DIR)
    no_context_responses = load_responses(NO_CONTEXT_DIR)
    
    # Group by calibration state
    state_improvements = defaultdict(list)
    
    for rec in hall_records:
        question = rec.get('question', '').strip()
        if not question:
            continue
        
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
        
        # Get correctness from hallucination judgment (iterative RAG)
        iter_correct = rec.get('is_correct', False)
        
        # Get correctness from no-context
        no_ctx_resp = no_context_responses.get(question)
        if not no_ctx_resp:
            continue
        
        no_ctx_correct = no_ctx_resp.get('reverified_correct', False)
        if not no_ctx_correct:
            no_ctx_correct = no_ctx_resp.get('is_correct', False)
        
        # Calculate improvement (0 if both wrong/right, +1 if improved, -1 if regressed)
        if iter_correct and not no_ctx_correct:
            improvement = 1.0  # Improved
        elif not iter_correct and no_ctx_correct:
            improvement = -1.0  # Regressed
        else:
            improvement = 0.0  # No change
        
        state_improvements[state].append(improvement)
    
    # Calculate average improvement percentage for each state
    states = ['Underconfident', 'Overconfident', 'Well-Calibrated']
    avg_improvements = []
    counts = []
    
    for state in states:
        improvements = state_improvements[state]
        if improvements:
            # Average improvement rate (% of questions that improved)
            avg_impr = 100 * np.mean(improvements)
            avg_improvements.append(avg_impr)
            counts.append(len(improvements))
        else:
            avg_improvements.append(0)
            counts.append(0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71']  # Red, Blue, Green
    bars = ax.bar(states, avg_improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=11, fontweight='bold')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('Average Improvement over No-Context (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Confidence Calibration State', fontsize=12, fontweight='bold')
    ax.set_title('Average Improvement by Calibration State\n(Across All Models and Questions)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(min(avg_improvements) - 5, max(avg_improvements) + 10)
    
    plt.tight_layout()
    output_path = PLOT_DIR / '13_calibration_state_vs_improvement_avg.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")
    
    # Print statistics
    print("\n" + "="*70)
    print("AVERAGE IMPROVEMENT BY CALIBRATION STATE")
    print("="*70)
    for state, avg_impr, count in zip(states, avg_improvements, counts):
        print(f"{state:<20} {avg_impr:>6.1f}%  (n={count:>4})")
    print("="*70)


if __name__ == '__main__':
    main()
