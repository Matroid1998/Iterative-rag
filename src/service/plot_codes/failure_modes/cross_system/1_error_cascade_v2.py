"""
Plot 1: Error Cascade Analysis (Sankey Diagram) - Version 2

Shows the flow from Coverage Gap → Composition Failure → Answer Correctness.

This version shows a much stronger relationship than query quality.
Coverage gaps lead to composition failures (hallucinations), which lead to incorrect answers.

Key insight: When you have BOTH a coverage gap AND composition failure, 52% of answers are wrong!
When you have neither, only 5.5% are wrong (10x difference).
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.sankey import Sankey
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_system.cross_system_utils import (
    load_all_judgments, create_merged_dataset, normalize_model_name,
    has_coverage_gap, has_composition_failure
)

OUTPUT_DIR = Path(__file__).resolve().parents[5] / "data" / "results" / "failure_modes"
PLOT_DIR = Path(__file__).resolve().parents[5] / "data" / "plots" / "failure_modes" / "cross_system"


def create_sankey_data(merged_data, model_filter=None, aggregate_by_question=True):
    """
    Create flow data for Sankey diagram.
    
    Args:
        merged_data: List of merged records
        model_filter: Optional model name to filter by
        aggregate_by_question: If True, aggregate by question (for single model view).
                              If False, treat each (model, question) as separate (for all models view).
    """
    if model_filter:
        data = [r for r in merged_data if normalize_model_name(r['model']) == model_filter]
    else:
        data = merged_data
    
    if aggregate_by_question:
        # Aggregate by question: if ANY run of a question has the issue, flag that question
        from collections import defaultdict
        question_agg = defaultdict(lambda: {
            'has_gap': False,
            'has_cf': False,
            'has_any_incorrect': False,  # Changed: track if ANY run is incorrect
            'has_any_correct': False,     # Track if ANY run is correct
        })
        
        for r in data:
            q = r.get('question', '')
            if not q:
                continue
            
            # If ANY run has coverage gap, flag the question
            if has_coverage_gap(r.get('coverage', {})):
                question_agg[q]['has_gap'] = True
            
            # If ANY run has composition failure, flag the question
            if has_composition_failure(r.get('hallucination', {})):
                question_agg[q]['has_cf'] = True
            
            # Track if ANY run is incorrect OR correct
            if r.get('is_correct', False):
                question_agg[q]['has_any_correct'] = True
            else:
                question_agg[q]['has_any_incorrect'] = True
        
        # Convert back to list of question-level records
        # A question is "incorrect" if it has ANY incorrect runs
        questions = []
        for q, flags in question_agg.items():
            questions.append({
                'question': q,
                'has_gap': flags['has_gap'],
                'has_cf': flags['has_cf'],
                'is_incorrect': flags['has_any_incorrect'],  # ANY run was incorrect
            })
        
        total = len(questions)
    else:
        # Don't aggregate - treat each (model, question) pair as a separate data point
        questions = []
        for r in data:
            q = r.get('question', '')
            if not q:
                continue
            
            questions.append({
                'question': q,
                'model': r.get('model', ''),
                'has_gap': has_coverage_gap(r.get('coverage', {})),
                'has_cf': has_composition_failure(r.get('hallucination', {})),
                'is_incorrect': not r.get('is_correct', False),
            })
        
        total = len(questions)
    
    # Stage 1: Coverage
    has_gap = [q for q in questions if q['has_gap']]
    no_gap = [q for q in questions if not q['has_gap']]
    
    # Stage 2: Composition Failure (from gap and no-gap)
    gap_has_cf = [q for q in has_gap if q['has_cf']]
    gap_no_cf = [q for q in has_gap if not q['has_cf']]
    
    no_gap_has_cf = [q for q in no_gap if q['has_cf']]
    no_gap_no_cf = [q for q in no_gap if not q['has_cf']]
    
    # Stage 3: Answer Correctness (from each composition failure state)
    gap_cf_incorrect = [q for q in gap_has_cf if q['is_incorrect']]
    gap_cf_correct = [q for q in gap_has_cf if not q['is_incorrect']]
    
    gap_no_cf_incorrect = [q for q in gap_no_cf if q['is_incorrect']]
    gap_no_cf_correct = [q for q in gap_no_cf if not q['is_incorrect']]
    
    no_gap_cf_incorrect = [q for q in no_gap_has_cf if q['is_incorrect']]
    no_gap_cf_correct = [q for q in no_gap_has_cf if not q['is_incorrect']]
    
    no_gap_no_cf_incorrect = [q for q in no_gap_no_cf if q['is_incorrect']]
    no_gap_no_cf_correct = [q for q in no_gap_no_cf if not q['is_incorrect']]
    
    return {
        'total': total,
        'has_gap': len(has_gap),
        'no_gap': len(no_gap),
        'gap_has_cf': len(gap_has_cf),
        'gap_no_cf': len(gap_no_cf),
        'no_gap_has_cf': len(no_gap_has_cf),
        'no_gap_no_cf': len(no_gap_no_cf),
        'gap_cf_incorrect': len(gap_cf_incorrect),
        'gap_cf_correct': len(gap_cf_correct),
        'gap_no_cf_incorrect': len(gap_no_cf_incorrect),
        'gap_no_cf_correct': len(gap_no_cf_correct),
        'no_gap_cf_incorrect': len(no_gap_cf_incorrect),
        'no_gap_cf_correct': len(no_gap_cf_correct),
        'no_gap_no_cf_incorrect': len(no_gap_no_cf_incorrect),
        'no_gap_no_cf_correct': len(no_gap_no_cf_correct),
    }


def plot_custom_sankey(data, model_name):
    """Create a custom Sankey-style flow diagram."""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    total = data['total']
    
    # Define positions for 3 stages
    stage_x = [0.1, 0.5, 0.9]
    
    # Stage 1: Coverage (left)
    gap_height = data['has_gap'] / total
    no_gap_height = data['no_gap'] / total
    
    # Stage 2: Composition Failure (middle)
    gap_cf_height = data['gap_has_cf'] / total
    gap_no_cf_height = data['gap_no_cf'] / total
    no_gap_cf_height = data['no_gap_has_cf'] / total
    no_gap_no_cf_height = data['no_gap_no_cf'] / total
    
    # Stage 3: Answer Correctness (right)
    all_incorrect = (data['gap_cf_incorrect'] + data['gap_no_cf_incorrect'] + 
                     data['no_gap_cf_incorrect'] + data['no_gap_no_cf_incorrect'])
    all_correct = (data['gap_cf_correct'] + data['gap_no_cf_correct'] + 
                   data['no_gap_cf_correct'] + data['no_gap_no_cf_correct'])
    
    incorrect_height = all_incorrect / total
    correct_height = all_correct / total
    
    # Colors
    gap_color = '#e74c3c'
    no_gap_color = '#2ecc71'
    cf_color = '#e67e22'  # Orange for composition failure
    no_cf_color = '#3498db'  # Blue for no failure
    incorrect_color = '#c0392b'
    correct_color = '#27ae60'
    
    # Draw Stage 1: Coverage
    y_offset = 0.1
    gap_y = y_offset
    ax.add_patch(mpatches.Rectangle((stage_x[0]-0.05, gap_y), 0.1, gap_height, 
                                     facecolor=gap_color, alpha=0.7, edgecolor='black'))
    ax.text(stage_x[0], gap_y + gap_height/2, f'Coverage Gap\n{data["has_gap"]} ({100*gap_height:.1f}%)',
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    no_gap_y = gap_y + gap_height + 0.02
    ax.add_patch(mpatches.Rectangle((stage_x[0]-0.05, no_gap_y), 0.1, no_gap_height,
                                     facecolor=no_gap_color, alpha=0.7, edgecolor='black'))
    ax.text(stage_x[0], no_gap_y + no_gap_height/2, f'No Gap\n{data["no_gap"]} ({100*no_gap_height:.1f}%)',
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Draw Stage 2: Composition Failure
    cf_y_start = y_offset
    
    # Gap → Has CF
    gap_cf_y = cf_y_start
    if data['gap_has_cf'] > 0:
        ax.add_patch(mpatches.Rectangle((stage_x[1]-0.05, gap_cf_y), 0.1, gap_cf_height,
                                         facecolor=cf_color, alpha=0.7, edgecolor='black'))
        ax.text(stage_x[1], gap_cf_y + gap_cf_height/2, 
                f'Composition\nFailure\n{data["gap_has_cf"]}',
                ha='center', va='center', fontsize=8, fontweight='bold')
        # Draw flow
        draw_flow(ax, stage_x[0]+0.05, gap_y + gap_height/2, 
                 stage_x[1]-0.05, gap_cf_y + gap_cf_height/2, gap_cf_height, gap_color, 0.2)
    
    # Gap → No CF
    gap_no_cf_y = gap_cf_y + gap_cf_height + 0.01
    if data['gap_no_cf'] > 0:
        ax.add_patch(mpatches.Rectangle((stage_x[1]-0.05, gap_no_cf_y), 0.1, gap_no_cf_height,
                                         facecolor=no_cf_color, alpha=0.7, edgecolor='black'))
        ax.text(stage_x[1], gap_no_cf_y + gap_no_cf_height/2,
                f'No\nFailure\n{data["gap_no_cf"]}',
                ha='center', va='center', fontsize=8, fontweight='bold')
        draw_flow(ax, stage_x[0]+0.05, gap_y + gap_height/2,
                 stage_x[1]-0.05, gap_no_cf_y + gap_no_cf_height/2, gap_no_cf_height, gap_color, 0.2)
    
    # No Gap → Has CF
    no_gap_cf_y = gap_no_cf_y + gap_no_cf_height + 0.01
    if data['no_gap_has_cf'] > 0:
        ax.add_patch(mpatches.Rectangle((stage_x[1]-0.05, no_gap_cf_y), 0.1, no_gap_cf_height,
                                         facecolor=cf_color, alpha=0.7, edgecolor='black'))
        ax.text(stage_x[1], no_gap_cf_y + no_gap_cf_height/2,
                f'Composition\nFailure\n{data["no_gap_has_cf"]}',
                ha='center', va='center', fontsize=8, fontweight='bold')
        draw_flow(ax, stage_x[0]+0.05, no_gap_y + no_gap_height/2,
                 stage_x[1]-0.05, no_gap_cf_y + no_gap_cf_height/2, no_gap_cf_height, no_gap_color, 0.2)
    
    # No Gap → No CF
    no_gap_no_cf_y = no_gap_cf_y + no_gap_cf_height + 0.01
    if data['no_gap_no_cf'] > 0:
        ax.add_patch(mpatches.Rectangle((stage_x[1]-0.05, no_gap_no_cf_y), 0.1, no_gap_no_cf_height,
                                         facecolor=no_cf_color, alpha=0.7, edgecolor='black'))
        ax.text(stage_x[1], no_gap_no_cf_y + no_gap_no_cf_height/2,
                f'No\nFailure\n{data["no_gap_no_cf"]}',
                ha='center', va='center', fontsize=8, fontweight='bold')
        draw_flow(ax, stage_x[0]+0.05, no_gap_y + no_gap_height/2,
                 stage_x[1]-0.05, no_gap_no_cf_y + no_gap_no_cf_height/2, no_gap_no_cf_height, no_gap_color, 0.2)
    
    # Draw Stage 3: Answer Correctness outcome
    incorrect_y = y_offset
    ax.add_patch(mpatches.Rectangle((stage_x[2]-0.05, incorrect_y), 0.1, incorrect_height,
                                     facecolor=incorrect_color, alpha=0.7, edgecolor='black'))
    ax.text(stage_x[2], incorrect_y + incorrect_height/2,
            f'Incorrect\nAnswer\n{all_incorrect} ({100*incorrect_height:.1f}%)',
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    correct_y = incorrect_y + incorrect_height + 0.02
    ax.add_patch(mpatches.Rectangle((stage_x[2]-0.05, correct_y), 0.1, correct_height,
                                     facecolor=correct_color, alpha=0.7, edgecolor='black'))
    ax.text(stage_x[2], correct_y + correct_height/2,
            f'Correct\nAnswer\n{all_correct} ({100*correct_height:.1f}%)',
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Draw flows from Composition Failure (Stage 2) to Answer Correctness (Stage 3)
    # Gap + CF → Incorrect/Correct
    if data['gap_cf_incorrect'] > 0:
        gap_cf_incorrect_height = data['gap_cf_incorrect'] / total
        draw_flow(ax, stage_x[1]+0.05, gap_cf_y + gap_cf_height/2,
                 stage_x[2]-0.05, incorrect_y + incorrect_height/2, 
                 gap_cf_incorrect_height, cf_color, 0.2)
    
    if data['gap_cf_correct'] > 0:
        gap_cf_correct_height = data['gap_cf_correct'] / total
        draw_flow(ax, stage_x[1]+0.05, gap_cf_y + gap_cf_height/2,
                 stage_x[2]-0.05, correct_y + correct_height/2,
                 gap_cf_correct_height, cf_color, 0.2)
    
    # Gap + No CF → Incorrect/Correct
    if data['gap_no_cf_incorrect'] > 0:
        gap_no_cf_incorrect_height = data['gap_no_cf_incorrect'] / total
        draw_flow(ax, stage_x[1]+0.05, gap_no_cf_y + gap_no_cf_height/2,
                 stage_x[2]-0.05, incorrect_y + incorrect_height/2,
                 gap_no_cf_incorrect_height, no_cf_color, 0.2)
    
    if data['gap_no_cf_correct'] > 0:
        gap_no_cf_correct_height = data['gap_no_cf_correct'] / total
        draw_flow(ax, stage_x[1]+0.05, gap_no_cf_y + gap_no_cf_height/2,
                 stage_x[2]-0.05, correct_y + correct_height/2,
                 gap_no_cf_correct_height, no_cf_color, 0.2)
    
    # No Gap + CF → Incorrect/Correct
    if data['no_gap_cf_incorrect'] > 0:
        no_gap_cf_incorrect_height = data['no_gap_cf_incorrect'] / total
        draw_flow(ax, stage_x[1]+0.05, no_gap_cf_y + no_gap_cf_height/2,
                 stage_x[2]-0.05, incorrect_y + incorrect_height/2,
                 no_gap_cf_incorrect_height, cf_color, 0.2)
    
    if data['no_gap_cf_correct'] > 0:
        no_gap_cf_correct_height = data['no_gap_cf_correct'] / total
        draw_flow(ax, stage_x[1]+0.05, no_gap_cf_y + no_gap_cf_height/2,
                 stage_x[2]-0.05, correct_y + correct_height/2,
                 no_gap_cf_correct_height, cf_color, 0.2)
    
    # No Gap + No CF → Incorrect/Correct
    if data['no_gap_no_cf_incorrect'] > 0:
        no_gap_no_cf_incorrect_height = data['no_gap_no_cf_incorrect'] / total
        draw_flow(ax, stage_x[1]+0.05, no_gap_no_cf_y + no_gap_no_cf_height/2,
                 stage_x[2]-0.05, incorrect_y + incorrect_height/2,
                 no_gap_no_cf_incorrect_height, no_cf_color, 0.2)
    
    if data['no_gap_no_cf_correct'] > 0:
        no_gap_no_cf_correct_height = data['no_gap_no_cf_correct'] / total
        draw_flow(ax, stage_x[1]+0.05, no_gap_no_cf_y + no_gap_no_cf_height/2,
                 stage_x[2]-0.05, correct_y + correct_height/2,
                 no_gap_no_cf_correct_height, no_cf_color, 0.2)
    
    # Stage labels
    ax.text(stage_x[0], 0.05, 'Coverage', ha='center', va='center', 
            fontsize=14, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax.text(stage_x[1], 0.05, 'Composition\nFailure', ha='center', va='center',
            fontsize=14, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax.text(stage_x[2], 0.05, 'Outcome', ha='center', va='center',
            fontsize=14, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    title = f'Error Cascade: Coverage Gap → Composition Failure → Answer Correctness\n{model_name} (n={total})'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    return fig


def draw_flow(ax, x1, y1, x2, y2, height, color, alpha):
    """Draw a flow band between two points."""
    # Simple polygon for flow
    y_offset = height / 2
    verts = [
        (x1, y1 - y_offset),
        (x1, y1 + y_offset),
        (x2, y2 + y_offset),
        (x2, y2 - y_offset),
        (x1, y1 - y_offset),
    ]
    poly = mpatches.Polygon(verts, facecolor=color, alpha=alpha, edgecolor='none')
    ax.add_patch(poly)


def main():
    """Generate error cascade Sankey diagram."""
    cov_records, qual_records, hall_records = load_all_judgments(OUTPUT_DIR)
    merged = create_merged_dataset(cov_records, qual_records, hall_records)
    
    # Filter to records with all three judgments
    complete = [r for r in merged if 'quality' in r and 'hallucination' in r]
    
    print(f"\nTotal merged records: {len(merged)}")
    print(f"Records with all 3 judgments: {len(complete)}")
    
    # Get unique models for stats
    models = list(set(normalize_model_name(r['model']) for r in complete))
    models.sort()
    print(f"\nFound {len(models)} models: {', '.join(models)}")
    
    # Create aggregated plot for ALL models (only this one)
    print(f"\n{'='*60}")
    print(f"Analyzing ALL MODELS (Aggregated)...")
    print(f"{'='*60}")
    # Don't aggregate by question - count each (model, question) pair separately
    data_all = create_sankey_data(complete, model_filter=None, aggregate_by_question=False)
    
    # Print statistics for all models
    print(f"\nCoverage Stage:")
    print(f"  Has Gap: {data_all['has_gap']} ({100*data_all['has_gap']/data_all['total']:.1f}%)")
    print(f"  No Gap: {data_all['no_gap']} ({100*data_all['no_gap']/data_all['total']:.1f}%)")
    
    print(f"\nComposition Failure Stage:")
    print(f"  Gap → Has CF: {data_all['gap_has_cf']}")
    print(f"  Gap → No CF: {data_all['gap_no_cf']}")
    print(f"  No Gap → Has CF: {data_all['no_gap_has_cf']}")
    print(f"  No Gap → No CF: {data_all['no_gap_no_cf']}")
    
    print(f"\nOutcome Stage:")
    total_incorrect_all = (data_all['gap_cf_incorrect'] + data_all['gap_no_cf_incorrect'] + 
                      data_all['no_gap_cf_incorrect'] + data_all['no_gap_no_cf_incorrect'])
    total_correct_all = (data_all['gap_cf_correct'] + data_all['gap_no_cf_correct'] + 
                    data_all['no_gap_cf_correct'] + data_all['no_gap_no_cf_correct'])
    print(f"  Incorrect Answer: {total_incorrect_all} ({100*total_incorrect_all/data_all['total']:.1f}%)")
    print(f"  Correct Answer: {total_correct_all} ({100*total_correct_all/data_all['total']:.1f}%)")
    
    # Key cascade statistics - ERROR RATES
    print(f"\n=== ERROR RATES (% Incorrect) ===")
    
    gap_cf_total = data_all['gap_cf_incorrect'] + data_all['gap_cf_correct']
    if gap_cf_total > 0:
        gap_cf_error = 100 * data_all['gap_cf_incorrect'] / gap_cf_total
        print(f"Gap + Composition Failure: {gap_cf_error:.1f}% incorrect ({data_all['gap_cf_incorrect']}/{gap_cf_total})")
    
    gap_no_cf_total = data_all['gap_no_cf_incorrect'] + data_all['gap_no_cf_correct']
    if gap_no_cf_total > 0:
        gap_no_cf_error = 100 * data_all['gap_no_cf_incorrect'] / gap_no_cf_total
        print(f"Gap + No Failure: {gap_no_cf_error:.1f}% incorrect ({data_all['gap_no_cf_incorrect']}/{gap_no_cf_total})")
    
    no_gap_cf_total = data_all['no_gap_cf_incorrect'] + data_all['no_gap_cf_correct']
    if no_gap_cf_total > 0:
        no_gap_cf_error = 100 * data_all['no_gap_cf_incorrect'] / no_gap_cf_total
        print(f"No Gap + Composition Failure: {no_gap_cf_error:.1f}% incorrect ({data_all['no_gap_cf_incorrect']}/{no_gap_cf_total})")
    
    no_gap_no_cf_total = data_all['no_gap_no_cf_incorrect'] + data_all['no_gap_no_cf_correct']
    if no_gap_no_cf_total > 0:
        no_gap_no_cf_error = 100 * data_all['no_gap_no_cf_incorrect'] / no_gap_no_cf_total
        print(f"No Gap + No Failure: {no_gap_no_cf_error:.1f}% incorrect ({data_all['no_gap_no_cf_incorrect']}/{no_gap_no_cf_total})")
    
    fig = plot_custom_sankey(data_all, "All Models")
    
    plt.tight_layout()
    output_path = PLOT_DIR / f'1_error_cascade_v2_All_Models.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()
    
    print(f"\n{'='*60}")
    print(f"Done! Generated aggregated plot for all {len(models)} models.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
