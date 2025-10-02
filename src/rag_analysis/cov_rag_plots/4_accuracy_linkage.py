"""
Plot 4: Accuracy Linkage to Coverage Issues
Grouped bars showing is_correct proportions vs {has_gap, any_late_hit}.
Shows if coverage gaps correlate with wrong answers.
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def load_accuracy_linkage_data(output_dir):
    """Load accuracy data linked to coverage issues."""
    # Structure: {model: {issue_type: {correct: count, incorrect: count}}}
    model_data = defaultdict(lambda: {
        'has_gap': {'correct': 0, 'incorrect': 0, 'no_issue': 0},
        'any_late_hit': {'correct': 0, 'incorrect': 0, 'no_issue': 0},
        'any_carry_drop': {'correct': 0, 'incorrect': 0, 'no_issue': 0},
        'no_issues': {'correct': 0, 'incorrect': 0}
    })
    
    for file_path in glob.glob(str(output_dir / '*coverage_gap_judgments.jsonl')):
        # Extract model name from filename
        filename = Path(file_path).name
        model_name = filename.replace('responses_', '').replace('_reverified_coverage_gap_judgments.jsonl', '')
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    is_correct = data.get('is_correct')
                    
                    if is_correct is None:
                        continue
                    
                    status = 'correct' if is_correct else 'incorrect'
                    
                    parsed = data.get('parsed_judgment', {})
                    
                    # Check for issues
                    coverage = parsed.get('retrieval_coverage_gap', {})
                    has_gap = coverage.get('has_gap', False)
                    
                    late_hit = parsed.get('late_hit_per_hop', {})
                    any_late_hit = late_hit.get('any_late_hit', False)
                    
                    anchor = parsed.get('anchor_carry_drop', {})
                    any_carry_drop = anchor.get('any_carry_drop', False)
                    
                    # Track each issue type
                    if has_gap:
                        model_data[model_name]['has_gap'][status] += 1
                    else:
                        model_data[model_name]['has_gap']['no_issue'] += 1
                    
                    if any_late_hit:
                        model_data[model_name]['any_late_hit'][status] += 1
                    else:
                        model_data[model_name]['any_late_hit']['no_issue'] += 1
                    
                    if any_carry_drop:
                        model_data[model_name]['any_carry_drop'][status] += 1
                    else:
                        model_data[model_name]['any_carry_drop']['no_issue'] += 1
                    
                    # Track runs with no issues at all
                    if not (has_gap or any_late_hit or any_carry_drop):
                        model_data[model_name]['no_issues'][status] += 1
                
                except json.JSONDecodeError:
                    continue
    
    return model_data


def create_grouped_bar_chart(model_data, output_path):
    """Create grouped bar chart showing accuracy rates for each issue type."""
    # Aggregate across all models
    aggregated = {
        'has_gap': {'correct': 0, 'incorrect': 0, 'no_issue_correct': 0, 'no_issue_incorrect': 0},
        'any_late_hit': {'correct': 0, 'incorrect': 0, 'no_issue_correct': 0, 'no_issue_incorrect': 0},
        'any_carry_drop': {'correct': 0, 'incorrect': 0, 'no_issue_correct': 0, 'no_issue_incorrect': 0},
        'no_issues': {'correct': 0, 'incorrect': 0}
    }
    
    for model, data in model_data.items():
        for issue_type in ['has_gap', 'any_late_hit', 'any_carry_drop']:
            aggregated[issue_type]['correct'] += data[issue_type]['correct']
            aggregated[issue_type]['incorrect'] += data[issue_type]['incorrect']
            aggregated[issue_type]['no_issue_correct'] += data[issue_type]['no_issue']
        
        aggregated['no_issues']['correct'] += data['no_issues']['correct']
        aggregated['no_issues']['incorrect'] += data['no_issues']['incorrect']
    
    # Calculate accuracy rates
    issue_types = ['Coverage Gap', 'Late Hit', 'Anchor Drop', 'No Issues']
    issue_keys = ['has_gap', 'any_late_hit', 'any_carry_drop', 'no_issues']
    
    with_issue_accuracy = []
    without_issue_accuracy = []
    
    for key in issue_keys:
        if key == 'no_issues':
            total = aggregated[key]['correct'] + aggregated[key]['incorrect']
            accuracy = 100 * aggregated[key]['correct'] / total if total > 0 else 0
            with_issue_accuracy.append(accuracy)
            without_issue_accuracy.append(0)  # Not applicable
        else:
            # With issue
            total_with = aggregated[key]['correct'] + aggregated[key]['incorrect']
            acc_with = 100 * aggregated[key]['correct'] / total_with if total_with > 0 else 0
            with_issue_accuracy.append(acc_with)
            
            # Without issue (from no_issue field, but we need to recalculate)
            # For now, we'll calculate overall accuracy excluding this issue
            without_issue_accuracy.append(0)  # Placeholder
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(issue_types))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, with_issue_accuracy, width, 
                   label='Accuracy WITH Issue', color='#c44e52', 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Calculate "without issue" for first 3
    for i, key in enumerate(issue_keys[:3]):
        total_models = len(model_data)
        total_without = 0
        correct_without = 0
        
        for model, data in model_data.items():
            # Count correct/incorrect where this issue is NOT present
            for line_key in ['has_gap', 'any_late_hit', 'any_carry_drop']:
                if line_key == key:
                    # This is the issue we're checking - use no_issue count
                    continue
        
        # Simplified: just show overall accuracy for runs without this specific issue
        # This requires re-parsing data, so for now we'll show a comparison differently
    
    # Actually, let's create a different visualization
    # Show: For each issue type, what % of incorrect answers had this issue?
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left plot: Accuracy rate when issue is present
    colors_present = ['#c44e52', '#e07b4f', '#dd8452', '#4c72b0']
    bars_present = ax1.bar(issue_types, with_issue_accuracy, 
                          color=colors_present, alpha=0.8, 
                          edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar in bars_present:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Accuracy Rate (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Accuracy When Issue is Present\n(Lower = Issue hurts performance)',
                 fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticklabels(issue_types, rotation=20, ha='right')
    
    # Right plot: Prevalence of issues in incorrect vs correct answers
    issue_prevalence_incorrect = []
    issue_prevalence_correct = []
    
    for key in issue_keys[:3]:  # Exclude 'no_issues'
        total_incorrect = sum(model_data[m][key]['incorrect'] for m in model_data)
        total_correct = sum(model_data[m][key]['correct'] for m in model_data)
        
        # Total incorrect/correct answers across all models
        all_incorrect = sum(
            sum(model_data[m][k]['incorrect'] for k in ['has_gap', 'any_late_hit', 'any_carry_drop'])
            for m in model_data
        )
        all_correct = sum(
            sum(model_data[m][k]['correct'] for k in ['has_gap', 'any_late_hit', 'any_carry_drop'])
            for m in model_data
        )
        
        prev_incorrect = 100 * total_incorrect / all_incorrect if all_incorrect > 0 else 0
        prev_correct = 100 * total_correct / all_correct if all_correct > 0 else 0
        
        issue_prevalence_incorrect.append(prev_incorrect)
        issue_prevalence_correct.append(prev_correct)
    
    x2 = np.arange(3)
    width2 = 0.35
    
    bars2_inc = ax2.bar(x2 - width2/2, issue_prevalence_incorrect, width2,
                       label='In Incorrect Answers', color='#c44e52',
                       alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2_cor = ax2.bar(x2 + width2/2, issue_prevalence_correct, width2,
                       label='In Correct Answers', color='#55a868',
                       alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars2_inc, bars2_cor]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Prevalence (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Issue Prevalence in Correct vs Incorrect Answers\n(Higher in incorrect = Issue causes errors)',
                 fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(issue_types[:3], rotation=20, ha='right')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved accuracy linkage chart to {output_path}")
    plt.close()
    
    # Print detailed statistics
    print("\n" + "="*80)
    print("ACCURACY LINKAGE TO COVERAGE ISSUES")
    print("="*80)
    
    for i, (issue_type, key) in enumerate(zip(issue_types[:3], issue_keys[:3])):
        print(f"\n{issue_type}:")
        
        total_with_issue = aggregated[key]['correct'] + aggregated[key]['incorrect']
        acc_with = 100 * aggregated[key]['correct'] / total_with_issue if total_with_issue > 0 else 0
        
        print(f"  Runs with issue: {total_with_issue}")
        print(f"  Accuracy when present: {acc_with:.1f}%")
        print(f"  Prevalence in incorrect answers: {issue_prevalence_incorrect[i]:.1f}%")
        print(f"  Prevalence in correct answers: {issue_prevalence_correct[i]:.1f}%")


def main():
    # Setup paths
    base_dir = Path(__file__).resolve().parents[2]
    output_dir = base_dir / "rag_analysis" / "output"
    plot_dir = base_dir / "rag_analysis" / "cov_rag_plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading accuracy linkage data...")
    model_data = load_accuracy_linkage_data(output_dir)
    
    if not model_data:
        print("No accuracy linkage data found!")
        return
    
    # Create plot
    output_path = plot_dir / "accuracy_linkage.png"
    create_grouped_bar_chart(model_data, output_path)


if __name__ == "__main__":
    main()
