"""
Plot 4b: Issue Prevalence in Correct vs Incorrect Answers (Per Model)
Shows what % of correct/incorrect answers have each issue, with 6 subplots (one per model).
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def normalize_model_name(model: str) -> str:
    """Normalize model name for display."""
    if 'gpt-5' in model.lower() or 'openai-gpt-5' in model.lower() or 'openai_gpt-5' in model.lower():
        return 'GPT-5'
    elif 'gpt-4o' in model.lower():
        return 'GPT-4o'
    elif 'deepseek' in model.lower() and 'r1' in model.lower():
        return 'DeepSeek R1'
    elif 'claude-3-7' in model.lower() and 'reasoning' in model.lower():
        return 'Claude 3.7 + Reasoning'
    elif 'claude-3-7' in model.lower():
        return 'Claude 3.7 Sonnet'
    elif 'claude-sonnet-4.5' in model.lower() or 'claude-4.5' in model.lower():
        return 'Claude Sonnet 4.5'
    elif 'claude-3-5' in model.lower():
        return 'Claude 3.5 Sonnet'
    elif 'gemini-2.5-pro' in model.lower() or 'gemini-2.5' in model.lower():
        return 'Gemini 2.5 Pro'
    elif 'grok-4' in model.lower():
        return 'Grok 4 Fast'
    elif 'glm-4.6' in model.lower() or 'glm-4' in model.lower():
        return 'GLM 4.6'
    elif 'mistral' in model.lower():
        return 'Mistral Large'
    elif 'llama' in model.lower():
        return 'Llama 3.3 70B'
    return model


def load_issue_prevalence_data(output_dir):
    """Load issue prevalence data for correct vs incorrect answers."""
    # Structure: {model: {issue_type: {'correct': count, 'incorrect': count}, 'totals': {'correct': count, 'incorrect': count}}}
    model_data = defaultdict(lambda: {
        'has_gap': {'correct': 0, 'incorrect': 0},
        'any_late_hit': {'correct': 0, 'incorrect': 0},
        'any_carry_drop': {'correct': 0, 'incorrect': 0},
        'totals': {'correct': 0, 'incorrect': 0}
    })
    
    for file_path in glob.glob(str(output_dir / '*coverage_gap_judgments.jsonl')):
        # Extract model name from filename
        filename = Path(file_path).name
        model_name = filename.replace('responses_', '').replace('_reverified_coverage_gap_judgments.jsonl', '').replace('_coverage_gap_judgments.jsonl', '')
        model_name = normalize_model_name(model_name)
        
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
                    
                    # Count totals
                    model_data[model_name]['totals'][status] += 1
                    
                    # Count issue occurrences
                    if has_gap:
                        model_data[model_name]['has_gap'][status] += 1
                    if any_late_hit:
                        model_data[model_name]['any_late_hit'][status] += 1
                    if any_carry_drop:
                        model_data[model_name]['any_carry_drop'][status] += 1
                
                except json.JSONDecodeError:
                    continue
    
    return model_data


def create_per_model_prevalence_plot(model_data, output_path):
    """Create multi-subplot figure showing issue prevalence for each model."""
    
    models = sorted(model_data.keys())
    
    if len(models) == 0:
        print("No model data found!")
        return
    
    # Calculate grid size (3 columns, enough rows to fit all models)
    num_models = len(models)
    ncols = 3
    nrows = (num_models + ncols - 1) // ncols  # Ceiling division
    
    # Create figure with calculated subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)  # Ensure 2D array
    axes = axes.flatten()
    
    issue_types = ['Coverage Gap', 'Late Hit', 'Anchor Drop']
    issue_keys = ['has_gap', 'any_late_hit', 'any_carry_drop']
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        data = model_data[model]
        
        # Calculate prevalence rates
        prevalence_incorrect = []
        prevalence_correct = []
        
        total_incorrect = data['totals']['incorrect']
        total_correct = data['totals']['correct']
        
        for key in issue_keys:
            # Prevalence in incorrect answers
            count_incorrect = data[key]['incorrect']
            prev_incorrect = 100 * count_incorrect / total_incorrect if total_incorrect > 0 else 0
            prevalence_incorrect.append(prev_incorrect)
            
            # Prevalence in correct answers
            count_correct = data[key]['correct']
            prev_correct = 100 * count_correct / total_correct if total_correct > 0 else 0
            prevalence_correct.append(prev_correct)
        
        # Create grouped bars
        x = np.arange(len(issue_types))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, prevalence_incorrect, width,
                      label='In INCORRECT', color='#c44e52',
                      alpha=0.8, edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x + width/2, prevalence_correct, width,
                      label='In CORRECT', color='#55a868',
                      alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%',
                           ha='center', va='bottom', fontsize=9)
        
        # Formatting
        ax.set_ylabel('Prevalence (%)', fontsize=11, fontweight='bold')
        ax.set_title(model, fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(issue_types, rotation=15, ha='right', fontsize=10)
        ax.set_ylim(0, max(max(prevalence_incorrect), max(prevalence_correct)) * 1.2)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right', fontsize=9)
        
        # Add sample sizes as text
        textstr = f"Correct: n={total_correct}\nIncorrect: n={total_incorrect}"
        ax.text(0.02, 0.97, textstr, transform=ax.transAxes,
               fontsize=8, verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Add difference annotations
        for i, (prev_inc, prev_cor) in enumerate(zip(prevalence_incorrect, prevalence_correct)):
            diff = prev_inc - prev_cor
            if abs(diff) > 2:  # Only show significant differences
                y_pos = max(prev_inc, prev_cor) + 2
                symbol = '⚠️' if diff > 5 else '↑' if diff > 0 else '↓'
                ax.text(i, y_pos, f'{symbol}{abs(diff):.0f}pp',
                       ha='center', fontsize=8, fontweight='bold',
                       color='red' if diff > 0 else 'green')
    
    # Hide unused subplots
    for idx in range(len(models), len(axes)):
        axes[idx].axis('off')
    
    # Overall title
    fig.suptitle('Issue Prevalence: Correct vs Incorrect Answers (Per Model)\nHigher in INCORRECT = Issue causes errors',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved per-model issue prevalence plot to {output_path}")
    plt.close()
    
    # Print statistics
    print("\n" + "="*80)
    print("ISSUE PREVALENCE IN CORRECT VS INCORRECT ANSWERS (PER MODEL)")
    print("="*80)
    
    for model in sorted(models):
        print(f"\n{model}:")
        data = model_data[model]
        
        total_incorrect = data['totals']['incorrect']
        total_correct = data['totals']['correct']
        
        print(f"  Total: {total_correct} correct, {total_incorrect} incorrect")
        
        for issue_name, key in zip(issue_types, issue_keys):
            count_incorrect = data[key]['incorrect']
            count_correct = data[key]['correct']
            
            prev_incorrect = 100 * count_incorrect / total_incorrect if total_incorrect > 0 else 0
            prev_correct = 100 * count_correct / total_correct if total_correct > 0 else 0
            
            diff = prev_incorrect - prev_correct
            
            print(f"\n  {issue_name}:")
            print(f"    In incorrect: {prev_incorrect:.1f}% ({count_incorrect}/{total_incorrect})")
            print(f"    In correct: {prev_correct:.1f}% ({count_correct}/{total_correct})")
            print(f"    Difference: {diff:+.1f}pp {'⚠️ ISSUE CAUSES ERRORS' if diff > 5 else '✓ Not significant' if abs(diff) < 2 else '~moderate'}")


def main():
    # Setup paths
    base_dir = Path(__file__).resolve().parents[5]
    output_dir = base_dir  / "data" / "results" / "failure_modes"
    plot_dir = base_dir  / "data" / "plots" / "failure_modes" / "coverage_gap"
    plot_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading issue prevalence data...")
    model_data = load_issue_prevalence_data(output_dir)
    
    if not model_data:
        print("No data found!")
        return
    
    # Create plot
    output_path = plot_dir / "4b_issue_prevalence_per_model.png"
    create_per_model_prevalence_plot(model_data, output_path)


if __name__ == "__main__":
    main()
