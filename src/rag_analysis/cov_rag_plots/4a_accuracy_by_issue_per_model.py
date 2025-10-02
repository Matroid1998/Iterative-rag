"""
Plot 4a: Accuracy by Issue Type (Per Model)
Shows accuracy rate when each coverage issue is present, with 6 subplots (one per model).
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
    elif 'claude-3-5' in model.lower():
        return 'Claude 3.5 Sonnet'
    elif 'mistral' in model.lower():
        return 'Mistral Large'
    elif 'llama' in model.lower():
        return 'Llama 3.3 70B'
    return model


def load_accuracy_by_issue_data(output_dir):
    """Load accuracy data by issue type for each model."""
    # Structure: {model: {issue_type: {'with_issue': {correct, total}, 'without_issue': {correct, total}}}}
    model_data = defaultdict(lambda: {
        'has_gap': {'with_issue': {'correct': 0, 'total': 0}, 'without_issue': {'correct': 0, 'total': 0}},
        'any_late_hit': {'with_issue': {'correct': 0, 'total': 0}, 'without_issue': {'correct': 0, 'total': 0}},
        'any_carry_drop': {'with_issue': {'correct': 0, 'total': 0}, 'without_issue': {'correct': 0, 'total': 0}},
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
                    
                    parsed = data.get('parsed_judgment', {})
                    
                    # Check for issues
                    coverage = parsed.get('retrieval_coverage_gap', {})
                    has_gap = coverage.get('has_gap', False)
                    
                    late_hit = parsed.get('late_hit_per_hop', {})
                    any_late_hit = late_hit.get('any_late_hit', False)
                    
                    anchor = parsed.get('anchor_carry_drop', {})
                    any_carry_drop = anchor.get('any_carry_drop', False)
                    
                    # Track each issue type
                    for issue_key, has_issue in [('has_gap', has_gap), ('any_late_hit', any_late_hit), ('any_carry_drop', any_carry_drop)]:
                        if has_issue:
                            model_data[model_name][issue_key]['with_issue']['total'] += 1
                            if is_correct:
                                model_data[model_name][issue_key]['with_issue']['correct'] += 1
                        else:
                            model_data[model_name][issue_key]['without_issue']['total'] += 1
                            if is_correct:
                                model_data[model_name][issue_key]['without_issue']['correct'] += 1
                
                except json.JSONDecodeError:
                    continue
    
    return model_data


def create_per_model_accuracy_plot(model_data, output_path):
    """Create 6-subplot figure showing accuracy by issue type for each model."""
    
    models = sorted(model_data.keys())
    
    if len(models) == 0:
        print("No model data found!")
        return
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    issue_types = ['Coverage Gap', 'Late Hit', 'Anchor Drop']
    issue_keys = ['has_gap', 'any_late_hit', 'any_carry_drop']
    colors = ['#c44e52', '#e07b4f', '#dd8452']
    
    for idx, model in enumerate(models):
        if idx >= 6:  # Only show first 6 models
            break
        
        ax = axes[idx]
        data = model_data[model]
        
        # Calculate accuracy rates
        with_issue_acc = []
        without_issue_acc = []
        
        for key in issue_keys:
            # With issue
            total_with = data[key]['with_issue']['total']
            correct_with = data[key]['with_issue']['correct']
            acc_with = 100 * correct_with / total_with if total_with > 0 else 0
            with_issue_acc.append(acc_with)
            
            # Without issue
            total_without = data[key]['without_issue']['total']
            correct_without = data[key]['without_issue']['correct']
            acc_without = 100 * correct_without / total_without if total_without > 0 else 0
            without_issue_acc.append(acc_without)
        
        # Create grouped bars
        x = np.arange(len(issue_types))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, with_issue_acc, width,
                      label='WITH Issue', color='#c44e52',
                      alpha=0.8, edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x + width/2, without_issue_acc, width,
                      label='WITHOUT Issue', color='#55a868',
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
        ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(model, fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(issue_types, rotation=15, ha='right', fontsize=10)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='lower left', fontsize=9)
        
        # Add sample sizes as text
        textstr = '\n'.join([
            f"Gap: n={data['has_gap']['with_issue']['total']}",
            f"Late: n={data['any_late_hit']['with_issue']['total']}",
            f"Drop: n={data['any_carry_drop']['with_issue']['total']}"
        ])
        ax.text(0.98, 0.97, textstr, transform=ax.transAxes,
               fontsize=8, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for idx in range(len(models), 6):
        axes[idx].axis('off')
    
    # Overall title
    fig.suptitle('Accuracy Rate by Coverage Issue Type (Per Model)\nLower "WITH Issue" bars indicate issue hurts performance',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved per-model accuracy by issue plot to {output_path}")
    plt.close()
    
    # Print statistics
    print("\n" + "="*80)
    print("ACCURACY BY ISSUE TYPE (PER MODEL)")
    print("="*80)
    
    for model in sorted(models):
        print(f"\n{model}:")
        data = model_data[model]
        
        for issue_name, key in zip(issue_types, issue_keys):
            total_with = data[key]['with_issue']['total']
            correct_with = data[key]['with_issue']['correct']
            acc_with = 100 * correct_with / total_with if total_with > 0 else 0
            
            total_without = data[key]['without_issue']['total']
            correct_without = data[key]['without_issue']['correct']
            acc_without = 100 * correct_without / total_without if total_without > 0 else 0
            
            diff = acc_without - acc_with
            
            print(f"  {issue_name}:")
            print(f"    WITH issue: {acc_with:.1f}% (n={total_with})")
            print(f"    WITHOUT issue: {acc_without:.1f}% (n={total_without})")
            print(f"    Impact: {diff:+.1f}pp {'⚠️ HURTS' if diff > 0 else '✓ OK' if diff < -2 else '~neutral'}")


def main():
    # Setup paths
    base_dir = Path(__file__).resolve().parents[2]
    output_dir = base_dir / "rag_analysis" / "output"
    plot_dir = base_dir / "rag_analysis" / "cov_rag_plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading accuracy by issue data...")
    model_data = load_accuracy_by_issue_data(output_dir)
    
    if not model_data:
        print("No data found!")
        return
    
    # Create plot
    output_path = plot_dir / "4a_accuracy_by_issue_per_model.png"
    create_per_model_accuracy_plot(model_data, output_path)


if __name__ == "__main__":
    main()
