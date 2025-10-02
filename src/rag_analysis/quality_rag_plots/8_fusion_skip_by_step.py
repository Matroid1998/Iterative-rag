"""
Plot 8: Fusion/Skip by Step
Bar chart of % fusion_or_skip by step; highlights multi-hop jumping behavior.
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def normalize_model_name(model_str):
    """Normalize model names for consistent display."""
    model_map = {
        'openai_gpt-5': 'GPT-5',
        'openai_gpt-4o': 'GPT-4o',
        'bedrock_us.deepseek.r1': 'DeepSeek R1',
        'bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning': 'Claude 3.7 Sonnet + Reasoning',
        'bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0': 'Claude 3.7 Sonnet',
        'bedrock_us.anthropic.claude-3-7-sonnet-reasoning': 'Claude 3.7 Sonnet + Reasoning',
        'bedrock_us.anthropic.claude-3-7-sonnet': 'Claude 3.7 Sonnet',
        'bedrock_mistral.mistral-large': 'Mistral Large'
    }
    
    # Try exact match first, then partial match
    if model_str in model_map:
        return model_map[model_str]
    
    for key, value in model_map.items():
        if key in model_str:
            return value
    return model_str


def load_correctness_map(output_dir):
    """Load is_correct information from coverage judgment files."""
    correctness = {}  # {(model, question): is_correct}
    
    for file_path in glob.glob(str(output_dir / '*coverage_gap_judgments.jsonl')):
        filename = Path(file_path).name
        model_from_file = filename.replace('responses_', '').replace('_reverified_coverage_gap_judgments.jsonl', '')
        model = normalize_model_name(model_from_file)
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    question = rec.get('question', '')
                    is_correct = rec.get('is_correct', False)
                    correctness[(model, question)] = is_correct
                except json.JSONDecodeError:
                    continue
    
    return correctness


def load_fusion_skip_by_step(output_dir):
    """Load fusion/skip rates by step, separated by correctness."""
    # Structure: {is_correct: {model: {step: {'fusion': count, 'total': count}}}}
    model_step_data = {
        True: defaultdict(lambda: defaultdict(lambda: {'fusion': 0, 'total': 0})),
        False: defaultdict(lambda: defaultdict(lambda: {'fusion': 0, 'total': 0}))
    }
    
    correctness_map = load_correctness_map(output_dir)
    
    matched = 0
    unmatched = 0
    
    for file_path in glob.glob(str(output_dir / '*quality_judement.jsonl')):
        filename = Path(file_path).name
        model_from_file = filename.replace('responses_', '').replace('_reverified_quality_judement.jsonl', '')
        model_name = normalize_model_name(model_from_file)
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    question = data.get('question', '')
                    
                    # Look up correctness
                    is_correct = correctness_map.get((model_name, question), None)
                    if is_correct is None:
                        unmatched += 1
                        continue
                    
                    matched += 1
                    parsed = data.get('parsed_judgment', {})
                    
                    for step_data in parsed.get('per_step', []):
                        step = step_data.get('step')
                        if step is None or step > 5:
                            continue
                        
                        fusion_or_skip = step_data.get('fusion_or_skip', False)
                        
                        model_step_data[is_correct][model_name][step]['total'] += 1
                        if fusion_or_skip:
                            model_step_data[is_correct][model_name][step]['fusion'] += 1
                
                except json.JSONDecodeError:
                    continue
    
    print(f"Matched: {matched}, Unmatched: {unmatched}")
    return model_step_data


def create_fusion_skip_bar_chart(model_step_data, output_path, title_suffix, is_correct):
    """Create bar chart of fusion/skip rates by step."""
    models = sorted(model_step_data.keys())
    
    max_step = max(max(steps.keys()) for steps in model_step_data.values())
    steps = list(range(1, max_step + 1))
    
    # Create subplots - one per model
    n_models = len(models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        step_data = model_step_data[model]
        
        rates = []
        totals = []
        for step in steps:
            data = step_data.get(step, {'fusion': 0, 'total': 0})
            total = data['total']
            fusion = data['fusion']
            rate = 100 * fusion / total if total > 0 else 0
            rates.append(rate)
            totals.append(total)
        
        # Create bar chart
        x = np.arange(len(steps))
        bars = ax.bar(x, rates, color='#dd8452', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, rate, total in zip(bars, rates, totals):
            if rate > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{rate:.1f}%\n(n={total})',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Customize subplot
        ax.set_title(model, fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel('Step Number', fontsize=10)
        ax.set_ylabel('% Fusion/Skip', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Step {s}' for s in steps])
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(rates) * 1.3 if rates else 30)
    
    # Hide empty subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    correctness_label = "CORRECT" if is_correct else "INCORRECT"
    plt.suptitle(f'Fusion/Skip Rate by Step ({correctness_label} Answers)\n(Multi-hop jumping behavior: Does the system skip or merge hops?)',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved fusion/skip by step chart to {output_path}")
    plt.close()
    
    # Print statistics
    correctness_label = "CORRECT" if is_correct else "INCORRECT"
    print("\n" + "="*80)
    print(f"FUSION/SKIP RATE BY STEP ({correctness_label} ANSWERS)")
    print("="*80)
    
    for model in models:
        step_data = model_step_data[model]
        
        print(f"\n{model}:")
        print(f"  {'Step':<6} {'Total':>8} {'Fusion/Skip':>12} {'Rate':>10}")
        print("  " + "-"*40)
        
        overall_fusion = 0
        overall_total = 0
        
        for step in steps:
            data = step_data.get(step, {'fusion': 0, 'total': 0})
            total = data['total']
            fusion = data['fusion']
            rate = 100 * fusion / total if total > 0 else 0
            
            if total > 0:
                print(f"  {step:<6} {total:>8} {fusion:>12} {rate:>9.1f}%")
                overall_fusion += fusion
                overall_total += total
        
        if overall_total > 0:
            print(f"  {'Overall':<6} {overall_total:>8} {overall_fusion:>12} {100*overall_fusion/overall_total:>9.1f}%")


def main():
    # Setup paths
    base_dir = Path(__file__).resolve().parents[2]
    output_dir = base_dir / "rag_analysis" / "output"
    plot_dir = base_dir / "rag_analysis" / "quality_rag_plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading fusion/skip data by step...")
    model_step_data = load_fusion_skip_by_step(output_dir)
    
    if not model_step_data:
        print("No fusion/skip data found!")
        return
    
    # Create plots for correct and incorrect answers
    output_path_correct = plot_dir / "fusion_skip_by_step_CORRECT.png"
    create_fusion_skip_bar_chart(model_step_data[True], output_path_correct, "CORRECT", True)
    
    output_path_incorrect = plot_dir / "fusion_skip_by_step_INCORRECT.png"
    create_fusion_skip_bar_chart(model_step_data[False], output_path_incorrect, "INCORRECT", False)


if __name__ == "__main__":
    main()
