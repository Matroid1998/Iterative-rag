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

def load_fusion_skip_by_step(output_dir):
    """Load fusion/skip rates by step."""
    # Structure: {model: {step: {'fusion': count, 'total': count}}}
    model_step_data = defaultdict(lambda: defaultdict(lambda: {'fusion': 0, 'total': 0}))
    
    for file_path in glob.glob(str(output_dir / '*quality_judement.jsonl')):
        filename = Path(file_path).name
        model_name = filename.replace('responses_', '').replace('_reverified_quality_judement.jsonl', '')
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    parsed = data.get('parsed_judgment', {})
                    
                    for step_data in parsed.get('per_step', []):
                        step = step_data.get('step')
                        if step is None or step > 5:
                            continue
                        
                        fusion_or_skip = step_data.get('fusion_or_skip', False)
                        
                        model_step_data[model_name][step]['total'] += 1
                        if fusion_or_skip:
                            model_step_data[model_name][step]['fusion'] += 1
                
                except json.JSONDecodeError:
                    continue
    
    return model_step_data


def create_fusion_skip_bar_chart(model_step_data, output_path):
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
        short_name = model.replace('bedrock_', '').replace('openai_', '').replace('us.anthropic.', '')
        ax.set_title(short_name, fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel('Step Number', fontsize=10)
        ax.set_ylabel('% Fusion/Skip', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Step {s}' for s in steps])
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(rates) * 1.3 if rates else 30)
    
    # Hide empty subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Fusion/Skip Rate by Step\n(Multi-hop jumping behavior: Does the system skip or merge hops?)',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved fusion/skip by step chart to {output_path}")
    plt.close()
    
    # Print statistics
    print("\n" + "="*80)
    print("FUSION/SKIP RATE BY STEP")
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
    
    # Create plot
    output_path = plot_dir / "fusion_skip_by_step.png"
    create_fusion_skip_bar_chart(model_step_data, output_path)


if __name__ == "__main__":
    main()
