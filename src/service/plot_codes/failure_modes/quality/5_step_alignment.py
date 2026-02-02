"""
Plot 5: Step Alignment Analysis
Bar chart of % is_next_logical_hop by step index; also overall per model.
Uses per_step[].is_next_logical_hop.
Also plots another comparison where is_next_logical_hop is counted as true if step number equals predicted hop number.
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def load_step_alignment_data(output_dir):
    """Load step alignment data (is_next_logical_hop)."""
    # Structure: {model: {step: {'next_hop_true': count, 'total': count, 'step_eq_hop': count}}}
    model_step_data = defaultdict(lambda: defaultdict(lambda: {'next_hop_true': 0, 'total': 0, 'step_eq_hop': 0}))
    
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
                        
                        is_next_logical = step_data.get('is_next_logical_hop', False)
                        predicted_hop = step_data.get('predicted_hop')
                        
                        model_step_data[model_name][step]['total'] += 1
                        
                        if is_next_logical:
                            model_step_data[model_name][step]['next_hop_true'] += 1
                        
                        # Alternative: step number equals predicted hop
                        if predicted_hop is not None and step == predicted_hop:
                            model_step_data[model_name][step]['step_eq_hop'] += 1
                
                except json.JSONDecodeError:
                    continue
    
    return model_step_data


def create_alignment_charts(model_step_data, output_path):
    """Create alignment bar charts."""
    models = sorted(model_step_data.keys())
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # === TOP PLOT: is_next_logical_hop rate by step ===
    ax1 = axes[0]
    
    max_step = max(max(steps.keys()) for steps in model_step_data.values())
    steps = list(range(1, max_step + 1))
    
    x = np.arange(len(steps))
    width = 0.15
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    for i, model in enumerate(models):
        rates = []
        for step in steps:
            data = model_step_data[model].get(step, {'next_hop_true': 0, 'total': 0})
            total = data['total']
            rate = 100 * data['next_hop_true'] / total if total > 0 else 0
            rates.append(rate)
        
        offset = (i - len(models) / 2) * width
        short_name = model.replace('bedrock_', '').replace('openai_', '').replace('us.anthropic.', '')
        bars = ax1.bar(x + offset, rates, width, label=short_name,
                      color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Step Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('% is_next_logical_hop = True', fontsize=12, fontweight='bold')
    ax1.set_title('Step Alignment: Is Query Targeting the Next Logical Hop?\n(Original: per_step[].is_next_logical_hop)',
                 fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Step {s}' for s in steps])
    ax1.legend(loc='upper right', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 100)
    
    # === BOTTOM PLOT: Alternative definition (step == predicted_hop) ===
    ax2 = axes[1]
    
    for i, model in enumerate(models):
        rates = []
        for step in steps:
            data = model_step_data[model].get(step, {'step_eq_hop': 0, 'total': 0})
            total = data['total']
            rate = 100 * data['step_eq_hop'] / total if total > 0 else 0
            rates.append(rate)
        
        offset = (i - len(models) / 2) * width
        short_name = model.replace('bedrock_', '').replace('openai_', '').replace('us.anthropic.', '')
        bars = ax2.bar(x + offset, rates, width, label=short_name,
                      color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Step Number', fontsize=12, fontweight='bold')
    ax2.set_ylabel('% (step == predicted_hop)', fontsize=12, fontweight='bold')
    ax2.set_title('Step Alignment: Alternative Definition\n(step number equals predicted hop number)',
                 fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Step {s}' for s in steps])
    ax2.legend(loc='upper right', fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved step alignment charts to {output_path}")
    plt.close()
    
    # Print statistics
    print("\n" + "="*80)
    print("STEP ALIGNMENT ANALYSIS")
    print("="*80)
    
    for model in models:
        print(f"\n{model}:")
        print(f"  {'Step':<6} {'Next Logical':>15} {'Step=Hop':>12} {'N':>8}")
        print("  " + "-"*45)
        
        overall_next = 0
        overall_eq = 0
        overall_total = 0
        
        for step in steps:
            data = model_step_data[model].get(step, {'next_hop_true': 0, 'step_eq_hop': 0, 'total': 0})
            total = data['total']
            next_rate = 100 * data['next_hop_true'] / total if total > 0 else 0
            eq_rate = 100 * data['step_eq_hop'] / total if total > 0 else 0
            
            if total > 0:
                print(f"  {step:<6} {next_rate:>14.1f}% {eq_rate:>11.1f}% {total:>8}")
                overall_next += data['next_hop_true']
                overall_eq += data['step_eq_hop']
                overall_total += total
        
        if overall_total > 0:
            print(f"  {'Overall':<6} {100*overall_next/overall_total:>14.1f}% {100*overall_eq/overall_total:>11.1f}% {overall_total:>8}")


def main():
    # Setup paths
    base_dir = Path(__file__).resolve().parents[5]
    output_dir = base_dir  / "data" / "results" / "failure_modes"
    plot_dir = base_dir  / "data" / "plots" / "failure_modes" / "quality"
    plot_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading step alignment data...")
    model_step_data = load_step_alignment_data(output_dir)
    
    if not model_step_data:
        print("No step alignment data found!")
        return
    
    # Create plot
    output_path = plot_dir / "step_alignment.png"
    create_alignment_charts(model_step_data, output_path)


if __name__ == "__main__":
    main()
