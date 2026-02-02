"""
Plot 3: Anchor Carry-Drop Rate by Step
Line chart of carry_drop rate by step using anchor_carry_drop.per_step.
One line per model to see where carry fails.
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def load_anchor_carry_data(output_dir):
    """Load anchor carry-drop data by step for each model."""
    model_step_data = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'carry_drop': 0}))
    
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
                    parsed = data.get('parsed_judgment', {})
                    
                    # Anchor carry-drop per step
                    anchor = parsed.get('anchor_carry_drop', {})
                    for step_data in anchor.get('per_step', []):
                        step = step_data.get('step')
                        carry_drop = step_data.get('carry_drop', False)
                        
                        if step is not None:
                            model_step_data[model_name][step]['total'] += 1
                            if carry_drop:
                                model_step_data[model_name][step]['carry_drop'] += 1
                
                except json.JSONDecodeError:
                    continue
    
    return model_step_data


def create_line_chart(model_step_data, output_path):
    """Create line chart showing carry-drop rate by step for each model."""
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Color palette for models
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    models = sorted(model_step_data.keys())
    max_step = max(max(steps.keys()) for steps in model_step_data.values() if steps)
    
    for i, model in enumerate(models):
        steps = sorted(model_step_data[model].keys())
        carry_drop_rates = []
        
        for step in steps:
            stats = model_step_data[model][step]
            total = stats['total']
            if total > 0:
                rate = 100 * stats['carry_drop'] / total
            else:
                rate = 0
            carry_drop_rates.append(rate)
        
        # Plot line
        short_name = model.replace('bedrock_', '').replace('openai_', '').replace('us.anthropic.', '')
        ax.plot(steps, carry_drop_rates, marker='o', linewidth=2.5, 
               markersize=8, label=short_name, color=colors[i % 10], alpha=0.8)
        
        # Add value labels for key points
        for step, rate in zip(steps, carry_drop_rates):
            # Only label GPT 4o and GPT 5 in steps 2 and 5
            is_target_model = 'gpt-4o' in short_name.lower() or 'gpt-5' in short_name.lower()
            is_target_step = step in [2, 5]
            
            if is_target_model and is_target_step:
                import matplotlib.patheffects as pe
                txt = ax.annotate(f'{rate:.1f}%', (step, rate), 
                          textcoords="offset points", xytext=(0, 10),
                          ha='center', fontsize=12, fontweight='bold', color='black')
                txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])
    
    # Customize plot
    ax.set_xlabel('Step Number', fontsize=16, fontweight='bold')
    ax.set_ylabel('Anchor Carry-Drop Rate (%)', fontsize=16, fontweight='bold')
    ax.set_title('Anchor Carry-Drop Rate by Step and Model\n(Higher % = More anchor loss)',
                fontsize=20, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=12, ncol=2)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, max_step + 0.5)
    ax.set_ylim(0, None)
    
    # Set integer ticks for steps
    ax.set_xticks(range(1, max_step + 1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved line chart to {output_path}")
    plt.close()
    
    # Print summary table
    print("\n" + "="*80)
    print("ANCHOR CARRY-DROP RATES BY STEP AND MODEL")
    print("="*80)
    
    for model in models:
        print(f"\n{model}:")
        print(f"  {'Step':<6} {'Total':>8} {'Carry-Drop':>12} {'Rate':>10}")
        print("  " + "-"*40)
        
        for step in sorted(model_step_data[model].keys()):
            stats = model_step_data[model][step]
            total = stats['total']
            carry_drop = stats['carry_drop']
            rate = 100 * carry_drop / total if total > 0 else 0
            print(f"  {step:<6} {total:>8} {carry_drop:>12} {rate:>9.1f}%")


def main():
    # Setup paths
    base_dir = Path(__file__).resolve().parents[5]
    output_dir = base_dir  / "data" / "results" / "failure_modes"
    plot_dir = base_dir  / "data" / "plots" / "failure_modes" / "coverage_gap"
    plot_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading anchor carry-drop data by step...")
    model_step_data = load_anchor_carry_data(output_dir)
    
    if not model_step_data:
        print("No anchor carry-drop data found!")
        return
    
    # Create plot
    output_path = plot_dir / "anchor_carry_by_step.png"
    create_line_chart(model_step_data, output_path)


if __name__ == "__main__":
    main()
