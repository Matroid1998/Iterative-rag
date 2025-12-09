"""
Plot 6: Anchor Carry-Drop Temporal Pattern (Aggregated)
Line chart showing carry-drop rate by step number across ALL models aggregated.
Insight: Does anchor degradation happen at specific steps?
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


TARGET_MODELS = [
    'openrouter_anthropic__claude-sonnet-4.5',
    'openrouter_google__gemini-2.5-pro',
    'openai_gpt-5',
    'openai_gpt-4o',
    'bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning'
]

def load_temporal_anchor_data(output_dir):
    """Load anchor carry-drop data aggregated across all models by step."""
    step_data = defaultdict(lambda: {'total': 0, 'carry_drop': 0})
    model_contributions = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'carry_drop': 0}))
    
    for file_path in glob.glob(str(output_dir / '*coverage_gap_judgments.jsonl')):
        # Extract model name for detailed breakdown
        filename = Path(file_path).name
        model_name = filename.replace('responses_', '').replace('_reverified_coverage_gap_judgments.jsonl', '')
        
        # Filter for target models
        if model_name not in TARGET_MODELS and f"2_{model_name}" not in TARGET_MODELS and model_name.replace("2_", "") not in TARGET_MODELS:
             # Handle potential "2_" prefix mismatch just in case, though usually extracting from filename handles it if done consistently.
             # Let's be strict but robust.
             # The extracted model_name from filename usually matches the list if we assume standard naming.
             # Let's allow exact match or match with slight variations if needed, but for now exact match against TARGET_MODELS.
             # Actually, let's normalize check to be safe.
             normalized_name = model_name.replace("2_", "") # Remove 2_ if present in filename produced model_name
             
             # Check if any target model matches
             match_found = False
             for target in TARGET_MODELS:
                 if target == model_name or target == normalized_name:
                     match_found = True
                     model_name = target # Unify name
                     break
             
             if not match_found:
                 continue
        
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
                    for step_data_item in anchor.get('per_step', []):
                        step = step_data_item.get('step')
                        carry_drop = step_data_item.get('carry_drop', False)
                        
                        if step is not None and step > 1:  # Only care about step 2+
                            step_data[step]['total'] += 1
                            model_contributions[model_name][step]['total'] += 1
                            
                            if carry_drop:
                                step_data[step]['carry_drop'] += 1
                                model_contributions[model_name][step]['carry_drop'] += 1
                
                except json.JSONDecodeError:
                    continue
    
    return step_data, model_contributions


def create_temporal_line_chart(step_data, model_contributions, output_path):
    """Create line chart showing temporal pattern of anchor carry-drop."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Top plot: Aggregated across all models
    steps = sorted(step_data.keys())
    carry_drop_rates = []
    total_counts = []
    
    for step in steps:
        total = step_data[step]['total']
        carry_drop = step_data[step]['carry_drop']
        rate = 100 * carry_drop / total if total > 0 else 0
        carry_drop_rates.append(rate)
        total_counts.append(total)
    
    # Plot main line
    line = ax1.plot(steps, carry_drop_rates, marker='o', linewidth=3.5, 
                    markersize=12, color='#c44e52', label='Carry-Drop Rate',
                    markeredgecolor='black', markeredgewidth=1.5)
    
    # Add value labels
    for step, rate, count in zip(steps, carry_drop_rates, total_counts):
        ax1.annotate(f'{rate:.1f}%\n(n={count})', (step, rate),
                    textcoords="offset points", xytext=(0, 15),
                    ha='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                             edgecolor='black', alpha=0.8))
    
    # Customize top plot
    ax1.set_xlabel('Step Number', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Anchor Carry-Drop Rate (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Anchor Carry-Drop Temporal Pattern (All Models Aggregated)\n' + 
                 'Does anchor loss increase with step number?',
                 fontsize=15, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(min(steps) - 0.5, max(steps) + 0.5)
    ax1.set_ylim(0, max(carry_drop_rates) * 1.3 if carry_drop_rates else 10)
    ax1.set_xticks(steps)
    
    # Trend line removed as requested.
    
    # Bottom plot: Breakdown by top models
    top_models = sorted(model_contributions.keys(), 
                       key=lambda m: sum(model_contributions[m][s]['total'] for s in steps),
                       reverse=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_models)))
    
    for i, model in enumerate(top_models):
        model_steps = []
        model_rates = []
        
        for step in steps:
            if step in model_contributions[model]:
                total = model_contributions[model][step]['total']
                carry_drop = model_contributions[model][step]['carry_drop']
                if total > 0:
                    rate = 100 * carry_drop / total
                    model_steps.append(step)
                    model_rates.append(rate)
        
        if model_steps:
            short_name = model.replace('bedrock_', '').replace('openai_', '').replace('us.anthropic.', '')
            ax2.plot(model_steps, model_rates, marker='o', linewidth=2.5,
                    markersize=8, label=short_name, color=colors[i], alpha=0.8)
    
    # Customize bottom plot
    ax2.set_xlabel('Step Number', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Anchor Carry-Drop Rate (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Anchor Carry-Drop by Step: Top Models Comparison',
                 fontsize=15, fontweight='bold', pad=15)
    ax2.legend(loc='best', fontsize=10, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(min(steps) - 0.5, max(steps) + 0.5)
    ax2.set_xticks(steps)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved temporal pattern chart to {output_path}")
    plt.close()
    
    # Print statistics
    print("\n" + "="*80)
    print("ANCHOR CARRY-DROP TEMPORAL PATTERN")
    print("="*80)
    
    print("\nAggregated across all models:")
    print(f"{'Step':<6} {'Total':>10} {'Carry-Drop':>12} {'Rate':>10}")
    print("-"*40)
    
    for step in steps:
        total = step_data[step]['total']
        carry_drop = step_data[step]['carry_drop']
        rate = 100 * carry_drop / total if total > 0 else 0
        print(f"{step:<6} {total:>10} {carry_drop:>12} {rate:>9.1f}%")
    
    # Calculate trend
    if len(steps) > 2:
        z = np.polyfit(steps, carry_drop_rates, 1)
        slope = z[0]
        if slope > 0.5:
            trend = "INCREASING"
        elif slope < -0.5:
            trend = "DECREASING"
        else:
            trend = "STABLE"
        
        print(f"\nTrend: {trend} (slope: {slope:.2f}% per step)")
        
        if trend == "INCREASING":
            print("⚠️  Anchor degradation worsens as RAG iterations increase!")
        elif trend == "STABLE":
            print("✓ Anchor carry-drop rate remains relatively stable across steps.")


def main():
    # Setup paths
    base_dir = Path(__file__).resolve().parents[2]
    output_dir = base_dir / "rag_analysis" / "output"
    plot_dir = base_dir / "rag_analysis" / "cov_rag_plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading temporal anchor carry-drop data...")
    step_data, model_contributions = load_temporal_anchor_data(output_dir)
    
    if not step_data:
        print("No temporal anchor data found!")
        return
    
    # Create plot
    output_path = plot_dir / "anchor_carry_temporal.png"
    create_temporal_line_chart(step_data, model_contributions, output_path)


if __name__ == "__main__":
    main()
