"""
Plot 9: Stability Analysis
Percent of runs with any partial_contradiction_with_prev=true.
Bar of run_level.distractor_latch per model.
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def load_stability_data(output_dir):
    """Load stability metrics: contradictions and distractor latch."""
    # Structure: {model: {'total': count, 'with_contradiction': count, 'distractor_latch': count}}
    model_stability = defaultdict(lambda: {'total': 0, 'with_contradiction': 0, 'distractor_latch': 0})
    
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
                    
                    model_stability[model_name]['total'] += 1
                    
                    # Check for any contradiction in steps
                    has_contradiction = False
                    for step_data in parsed.get('per_step', []):
                        if step_data.get('partial_contradiction_with_prev'):
                            has_contradiction = True
                            break
                    
                    if has_contradiction:
                        model_stability[model_name]['with_contradiction'] += 1
                    
                    # Check distractor latch
                    run_level = parsed.get('run_level', {})
                    if run_level.get('distractor_latch'):
                        model_stability[model_name]['distractor_latch'] += 1
                
                except json.JSONDecodeError:
                    continue
    
    return model_stability


def create_stability_chart(model_stability, output_path):
    """Create bar chart showing stability metrics."""
    models = sorted(model_stability.keys())
    
    # Prepare data
    contradiction_rates = []
    distractor_rates = []
    
    for model in models:
        stats = model_stability[model]
        total = stats['total']
        
        contra_rate = 100 * stats['with_contradiction'] / total if total > 0 else 0
        distractor_rate = 100 * stats['distractor_latch'] / total if total > 0 else 0
        
        contradiction_rates.append(contra_rate)
        distractor_rates.append(distractor_rate)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    x = np.arange(len(models))
    width = 0.6
    
    # === TOP: Partial Contradictions ===
    bars1 = ax1.bar(x, contradiction_rates, width, color='#c44e52', alpha=0.8,
                    edgecolor='black', linewidth=1.5,
                    label='Runs with Contradictions')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('% of Runs with Contradictions', fontsize=12, fontweight='bold')
    ax1.set_title('Partial Answer Contradictions by Model\n(How often do models contradict themselves across steps?)',
                 fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('bedrock_', '').replace('openai_', '').replace('us.anthropic.', '')
                         for m in models], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(contradiction_rates) * 1.3 if contradiction_rates else 5)
    
    # === BOTTOM: Distractor Latch ===
    bars2 = ax2.bar(x, distractor_rates, width, color='#dd8452', alpha=0.8,
                    edgecolor='black', linewidth=1.5,
                    label='Runs with Distractor Latch')
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('% of Runs with Distractor Latch', fontsize=12, fontweight='bold')
    ax2.set_title('Distractor Latch (Scaffold Trap) by Model\n(How often do models get stuck on wrong chemical families?)',
                 fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace('bedrock_', '').replace('openai_', '').replace('us.anthropic.', '')
                         for m in models], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(distractor_rates) * 1.3 if distractor_rates else 20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved stability analysis chart to {output_path}")
    plt.close()
    
    # Print statistics
    print("\n" + "="*80)
    print("STABILITY ANALYSIS")
    print("="*80)
    print(f"{'Model':<50} {'Contradictions':>16} {'Distractor Latch':>18} {'Total':>8}")
    print("-"*96)
    
    for model in models:
        stats = model_stability[model]
        total = stats['total']
        contra_rate = 100 * stats['with_contradiction'] / total if total > 0 else 0
        distractor_rate = 100 * stats['distractor_latch'] / total if total > 0 else 0
        
        print(f"{model:<50} {contra_rate:>15.1f}% {distractor_rate:>17.1f}% {total:>8}")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("\nPartial Contradictions:")
    print("  - Indicates when a model's answer at step t contradicts its answer at step t-1")
    print("  - Lower is better (more stable reasoning)")
    
    print("\nDistractor Latch (Scaffold Trap):")
    print("  - Occurs when the system locks onto a chemically similar but wrong compound family")
    print("  - Lower is better (better target selectivity)")


def main():
    # Setup paths
    base_dir = Path(__file__).resolve().parents[5]
    output_dir = base_dir  / "data" / "results" / "failure_modes"
    plot_dir = base_dir  / "data" / "plots" / "failure_modes" / "quality"
    plot_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading stability data...")
    model_stability = load_stability_data(output_dir)
    
    if not model_stability:
        print("No stability data found!")
        return
    
    # Create plot
    output_path = plot_dir / "stability_analysis.png"
    create_stability_chart(model_stability, output_path)


if __name__ == "__main__":
    main()
