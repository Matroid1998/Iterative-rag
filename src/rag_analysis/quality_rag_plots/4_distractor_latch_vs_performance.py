"""
Plot 4: Distractor Latch vs Model Performance
Bar chart showing distractor_latch rate by model, with accuracy overlay line.
Insight: Models with fewer distractions perform better?
Model accuracies from CSV files in /media/torontoai/Iterative-rag/src/results/new_results_csv
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import csv

def load_model_accuracy(results_dir):
    """Load model accuracy from CSV files."""
    model_accuracy = {}
    
    for csv_file in glob.glob(str(results_dir / '*.csv')):
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try different column names
                    model = row.get('Model', '') or row.get('model', '')
                    accuracy = row.get('Accuracy (%)', '') or row.get('accuracy', '') or row.get('Accuracy', '')
                    
                    model = model.strip()
                    if model and accuracy:
                        try:
                            # Handle percentage format
                            acc_str = str(accuracy).strip()
                            if '%' in acc_str:
                                acc_val = float(acc_str.strip('%'))
                            else:
                                acc_val = float(acc_str)
                            
                            # Normalize model name to match quality file naming
                            normalized_model = model.replace('openai-', 'openai_').replace('bedrock-', 'bedrock_')
                            model_accuracy[normalized_model] = acc_val
                        except ValueError:
                            pass
        except Exception as e:
            print(f"Warning: Error reading {csv_file}: {e}")
    
    return model_accuracy


def load_distractor_latch_data(output_dir):
    """Load distractor latch rates per model."""
    model_stats = defaultdict(lambda: {'total': 0, 'distractor_latch': 0})
    
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
                    
                    model_stats[model_name]['total'] += 1
                    
                    run_level = parsed.get('run_level', {})
                    if run_level.get('distractor_latch'):
                        model_stats[model_name]['distractor_latch'] += 1
                
                except json.JSONDecodeError:
                    continue
    
    return model_stats


def create_bar_with_line_overlay(model_stats, model_accuracy, output_path):
    """Create bar chart with line overlay."""
    # Prepare data
    models = sorted(model_stats.keys())
    distractor_rates = []
    accuracies = []
    
    for model in models:
        stats = model_stats[model]
        total = stats['total']
        distractor_count = stats['distractor_latch']
        rate = 100 * distractor_count / total if total > 0 else 0
        distractor_rates.append(rate)
        
        # Try to find accuracy
        acc = model_accuracy.get(model, None)
        if acc is None:
            # Try with different naming conventions
            for key in model_accuracy.keys():
                if model in key or key in model:
                    acc = model_accuracy[key]
                    break
        accuracies.append(acc)
    
    # Create plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(models))
    width = 0.6
    
    # Bar chart for distractor latch rate
    bars = ax1.bar(x, distractor_rates, width, label='Distractor Latch Rate',
                   color='#c44e52', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Customize left y-axis
    ax1.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Distractor Latch Rate (%)', fontsize=13, fontweight='bold', color='#c44e52')
    ax1.tick_params(axis='y', labelcolor='#c44e52')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('bedrock_', '').replace('openai_', '').replace('us.anthropic.', '')
                         for m in models], rotation=45, ha='right')
    ax1.set_ylim(0, max(distractor_rates) * 1.3 if distractor_rates else 20)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Create second y-axis for accuracy
    ax2 = ax1.twinx()
    
    # Line plot for accuracy
    if any(a is not None for a in accuracies):
        # Handle None values
        valid_x = [i for i, a in enumerate(accuracies) if a is not None]
        valid_acc = [a for a in accuracies if a is not None]
        
        if valid_x:
            line = ax2.plot(valid_x, valid_acc, marker='o', linewidth=3, markersize=10,
                          color='#4c72b0', label='Accuracy', markeredgecolor='white',
                          markeredgewidth=2)
            
            # Add value labels
            for xi, acc in zip(valid_x, valid_acc):
                ax2.text(xi, acc + 1, f'{acc:.1f}%',
                        ha='center', va='bottom', fontsize=10, fontweight='bold',
                        color='#4c72b0')
    
    # Customize right y-axis
    ax2.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold', color='#4c72b0')
    ax2.tick_params(axis='y', labelcolor='#4c72b0')
    ax2.set_ylim(0, 100)
    
    # Title
    ax1.set_title('Distractor Latch Rate vs Model Accuracy\n(Do fewer distractors lead to better performance?)',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved distractor latch vs performance plot to {output_path}")
    plt.close()
    
    # Print statistics
    print("\n" + "="*80)
    print("DISTRACTOR LATCH VS MODEL PERFORMANCE")
    print("="*80)
    print(f"{'Model':<50} {'Distractor Rate':>18} {'Accuracy':>12} {'N':>8}")
    print("-"*90)
    
    for model, rate, acc in zip(models, distractor_rates, accuracies):
        acc_str = f"{acc:.1f}%" if acc is not None else "N/A"
        total = model_stats[model]['total']
        print(f"{model:<50} {rate:>17.1f}% {acc_str:>12} {total:>8}")
    
    # Calculate correlation if we have accuracy data
    valid_data = [(r, a) for r, a in zip(distractor_rates, accuracies) if a is not None]
    if len(valid_data) > 2:
        rates_clean = [r for r, a in valid_data]
        accs_clean = [a for r, a in valid_data]
        correlation = np.corrcoef(rates_clean, accs_clean)[0, 1]
        print(f"\nCorrelation between distractor rate and accuracy: {correlation:.3f}")
        if correlation < -0.3:
            print("✓ Strong negative correlation: Fewer distractors → Better accuracy")
        elif correlation > 0.3:
            print("⚠️  Positive correlation: More distractors → Better accuracy (unexpected!)")
        else:
            print("~ Weak correlation: Distractor rate not strongly related to accuracy")


def main():
    # Setup paths
    base_dir = Path(__file__).resolve().parents[2]
    output_dir = base_dir / "rag_analysis" / "output"
    results_dir = base_dir / "results" / "new_results_csv"
    plot_dir = base_dir / "rag_analysis" / "quality_rag_plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading model accuracy from CSV files...")
    model_accuracy = load_model_accuracy(results_dir)
    print(f"Loaded accuracy for {len(model_accuracy)} models")
    
    print("Loading distractor latch data...")
    model_stats = load_distractor_latch_data(output_dir)
    
    if not model_stats:
        print("No distractor latch data found!")
        return
    
    # Create plot
    output_path = plot_dir / "distractor_latch_vs_performance.png"
    create_bar_with_line_overlay(model_stats, model_accuracy, output_path)


if __name__ == "__main__":
    main()
