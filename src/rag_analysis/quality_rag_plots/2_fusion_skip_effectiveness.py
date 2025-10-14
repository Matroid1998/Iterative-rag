"""
Plot 2: Fusion/Skip Effectiveness (Per Model)
Box plot comparing accuracy of runs with vs without fusion/skip, grouped by number_of_hops.
6 subplots, one for each model.
Insight: Is fusion/skip a good strategy or does it hurt accuracy?
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import csv


def normalize_model_name(model: str) -> str:
    """Normalize model name for display."""
    if 'gpt-5' in model.lower():
        return 'GPT-5'
    elif 'gpt-4o' in model.lower():
        return 'GPT-4o'
    elif 'deepseek' in model.lower() and 'r1' in model.lower():
        return 'DeepSeek R1'
    elif 'claude-3-7' in model.lower() and 'reasoning' in model.lower():
        return 'Claude 3.7 Sonnet + Reasoning'
    elif 'claude-3-7' in model.lower():
        return 'Claude 3.7 Sonnet'
    elif 'claude-sonnet-4.5' in model.lower() or 'claude_sonnet_4_5' in model.lower():
        return 'Claude Sonnet 4.5'
    elif 'gemini-2.5-pro' in model.lower():
        return 'Gemini 2.5 Pro'
    elif 'grok-4' in model.lower():
        return 'Grok 4 Fast'
    elif 'mistral' in model.lower():
        return 'Mistral Large'
    return model

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
                                acc_val = float(acc_str.strip('%')) / 100
                            else:
                                acc_val = float(acc_str) / 100
                            
                            # Normalize model name
                            normalized_model = model.replace('openai-', 'openai_').replace('bedrock-', 'bedrock_')
                            model_accuracy[normalized_model] = acc_val
                        except ValueError:
                            pass
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    return model_accuracy


def load_fusion_skip_data(output_dir, model_accuracy):
    """Load fusion/skip data with correctness information per model."""
    # Structure: {model: {num_hops: {'with_fusion': [is_correct, ...], 'without_fusion': [is_correct, ...]}}}
    model_fusion_data = defaultdict(lambda: defaultdict(lambda: {'with_fusion': [], 'without_fusion': []}))
    
    for file_path in glob.glob(str(output_dir / '*quality_judement.jsonl')):
        filename = Path(file_path).name
        model_name = filename.replace('responses_', '').replace('_reverified_quality_judement.jsonl', '').replace('_quality_judement.jsonl', '')
        normalized_model = normalize_model_name(model_name)
        
        # Try to find corresponding coverage gap file for is_correct
        coverage_file = file_path.replace('quality_judement', 'coverage_gap_judgments')
        coverage_data = {}
        
        if Path(coverage_file).exists():
            with open(coverage_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        question = data.get('question', '').strip()
                        is_correct = data.get('is_correct')
                        if question and is_correct is not None:
                            coverage_data[question] = is_correct
                    except:
                        pass
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    question = data.get('question', '').strip()
                    num_hops = data.get('number_of_hops', 1)
                    parsed = data.get('parsed_judgment', {})
                    
                    # Check if any step has fusion_or_skip
                    has_fusion = False
                    for step_data in parsed.get('per_step', []):
                        if step_data.get('fusion_or_skip'):
                            has_fusion = True
                            break
                    
                    # Get correctness
                    is_correct = coverage_data.get(question)
                    if is_correct is None:
                        continue  # Skip if we don't have correctness info
                    
                    if has_fusion:
                        model_fusion_data[normalized_model][num_hops]['with_fusion'].append(int(is_correct))
                    else:
                        model_fusion_data[normalized_model][num_hops]['without_fusion'].append(int(is_correct))
                
                except json.JSONDecodeError:
                    continue
    
    return model_fusion_data


def create_box_plot(model_fusion_data, output_path):
    """Create box plot comparing accuracy with/without fusion for each model."""
    models = sorted(model_fusion_data.keys())
    
    if len(models) == 0:
        print("No model data found!")
        return
    
    # Create figure with dynamic subplots (3 columns)
    n_models = len(models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Plot each model
    for idx, model in enumerate(models):
        
        ax = axes[idx]
        fusion_data = model_fusion_data[model]
        hop_counts = sorted(fusion_data.keys())
        
        if not hop_counts:
            ax.text(0.5, 0.5, f'No data for {model}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model, fontsize=12, fontweight='bold')
            continue
        
        # Prepare data for box plot
        positions = []
        data_to_plot = []
        labels = []
        colors = []
        
        x_pos = 1
        for num_hops in hop_counts:
            with_fusion = fusion_data[num_hops]['with_fusion']
            without_fusion = fusion_data[num_hops]['without_fusion']
            
            if with_fusion and len(with_fusion) > 0:
                positions.append(x_pos)
                data_to_plot.append(with_fusion)
                labels.append(f'{num_hops}h\nFusion\n(n={len(with_fusion)})')
                colors.append('#c44e52')
                x_pos += 1
            
            if without_fusion and len(without_fusion) > 0:
                positions.append(x_pos)
                data_to_plot.append(without_fusion)
                labels.append(f'{num_hops}h\nNo\n(n={len(without_fusion)})')
                colors.append('#4c72b0')
                x_pos += 1
            
            x_pos += 0.3  # Small gap between hop groups
        
        if not data_to_plot:
            ax.text(0.5, 0.5, f'No data for {model}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model, fontsize=12, fontweight='bold')
            continue
        
        # Create box plot
        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.5,
                        patch_artist=True, showmeans=True, meanline=True,
                        boxprops=dict(linewidth=1),
                        whiskerprops=dict(linewidth=1),
                        capprops=dict(linewidth=1),
                        medianprops=dict(color='black', linewidth=1.5),
                        meanprops=dict(color='darkred', linewidth=1.5, linestyle='--'))
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add mean accuracy labels
        for i, (pos, data) in enumerate(zip(positions, data_to_plot)):
            if len(data) > 0:
                mean_acc = np.mean(data) * 100
                ax.text(pos, 1.02, f'{mean_acc:.0f}%', ha='center', va='bottom',
                       fontsize=7, fontweight='bold')
        
        # Formatting
        ax.set_ylabel('Correctness', fontsize=9, fontweight='bold')
        ax.set_title(model, fontsize=11, fontweight='bold', pad=10)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylim(-0.05, 1.12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.4)
    
    # Hide unused subplots
    for idx in range(len(models), len(axes)):
        axes[idx].axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#c44e52', alpha=0.7, label='With Fusion/Skip'),
        Patch(facecolor='#4c72b0', alpha=0.7, label='Without Fusion/Skip')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, framealpha=0.95, 
              fontsize=10, bbox_to_anchor=(0.5, -0.02))
    
    # Overall title
    fig.suptitle('Fusion/Skip Strategy Effectiveness (Per Model)\nDoes fusion/skip improve or hurt accuracy?',
                fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.985])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved fusion/skip effectiveness plot to {output_path}")
    plt.close()
    
    # Print statistics
    print("\n" + "="*80)
    print("FUSION/SKIP EFFECTIVENESS ANALYSIS (PER MODEL)")
    print("="*80)
    
    for model in sorted(models):
        print(f"\n{model}:")
        fusion_data = model_fusion_data[model]
        hop_counts = sorted(fusion_data.keys())
        
        for num_hops in hop_counts:
            with_fusion = fusion_data[num_hops]['with_fusion']
            without_fusion = fusion_data[num_hops]['without_fusion']
            
            if not with_fusion and not without_fusion:
                continue
            
            print(f"  {num_hops}-hop questions:")
            
            if with_fusion:
                acc = np.mean(with_fusion) * 100
                print(f"    With Fusion/Skip: {acc:.1f}% accuracy (n={len(with_fusion)})")
            
            if without_fusion:
                acc = np.mean(without_fusion) * 100
                print(f"    Without Fusion/Skip: {acc:.1f}% accuracy (n={len(without_fusion)})")
            
            if with_fusion and without_fusion:
                diff = (np.mean(with_fusion) - np.mean(without_fusion)) * 100
                if diff > 2:
                    verdict = "✓ Fusion/Skip HELPS"
                elif diff < -2:
                    verdict = "⚠️  Fusion/Skip HURTS"
                else:
                    verdict = "~ Fusion/Skip NEUTRAL"
                print(f"    Difference: {diff:+.1f}% - {verdict}")


def main():
    # Setup paths
    base_dir = Path(__file__).resolve().parents[2]
    output_dir = base_dir / "rag_analysis" / "output"
    results_dir = base_dir / "results" / "new_results_csv"
    plot_dir = base_dir / "rag_analysis" / "quality_rag_plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading model accuracy...")
    model_accuracy = load_model_accuracy(results_dir)
    
    print("Loading fusion/skip data...")
    model_fusion_data = load_fusion_skip_data(output_dir, model_accuracy)
    
    if not model_fusion_data:
        print("No fusion/skip data found!")
        return
    
    # Create plot
    output_path = plot_dir / "fusion_skip_effectiveness.png"
    create_box_plot(model_fusion_data, output_path)


if __name__ == "__main__":
    main()
