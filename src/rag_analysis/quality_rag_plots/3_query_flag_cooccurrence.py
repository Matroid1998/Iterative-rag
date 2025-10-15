"""
Plot 3: Query Flag Co-occurrence Matrix
Heatmap showing how often query flags (vague, over_broad, compound, off_topic) appear together.
Insight: Are certain query problems correlated?
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

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


def load_query_flags(output_dir):
    """Load all query flag combinations per model."""
    model_flag_combinations = defaultdict(list)
    
    for file_path in glob.glob(str(output_dir / '*quality_judement.jsonl')):
        filename = Path(file_path).name
        model_name = filename.replace('responses_', '').replace('_reverified_quality_judement.jsonl', '').replace('_quality_judement.jsonl', '')
        normalized_model = normalize_model_name(model_name)
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    parsed = data.get('parsed_judgment', {})
                    
                    for step_data in parsed.get('per_step', []):
                        quality = step_data.get('query_quality', {})
                        
                        flags = {
                            'vague': quality.get('vague', False),
                            'over_broad': quality.get('over_broad', False),
                            'compound': quality.get('compound', False),
                            'off_topic': quality.get('off_topic', False),
                        }
                        
                        model_flag_combinations[normalized_model].append(flags)
                
                except json.JSONDecodeError:
                    continue
    
    return model_flag_combinations


def create_heatmap(model_flag_combinations, output_path):
    """Create heatmap of flag co-occurrences for each model."""
    models = sorted(model_flag_combinations.keys())
    flags = ['vague', 'over_broad', 'compound', 'off_topic']
    
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
        flag_combinations = model_flag_combinations[model]
        
        if not flag_combinations:
            ax.text(0.5, 0.5, f'No data for {model}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model, fontsize=12, fontweight='bold')
            continue
        
        # Create co-occurrence matrix
        n = len(flags)
        matrix = np.zeros((n, n))
        
        # Count co-occurrences
        for combo in flag_combinations:
            for i, flag1 in enumerate(flags):
                for j, flag2 in enumerate(flags):
                    if combo.get(flag1) and combo.get(flag2):
                        matrix[i, j] += 1
        
        # Calculate percentages
        total_steps = len(flag_combinations)
        matrix_pct = (matrix / total_steps) * 100
        
        # Create heatmap
        im = ax.imshow(matrix_pct, cmap='YlOrRd', aspect='auto', vmin=0, vmax=min(20, matrix_pct.max()))
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(flags)))
        ax.set_yticks(np.arange(len(flags)))
        ax.set_xticklabels(flags, fontsize=8)
        ax.set_yticklabels(flags, fontsize=8)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(flags)):
            for j in range(len(flags)):
                value = matrix_pct[i, j]
                weight = 'bold' if i == j else 'normal'
                color = 'white' if value > matrix_pct.max() * 0.5 else 'black'
                ax.text(j, i, f'{value:.1f}',
                       ha="center", va="center", color=color,
                       fontsize=7, fontweight=weight)
        
        # Add title
        ax.set_title(model, fontsize=11, fontweight='bold', pad=10)
        
        # Add grid
        ax.set_xticks(np.arange(len(flags))-.5, minor=True)
        ax.set_yticks(np.arange(len(flags))-.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", size=0)
    
    # Hide unused subplots
    for idx in range(len(models), len(axes)):
        axes[idx].axis('off')
    
    # Overall title
    fig.suptitle('Query Flag Co-occurrence Matrix (Per Model)\nHow often do query problems appear together? (%)',
                fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved co-occurrence heatmap to {output_path}")
    plt.close()
    
    # Print correlation analysis
    print("\n" + "="*80)
    print("QUERY FLAG CO-OCCURRENCE ANALYSIS (PER MODEL)")
    print("="*80)
    
    for model in sorted(models):
        flag_combinations = model_flag_combinations[model]
        if not flag_combinations:
            continue
        
        # Create matrix
        n = len(flags)
        matrix = np.zeros((n, n))
        for combo in flag_combinations:
            for i, flag1 in enumerate(flags):
                for j, flag2 in enumerate(flags):
                    if combo.get(flag1) and combo.get(flag2):
                        matrix[i, j] += 1
        
        total_steps = len(flag_combinations)
        matrix_pct = (matrix / total_steps) * 100
        
        print(f"\n{model} (n={total_steps} steps):")
        print("  Individual flag rates:")
        for i, flag in enumerate(flags):
            print(f"    {flag}: {matrix_pct[i, i]:.1f}%")
        
        print("  Top co-occurrences:")
        correlations = []
        for i in range(len(flags)):
            for j in range(i + 1, len(flags)):
                if matrix_pct[i, j] > 0.1:
                    correlations.append((flags[i], flags[j], matrix_pct[i, j]))
        
        correlations.sort(key=lambda x: x[2], reverse=True)
        for flag1, flag2, rate in correlations[:3]:
            print(f"    {flag1} + {flag2}: {rate:.1f}%")


def main():
    # Setup paths
    base_dir = Path(__file__).resolve().parents[2]
    output_dir = base_dir / "rag_analysis" / "output"
    plot_dir = base_dir / "rag_analysis" / "quality_rag_plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading query flags...")
    model_flag_combinations = load_query_flags(output_dir)
    
    if not model_flag_combinations:
        print("No query flag data found!")
        return
    
    total_steps = sum(len(flags) for flags in model_flag_combinations.values())
    print(f"Loaded {total_steps} query steps across {len(model_flag_combinations)} models")
    
    # Create plot
    output_path = plot_dir / "query_flag_cooccurrence.png"
    create_heatmap(model_flag_combinations, output_path)


if __name__ == "__main__":
    main()
