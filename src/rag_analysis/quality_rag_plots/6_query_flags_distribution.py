"""
Plot 6: Query Flags Distribution
Stacked percentage for vague/over_broad/compound/off_topic/anchored across all steps per model.
From per_step[].query_quality.
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def load_query_flags_by_model(output_dir):
    """Load query flags aggregated by model."""
    # Structure: {model: {flag: count, 'total': count}}
    model_flags = defaultdict(lambda: defaultdict(int))
    
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
                        quality = step_data.get('query_quality', {})
                        
                        model_flags[model_name]['total'] += 1
                        
                        if quality.get('vague'):
                            model_flags[model_name]['vague'] += 1
                        if quality.get('over_broad'):
                            model_flags[model_name]['over_broad'] += 1
                        if quality.get('compound'):
                            model_flags[model_name]['compound'] += 1
                        if quality.get('off_topic'):
                            model_flags[model_name]['off_topic'] += 1
                        if quality.get('anchored'):
                            model_flags[model_name]['anchored'] += 1
                
                except json.JSONDecodeError:
                    continue
    
    return model_flags


def create_stacked_bar_chart(model_flags, output_path):
    """Create stacked bar chart of query flags per model."""
    models = sorted(model_flags.keys())
    flags = ['vague', 'over_broad', 'compound', 'off_topic', 'anchored']
    flag_labels = ['Vague', 'Over-broad', 'Compound', 'Off-topic', 'Anchored']
    
    # Prepare data
    data_matrix = []
    for flag in flags:
        row = []
        for model in models:
            total = model_flags[model]['total']
            count = model_flags[model][flag]
            percentage = 100 * count / total if total > 0 else 0
            row.append(percentage)
        data_matrix.append(row)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(models))
    width = 0.6
    
    # Colors for each flag
    colors = ['#c44e52', '#dd8452', '#e9a26e', '#4c72b0', '#55a868']
    
    bottom = np.zeros(len(models))
    bars = []
    
    for flag_idx, (flag, label, color) in enumerate(zip(flags, flag_labels, colors)):
        percentages = data_matrix[flag_idx]
        bars_layer = ax.bar(x, percentages, width, label=label, bottom=bottom,
                           color=color, edgecolor='white', linewidth=1.5, alpha=0.85)
        bars.append(bars_layer)
        
        # Add value labels for significant values
        for i, (bar, pct) in enumerate(zip(bars_layer, percentages)):
            if pct > 3:  # Only label if > 3%
                height = bar.get_height()
                y_pos = bottom[i] + height / 2
                ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                       f'{pct:.1f}%',
                       ha='center', va='center', fontsize=8, fontweight='bold',
                       color='white' if pct > 8 else 'black')
        
        bottom += percentages
    
    # Customize plot
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('Percentage of Query Steps (%)', fontsize=13, fontweight='bold')
    ax.set_title('Query Quality Flags Distribution by Model\n(What query problems does each model exhibit?)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('bedrock_', '').replace('openai_', '').replace('us.anthropic.', '')
                        for m in models], rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=11, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved query flags distribution to {output_path}")
    plt.close()
    
    # Print statistics
    print("\n" + "="*80)
    print("QUERY FLAGS DISTRIBUTION BY MODEL")
    print("="*80)
    print(f"{'Model':<50} {'Vague':>8} {'Over-Br':>8} {'Compound':>10} {'Off-Top':>8} {'Anchored':>10} {'Total':>8}")
    print("-"*106)
    
    for model in models:
        total = model_flags[model]['total']
        vague = 100 * model_flags[model]['vague'] / total if total > 0 else 0
        over_broad = 100 * model_flags[model]['over_broad'] / total if total > 0 else 0
        compound = 100 * model_flags[model]['compound'] / total if total > 0 else 0
        off_topic = 100 * model_flags[model]['off_topic'] / total if total > 0 else 0
        anchored = 100 * model_flags[model]['anchored'] / total if total > 0 else 0
        
        print(f"{model:<50} {vague:>7.1f}% {over_broad:>7.1f}% {compound:>9.1f}% {off_topic:>7.1f}% {anchored:>9.1f}% {total:>8}")


def main():
    # Setup paths
    base_dir = Path(__file__).resolve().parents[2]
    output_dir = base_dir / "rag_analysis" / "output"
    plot_dir = base_dir / "rag_analysis" / "quality_rag_plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading query flags by model...")
    model_flags = load_query_flags_by_model(output_dir)
    
    if not model_flags:
        print("No query flag data found!")
        return
    
    # Create plot
    output_path = plot_dir / "query_flags_distribution.png"
    create_stacked_bar_chart(model_flags, output_path)


if __name__ == "__main__":
    main()
