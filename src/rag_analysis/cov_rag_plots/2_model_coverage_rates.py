"""
Plot 2: Model-level Coverage Gap Rates
Bar chart of % has_gap and % any_late_hit per model.
From parsed_judgment.retrieval_coverage_gap.has_gap and late_hit_per_hop.any_late_hit.
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def load_model_coverage_data(output_dir):
    """Load coverage gap and late hit rates per model."""
    model_stats = defaultdict(lambda: {'total': 0, 'has_gap': 0, 'any_late_hit': 0})
    
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
                    
                    model_stats[model_name]['total'] += 1
                    
                    # Coverage gap
                    coverage = parsed.get('retrieval_coverage_gap', {})
                    if coverage.get('has_gap'):
                        model_stats[model_name]['has_gap'] += 1
                    
                    # Late hit
                    late_hit = parsed.get('late_hit_per_hop', {})
                    if late_hit.get('any_late_hit'):
                        model_stats[model_name]['any_late_hit'] += 1
                
                except json.JSONDecodeError:
                    continue
    
    return model_stats


def create_bar_chart(model_stats, output_path):
    """Create grouped bar chart of coverage gap and late hit rates."""
    # Prepare data
    models = sorted(model_stats.keys())
    has_gap_rates = []
    late_hit_rates = []
    
    for model in models:
        stats = model_stats[model]
        total = stats['total']
        if total > 0:
            has_gap_rates.append(100 * stats['has_gap'] / total)
            late_hit_rates.append(100 * stats['any_late_hit'] / total)
        else:
            has_gap_rates.append(0)
            late_hit_rates.append(0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, has_gap_rates, width, label='Has Coverage Gap',
                   color='#c44e52', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, late_hit_rates, width, label='Has Late Hit',
                   color='#55a868', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    add_labels(bars1)
    add_labels(bars2)
    
    # Customize plot
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage of Runs (%)', fontsize=14, fontweight='bold')
    ax.set_title('Coverage Gap and Late Hit Rates by Model\n(Lower is better)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('bedrock_', '').replace('openai_', '') for m in models],
                       rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(max(has_gap_rates), max(late_hit_rates)) * 1.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved bar chart to {output_path}")
    plt.close()
    
    # Print summary table
    print("\n" + "="*80)
    print("MODEL COVERAGE GAP AND LATE HIT RATES")
    print("="*80)
    print(f"{'Model':<50} {'Has Gap':>12} {'Late Hit':>12} {'Total':>8}")
    print("-"*80)
    for model in models:
        stats = model_stats[model]
        total = stats['total']
        gap_pct = 100 * stats['has_gap'] / total if total > 0 else 0
        late_pct = 100 * stats['any_late_hit'] / total if total > 0 else 0
        print(f"{model:<50} {gap_pct:>11.1f}% {late_pct:>11.1f}% {total:>8}")


def main():
    # Setup paths
    base_dir = Path(__file__).resolve().parents[2]
    output_dir = base_dir / "rag_analysis" / "output"
    plot_dir = base_dir / "rag_analysis" / "cov_rag_plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading model coverage data...")
    model_stats = load_model_coverage_data(output_dir)
    
    if not model_stats:
        print("No coverage data found!")
        return
    
    # Create plot
    output_path = plot_dir / "model_coverage_rates.png"
    create_bar_chart(model_stats, output_path)


if __name__ == "__main__":
    main()
