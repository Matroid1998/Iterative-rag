"""
Plot 5: Composition Failure Rate by Model

Bar chart showing percentage of incorrect answers with composition failure per model.
Denominator is the number of incorrect answers (not total runs).

Insight: Among incorrect answers, which models have higher composition failure rates?
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hallucination.hall_plot_utils import (
    load_hallucination_judgments, load_coverage_judgments, 
    create_merged_dataset, normalize_model_name
)

OUTPUT_DIR = Path(__file__).resolve().parents[5] / "data" / "results" / "failure_modes"
PLOT_DIR = Path(__file__).resolve().parents[5] / "data" / "plots" / "failure_modes" / "hallucination"
REVERIFIED_DIR = Path(__file__).resolve().parents[4] / "responses_reverified"

def load_reverified_correctness(reverified_dir: Path):
    """Load correctness data from valid files in src/responses_reverified."""
    records = []
    for f in reverified_dir.glob('*.jsonl'):
        filename = f.name
        # Extract model name: remove 'responses_' and '_reverified.jsonl'
        # Handle the one consistent anomaly: sonnet_4_5_reasoning.jsonl
        if 'sonnet_4_5' in filename:
             model_raw = 'openrouter_anthropic_claude_sonnet_4_5_reasoning'
        else:
             model_raw = filename.replace('responses_', '').replace('_reverified.jsonl', '')
        
        normalized_model = normalize_model_name(model_raw)
        
        with open(f, 'r', encoding='utf-8') as file:
            for line in file:
                if not line.strip(): continue
                try:
                    rec = json.loads(line)
                    # Extract question from raw if not at top level
                    question = rec.get('question', '')
                    if not question and 'raw' in rec:
                        question = rec['raw'].get('question', '')
                        
                    entry = {
                        'model': normalized_model,
                        'question': question,
                        'is_correct': rec.get('is_correct', False),
                        # Dummy parsed_judgment to satisfy create_merged_dataset structure if accessed
                        'parsed_judgment': {'is_correct': rec.get('is_correct', False)}
                    }
                    records.append(entry)
                except json.JSONDecodeError:
                    continue
    return records


def main():
    """Generate composition failure rate by model plot."""
    # Load hallucination and coverage judgments
    # Load hallucination judgments (numerator source)
    hall_records = load_hallucination_judgments(OUTPUT_DIR)
    
    # Load correctly from both sources, prioritizing reverified
    # output_cov: defaults/backups (e.g. if reverified is broken for a model)
    output_cov = load_coverage_judgments(OUTPUT_DIR)
    # reverified_cov: user requested source (overwrites output_cov)
    reverified_cov = load_reverified_correctness(REVERIFIED_DIR)
    
    cov_records = output_cov + reverified_cov
    
    # Merge datasets to get is_correct field
    merged = create_merged_dataset(hall_records, cov_records, [])
    
    # Group by model
    model_stats = defaultdict(lambda: {'incorrects': 0, 'failures': 0})
    
    for rec in merged:
        model = normalize_model_name(rec.get('model', ''))
        is_correct = rec.get('is_correct', False)
        cf = rec.get('hallucination', {}).get('composition_and_faithfulness', {})
        
        # Only count incorrect answers
        if not is_correct:
            model_stats[model]['incorrects'] += 1
            
            # Count composition failures among incorrect answers
            if cf.get('composition_failure', False):
                model_stats[model]['failures'] += 1
    
    # Calculate percentages
    models = sorted(model_stats.keys())
    failure_rates = []
    failure_counts = []
    total_counts = []
    
    for model in models:
        stats = model_stats[model]
        rate = 100 * stats['failures'] / stats['incorrects'] if stats['incorrects'] > 0 else 0
        failure_rates.append(rate)
        failure_counts.append(stats['failures'])
        total_counts.append(stats['incorrects'])
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(models))
    bars = ax.bar(x, failure_rates, color='#748899', alpha=0.8,
                  edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, rate, fail, total) in enumerate(zip(bars, failure_rates, failure_counts, total_counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{rate:.1f}%\n({fail}/{total})',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add average line
    avg_rate = np.mean(failure_rates)
    ax.axhline(y=avg_rate, color='red', linestyle='--', linewidth=2,
              label=f'Average: {avg_rate:.1f}%', alpha=0.7)
    
    ax.set_ylabel('Composition Failure Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Composition Failure Rate by Model\n(% of incorrect answers with composition failure)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.set_ylim(0, max(failure_rates) * 1.2 if failure_rates else 10)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = PLOT_DIR / '5_composition_failure_rate.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n=== Composition Failure Rate by Model ===")
    print("(% of incorrect answers with composition failure)")
    for model in models:
        stats = model_stats[model]
        rate = 100 * stats['failures'] / stats['incorrects'] if stats['incorrects'] > 0 else 0
        print(f"\n{model}:")
        print(f"  Incorrect answers: {stats['incorrects']}")
        print(f"  Composition failures: {stats['failures']}")
        print(f"  Failure rate: {rate:.1f}%")


if __name__ == '__main__':
    main()
