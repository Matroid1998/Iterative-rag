"""
Plot 3: Unsupported Claims Distribution

Histogram showing the distribution of unsupported claims per run, faceted by model.

Insight: Which models make more unsupported claims?
"""
import json
import sys
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hallucination_rag_plots.hall_plot_utils import (
    load_hallucination_judgments, count_unsupported_claims, normalize_model_name
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'rag_analysis' / 'output'
PLOT_DIR = Path(__file__).resolve().parent


def load_correctness_map(output_dir):
    """Load is_correct information from coverage judgment files."""
    correctness = {}  # {(model, question): is_correct}
    
    for file_path in glob.glob(str(output_dir / '*coverage_gap_judgments.jsonl')):
        filename = Path(file_path).name
        model_from_file = filename.replace('responses_', '').replace('_reverified_coverage_gap_judgments.jsonl', '')
        model = normalize_model_name(model_from_file)
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    question = rec.get('question', '')
                    is_correct = rec.get('is_correct', False)
                    correctness[(model, question)] = is_correct
                except json.JSONDecodeError:
                    continue
    
    return correctness


def create_distribution_plot(model_unsupported, models, title, output_path, max_y_limit):
    """Create a distribution plot for a subset of data."""
    n_models = len(models)
    
    # Create subplots
    fig, axes = plt.subplots(n_models, 1, figsize=(12, 3 * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]
    
    # Calculate max unsupported value across all models
    all_values = []
    for vals in model_unsupported.values():
        all_values.extend(vals)
    max_unsupported = max(all_values) if all_values else 0
    bins = range(0, max_unsupported + 2)
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_models))
    
    for i, (model, ax) in enumerate(zip(models, axes)):
        unsupported = model_unsupported[model]
        
        if not unsupported:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=14)
            ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
            ax.set_title(model, fontsize=12, fontweight='bold', pad=10)
            continue
        
        # Create histogram with percentages instead of counts
        n_total = len(unsupported)
        weights = np.ones_like(unsupported) * 100.0 / n_total
        ax.hist(unsupported, bins=bins, alpha=0.75, color=colors[i], 
               edgecolor='black', linewidth=0.5, weights=weights)
        
        # Add statistics
        mean_val = np.mean(unsupported)
        median_val = np.median(unsupported)
        zero_pct = 100 * sum(1 for u in unsupported if u == 0) / len(unsupported)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='blue', linestyle=':', linewidth=2,
                  label=f'Median: {median_val:.0f}')
        
        # Add text box with statistics (positioned on the left)
        stats_text = f'n = {len(unsupported)}\n'
        stats_text += f'Zero claims: {zero_pct:.1f}%\n'
        stats_text += f'Mean: {mean_val:.2f}\n'
        stats_text += f'Max: {max(unsupported)}'
        
        ax.text(0.02, 0.97, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
        ax.set_title(model, fontsize=12, fontweight='bold', pad=10)
        # Position legend on upper right to avoid overlap with text box
        ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Set same y-axis range for all subplots
        ax.set_ylim(0, max_y_limit)
    
    axes[-1].set_xlabel('Number of Unsupported Claims per Run', 
                        fontsize=12, fontweight='bold')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Generate unsupported claims distribution plots separated by correctness."""
    records = load_hallucination_judgments(OUTPUT_DIR)
    correctness_map = load_correctness_map(OUTPUT_DIR)
    
    print(f"Loaded {len(records)} hallucination records")
    print(f"Loaded {len(correctness_map)} correctness mappings")
    
    # Group by model and correctness
    model_unsupported_correct = defaultdict(list)
    model_unsupported_incorrect = defaultdict(list)
    matched_count = 0
    unmatched_count = 0
    
    for rec in records:
        model = normalize_model_name(rec.get('model', ''))
        question = rec.get('question', '')
        
        # Look up correctness from coverage data
        is_correct = correctness_map.get((model, question), None)
        
        if is_correct is None:
            unmatched_count += 1
            continue
        
        matched_count += 1
        judgment = rec.get('parsed_judgment', {})
        unsupported = count_unsupported_claims(judgment)
        
        if is_correct:
            model_unsupported_correct[model].append(unsupported)
        else:
            model_unsupported_incorrect[model].append(unsupported)
    
    print(f"Matched: {matched_count}, Unmatched: {unmatched_count}")
    
    models = sorted(set(model_unsupported_correct.keys()) | set(model_unsupported_incorrect.keys()))
    
    # Calculate global max percentage for consistent y-axis
    max_pct = 0
    for model in models:
        for unsupported_list in [model_unsupported_correct[model], model_unsupported_incorrect[model]]:
            if unsupported_list:
                hist, _ = np.histogram(unsupported_list, bins=range(0, max(unsupported_list) + 2))
                # Convert to percentage
                hist_pct = hist * 100.0 / len(unsupported_list)
                max_pct = max(max_pct, max(hist_pct))
    
    # Add 10% padding to max percentage
    max_y_limit = max_pct * 1.1
    
    # Create plot for CORRECT answers
    output_path_correct = PLOT_DIR / '3_unsupported_claims_distribution_CORRECT.png'
    create_distribution_plot(
        model_unsupported_correct, 
        models,
        'Distribution of Unsupported Claims by Model (CORRECT Answers)',
        output_path_correct,
        max_y_limit
    )
    
    # Create plot for INCORRECT answers
    output_path_incorrect = PLOT_DIR / '3_unsupported_claims_distribution_INCORRECT.png'
    create_distribution_plot(
        model_unsupported_incorrect,
        models,
        'Distribution of Unsupported Claims by Model (INCORRECT Answers)',
        output_path_incorrect,
        max_y_limit
    )
    
    # Print statistics
    print("\n=== Unsupported Claims Statistics ===")
    for model in models:
        print(f"\n{model}:")
        
        # Correct answers
        correct_unsupported = model_unsupported_correct[model]
        if correct_unsupported:
            zero_pct = 100 * sum(1 for u in correct_unsupported if u == 0) / len(correct_unsupported)
            print(f"  CORRECT answers (n={len(correct_unsupported)}):")
            print(f"    Runs with 0 unsupported: {zero_pct:.1f}%")
            print(f"    Mean unsupported: {np.mean(correct_unsupported):.2f}")
            print(f"    Median unsupported: {np.median(correct_unsupported):.0f}")
            print(f"    Max unsupported: {max(correct_unsupported)}")
        else:
            print(f"  CORRECT answers: No data")
        
        # Incorrect answers
        incorrect_unsupported = model_unsupported_incorrect[model]
        if incorrect_unsupported:
            zero_pct = 100 * sum(1 for u in incorrect_unsupported if u == 0) / len(incorrect_unsupported)
            print(f"  INCORRECT answers (n={len(incorrect_unsupported)}):")
            print(f"    Runs with 0 unsupported: {zero_pct:.1f}%")
            print(f"    Mean unsupported: {np.mean(incorrect_unsupported):.2f}")
            print(f"    Median unsupported: {np.median(incorrect_unsupported):.0f}")
            print(f"    Max unsupported: {max(incorrect_unsupported)}")
        else:
            print(f"  INCORRECT answers: No data")


if __name__ == '__main__':
    main()
