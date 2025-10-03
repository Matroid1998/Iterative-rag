"""
Plot 10: Overconfidence vs Accuracy

Shows the relationship between overconfidence rate and model accuracy.

Insight: Do overconfident models have lower accuracy?
"""
import json
import sys
import csv
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hallucination_rag_plots.hall_plot_utils import (
    load_hallucination_judgments, normalize_model_name
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'rag_analysis' / 'output'
PLOT_DIR = Path(__file__).resolve().parent
CSV_PATH = Path(__file__).resolve().parents[2] / 'results' / 'reverify_accuracies.csv'


def is_overconfident(hallucination_judgment: dict) -> bool:
    """Check if run is overconfident."""
    cm = hallucination_judgment.get('confidence_miscalibration', {})
    direction = cm.get('direction', '')
    return direction == 'overconfident_finalize'


def load_accuracies():
    """Load accuracies from CSV file."""
    iterative_rag = {}
    
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = row['folder']
            file_name = row['file_name']
            accuracy = float(row['accuracy'])
            
            # Extract model name from file_name
            model_key = file_name.replace('responses_', '').replace('_reverified.jsonl', '')
            
            if folder == 'Iterative-RAG':
                iterative_rag[model_key] = accuracy
    
    return iterative_rag


def main():
    """Generate overconfidence vs accuracy plot."""
    # Load hallucination judgments
    hall_records = load_hallucination_judgments(OUTPUT_DIR)
    
    # Calculate overconfidence rate per model
    model_stats = defaultdict(lambda: {'total': 0, 'overconfident': 0})
    
    for rec in hall_records:
        model = normalize_model_name(rec.get('model', ''))
        model_stats[model]['total'] += 1
        
        if is_overconfident(rec.get('parsed_judgment', {})):
            model_stats[model]['overconfident'] += 1
    
    # Calculate overconfidence percentages
    model_overconf_rates = {}
    for model, stats in model_stats.items():
        if stats['total'] > 0:
            model_overconf_rates[model] = 100 * stats['overconfident'] / stats['total']
    
    # Load accuracies
    iterative_rag = load_accuracies()
    
    # Model name mapping for matching
    model_mapping = {
        'bedrock_mistral.mistral-large-2402-v1:0': 'Mistral Large',
        'bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0': 'Claude 3.7 Sonnet',
        'bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning': 'Claude 3.7 Sonnet + Reasoning',
        'bedrock_us.deepseek.r1-v1:0-reasoning': 'DeepSeek R1',
        'openai_gpt-4o': 'GPT-4o',
        'openai_gpt-5': 'GPT-5',
    }
    
    # Prepare data for plotting
    plot_data = []
    
    for model_key, normalized_name in model_mapping.items():
        if model_key in iterative_rag:
            if normalized_name in model_overconf_rates:
                overconf_rate = model_overconf_rates[normalized_name]
                accuracy = iterative_rag[model_key] * 100  # Convert to percentage
                plot_data.append({
                    'model': normalized_name,
                    'overconfidence': overconf_rate,
                    'accuracy': accuracy
                })
    
    if not plot_data:
        print("No matching data found!")
        return
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract data
    models = [d['model'] for d in plot_data]
    overconfidence = [d['overconfidence'] for d in plot_data]
    accuracies = [d['accuracy'] for d in plot_data]
    
    # Define colors for each model
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    # Plot points
    for i, (model, overconf, acc, color) in enumerate(zip(models, overconfidence, accuracies, colors)):
        ax.scatter(overconf, acc, s=400, alpha=0.7, color=color, 
                  edgecolors='black', linewidth=2, label=model)
        
        # Add model name annotation
        ax.annotate(model, (overconf, acc), 
                   textcoords="offset points", xytext=(0, -20),
                   ha='center', fontsize=9, fontweight='bold')
    
    # Calculate and display correlation
    if len(overconfidence) > 1:
        correlation = np.corrcoef(overconfidence, accuracies)[0, 1]
        ax.text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Formatting
    ax.set_xlabel('Overconfidence Rate (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Relationship Between Overconfidence and Accuracy\nIterative RAG Performance', 
                fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    
    # Set reasonable axis limits
    ax.set_xlim(min(overconfidence) - 2, max(overconfidence) + 2)
    ax.set_ylim(min(accuracies) - 3, max(accuracies) + 3)
    
    plt.tight_layout()
    output_path = PLOT_DIR / '10_overconfidence_vs_accuracy.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n=== Overconfidence vs Accuracy ===")
    for data in sorted(plot_data, key=lambda x: x['overconfidence'], reverse=True):
        model = data['model']
        stats = model_stats[model]
        print(f"\n{data['model']}:")
        print(f"  Overconfidence Rate: {data['overconfidence']:.1f}% ({stats['overconfident']}/{stats['total']} runs)")
        print(f"  Accuracy: {data['accuracy']:.2f}%")


if __name__ == '__main__':
    main()