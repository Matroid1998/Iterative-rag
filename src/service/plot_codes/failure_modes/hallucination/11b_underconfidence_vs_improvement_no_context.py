"""
Plot 11b: Underconfidence vs Accuracy Improvement (Over No Context)

Shows the relationship between underconfidence rate and accuracy improvement 
from Iterative RAG over No Context baseline.
"""
import json
import sys
import csv
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hallucination.hall_plot_utils import (
    load_hallucination_judgments, normalize_model_name
)

OUTPUT_DIR = Path(__file__).resolve().parents[5] / "data" / "results" / "failure_modes"
PLOT_DIR = Path(__file__).resolve().parents[5] / "data" / "plots" / "failure_modes" / "hallucination"
CSV_PATH = Path(__file__).resolve().parents[4] / "results" / "reverify_accuracies.csv"


def is_underconfident(hallucination_judgment: dict) -> bool:
    """Check if run is underconfident."""
    cm = hallucination_judgment.get('confidence_miscalibration', {})
    direction = cm.get('direction', '')
    return direction == 'underconfident_continue'


def load_accuracies():
    """Load accuracies from CSV file."""
    iterative_rag = {}
    no_context = {}
    
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = row['folder']
            file_name = row['file_name']
            accuracy = float(row['accuracy'])
            
            model_key = file_name.replace('responses_', '').replace('_reverified.jsonl', '').replace('.jsonl', '')
            
            if folder == 'Iterative-RAG':
                iterative_rag[model_key] = accuracy
            elif folder == 'response-jsonl-without-context':
                no_context[model_key] = accuracy
    
    return iterative_rag, no_context


def main():
    """Generate underconfidence vs accuracy improvement plot (over no context)."""
    hall_records = load_hallucination_judgments(OUTPUT_DIR)
    
    model_stats = defaultdict(lambda: {'total': 0, 'underconfident': 0})
    
    for rec in hall_records:
        model = normalize_model_name(rec.get('model', ''))
        model_stats[model]['total'] += 1
        
        if is_underconfident(rec.get('parsed_judgment', {})):
            model_stats[model]['underconfident'] += 1
    
    model_underconf_rates = {}
    for model, stats in model_stats.items():
        if stats['total'] > 0:
            model_underconf_rates[model] = 100 * stats['underconfident'] / stats['total']
    
    iterative_rag, no_context = load_accuracies()
    
    model_mapping = {
        'bedrock_mistral.mistral-large-2402-v1:0': 'Mistral Large',
        'bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0': 'Claude 3.7 Sonnet',
        'bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning': 'Claude 3.7 Sonnet + Reasoning',
        'bedrock_us.deepseek.r1-v1:0-reasoning': 'DeepSeek R1',
        'openai_gpt-4o': 'GPT-4o',
        'openai_gpt-5': 'GPT-5',
        'openrouter_anthropic__claude-sonnet-4.5': 'Claude Sonnet 4.5',
        'openrouter_google__gemini-2.5-pro': 'Gemini 2.5 Pro',
        'openrouter_x-ai__grok-4-fast': 'Grok 4 Fast',
    }
    
    plot_data = []
    
    for model_key, normalized_name in model_mapping.items():
        if model_key in iterative_rag and model_key in no_context:
            if normalized_name in model_underconf_rates:
                underconf_rate = model_underconf_rates[normalized_name]
                improvement = (iterative_rag[model_key] - no_context[model_key]) * 100
                plot_data.append({
                    'model': normalized_name,
                    'underconfidence': underconf_rate,
                    'improvement': improvement
                })
    
    if not plot_data:
        print("No matching data found!")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = [d['model'] for d in plot_data]
    underconfidence = [d['underconfidence'] for d in plot_data]
    improvements = [d['improvement'] for d in plot_data]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    for i, (model, underconf, imp, color) in enumerate(zip(models, underconfidence, improvements, colors)):
        ax.scatter(underconf, imp, s=400, alpha=0.7, color=color, 
                  edgecolors='black', linewidth=2, label=model)
        ax.annotate(model, (underconf, imp), 
                   textcoords="offset points", xytext=(0, -20),
                   ha='center', fontsize=9, fontweight='bold')
    
    if len(underconfidence) > 1:
        correlation = np.corrcoef(underconfidence, improvements)[0, 1]
        ax.text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.axhline(y=0, color='red', linestyle='-', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Underconfidence Rate (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy Improvement (Iterative RAG - No Context) [%]', 
                 fontsize=13, fontweight='bold')
    ax.set_title('Relationship Between Underconfidence and Iterative RAG Improvement\nOver No Context Baseline', 
                fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=10)
    
    ax.set_xlim(min(underconfidence) - 2, max(underconfidence) + 2)
    y_range = max(improvements) - min(improvements)
    ax.set_ylim(min(improvements) - 0.1*y_range, max(improvements) + 0.1*y_range)
    
    plt.tight_layout()
    output_path = PLOT_DIR / '11b_underconfidence_vs_improvement_no_context.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    print("\n=== Underconfidence vs Accuracy Improvement (Over No Context) ===")
    for data in sorted(plot_data, key=lambda x: x['underconfidence'], reverse=True):
        model = data['model']
        stats = model_stats[model]
        print(f"\n{data['model']}:")
        print(f"  Underconfidence Rate: {data['underconfidence']:.1f}% ({stats['underconfident']}/{stats['total']} runs)")
        print(f"  Improvement: {data['improvement']:.2f} percentage points")


if __name__ == '__main__':
    main()
