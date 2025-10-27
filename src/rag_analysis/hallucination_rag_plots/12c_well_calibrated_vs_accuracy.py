"""
Plot 12c: Well-Calibrated vs Iterative RAG Accuracy

Shows the relationship between well-calibrated rate and absolute accuracy 
achieved by Iterative RAG (not improvement).
"""
import json
import sys
import csv
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hallucination_rag_plots.hall_plot_utils import (
    load_hallucination_judgments, normalize_model_name
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'rag_analysis' / 'output'
PLOT_DIR = Path(__file__).resolve().parent
CSV_PATH = Path(__file__).resolve().parents[2] / 'results' / 'reverify_accuracies.csv'


def is_well_calibrated(hallucination_judgment: dict) -> bool:
    """Check if run is well-calibrated (ok)."""
    cm = hallucination_judgment.get('confidence_miscalibration', {})
    direction = cm.get('direction', '')
    return direction == 'ok'


def load_accuracies():
    """Load accuracies from CSV file."""
    iterative_rag = {}
    
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = row['folder']
            file_name = row['file_name']
            accuracy = float(row['accuracy'])
            
            model_key = file_name.replace('responses_', '').replace('_reverified.jsonl', '').replace('.jsonl', '')
            
            if folder == 'Iterative-RAG':
                iterative_rag[model_key] = accuracy
    
    return iterative_rag


def main():
    """Generate well-calibrated vs iterative RAG accuracy plot."""
    hall_records = load_hallucination_judgments(OUTPUT_DIR)
    
    model_stats = defaultdict(lambda: {'total': 0, 'well_calibrated': 0})
    
    for rec in hall_records:
        model = normalize_model_name(rec.get('model', ''))
        model_stats[model]['total'] += 1
        
        if is_well_calibrated(rec.get('parsed_judgment', {})):
            model_stats[model]['well_calibrated'] += 1
    
    model_calibrated_rates = {}
    for model, stats in model_stats.items():
        if stats['total'] > 0:
            model_calibrated_rates[model] = 100 * stats['well_calibrated'] / stats['total']
    
    iterative_rag = load_accuracies()
    
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
        if model_key in iterative_rag:
            if normalized_name in model_calibrated_rates:
                calibrated_rate = model_calibrated_rates[normalized_name]
                accuracy = iterative_rag[model_key] * 100
                plot_data.append({
                    'model': normalized_name,
                    'well_calibrated': calibrated_rate,
                    'accuracy': accuracy
                })
    
    if not plot_data:
        print("No matching data found!")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = [d['model'] for d in plot_data]
    well_calibrated = [d['well_calibrated'] for d in plot_data]
    accuracies = [d['accuracy'] for d in plot_data]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    for i, (model, calibrated, acc, color) in enumerate(zip(models, well_calibrated, accuracies, colors)):
        ax.scatter(calibrated, acc, s=400, alpha=0.7, color=color, 
                  edgecolors='black', linewidth=2, label=model)
        ax.annotate(model, (calibrated, acc), 
                   textcoords="offset points", xytext=(0, -20),
                   ha='center', fontsize=9, fontweight='bold')
    
    if len(well_calibrated) > 1:
        correlation = np.corrcoef(well_calibrated, accuracies)[0, 1]
        ax.text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.set_xlabel('Well-Calibrated Rate (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Iterative RAG Accuracy (%)', 
                 fontsize=13, fontweight='bold')
    ax.set_title('Relationship Between Well-Calibration and Iterative RAG Accuracy', 
                fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=10)
    
    ax.set_xlim(min(well_calibrated) - 2, max(well_calibrated) + 2)
    ax.set_ylim(min(accuracies) - 5, max(accuracies) + 5)
    
    plt.tight_layout()
    output_path = PLOT_DIR / '12c_well_calibrated_vs_accuracy.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    print("\n=== Well-Calibrated vs Iterative RAG Accuracy ===")
    for data in sorted(plot_data, key=lambda x: x['well_calibrated'], reverse=True):
        model = data['model']
        stats = model_stats[model]
        print(f"\n{data['model']}:")
        print(f"  Well-Calibrated Rate: {data['well_calibrated']:.1f}% ({stats['well_calibrated']}/{stats['total']} runs)")
        print(f"  Accuracy: {data['accuracy']:.2f}%")


if __name__ == '__main__':
    main()
