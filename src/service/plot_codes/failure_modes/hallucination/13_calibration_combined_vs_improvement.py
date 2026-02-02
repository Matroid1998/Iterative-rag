"""
Plot 13: Calibration Types vs Accuracy Improvement (Combined)

Shows the relationship between all three calibration types and accuracy improvement 
from Iterative RAG over Gold Context in a single plot.

Insight: How do different calibration behaviors affect iterative retrieval benefits?
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
from hallucination.hall_plot_utils import (
    load_hallucination_judgments, normalize_model_name
)

OUTPUT_DIR = Path(__file__).resolve().parents[5] / "data" / "results" / "failure_modes"
PLOT_DIR = Path(__file__).resolve().parents[5] / "data" / "plots" / "failure_modes" / "hallucination"
CSV_PATH = Path(__file__).resolve().parents[4] / "results" / "reverify_accuracies.csv"


def get_calibration_direction(hallucination_judgment: dict) -> str:
    """Get calibration direction."""
    cm = hallucination_judgment.get('confidence_miscalibration', {})
    return cm.get('direction', '')


def load_accuracies():
    """Load accuracies from CSV file."""
    iterative_rag = {}
    gold_context = {}
    
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = row['folder']
            file_name = row['file_name']
            accuracy = float(row['accuracy'])
            
            # Extract model name from file_name
            model_key = file_name.replace('responses_', '').replace('_reverified.jsonl', '').replace('.jsonl', '')
            
            if folder == 'Iterative-RAG':
                iterative_rag[model_key] = accuracy
            elif folder == 'response-jsonl-with-context':
                gold_context[model_key] = accuracy
                
                # Also handle alternative naming patterns for new models
                # Claude Sonnet 4.5: openrouter_anthropic_claude_sonnet_4_5_reasoning → openrouter_anthropic__claude-sonnet-4.5
                if 'openrouter_anthropic_claude_sonnet_4_5' in model_key:
                    gold_context['openrouter_anthropic__claude-sonnet-4.5'] = accuracy
                # Gemini 2.5 Pro: openrouter_google__gemini-2.5-pro-reasoning → openrouter_google__gemini-2.5-pro
                elif 'openrouter_google__gemini-2.5-pro' in model_key:
                    gold_context['openrouter_google__gemini-2.5-pro'] = accuracy
                # Grok 4 Fast: openrouter_x-ai__grok-4-fast-reasoning → openrouter_x-ai__grok-4-fast
                elif 'openrouter_x-ai__grok-4-fast' in model_key:
                    gold_context['openrouter_x-ai__grok-4-fast'] = accuracy
    
    # Manually add GPT-5 gold context accuracy
    gold_context['openai_gpt-5'] = 0.7168
    
    return iterative_rag, gold_context


def main():
    """Generate combined calibration vs accuracy improvement plot."""
    # Load hallucination judgments
    hall_records = load_hallucination_judgments(OUTPUT_DIR)
    
    # Calculate calibration rates per model
    model_stats = defaultdict(lambda: {
        'total': 0, 
        'overconfident': 0, 
        'underconfident': 0, 
        'well_calibrated': 0
    })
    
    for rec in hall_records:
        model = normalize_model_name(rec.get('model', ''))
        direction = get_calibration_direction(rec.get('parsed_judgment', {}))
        
        model_stats[model]['total'] += 1
        
        if direction == 'overconfident_finalize':
            model_stats[model]['overconfident'] += 1
        elif direction == 'underconfident_continue':
            model_stats[model]['underconfident'] += 1
        elif direction == 'ok':
            model_stats[model]['well_calibrated'] += 1
    
    # Calculate percentages
    model_calibration_rates = {}
    for model, stats in model_stats.items():
        if stats['total'] > 0:
            model_calibration_rates[model] = {
                'overconfident': 100 * stats['overconfident'] / stats['total'],
                'underconfident': 100 * stats['underconfident'] / stats['total'],
                'well_calibrated': 100 * stats['well_calibrated'] / stats['total']
            }
    
    # Load accuracies
    iterative_rag, gold_context = load_accuracies()
    
    # Model name mapping for matching
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
    
    # Prepare data for plotting
    plot_data = []
    
    for model_key, normalized_name in model_mapping.items():
        if model_key in iterative_rag and model_key in gold_context:
            if normalized_name in model_calibration_rates:
                improvement = (iterative_rag[model_key] - gold_context[model_key]) * 100
                calibration = model_calibration_rates[normalized_name]
                
                plot_data.append({
                    'model': normalized_name,
                    'improvement': improvement,
                    'overconfident': calibration['overconfident'],
                    'underconfident': calibration['underconfident'],
                    'well_calibrated': calibration['well_calibrated']
                })
    
    if not plot_data:
        print("No matching data found!")
        return
    
    # Create subplot with 3 panels
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Extract data
    models = [d['model'] for d in plot_data]
    improvements = [d['improvement'] for d in plot_data]
    overconfident = [d['overconfident'] for d in plot_data]
    underconfident = [d['underconfident'] for d in plot_data]
    well_calibrated = [d['well_calibrated'] for d in plot_data]
    
    # Define colors for each model (consistent across subplots)
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    # Plot 1: Overconfident vs Improvement
    for i, (model, overconf, imp, color) in enumerate(zip(models, overconfident, improvements, colors)):
        ax1.scatter(overconf, imp, s=300, alpha=0.7, color=color, 
                   edgecolors='black', linewidth=1.5, label=model if i < 6 else "")
        ax1.annotate(model, (overconf, imp), textcoords="offset points", 
                    xytext=(0, -15), ha='center', fontsize=8, fontweight='bold')
    
    if len(overconfident) > 1:
        corr1 = np.corrcoef(overconfident, improvements)[0, 1]
        ax1.text(0.02, 0.98, f'r = {corr1:.3f}', transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax1.axhline(y=0, color='red', linestyle='-', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Overconfident Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy Improvement (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Overconfident vs Improvement', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Underconfident vs Improvement
    for i, (model, underconf, imp, color) in enumerate(zip(models, underconfident, improvements, colors)):
        ax2.scatter(underconf, imp, s=300, alpha=0.7, color=color, 
                   edgecolors='black', linewidth=1.5)
        ax2.annotate(model, (underconf, imp), textcoords="offset points", 
                    xytext=(0, -15), ha='center', fontsize=8, fontweight='bold')
    
    if len(underconfident) > 1:
        corr2 = np.corrcoef(underconfident, improvements)[0, 1]
        ax2.text(0.02, 0.98, f'r = {corr2:.3f}', transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax2.axhline(y=0, color='red', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Underconfident Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Accuracy Improvement (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Underconfident vs Improvement', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 3: Well-calibrated vs Improvement
    for i, (model, calibrated, imp, color) in enumerate(zip(models, well_calibrated, improvements, colors)):
        ax3.scatter(calibrated, imp, s=300, alpha=0.7, color=color, 
                   edgecolors='black', linewidth=1.5)
        ax3.annotate(model, (calibrated, imp), textcoords="offset points", 
                    xytext=(0, -15), ha='center', fontsize=8, fontweight='bold')
    
    if len(well_calibrated) > 1:
        corr3 = np.corrcoef(well_calibrated, improvements)[0, 1]
        ax3.text(0.02, 0.98, f'r = {corr3:.3f}', transform=ax3.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax3.axhline(y=0, color='red', linestyle='-', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Well-Calibrated Rate (%)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Accuracy Improvement (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Well-Calibrated vs Improvement', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Overall title and legend
    fig.suptitle('Calibration Types vs Iterative RAG Improvement Over Gold Context\n(Each point represents one model)', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Add a single legend for all subplots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02), 
              framealpha=0.9, fontsize=10)
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.92])
    
    output_path = PLOT_DIR / '13_calibration_combined_vs_improvement.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print comprehensive statistics
    print("\n=== Combined Calibration Analysis ===")
    for data in sorted(plot_data, key=lambda x: x['improvement'], reverse=True):
        model = data['model']
        stats = model_stats[model]
        print(f"\n{model}:")
        print(f"  Improvement: {data['improvement']:.2f} percentage points")
        print(f"  Overconfident: {data['overconfident']:.1f}% ({stats['overconfident']}/{stats['total']})")
        print(f"  Underconfident: {data['underconfident']:.1f}% ({stats['underconfident']}/{stats['total']})")
        print(f"  Well-calibrated: {data['well_calibrated']:.1f}% ({stats['well_calibrated']}/{stats['total']})")


if __name__ == '__main__':
    main()