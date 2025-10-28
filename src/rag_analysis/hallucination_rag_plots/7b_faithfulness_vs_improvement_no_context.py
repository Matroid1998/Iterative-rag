"""
Plot 7b: Faithfulness vs Accuracy Improvement Over No Context

Scatter plot showing relationship between average faithfulness score and 
accuracy improvement (Iterative RAG - No Context baseline).

Insight: Does higher faithfulness correlate with better improvement from iterative approach over baseline?
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
            
            # Extract model name from file_name
            model_key = file_name.replace('responses_', '').replace('_reverified.jsonl', '').replace('.jsonl', '')
            
            if folder == 'Iterative-RAG':
                iterative_rag[model_key] = accuracy
            elif folder == 'response-jsonl-without-context':
                no_context[model_key] = accuracy
                
                # Also handle alternative naming patterns for new models
                # Claude Sonnet 4.5: openrouter_anthropic_claude_sonnet_4_5_reasoning → openrouter_anthropic__claude-sonnet-4.5
                if 'openrouter_anthropic_claude_sonnet_4_5' in model_key:
                    no_context['openrouter_anthropic__claude-sonnet-4.5'] = accuracy
                # Gemini 2.5 Pro: openrouter_google__gemini-2.5-pro-reasoning → openrouter_google__gemini-2.5-pro
                elif 'openrouter_google__gemini-2.5-pro' in model_key:
                    no_context['openrouter_google__gemini-2.5-pro'] = accuracy
                # Grok 4 Fast: openrouter_x-ai__grok-4-fast-reasoning → openrouter_x-ai__grok-4-fast
                elif 'openrouter_x-ai__grok-4-fast' in model_key:
                    no_context['openrouter_x-ai__grok-4-fast'] = accuracy
    
    # Manually add GPT-5 no context accuracy if needed
    # no_context['openai_gpt-5'] = 0.XXXX  # Add if available
    
    return iterative_rag, no_context


def main():
    """Generate faithfulness vs accuracy improvement over no context plot."""
    # Load faithfulness scores
    records = load_hallucination_judgments(OUTPUT_DIR)
    
    # Group faithfulness scores by model
    model_faithfulness = defaultdict(list)
    
    for rec in records:
        model = normalize_model_name(rec.get('model', ''))
        cf = rec.get('parsed_judgment', {}).get('composition_and_faithfulness', {})
        suff = cf.get('sufficiency_score_est')
        if suff is not None:
            model_faithfulness[model].append(float(suff))
    
    # Calculate average faithfulness per model
    avg_faithfulness = {}
    for model, scores in model_faithfulness.items():
        avg_faithfulness[model] = np.mean(scores)
    
    # Load accuracies
    iterative_rag, no_context = load_accuracies()
    
    # Model name mapping for matching
    model_mapping = {
        'bedrock_mistral.mistral-large-2402-v1:0': 'Mistral Large',
        'bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0': 'Claude 3.7 Sonnet',
        'bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning': 'Claude 3.7 Sonnet + Reasoning',
        'bedrock_us.deepseek.r1-v1:0-reasoning': 'DeepSeek R1',
        'bedrock_us.meta.llama3-3-70b-instruct-v1:0': 'Llama 3.3 70B',
        'openai_gpt-4o': 'GPT-4o',
        'openai_gpt-5': 'GPT-5',
        'openrouter_anthropic__claude-sonnet-4.5': 'Claude Sonnet 4.5',
        'openrouter_google__gemini-2.5-pro': 'Gemini 2.5 Pro',
        'openrouter_x-ai__grok-4-fast': 'Grok 4 Fast',
        'openrouter_z-ai__glm-4.6': 'GLM 4.6',
    }
    
    # Prepare data for plotting
    plot_data = []
    
    for model_key, normalized_name in model_mapping.items():
        if model_key in iterative_rag and model_key in no_context:
            if normalized_name in avg_faithfulness:
                faith = avg_faithfulness[normalized_name]
                improvement = (iterative_rag[model_key] - no_context[model_key]) * 100  # Convert to percentage points
                plot_data.append({
                    'model': normalized_name,
                    'faithfulness': faith,
                    'improvement': improvement
                })
    
    if not plot_data:
        print("No matching data found!")
        return
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract data
    models = [d['model'] for d in plot_data]
    faithfulness = [d['faithfulness'] for d in plot_data]
    improvements = [d['improvement'] for d in plot_data]
    
    # Define colors for each model
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    # Plot points
    for i, (model, faith, imp, color) in enumerate(zip(models, faithfulness, improvements, colors)):
        ax.scatter(faith, imp, s=400, alpha=0.7, color=color, 
                  edgecolors='black', linewidth=2, label=model)
        
        # Add model name annotation
        ax.annotate(model, (faith, imp), 
                   textcoords="offset points", xytext=(0, -20),
                   ha='center', fontsize=9, fontweight='bold')
    
    # Calculate and display correlation
    if len(faithfulness) > 1:
        correlation = np.corrcoef(faithfulness, improvements)[0, 1]
        ax.text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Add zero line
    ax.axhline(y=0, color='red', linestyle='-', linewidth=1, alpha=0.5)
    
    # Formatting
    ax.set_xlabel('Average Faithfulness Score (Sufficiency)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy Improvement (Iterative RAG - No Context) [%]', 
                 fontsize=13, fontweight='bold')
    ax.set_title('Relationship Between Faithfulness and Iterative RAG Improvement\nOver No Context Baseline', 
                fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.9, fontsize=10)
    
    # Set reasonable axis limits
    ax.set_xlim(min(faithfulness) - 0.05, max(faithfulness) + 0.05)
    y_range = max(improvements) - min(improvements)
    ax.set_ylim(min(improvements) - 0.1*y_range, max(improvements) + 0.1*y_range)
    
    plt.tight_layout()
    output_path = PLOT_DIR / '7b_faithfulness_vs_improvement_no_context.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n=== Faithfulness vs Accuracy Improvement Over No Context ===")
    for data in sorted(plot_data, key=lambda x: x['faithfulness'], reverse=True):
        print(f"\n{data['model']}:")
        print(f"  Avg Faithfulness: {data['faithfulness']:.3f}")
        print(f"  Improvement: {data['improvement']:.2f} percentage points")


if __name__ == '__main__':
    main()
