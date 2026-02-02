"""
Plot 7c: Faithfulness (Unsupported Claims) vs Accuracy Improvement Over No Context

Scatter plot showing relationship between faithfulness (based on unsupported claims) and 
accuracy improvement (Iterative RAG - No Context baseline).

Faithfulness = 1 - (count of steps with unsupported claims / total steps taken)
Lower unsupported claim rate = higher faithfulness

Insight: Does lower hallucination rate correlate with better improvement from iterative approach?
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
RESPONSES_DIR = Path(__file__).resolve().parents[4] / "responses_reverified"
PLOT_DIR = Path(__file__).resolve().parents[5] / "data" / "plots" / "failure_modes" / "hallucination"
CSV_PATH = Path(__file__).resolve().parents[4] / "results" / "reverify_accuracies.csv"

# Model name mapping
MODEL_NAME_MAP = {
    'responses_openai_gpt-4o_reverified': 'GPT-4o',
    'responses_openai_gpt-5_reverified': 'GPT-5',
    'responses_openrouter_google__gemini-2.5-pro_reverified': 'Gemini 2.5 Pro',
    'responses_openrouter_x-ai__grok-4-fast_reverified': 'Grok 4 Fast',
    'responses_openrouter_z-ai__glm-4.6_reverified': 'GLM 4.6',
    'responses_bedrock_us.deepseek.r1-v1:0-reasoning_reverified': 'DeepSeek R1',
    'responses_bedrock_mistral.mistral-large-2402-v1:0_reverified': 'Mistral Large',
    'responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0_reverified': 'Claude 3.7 Sonnet',
    'responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning_reverified': 'Claude 3.7 Sonnet + Reasoning',
    'responses_openrouter_anthropic__claude-sonnet-4.5_reverified': 'Claude Sonnet 4.5',
    '2_responses_bedrock_us.meta.llama3-3-70b-instruct-v1:0_reverified': 'Llama 3.3 70B',
    'responses_bedrock_us.meta.llama3-3-70b-instruct-v1:0_reverified': 'Llama 3.3 70B',
}


def get_max_source_step(evidence_list):
    """Extract the maximum source_step from evidence array."""
    if not evidence_list:
        return 0
    
    max_step = 0
    for evidence in evidence_list:
        step = evidence.get('source_step', 0)
        if step > max_step:
            max_step = step
    
    return max_step


def load_response_file(file_path):
    """Load response file and return dict keyed by question."""
    responses = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    raw_response = record.get('raw_response', {})
                    question = raw_response.get('question', '')
                    if question:
                        responses[question] = record
                except json.JSONDecodeError:
                    continue
    return responses


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
                
                # Handle alternative naming patterns
                if 'openrouter_anthropic_claude_sonnet_4_5' in model_key:
                    no_context['openrouter_anthropic__claude-sonnet-4.5'] = accuracy
                elif 'openrouter_google__gemini-2.5-pro' in model_key:
                    no_context['openrouter_google__gemini-2.5-pro'] = accuracy
                elif 'openrouter_x-ai__grok-4-fast' in model_key:
                    no_context['openrouter_x-ai__grok-4-fast'] = accuracy
    
    return iterative_rag, no_context


def calculate_faithfulness_by_unsupported_claims():
    """
    Calculate faithfulness as: 1 - (steps with unsupported claims / total steps)
    Returns dict: {model_name: faithfulness_score}
    """
    model_faithfulness = defaultdict(lambda: {'unsupported_steps': 0, 'total_steps': 0})
    
    # Load hallucination judgments
    import glob
    for hall_file in glob.glob(str(OUTPUT_DIR / '*hallucination_judgment.jsonl')):
        filename = Path(hall_file).stem
        base_filename = filename.replace('_hallucination_judgment', '')
        model_name = MODEL_NAME_MAP.get(base_filename, base_filename)
        
        # Load corresponding response file to get max source steps
        response_file = RESPONSES_DIR / f"{base_filename}.jsonl"
        if not response_file.exists():
            continue
        
        responses = load_response_file(response_file)
        
        # Process hallucination judgments
        with open(hall_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        question = record.get('question', '')
                        
                        # Get max source step from response file
                        if question in responses:
                            resp_record = responses[question]
                            raw_response = resp_record.get('raw_response', {})
                            evidence = raw_response.get('evidence', [])
                            max_step = get_max_source_step(evidence)
                            
                            if max_step > 0:
                                model_faithfulness[model_name]['total_steps'] += max_step
                                
                                # Count steps with unsupported claims
                                parsed = record.get('parsed_judgment', {})
                                cf = parsed.get('composition_and_faithfulness', {})
                                unsupported_claims = cf.get('unsupported_claims', [])
                                
                                # Get unique steps with unsupported claims
                                unsupported_steps = set()
                                for claim in unsupported_claims:
                                    if not claim.get('is_supported', True):  # False or missing = unsupported
                                        step = claim.get('source_step')
                                        if step:
                                            unsupported_steps.add(step)
                                
                                model_faithfulness[model_name]['unsupported_steps'] += len(unsupported_steps)
                    
                    except json.JSONDecodeError:
                        continue
    
    # Calculate faithfulness score for each model
    faithfulness_scores = {}
    for model, data in model_faithfulness.items():
        total = data['total_steps']
        unsupported = data['unsupported_steps']
        if total > 0:
            # Faithfulness = 1 - (unsupported / total)
            # Higher score = more faithful (fewer unsupported claims)
            faithfulness_scores[model] = 1.0 - (unsupported / total)
    
    return faithfulness_scores


def main():
    """Generate faithfulness (unsupported claims) vs accuracy improvement plot."""
    # Calculate faithfulness based on unsupported claims
    print("Calculating faithfulness scores based on unsupported claims...")
    avg_faithfulness = calculate_faithfulness_by_unsupported_claims()
    
    if not avg_faithfulness:
        print("No faithfulness data found!")
        return
    
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
                improvement = (iterative_rag[model_key] - no_context[model_key]) * 100
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
    ax.set_xlabel('Faithfulness Score (1 - Unsupported Claims Rate)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy Improvement (Iterative RAG - No Context) [%]', 
                 fontsize=13, fontweight='bold')
    ax.set_title('Relationship Between Faithfulness (Unsupported Claims) and Iterative RAG Improvement\nOver No Context Baseline', 
                fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.9, fontsize=10)
    
    # Set reasonable axis limits
    ax.set_xlim(min(faithfulness) - 0.02, max(faithfulness) + 0.02)
    y_range = max(improvements) - min(improvements)
    ax.set_ylim(min(improvements) - 0.1*y_range, max(improvements) + 0.1*y_range)
    
    plt.tight_layout()
    output_path = PLOT_DIR / '7c_faithfulness_unsupported_vs_improvement_no_context.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n=== Faithfulness (Unsupported Claims) vs Accuracy Improvement Over No Context ===")
    for data in sorted(plot_data, key=lambda x: x['faithfulness'], reverse=True):
        print(f"\n{data['model']}:")
        print(f"  Faithfulness Score: {data['faithfulness']:.3f}")
        print(f"  Improvement: {data['improvement']:.2f} percentage points")


if __name__ == '__main__':
    main()
