"""
Plot 6: Anchor Carry-Drop Temporal Pattern (Aggregated) - GOLD CONTEXT WRONG ONLY
Line chart showing carry-drop rate by step number across models.
Top Plot: Average of ALL available models.
Bottom Plot: Breakdown of selected TARGET models.
Filtered to include only questions where the model was INCORRECT in the Gold Context setup.
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Target models for the breakdown plot (Bottom Chart)
TARGET_MODELS_FILES = {
    'openrouter_anthropic__claude-sonnet-4.5': 'responses_openrouter_anthropic_claude_sonnet_4_5_reasoning.jsonl',
    'openrouter_google__gemini-2.5-pro': 'responses_openrouter_google__gemini-2.5-pro-reasoning.jsonl',
    'bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning': 'responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning_reverified.jsonl',
    'bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0': 'responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0_reverified.jsonl'
}

TARGET_MODELS = list(TARGET_MODELS_FILES.keys())

def extract_question(record):
    """Extract question text from a record."""
    question = record.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()
    for key in ("raw", "raw_response"):
        raw = record.get(key)
        if isinstance(raw, dict):
            q = raw.get("question")
            if isinstance(q, str) and q.strip():
                return q.strip()
    return None

def find_gold_context_file(model_name_base, gold_context_dir):
    """Try to locate the Gold Context file for a given model base name."""
    # Special mappings
    gold_mappings = {
        # Add any known special mappings here if auto-discovery fails
        "openrouter_anthropic__claude-sonnet-4.5": "responses_openrouter_anthropic_claude_sonnet_4_5_reasoning.jsonl",
        "openrouter_google__gemini-2.5-pro": "responses_openrouter_google__gemini-2.5-pro-reasoning.jsonl",
        "bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning": "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning_reverified.jsonl",
        # Grok and Gemini mappings from other scripts
         "openrouter_x-ai__grok-4-fast": "responses_openrouter_x-ai__grok-4-fast-reasoning.jsonl",
         "openrouter_z-ai__glm-4.6": "responses_openrouter_z-ai__glm-4.6-reasoning_reverified.jsonl"
    }
    
    # 1. Check special mapping
    if model_name_base in gold_mappings:
        candidate = gold_context_dir / gold_mappings[model_name_base]
        if candidate.exists():
            return candidate
            
    # 2. Try constructing filename: responses_[model].jsonl
    candidate = gold_context_dir / f"responses_{model_name_base}.jsonl"
    if candidate.exists():
        return candidate
    
    # 3. Try with _reverified suffix
    candidate = gold_context_dir / f"responses_{model_name_base}_reverified.jsonl"
    if candidate.exists():
        return candidate

    # 4. Try removing _reverified if it was part of base
    clean_base = model_name_base.replace("_reverified", "")
    candidate = gold_context_dir / f"responses_{clean_base}.jsonl"
    if candidate.exists():
        return candidate
        
    return None

def load_gold_context_wrong_questions(gold_context_dir, model_name_base):
    """
    Load set of wrong questions for each target model.
    """
    path = find_gold_context_file(model_name_base, gold_context_dir)
    wrong_questions = set()
    
    if not path:
        print(f"Warning: Gold context file not found for {model_name_base}")
        return wrong_questions
        
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                question = extract_question(record)
                if not question:
                    continue
                
                is_correct = bool(record.get("is_correct", False))
                if not is_correct:
                    wrong_questions.add(question)
            except json.JSONDecodeError:
                continue
    
    if wrong_questions:
        print(f"Loaded {len(wrong_questions)} wrong gold questions for {model_name_base}")
    else:
        print(f"Warning: No wrong questions found (or file empty) for {model_name_base}")
        
    return wrong_questions

def load_temporal_anchor_data(output_dir, gold_context_dir):
    """Load anchor carry-drop data aggregated across ALL models by step, filtered by gold wrong."""
    
    step_data = defaultdict(lambda: {'total': 0, 'carry_drop': 0})
    model_contributions = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'carry_drop': 0}))
    
    loaded_models_count = 0
    
    for file_path in glob.glob(str(output_dir / '*coverage_gap_judgments.jsonl')):
        # Extract model name
        filename = Path(file_path).name
        model_name_base = filename.replace('responses_', '').replace('_reverified_coverage_gap_judgments.jsonl', '')
        
        # Normalize simple case for TARGET_MODELS matching later
        normalized_name = model_name_base.replace("2_", "")
        
        # Load wrong questions for this specific model
        wrong_questions = load_gold_context_wrong_questions(gold_context_dir, model_name_base)
        
        if not wrong_questions:
            continue
            
        loaded_models_count += 1
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    
                    # Check if this question is in the wrong set
                    question = data.get("question", "").strip()
                    if not question or question not in wrong_questions:
                        continue
                        
                    parsed = data.get('parsed_judgment', {})
                    
                    # Anchor carry-drop per step
                    anchor = parsed.get('anchor_carry_drop', {})
                    for step_data_item in anchor.get('per_step', []):
                        step = step_data_item.get('step')
                        carry_drop = step_data_item.get('carry_drop', False)
                        
                        if step is not None and step > 1:  # Only care about step 2+
                            # Aggregate to GLOBAL stats (All models)
                            step_data[step]['total'] += 1
                            if carry_drop:
                                step_data[step]['carry_drop'] += 1
                            
                            # Aggregate to MODEL stats (for breakdown)
                            # We store data for ALL models here, but will only plot TARGET_MODELS later
                            # Use normalized name for consistency with target list
                            model_key = normalized_name # Use normalized name for consistency in model_contributions keys
                            
                            model_contributions[model_key][step]['total'] += 1
                            if carry_drop:
                                model_contributions[model_key][step]['carry_drop'] += 1
                
                except json.JSONDecodeError:
                    continue
    
    print(f"\nSuccessfully loaded data for {loaded_models_count} models.\n")
    return step_data, model_contributions


def create_temporal_line_chart(step_data, model_contributions, output_path):
    """Create line chart showing temporal pattern of anchor carry-drop (Filtered)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Top plot: Aggregated across ALL models
    steps = sorted(step_data.keys())
    carry_drop_rates = []
    total_counts = []
    
    for step in steps:
        total = step_data[step]['total']
        carry_drop = step_data[step]['carry_drop']
        rate = 100 * carry_drop / total if total > 0 else 0
        carry_drop_rates.append(rate)
        total_counts.append(total)
    
    # Plot main line
    line = ax1.plot(steps, carry_drop_rates, marker='o', linewidth=3.5, 
                    markersize=12, color='#c44e52', label='Carry-Drop Rate',
                    markeredgecolor='black', markeredgewidth=1.5)
    
    # Add value labels
    for step, rate, count in zip(steps, carry_drop_rates, total_counts):
        ax1.annotate(f'{rate:.1f}%\n(n={count})', (step, rate),
                    textcoords="offset points", xytext=(0, 15),
                    ha='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                             edgecolor='black', alpha=0.8))
    
    # Customize top plot
    ax1.set_xlabel('Step Number', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Anchor Carry-Drop Rate (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Anchor Carry-Drop Temporal Pattern (Gold Wrong - Average of ALL Models)\n' + 
                 'Does anchor degradation happen over time on hard questions?',
                 fontsize=15, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    if steps:
        ax1.set_xlim(min(steps) - 0.5, max(steps) + 0.5)
        ax1.set_ylim(0, max(carry_drop_rates) * 1.3 if carry_drop_rates else 10)
        ax1.set_xticks(steps)
    
    # Bottom plot: Breakdown by TARGET models only
    colors = plt.cm.tab10(np.linspace(0, 1, len(TARGET_MODELS)))
    
    for i, model in enumerate(TARGET_MODELS):
        if model not in model_contributions:
            print(f"Warning: No data found for target model {model}")
            continue
            
        model_steps = []
        model_rates = []
        
        for step in steps:
            if step in model_contributions[model]:
                total = model_contributions[model][step]['total']
                carry_drop = model_contributions[model][step]['carry_drop']
                if total > 0:
                    rate = 100 * carry_drop / total
                    model_steps.append(step)
                    model_rates.append(rate)
        
        if model_steps:
            short_name = model.replace('bedrock_', '').replace('openai_', '').replace('us.anthropic.', '').replace('openrouter_', '')
            ax2.plot(model_steps, model_rates, marker='o', linewidth=2.5,
                    markersize=8, label=short_name, color=colors[i], alpha=0.8)
    
    # Customize bottom plot
    ax2.set_xlabel('Step Number', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Anchor Carry-Drop Rate (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Anchor Carry-Drop by Step: Selected Models Breakdown',
                 fontsize=15, fontweight='bold', pad=15)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=10, ncol=2, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    if steps:
        ax2.set_xlim(min(steps) - 0.5, max(steps) + 0.5)
        ax2.set_xticks(steps)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved temporal pattern chart to {output_path}")
    plt.close()
    
    # Print statistics
    print("\n" + "="*80)
    print("ANCHOR CARRY-DROP TEMPORAL PATTERN (GOLD WRONG - ALL MODELS)")
    print("="*80)
    
    print("\nAggregated across ALL models:")
    print(f"{'Step':<6} {'Total':>10} {'Carry-Drop':>12} {'Rate':>10}")
    print("-"*40)
    
    for step in steps:
        rate = 100 * carry_drop / total if total > 0 else 0
        print(f"{step:<6} {total:>10} {carry_drop:>12} {rate:>9.1f}%")

def main():
    # Setup paths
    base_dir = Path(__file__).resolve().parents[2]
    output_dir = base_dir / "rag_analysis" / "output"
    gold_context_dir = base_dir / "response-jsonl-with-context"
    plot_dir = base_dir / "rag_analysis" / "cov_rag_plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading temporal anchor carry-drop data (Gold filtered)...")
    step_data, model_contributions = load_temporal_anchor_data(output_dir, gold_context_dir)
    
    if not step_data:
        print("No temporal anchor data found matching filters!")
        return
    
    # Create plot
    output_path = plot_dir / "anchor_carry_temporal_gold_wrong.png"
    create_temporal_line_chart(step_data, model_contributions, output_path)


if __name__ == "__main__":
    main()
