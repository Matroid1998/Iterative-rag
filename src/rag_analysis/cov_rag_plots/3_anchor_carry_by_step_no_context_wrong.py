"""
Plot 3: Anchor Carry-Drop Rate by Step - NO CONTEXT WRONG ONLY
Line chart of carry_drop rate by step using anchor_carry_drop.per_step.
Filtered to include only questions where the model was INCORRECT in the No-Context setup.
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

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

def find_no_context_file(model_name_base, no_context_dir):
    """Try to locate the No Context file for a given model base name."""
    # Special mappings (if any strictly required, though dynamic should cover most)
    # Re-using the known mapping as a fallback/first check
    NO_CONTEXT_MAPPING = {
        "responses_openrouter_google__gemini-2.5-pro_reverified.jsonl": "responses_openrouter_google__gemini-2.5-pro-reasoning.jsonl",
        "responses_openrouter_x-ai__grok-4-fast_reverified.jsonl": "responses_openrouter_x-ai__grok-4-fast-reasoning.jsonl",
        "responses_openrouter_z-ai__glm-4.6_reverified.jsonl": "responses_openrouter_z-ai__glm-4.6-reasoning_reverified.jsonl",
    }
    
    # 1. Check special mapping
    candidate_name = f"responses_{model_name_base}.jsonl"
    if candidate_name in NO_CONTEXT_MAPPING:
        return no_context_dir / NO_CONTEXT_MAPPING[candidate_name]
    
    candidate_name_rev = f"responses_{model_name_base}_reverified.jsonl"
    if candidate_name_rev in NO_CONTEXT_MAPPING:
         return no_context_dir / NO_CONTEXT_MAPPING[candidate_name_rev]

    # 2. Try same filename
    candidate = no_context_dir / f"responses_{model_name_base}.jsonl"
    if candidate.exists():
        return candidate
    
    # 3. Try with _reverified suffix
    candidate = no_context_dir / f"responses_{model_name_base}_reverified.jsonl"
    if candidate.exists():
        return candidate

    # 4. Try removing _reverified
    clean_base = model_name_base.replace("_reverified", "")
    candidate = no_context_dir / f"responses_{clean_base}.jsonl"
    if candidate.exists():
        return candidate
        
    return None

def load_no_context_wrong_questions(no_context_dir, model_name_base):
    """Load set of wrong questions for a specific model from No Context data."""
    path = find_no_context_file(model_name_base, no_context_dir)
    wrong_questions = set()
    
    if not path:
        print(f"Warning: No-context file not found for {model_name_base}")
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
        print(f"Loaded {len(wrong_questions)} wrong no-context questions for {model_name_base}")
    else:
        print(f"Warning: No wrong questions found (or file empty) for {model_name_base}")
        
    return wrong_questions

def load_anchor_carry_data(output_dir, no_context_dir):
    """Load anchor carry-drop data by step for ALL models, filtered by no-context wrong."""
    
    model_step_data = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'carry_drop': 0}))
    
    # Iterate over all coverage files
    for file_path in glob.glob(str(output_dir / '*coverage_gap_judgments.jsonl')):
        # Extract model name
        filename = Path(file_path).name
        model_name_base = filename.replace('responses_', '').replace('_reverified_coverage_gap_judgments.jsonl', '')
        # Remove "2_" prefix
        if model_name_base.startswith("2_"):
             model_name_base = model_name_base[2:]

        # Load wrong questions for this specific model
        wrong_questions = load_no_context_wrong_questions(no_context_dir, model_name_base)
        
        if not wrong_questions:
            continue
        
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
                    for step_data in anchor.get('per_step', []):
                        step = step_data.get('step')
                        carry_drop = step_data.get('carry_drop', False)
                        
                        if step is not None:
                            model_step_data[model_name_base][step]['total'] += 1
                            if carry_drop:
                                model_step_data[model_name_base][step]['carry_drop'] += 1
                
                except json.JSONDecodeError:
                    continue
    
    return model_step_data


def create_line_chart(model_step_data, output_path):
    """Create line chart showing carry-drop rate by step for each model."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Color palette
    models = sorted(model_step_data.keys())
    if not models:
        print("No models to plot.")
        return

    # Use a larger colormap for many models
    colors = plt.cm.tab20(np.linspace(0, 1, len(models)))
    
    max_step = 0
    for m in models:
        steps = model_step_data[m].keys()
        if steps:
            max_step = max(max_step, max(steps))
    
    for i, model in enumerate(models):
        steps = sorted(model_step_data[model].keys())
        if not steps:
            continue
        
        carry_drop_rates = []
        
        for step in steps:
            stats = model_step_data[model][step]
            total = stats['total']
            if total > 0:
                rate = 100 * stats['carry_drop'] / total
            else:
                rate = 0
            carry_drop_rates.append(rate)
        
        # Plot line
        short_name = model.replace('bedrock_', '').replace('openai_', '').replace('us.anthropic.', '').replace('openrouter_', '')
        ax.plot(steps, carry_drop_rates, marker='o', linewidth=2.5, 
               markersize=8, label=short_name, color=colors[i], alpha=0.8)
    
    # Customize plot
    ax.set_xlabel('Step Number', fontsize=14, fontweight='bold')
    ax.set_ylabel('Anchor Carry-Drop Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Anchor Carry-Drop Rate by Step (No-Context Wrong - All Models)\n(Higher % = More anchor loss on questions requiring context)',
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=9, ncol=3, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, max_step + 0.5)
    ax.set_ylim(0, None)
    
    # Set integer ticks for steps
    if max_step > 0:
        ax.set_xticks(range(1, max_step + 1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved line chart to {output_path}")
    plt.close()
    
    # Print summary table
    print("\n" + "="*80)
    print("ANCHOR CARRY-DROP RATES BY STEP (NO CONTEXT WRONG - ALL MODELS)")
    print("="*80)
    
    for model in models:
        print(f"\n{model}:")
        print(f"  {'Step':<6} {'Total':>8} {'Carry-Drop':>12} {'Rate':>10}")
        print("  " + "-"*40)
        
        for step in sorted(model_step_data[model].keys()):
            stats = model_step_data[model][step]
            total = stats['total']
            carry_drop = stats['carry_drop']
            rate = 100 * carry_drop / total if total > 0 else 0
            print(f"  {step:<6} {total:>8} {carry_drop:>12} {rate:>9.1f}%")

def main():
    # Setup paths
    base_dir = Path(__file__).resolve().parents[2]
    output_dir = base_dir / "rag_analysis" / "output"
    no_context_dir = base_dir / "response-jsonl-without-context"
    plot_dir = base_dir / "rag_analysis" / "cov_rag_plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading anchor carry-drop data by step (No-Context filtered - ALL MODELS)...")
    model_step_data = load_anchor_carry_data(output_dir, no_context_dir)
    
    if not model_step_data:
        print("No anchor carry-drop data found matching filters!")
        return
    
    # Create plot
    output_path = plot_dir / "anchor_carry_by_step_no_context_wrong.png"
    create_line_chart(model_step_data, output_path)

if __name__ == "__main__":
    main()
