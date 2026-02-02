"""
Plot: Coverage Gap Impact by Model - Gold Context Wrong Questions Only

Shows the difference in accuracy between with/without coverage gap for each model,
filtered to only include questions that were answered incorrectly even with full gold context.

This isolates questions where models struggle even with complete information,
to see if coverage gaps make these already-difficult questions even worse.

Impact = Accuracy_without_gap - Accuracy_with_gap
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


def normalize_model_name(model: str) -> str:
    """Normalize model name for display."""
    if 'gpt-5' in model.lower() or 'openai-gpt-5' in model.lower() or 'openai_gpt-5' in model.lower():
        return 'GPT-5'
    elif 'gpt-4o' in model.lower():
        return 'GPT-4o'
    elif 'deepseek' in model.lower() and 'r1' in model.lower():
        return 'DeepSeek R1'
    elif 'claude-3-7' in model.lower() and 'reasoning' in model.lower():
        return 'Claude 3.7 + Reasoning'
    elif 'claude-3-7' in model.lower():
        return 'Claude 3.7 Sonnet'
    elif 'claude-sonnet-4.5' in model.lower() or 'claude-4.5' in model.lower():
        return 'Claude Sonnet 4.5'
    elif 'claude-3-5' in model.lower():
        return 'Claude 3.5 Sonnet'
    elif 'gemini-2.5-pro' in model.lower() or 'gemini-2.5' in model.lower():
        return 'Gemini 2.5 Pro'
    elif 'grok-4' in model.lower():
        return 'Grok 4 Fast'
    elif 'glm-4.6' in model.lower() or 'glm-4' in model.lower():
        return 'GLM 4.6'
    elif 'mistral' in model.lower():
        return 'Mistral Large'
    elif 'llama' in model.lower():
        return 'Llama 3.3 70B'
    return model


def extract_question_from_baseline(record):
    """Extract question from baseline record."""
    if 'raw' in record and isinstance(record['raw'], dict):
        return record['raw'].get('question', '')
    return record.get('question', '')


def load_gold_context_wrong_questions(base_dir):
    """Load questions that were answered incorrectly with full gold context."""
    gold_context_dir = base_dir / "src" / "response-jsonl-with-context"
    
    if not gold_context_dir.exists():
        print(f"Warning: Gold context directory not found: {gold_context_dir}")
        return {}
    
    wrong_questions = defaultdict(set)
    
    for file_path in gold_context_dir.glob("*.jsonl"):
        model_name = file_path.stem
        if 'responses_' in model_name:
            model_name = model_name.replace('responses_', '')
        if '_reverified' in model_name:
            model_name = model_name.replace('_reverified', '')
        
        model_name = normalize_model_name(model_name)
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    question = extract_question_from_baseline(data)
                    is_correct = data.get('is_correct', False)
                    
                    if not is_correct and question:
                        wrong_questions[model_name].add(question)
                
                except json.JSONDecodeError:
                    continue
    
    print(f"Loaded gold-context wrong questions for {len(wrong_questions)} models")
    for model, questions in sorted(wrong_questions.items()):
        print(f"  {model}: {len(questions)} wrong questions")
    
    return wrong_questions


def extract_question_from_iterative(record):
    """Extract question from iterative record."""
    if 'question_dict' in record:
        return record['question_dict'].get('question', '')
    if 'raw' in record and isinstance(record['raw'], dict):
        return record['raw'].get('question', '')
    return record.get('question', '')


def load_accuracy_by_issue_data_filtered(output_dir, base_dir, wrong_questions):
    """Load accuracy data filtered to gold-context wrong questions only."""
    model_data = defaultdict(lambda: {
        'has_gap': {'with_issue': {'correct': 0, 'total': 0}, 'without_issue': {'correct': 0, 'total': 0}},
    })
    
    for file_path in glob.glob(str(output_dir / '*coverage_gap_judgments.jsonl')):
        filename = Path(file_path).name
        model_name = filename.replace('responses_', '').replace('_reverified_coverage_gap_judgments.jsonl', '').replace('_coverage_gap_judgments.jsonl', '')
        model_name = normalize_model_name(model_name)
        
        if model_name not in wrong_questions:
            print(f"  Warning: No gold context baseline data for {model_name}, skipping...")
            continue
        
        model_wrong_set = wrong_questions[model_name]
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    
                    question = extract_question_from_iterative(data)
                    if question not in model_wrong_set:
                        continue
                    
                    is_correct = data.get('is_correct')
                    
                    if is_correct is None:
                        continue
                    
                    parsed = data.get('parsed_judgment', {})
                    coverage = parsed.get('retrieval_coverage_gap', {})
                    has_gap = coverage.get('has_gap', False)
                    
                    if has_gap:
                        model_data[model_name]['has_gap']['with_issue']['total'] += 1
                        if is_correct:
                            model_data[model_name]['has_gap']['with_issue']['correct'] += 1
                    else:
                        model_data[model_name]['has_gap']['without_issue']['total'] += 1
                        if is_correct:
                            model_data[model_name]['has_gap']['without_issue']['correct'] += 1
                
                except json.JSONDecodeError:
                    continue
    
    return model_data


def calculate_impact_data(model_data):
    """Calculate impact (difference) for each model."""
    impacts = {}
    
    for model in model_data:
        data = model_data[model]
        
        with_total = data['has_gap']['with_issue']['total']
        with_correct = data['has_gap']['with_issue']['correct']
        acc_with = 100 * with_correct / with_total if with_total > 0 else 0
        
        without_total = data['has_gap']['without_issue']['total']
        without_correct = data['has_gap']['without_issue']['correct']
        acc_without = 100 * without_correct / without_total if without_total > 0 else 0
        
        impact = acc_without - acc_with
        
        impacts[model] = {
            'impact': impact,
            'acc_with': acc_with,
            'acc_without': acc_without,
            'n_with': with_total,
            'n_without': without_total
        }
    
    return impacts


def create_horizontal_bar_chart(impacts, output_path):
    """Plot: Horizontal bar chart sorted by impact."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sorted_items = sorted(impacts.items(), key=lambda x: x[1]['impact'], reverse=True)
    models = [item[0] for item in sorted_items]
    impact_values = [item[1]['impact'] for item in sorted_items]
    
    # Color gradient based on impact magnitude
    colors = []
    for val in impact_values:
        if val > 25:
            colors.append('#8b0000')  # Dark red - severe
        elif val > 20:
            colors.append('#d62728')  # Red - very high
        elif val > 15:
            colors.append('#ff7f0e')  # Orange - high
        elif val > 10:
            colors.append('#ffbb78')  # Light orange - moderate
        elif val > 5:
            colors.append('#ffd700')  # Gold - low
        else:
            colors.append('#2ca02c')  # Green - minimal
    
    y_pos = np.arange(len(models))
    bars = ax.barh(y_pos, impact_values, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, impact_values)):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
               f'{val:.1f}pp',
               va='center', ha='left', fontsize=11, fontweight='bold')
    
    # Add average reference line
    avg_impact = np.mean(impact_values)
    ax.axvline(x=avg_impact, color='blue', linestyle='--', linewidth=2, alpha=0.7,
              label=f'Average: {avg_impact:.1f}pp')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=11)
    ax.set_xlabel('Accuracy Impact (percentage points)', fontsize=13, fontweight='bold')
    ax.set_title('Coverage Gap Impact by Model (Gold Context Wrong Questions Only)\n(Sorted by Impact: Accuracy without gap - Accuracy with gap)',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(0, max(impact_values) * 1.15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add color legend
    legend_elements = [
        Line2D([0], [0], color='blue', linestyle='--', linewidth=2, 
               alpha=0.7, label=f'Average: {avg_impact:.1f}pp'),
        Rectangle((0, 0), 1, 1, fc='#8b0000', alpha=0.85, label='Severe (>25pp)'),
        Rectangle((0, 0), 1, 1, fc='#d62728', alpha=0.85, label='Very High (20-25pp)'),
        Rectangle((0, 0), 1, 1, fc='#ff7f0e', alpha=0.85, label='High (15-20pp)'),
        Rectangle((0, 0), 1, 1, fc='#ffbb78', alpha=0.85, label='Moderate (10-15pp)'),
        Rectangle((0, 0), 1, 1, fc='#ffd700', alpha=0.85, label='Low (5-10pp)'),
        Rectangle((0, 0), 1, 1, fc='#2ca02c', alpha=0.85, label='Minimal (<5pp)')
    ]
    
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), 
             fontsize=10, title='Impact Level', frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved horizontal bar chart to {output_path}")
    plt.close()


def print_summary_statistics(impacts):
    """Print detailed statistics."""
    print("\n" + "="*80)
    print("COVERAGE GAP IMPACT ANALYSIS (GOLD CONTEXT WRONG QUESTIONS ONLY)")
    print("="*80)
    
    impact_values = [data['impact'] for data in impacts.values()]
    
    print(f"\nOverall Statistics:")
    print(f"  Average Impact: {np.mean(impact_values):.1f} pp")
    print(f"  Median Impact: {np.median(impact_values):.1f} pp")
    print(f"  Std Dev: {np.std(impact_values):.1f} pp")
    print(f"  Min Impact: {min(impact_values):.1f} pp ({[m for m, d in impacts.items() if d['impact'] == min(impact_values)][0]})")
    print(f"  Max Impact: {max(impact_values):.1f} pp ({[m for m, d in impacts.items() if d['impact'] == max(impact_values)][0]})")
    
    print("\n" + "="*80)
    print("PER-MODEL BREAKDOWN (Sorted by Impact)")
    print("="*80)
    
    sorted_items = sorted(impacts.items(), key=lambda x: x[1]['impact'], reverse=True)
    
    for rank, (model, data) in enumerate(sorted_items, 1):
        print(f"\n{rank}. {model}:")
        print(f"   With Gap: {data['acc_with']:.1f}% (n={data['n_with']})")
        print(f"   Without Gap: {data['acc_without']:.1f}% (n={data['n_without']})")
        print(f"   Impact: {data['impact']:+.1f} pp")


def main():
    base_dir = Path(__file__).resolve().parents[5]
    output_dir = base_dir  / "data" / "results" / "failure_modes"
    plot_dir = base_dir  / "data" / "plots" / "failure_modes" / "coverage_gap"
    plot_dir.mkdir(exist_ok=True)
    
    print("Loading gold-context wrong questions...")
    wrong_questions = load_gold_context_wrong_questions(base_dir)
    
    if not wrong_questions:
        print("No gold context baseline data found!")
        return
    
    print("\nLoading coverage gap data (filtered to gold-context wrong)...")
    model_data = load_accuracy_by_issue_data_filtered(output_dir, base_dir, wrong_questions)
    
    if not model_data:
        print("No data found!")
        return
    
    impacts = calculate_impact_data(model_data)
    
    if not impacts:
        print("No impact data calculated!")
        return
    
    print("\nGenerating horizontal bar chart...")
    output_path = plot_dir / "4a_impact_by_model_horizontal_bars_gold_context_wrong.png"
    create_horizontal_bar_chart(impacts, output_path)
    
    print_summary_statistics(impacts)
    
    print("\n" + "="*80)
    print("Plot generated successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
