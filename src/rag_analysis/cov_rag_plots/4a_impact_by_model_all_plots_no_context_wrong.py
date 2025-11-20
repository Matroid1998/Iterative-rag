"""
Plot: Coverage Gap Impact by Model (Multiple Visualizations) - No Context Wrong Questions Only

Shows the difference in accuracy between with/without coverage gap for each model,
filtered to only include questions that were answered incorrectly in the no-context baseline.

Generates multiple plot types:
1. Horizontal bar chart (sorted by impact)
2. Lollipop chart
3. Scatter plot with diagonal reference line
4. Waterfall chart

Impact = Accuracy_without_gap - Accuracy_with_gap
Higher impact = coverage gaps hurt performance more
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


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
    elif 'claude-sonnet-4.5' in model.lower() or 'claude_sonnet_4_5' in model.lower() or 'claude-4.5' in model.lower():
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


def load_no_context_wrong_questions(base_dir):
    """Load questions that were answered incorrectly in no-context baseline."""
    no_context_dir = base_dir / "response-jsonl-without-context"
    
    if not no_context_dir.exists():
        print(f"Warning: No-context directory not found: {no_context_dir}")
        return {}
    
    wrong_questions = defaultdict(set)
    
    for file_path in no_context_dir.glob("*.jsonl"):
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
    
    return wrong_questions


def extract_question_from_iterative(record):
    """Extract question from iterative record."""
    if 'question_dict' in record:
        return record['question_dict'].get('question', '')
    if 'raw' in record and isinstance(record['raw'], dict):
        return record['raw'].get('question', '')
    return record.get('question', '')


def load_accuracy_by_issue_data_filtered(output_dir, base_dir, wrong_questions):
    """Load accuracy data filtered to no-context wrong questions only."""
    model_data = defaultdict(lambda: {
        'has_gap': {'with_issue': {'correct': 0, 'total': 0}, 'without_issue': {'correct': 0, 'total': 0}},
    })
    
    for file_path in glob.glob(str(output_dir / '*coverage_gap_judgments.jsonl')):
        filename = Path(file_path).name
        model_name = filename.replace('responses_', '').replace('_reverified_coverage_gap_judgments.jsonl', '').replace('_coverage_gap_judgments.jsonl', '')
        model_name = normalize_model_name(model_name)
        
        if model_name not in wrong_questions:
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
    """Plot 1: Horizontal bar chart sorted by impact."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sorted_items = sorted(impacts.items(), key=lambda x: x[1]['impact'], reverse=True)
    models = [item[0] for item in sorted_items]
    impact_values = [item[1]['impact'] for item in sorted_items]
    
    # Color gradient based on impact magnitude
    colors = []
    for val in impact_values:
        if val > 35:
            colors.append('#8b0000')  # Dark red - severe
        elif val > 30:
            colors.append('#d62728')  # Red - very high
        elif val > 25:
            colors.append('#ff7f0e')  # Orange - high
        elif val > 20:
            colors.append('#ffbb78')  # Light orange - moderate
        else:
            colors.append('#2ca02c')  # Green - low
    
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
    ax.set_title('Coverage Gap Impact by Model (No-Context Wrong Questions Only)\n(Sorted by Impact: Accuracy without gap - Accuracy with gap)',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(0, max(impact_values) * 1.15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add color legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc='#8b0000', alpha=0.85, label='Severe (>35pp)'),
        Rectangle((0, 0), 1, 1, fc='#d62728', alpha=0.85, label='Very High (30-35pp)'),
        Rectangle((0, 0), 1, 1, fc='#ff7f0e', alpha=0.85, label='High (25-30pp)'),
        Rectangle((0, 0), 1, 1, fc='#ffbb78', alpha=0.85, label='Moderate (20-25pp)'),
        Rectangle((0, 0), 1, 1, fc='#2ca02c', alpha=0.85, label='Low (<20pp)')
    ]
    from matplotlib.lines import Line2D
    legend_elements.insert(0, Line2D([0], [0], color='blue', linestyle='--', linewidth=2, 
                                     alpha=0.7, label=f'Average: {avg_impact:.1f}pp'))
    
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), 
             fontsize=10, title='Impact Level', frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved horizontal bar chart to {output_path}")
    plt.close()


def create_lollipop_chart(impacts, output_path):
    """Plot 2: Lollipop chart."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sorted_items = sorted(impacts.items(), key=lambda x: x[1]['impact'], reverse=True)
    models = [item[0] for item in sorted_items]
    impact_values = [item[1]['impact'] for item in sorted_items]
    
    y_pos = np.arange(len(models))
    
    ax.hlines(y=y_pos, xmin=0, xmax=impact_values, color='gray', alpha=0.4, linewidth=2)
    
    colors = []
    for val in impact_values:
        if val > 35:
            colors.append('#8b0000')
        elif val > 30:
            colors.append('#d62728')
        elif val > 25:
            colors.append('#ff7f0e')
        elif val > 20:
            colors.append('#ffbb78')
        else:
            colors.append('#2ca02c')
    
    ax.scatter(impact_values, y_pos, s=200, c=colors, alpha=0.85, edgecolors='black', linewidth=2, zorder=5)
    
    for i, val in enumerate(impact_values):
        ax.text(val + 0.5, i, f'{val:.1f}pp',
               va='center', ha='left', fontsize=11, fontweight='bold')
    
    avg_impact = np.mean(impact_values)
    ax.axvline(x=avg_impact, color='blue', linestyle='--', linewidth=2, alpha=0.7,
              label=f'Average: {avg_impact:.1f}pp')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=11)
    ax.set_xlabel('Accuracy Impact (percentage points)', fontsize=13, fontweight='bold')
    ax.set_title('Coverage Gap Impact by Model - Lollipop Chart\n(No-Context Wrong Questions Only)',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(0, max(impact_values) * 1.15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved lollipop chart to {output_path}")
    plt.close()


def create_scatter_plot(impacts, output_path):
    """Plot 3: Scatter plot with diagonal reference line."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    models = list(impacts.keys())
    acc_with = [impacts[m]['acc_with'] for m in models]
    acc_without = [impacts[m]['acc_without'] for m in models]
    impact_values = [impacts[m]['impact'] for m in models]
    
    colors = []
    for val in impact_values:
        if val > 35:
            colors.append('#8b0000')
        elif val > 30:
            colors.append('#d62728')
        elif val > 25:
            colors.append('#ff7f0e')
        elif val > 20:
            colors.append('#ffbb78')
        else:
            colors.append('#2ca02c')
    
    scatter = ax.scatter(acc_without, acc_with, s=250, c=colors, alpha=0.8,
                        edgecolors='black', linewidth=2, zorder=5)
    
    max_val = max(max(acc_with), max(acc_without))
    min_val = min(min(acc_with), min(acc_without))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5,
           label='No Impact (y=x)')
    
    for i, model in enumerate(models):
        ax.annotate(model, (acc_without[i], acc_with[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))
    
    ax.set_xlabel('Accuracy WITHOUT Coverage Gap (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy WITH Coverage Gap (%)', fontsize=13, fontweight='bold')
    ax.set_title('Coverage Gap Impact: Accuracy Comparison (No-Context Wrong Questions)\n(Points below diagonal = coverage gaps hurt performance)',
                fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)
    
    ax.set_aspect('equal')
    margin = 5
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)
    
    mid_x = (min_val + max_val) / 2
    mid_y = (min_val + max_val) / 2
    ax.text(max_val - 5, min_val + 5, 'Severely\nImpacted',
           ha='right', va='bottom', fontsize=11, style='italic',
           bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved scatter plot to {output_path}")
    plt.close()


def create_waterfall_chart(impacts, output_path):
    """Plot 4: Waterfall chart showing cumulative impact."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sorted_items = sorted(impacts.items(), key=lambda x: x[1]['impact'], reverse=True)
    models = [item[0] for item in sorted_items]
    impact_values = [item[1]['impact'] for item in sorted_items]
    
    cumulative = 0
    positions = []
    for val in impact_values:
        positions.append(cumulative)
        cumulative += val
    
    x_pos = np.arange(len(models))
    colors = ['#c44e52' for _ in impact_values]
    
    for i, (pos, val) in enumerate(zip(positions, impact_values)):
        bar = ax.bar(i, val, bottom=pos, color=colors[i], alpha=0.85,
                    edgecolor='black', linewidth=1.5)
        
        ax.text(i, pos + val/2, f'{val:.1f}pp',
               ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        
        if i < len(models) - 1:
            ax.plot([i + 0.4, i + 0.6], [pos + val, pos + val],
                   'k--', linewidth=1, alpha=0.5)
    
    ax.axhline(y=cumulative, color='blue', linestyle='-', linewidth=3, alpha=0.7,
              label=f'Total Cumulative Impact: {cumulative:.1f}pp')
    
    avg_impact = np.mean(impact_values)
    ax.axhline(y=avg_impact * len(models), color='green', linestyle='--', linewidth=2, alpha=0.7,
              label=f'Average × {len(models)} models: {avg_impact * len(models):.1f}pp')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Cumulative Accuracy Impact (pp)', fontsize=13, fontweight='bold')
    ax.set_title('Waterfall Chart: Cumulative Coverage Gap Impact\n(No-Context Wrong Questions Only)',
                fontsize=15, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved waterfall chart to {output_path}")
    plt.close()


def print_summary_statistics(impacts):
    """Print detailed statistics."""
    print("\n" + "="*80)
    print("COVERAGE GAP IMPACT ANALYSIS - SUMMARY (NO-CONTEXT WRONG QUESTIONS ONLY)")
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
    base_dir = Path(__file__).resolve().parents[2]
    output_dir = base_dir / "rag_analysis" / "output"
    plot_dir = base_dir / "rag_analysis" / "cov_rag_plots"
    plot_dir.mkdir(exist_ok=True)
    
    print("Loading no-context wrong questions...")
    wrong_questions = load_no_context_wrong_questions(base_dir)
    
    if not wrong_questions:
        print("No baseline data found!")
        return
    
    print("\nLoading coverage gap data (filtered)...")
    model_data = load_accuracy_by_issue_data_filtered(output_dir, base_dir, wrong_questions)
    
    if not model_data:
        print("No data found!")
        return
    
    impacts = calculate_impact_data(model_data)
    
    print("\nGenerating plots...")
    
    # 1. Horizontal bar chart
    output_path_1 = plot_dir / "4a_impact_by_model_horizontal_bars_no_context_wrong.png"
    create_horizontal_bar_chart(impacts, output_path_1)
    
    # 2. Lollipop chart
    output_path_2 = plot_dir / "4a_impact_by_model_lollipop_no_context_wrong.png"
    create_lollipop_chart(impacts, output_path_2)
    
    # 3. Scatter plot
    output_path_3 = plot_dir / "4a_impact_by_model_scatter_no_context_wrong.png"
    create_scatter_plot(impacts, output_path_3)
    
    # 4. Waterfall chart
    output_path_4 = plot_dir / "4a_impact_by_model_waterfall_no_context_wrong.png"
    create_waterfall_chart(impacts, output_path_4)
    
    print_summary_statistics(impacts)
    
    print("\n" + "="*80)
    print("All plots generated successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
