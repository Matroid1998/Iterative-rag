"""
Plot: Aggregate Accuracy by Coverage Gap - No Context Wrong Questions Only

Same as 4a but filtered to only include questions that were answered incorrectly 
in the no-context (baseline) scenario. Shows if coverage gaps matter more for 
questions that need context.
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


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


def load_no_context_wrong_questions(base_dir):
    """Load questions that were answered incorrectly in no-context baseline."""
    no_context_dir = base_dir / "response-jsonl-without-context"
    
    if not no_context_dir.exists():
        print(f"Warning: No-context directory not found: {no_context_dir}")
        return {}
    
    # Structure: {model: set(questions)}
    wrong_questions = defaultdict(set)
    
    for file_path in no_context_dir.glob("*.jsonl"):
        model_name = file_path.stem
        # Match model name pattern
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
    
    print(f"Loaded no-context wrong questions for {len(wrong_questions)} models")
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
    """Load accuracy data filtered to no-context wrong questions only."""
    # Structure: {model: {issue_type: {'with_issue': {correct, total}, 'without_issue': {correct, total}}}}
    model_data = defaultdict(lambda: {
        'has_gap': {'with_issue': {'correct': 0, 'total': 0}, 'without_issue': {'correct': 0, 'total': 0}},
    })
    
    for file_path in glob.glob(str(output_dir / '*coverage_gap_judgments.jsonl')):
        # Extract model name from filename
        filename = Path(file_path).name
        model_name = filename.replace('responses_', '').replace('_reverified_coverage_gap_judgments.jsonl', '').replace('_coverage_gap_judgments.jsonl', '')
        model_name = normalize_model_name(model_name)
        
        # Get wrong questions for this model
        if model_name not in wrong_questions:
            print(f"  Warning: No baseline data for {model_name}, skipping...")
            continue
        
        model_wrong_set = wrong_questions[model_name]
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    
                    # Extract question and filter
                    question = extract_question_from_iterative(data)
                    if question not in model_wrong_set:
                        continue  # Skip questions that were correct in no-context
                    
                    is_correct = data.get('is_correct')
                    
                    if is_correct is None:
                        continue
                    
                    parsed = data.get('parsed_judgment', {})
                    
                    # Check for issues
                    coverage = parsed.get('retrieval_coverage_gap', {})
                    has_gap = coverage.get('has_gap', False)
                    
                    # Track coverage gap issue only
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


def create_aggregate_accuracy_plot(model_data, output_path):
    """Create single bar chart showing average accuracy with/without coverage gap."""
    
    models = sorted(model_data.keys())
    
    if len(models) == 0:
        print("No model data found!")
        return
    
    # Calculate aggregate accuracy across all models
    total_with_correct = 0
    total_with_all = 0
    total_without_correct = 0
    total_without_all = 0
    
    model_accuracies_with = []
    model_accuracies_without = []
    
    for model in models:
        data = model_data[model]
        
        # With coverage gap
        with_total = data['has_gap']['with_issue']['total']
        with_correct = data['has_gap']['with_issue']['correct']
        if with_total > 0:
            acc_with = 100 * with_correct / with_total
            model_accuracies_with.append(acc_with)
            total_with_correct += with_correct
            total_with_all += with_total
        
        # Without coverage gap
        without_total = data['has_gap']['without_issue']['total']
        without_correct = data['has_gap']['without_issue']['correct']
        if without_total > 0:
            acc_without = 100 * without_correct / without_total
            model_accuracies_without.append(acc_without)
            total_without_correct += without_correct
            total_without_all += without_total
    
    # Calculate average accuracy
    avg_with = np.mean(model_accuracies_with) if model_accuracies_with else 0
    avg_without = np.mean(model_accuracies_without) if model_accuracies_without else 0
    
    # Calculate standard error
    stderr_with = np.std(model_accuracies_with) / np.sqrt(len(model_accuracies_with)) if len(model_accuracies_with) > 1 else 0
    stderr_without = np.std(model_accuracies_without) / np.sqrt(len(model_accuracies_without)) if len(model_accuracies_without) > 1 else 0
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    categories = ['With Coverage Gap', 'Without Coverage Gap']
    accuracies = [avg_with, avg_without]
    errors = [stderr_with, stderr_without]
    colors = ['#c44e52', '#55a868']
    
    x = np.arange(len(categories))
    width = 0.5
    
    bars = ax.bar(x, accuracies, width, yerr=errors, capsize=10,
                  color=colors, alpha=0.85, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for i, (bar, acc, err) in enumerate(zip(bars, accuracies, errors)):
        height = bar.get_height()
        # Main accuracy value
        ax.text(bar.get_x() + bar.get_width()/2., height + err + 2,
               f'{acc:.1f}%',
               ha='center', va='bottom', fontsize=18, fontweight='bold')
    
    # Calculate impact
    impact = avg_without - avg_with
    
    # Formatting
    ax.set_ylabel('Average Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Impact of Coverage Gaps on Model Accuracy\n(No-Context Wrong Questions Only)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved aggregate accuracy plot (filtered) to {output_path}")
    plt.close()
    
    # Print statistics
    print("\n" + "="*80)
    print("AGGREGATE ACCURACY BY COVERAGE GAP (NO-CONTEXT WRONG QUESTIONS ONLY)")
    print("="*80)
    print(f"\nWith Coverage Gap:")
    print(f"  Average Accuracy: {avg_with:.1f}% ± {stderr_with:.1f}% (SE)")
    print(f"  Total Questions: {total_with_all}")
    print(f"  Correct: {total_with_correct}")
    print(f"  Models: {len(model_accuracies_with)}")
    
    print(f"\nWithout Coverage Gap:")
    print(f"  Average Accuracy: {avg_without:.1f}% ± {stderr_without:.1f}% (SE)")
    print(f"  Total Questions: {total_without_all}")
    print(f"  Correct: {total_without_correct}")
    print(f"  Models: {len(model_accuracies_without)}")
    
    print(f"\nImpact: {impact:+.1f} percentage points")
    print(f"{'⚠️  Coverage gaps significantly hurt performance' if impact > 5 else '✓ Coverage gaps have minimal impact' if abs(impact) < 2 else '~ Moderate impact'}")
    
    print("\n" + "="*80)
    print("PER-MODEL BREAKDOWN (NO-CONTEXT WRONG QUESTIONS ONLY)")
    print("="*80)
    for model in sorted(models):
        data = model_data[model]
        
        with_total = data['has_gap']['with_issue']['total']
        with_correct = data['has_gap']['with_issue']['correct']
        acc_with = 100 * with_correct / with_total if with_total > 0 else 0
        
        without_total = data['has_gap']['without_issue']['total']
        without_correct = data['has_gap']['without_issue']['correct']
        acc_without = 100 * without_correct / without_total if without_total > 0 else 0
        
        diff = acc_without - acc_with
        
        print(f"\n{model}:")
        print(f"  With Gap: {acc_with:.1f}% (n={with_total})")
        print(f"  Without Gap: {acc_without:.1f}% (n={without_total})")
        print(f"  Impact: {diff:+.1f}pp")


def main():
    # Setup paths
    base_dir = Path(__file__).resolve().parents[2]
    output_dir = base_dir / "rag_analysis" / "output"
    plot_dir = base_dir / "rag_analysis" / "cov_rag_plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Load no-context wrong questions
    print("Loading no-context wrong questions...")
    wrong_questions = load_no_context_wrong_questions(base_dir)
    
    if not wrong_questions:
        print("No baseline data found!")
        return
    
    # Load data filtered by no-context wrong questions
    print("\nLoading accuracy by coverage gap data (filtered)...")
    model_data = load_accuracy_by_issue_data_filtered(output_dir, base_dir, wrong_questions)
    
    if not model_data:
        print("No data found!")
        return
    
    # Create aggregate plot
    output_path = plot_dir / "4a_aggregate_accuracy_by_coverage_gap_no_context_wrong.png"
    create_aggregate_accuracy_plot(model_data, output_path)


if __name__ == "__main__":
    main()
