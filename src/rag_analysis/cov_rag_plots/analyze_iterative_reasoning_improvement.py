"""
Analysis: Iterative RAG Reasoning Improvement (Coverage Gap Filtered)

For each model, finds questions that were:
1. Answered CORRECTLY in Iterative RAG
2. Answered INCORRECTLY in Gold Context
3. Have NO coverage gaps (so improvement is purely from iterative reasoning, not better retrieval)

Then calculates the percentage of such questions as a measure of pure reasoning benefit.
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import pandas as pd


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


def extract_question(record):
    """Extract question from record."""
    if 'question_dict' in record:
        return record['question_dict'].get('question', '')
    if 'raw' in record and isinstance(record['raw'], dict):
        return record['raw'].get('question', '')
    return record.get('question', '')


def load_gold_context_results(base_dir):
    """Load gold context results (with full context)."""
    gold_dir = base_dir / "response-jsonl-with-context"
    
    if not gold_dir.exists():
        print(f"Warning: Gold context directory not found: {gold_dir}")
        return {}
    
    # Structure: {model: {question: is_correct}}
    gold_results = defaultdict(dict)
    
    for file_path in gold_dir.glob("*.jsonl"):
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
                    question = extract_question(data)
                    is_correct = data.get('is_correct', False)
                    
                    if question:
                        gold_results[model_name][question] = is_correct
                
                except json.JSONDecodeError:
                    continue
    
    return gold_results


def load_iterative_rag_results_with_coverage(output_dir):
    """Load iterative RAG results with coverage gap information."""
    # Structure: {model: {question: {'is_correct': bool, 'has_gap': bool}}}
    iterative_results = defaultdict(dict)
    
    for file_path in glob.glob(str(output_dir / '*coverage_gap_judgments.jsonl')):
        filename = Path(file_path).name
        model_name = filename.replace('responses_', '').replace('_reverified_coverage_gap_judgments.jsonl', '').replace('_coverage_gap_judgments.jsonl', '')
        model_name = normalize_model_name(model_name)
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    question = extract_question(data)
                    is_correct = data.get('is_correct')
                    
                    if question and is_correct is not None:
                        parsed = data.get('parsed_judgment', {})
                        coverage = parsed.get('retrieval_coverage_gap', {})
                        has_gap = coverage.get('has_gap', False)
                        
                        iterative_results[model_name][question] = {
                            'is_correct': is_correct,
                            'has_gap': has_gap
                        }
                
                except json.JSONDecodeError:
                    continue
    
    return iterative_results


def calculate_reasoning_improvement(gold_results, iterative_results):
    """Calculate pure reasoning improvement for each model."""
    
    results = []
    
    for model in sorted(set(gold_results.keys()) & set(iterative_results.keys())):
        gold = gold_results[model]
        iterative = iterative_results[model]
        
        # Find common questions
        common_questions = set(gold.keys()) & set(iterative.keys())
        
        if not common_questions:
            continue
        
        # Count different categories
        total_questions = len(common_questions)
        
        # Questions correct in iterative but wrong in gold
        iter_correct_gold_wrong = []
        for q in common_questions:
            if iterative[q]['is_correct'] and not gold[q]:
                iter_correct_gold_wrong.append(q)
        
        # Among those, filter out ones with coverage gaps
        iter_correct_gold_wrong_no_gap = []
        for q in iter_correct_gold_wrong:
            if not iterative[q]['has_gap']:
                iter_correct_gold_wrong_no_gap.append(q)
        
        # Also count the ones WITH gaps for comparison
        iter_correct_gold_wrong_with_gap = []
        for q in iter_correct_gold_wrong:
            if iterative[q]['has_gap']:
                iter_correct_gold_wrong_with_gap.append(q)
        
        # Calculate percentages
        n_iter_correct_gold_wrong = len(iter_correct_gold_wrong)
        n_no_gap = len(iter_correct_gold_wrong_no_gap)
        n_with_gap = len(iter_correct_gold_wrong_with_gap)
        
        pct_improvement_total = 100 * n_iter_correct_gold_wrong / total_questions if total_questions > 0 else 0
        pct_improvement_no_gap = 100 * n_no_gap / total_questions if total_questions > 0 else 0
        pct_improvement_with_gap = 100 * n_with_gap / total_questions if total_questions > 0 else 0
        
        results.append({
            'Model': model,
            'Total Questions': total_questions,
            'Iter Correct, Gold Wrong (Total)': n_iter_correct_gold_wrong,
            'Iter Correct, Gold Wrong (No Gap)': n_no_gap,
            'Iter Correct, Gold Wrong (With Gap)': n_with_gap,
            'Pure Reasoning Improvement (%)': round(pct_improvement_no_gap, 2),
            'Retrieval-Aided Improvement (%)': round(pct_improvement_with_gap, 2),
            'Total Improvement (%)': round(pct_improvement_total, 2)
        })
    
    return results


def main():
    # Setup paths
    base_dir = Path(__file__).resolve().parents[2]
    output_dir = base_dir / "rag_analysis" / "output"
    
    print("Loading gold context results...")
    gold_results = load_gold_context_results(base_dir)
    print(f"Loaded gold context results for {len(gold_results)} models")
    
    print("\nLoading iterative RAG results with coverage gap info...")
    iterative_results = load_iterative_rag_results_with_coverage(output_dir)
    print(f"Loaded iterative RAG results for {len(iterative_results)} models")
    
    print("\nCalculating reasoning improvement (coverage gap filtered)...")
    results = calculate_reasoning_improvement(gold_results, iterative_results)
    
    if not results:
        print("No results to display!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by pure reasoning improvement
    df = df.sort_values('Pure Reasoning Improvement (%)', ascending=False)
    
    print("\n" + "="*100)
    print("ITERATIVE RAG REASONING IMPROVEMENT ANALYSIS")
    print("="*100)
    print("\nQuestions Correct in Iterative RAG but Wrong in Gold Context")
    print("(Filtered by Coverage Gap Status)")
    print("-"*100)
    print(df.to_string(index=False))
    
    # Summary statistics
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    
    print(f"\nAverage Pure Reasoning Improvement: {df['Pure Reasoning Improvement (%)'].mean():.2f}%")
    print(f"Average Retrieval-Aided Improvement: {df['Retrieval-Aided Improvement (%)'].mean():.2f}%")
    print(f"Average Total Improvement: {df['Total Improvement (%)'].mean():.2f}%")
    
    print(f"\nBest Pure Reasoning: {df.iloc[0]['Model']} ({df.iloc[0]['Pure Reasoning Improvement (%)']}%)")
    print(f"Worst Pure Reasoning: {df.iloc[-1]['Model']} ({df.iloc[-1]['Pure Reasoning Improvement (%)']}%)")
    
    # Calculate what % of iterative improvements are due to reasoning vs retrieval
    total_no_gap = df['Iter Correct, Gold Wrong (No Gap)'].sum()
    total_with_gap = df['Iter Correct, Gold Wrong (With Gap)'].sum()
    total_improvements = total_no_gap + total_with_gap
    
    if total_improvements > 0:
        pct_reasoning = 100 * total_no_gap / total_improvements
        pct_retrieval = 100 * total_with_gap / total_improvements
        
        print(f"\n" + "-"*100)
        print("IMPROVEMENT ATTRIBUTION (Across All Models)")
        print("-"*100)
        print(f"Total Iterative Improvements over Gold Context: {total_improvements}")
        print(f"  - Due to Pure Reasoning (No Gap): {total_no_gap} ({pct_reasoning:.1f}%)")
        print(f"  - Due to Better Retrieval (With Gap): {total_with_gap} ({pct_retrieval:.1f}%)")
    
    # Save to CSV
    output_path = base_dir / "rag_analysis" / "cov_rag_plots" / "iterative_reasoning_improvement.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")
    
    # Create a simpler table for the paper/presentation
    print("\n" + "="*100)
    print("SIMPLIFIED TABLE (For Paper/Presentation)")
    print("="*100)
    
    simple_df = df[['Model', 'Total Questions', 
                    'Iter Correct, Gold Wrong (No Gap)', 
                    'Pure Reasoning Improvement (%)']].copy()
    simple_df.columns = ['Model', 'Total Questions', 'Pure Reasoning Wins', 'Improvement (%)']
    
    print(simple_df.to_string(index=False))
    
    print("\n" + "="*100)


if __name__ == "__main__":
    main()
