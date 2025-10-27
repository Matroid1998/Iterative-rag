"""
Analysis: Coverage Gap Prevalence by Model

Calculates the percentage of questions that have coverage gaps for each model
in the iterative RAG setup.
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


def calculate_coverage_gap_prevalence(output_dir):
    """Calculate coverage gap statistics for each model."""
    
    results = []
    
    for file_path in glob.glob(str(output_dir / '*coverage_gap_judgments.jsonl')):
        filename = Path(file_path).name
        model_name = filename.replace('responses_', '').replace('_reverified_coverage_gap_judgments.jsonl', '').replace('_coverage_gap_judgments.jsonl', '')
        model_name = normalize_model_name(model_name)
        
        total_questions = 0
        questions_with_gap = 0
        questions_without_gap = 0
        
        # Also track by correctness
        correct_with_gap = 0
        correct_without_gap = 0
        incorrect_with_gap = 0
        incorrect_without_gap = 0
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    is_correct = data.get('is_correct')
                    
                    if is_correct is None:
                        continue
                    
                    parsed = data.get('parsed_judgment', {})
                    coverage = parsed.get('retrieval_coverage_gap', {})
                    has_gap = coverage.get('has_gap', False)
                    
                    total_questions += 1
                    
                    if has_gap:
                        questions_with_gap += 1
                        if is_correct:
                            correct_with_gap += 1
                        else:
                            incorrect_with_gap += 1
                    else:
                        questions_without_gap += 1
                        if is_correct:
                            correct_without_gap += 1
                        else:
                            incorrect_without_gap += 1
                
                except json.JSONDecodeError:
                    continue
        
        if total_questions > 0:
            gap_percentage = 100 * questions_with_gap / total_questions
            no_gap_percentage = 100 * questions_without_gap / total_questions
            
            # Accuracy calculations
            acc_with_gap = 100 * correct_with_gap / questions_with_gap if questions_with_gap > 0 else 0
            acc_without_gap = 100 * correct_without_gap / questions_without_gap if questions_without_gap > 0 else 0
            
            results.append({
                'Model': model_name,
                'Total Questions': total_questions,
                'With Coverage Gap': questions_with_gap,
                'Without Coverage Gap': questions_without_gap,
                'Coverage Gap (%)': round(gap_percentage, 2),
                'No Gap (%)': round(no_gap_percentage, 2),
                'Accuracy With Gap (%)': round(acc_with_gap, 2),
                'Accuracy Without Gap (%)': round(acc_without_gap, 2),
                'Impact (pp)': round(acc_without_gap - acc_with_gap, 2)
            })
    
    return results


def main():
    # Setup paths
    base_dir = Path(__file__).resolve().parents[2]
    output_dir = base_dir / "rag_analysis" / "output"
    
    print("Calculating coverage gap prevalence by model...")
    results = calculate_coverage_gap_prevalence(output_dir)
    
    if not results:
        print("No data found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by coverage gap percentage (descending)
    df = df.sort_values('Coverage Gap (%)', ascending=False)
    
    print("\n" + "="*120)
    print("COVERAGE GAP PREVALENCE BY MODEL")
    print("="*120)
    print(df.to_string(index=False))
    
    # Summary statistics
    print("\n" + "="*120)
    print("SUMMARY STATISTICS")
    print("="*120)
    
    avg_gap_pct = df['Coverage Gap (%)'].mean()
    median_gap_pct = df['Coverage Gap (%)'].median()
    min_gap = df['Coverage Gap (%)'].min()
    max_gap = df['Coverage Gap (%)'].max()
    
    print(f"\nAverage Coverage Gap Rate: {avg_gap_pct:.2f}%")
    print(f"Median Coverage Gap Rate: {median_gap_pct:.2f}%")
    print(f"Range: {min_gap:.2f}% - {max_gap:.2f}%")
    
    model_with_min = df[df['Coverage Gap (%)'] == min_gap].iloc[0]['Model']
    model_with_max = df[df['Coverage Gap (%)'] == max_gap].iloc[0]['Model']
    
    print(f"\nLowest Gap Rate: {model_with_min} ({min_gap:.2f}%)")
    print(f"Highest Gap Rate: {model_with_max} ({max_gap:.2f}%)")
    
    # Average impact
    avg_impact = df['Impact (pp)'].mean()
    print(f"\nAverage Accuracy Impact of Coverage Gaps: {avg_impact:.2f} percentage points")
    
    # Total questions across all models
    total_all = df['Total Questions'].sum()
    total_with_gap = df['With Coverage Gap'].sum()
    total_without_gap = df['Without Coverage Gap'].sum()
    
    print(f"\n" + "-"*120)
    print("AGGREGATE ACROSS ALL MODELS")
    print("-"*120)
    print(f"Total Questions: {total_all:,}")
    print(f"With Coverage Gap: {total_with_gap:,} ({100*total_with_gap/total_all:.2f}%)")
    print(f"Without Coverage Gap: {total_without_gap:,} ({100*total_without_gap/total_all:.2f}%)")
    
    # Save to CSV
    output_path = base_dir / "rag_analysis" / "cov_rag_plots" / "coverage_gap_prevalence.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")
    
    # Create simplified table for presentation
    print("\n" + "="*120)
    print("SIMPLIFIED TABLE (For Paper/Presentation)")
    print("="*120)
    
    simple_df = df[['Model', 'Total Questions', 'Coverage Gap (%)', 
                    'Accuracy With Gap (%)', 'Accuracy Without Gap (%)', 'Impact (pp)']].copy()
    
    print(simple_df.to_string(index=False))
    
    print("\n" + "="*120)


if __name__ == "__main__":
    main()
