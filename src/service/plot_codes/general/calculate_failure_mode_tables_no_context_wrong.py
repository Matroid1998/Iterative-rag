#!/usr/bin/env python3
"""
Calculate failure mode analysis tables (Prevalence, Impact, Damage Index)
ONLY for questions that were answered incorrectly in no-context mode.

This shows how failure modes specifically affect questions that needed context.

Formulas:
- Prevalence: p_{m,f} = (# runs with failure f) / (# all runs for model m)
- Impact: Δ_{m,f} = Accuracy_m(¬f) - Accuracy_m(f)  [in percentage points]
- Damage Index: d_{m,f} = p_{m,f} × Δ_{m,f}
"""

import json
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd


def get_base_path() -> Path:
    """Get the base path for the project."""
    return Path(__file__).resolve().parents[4]


def load_no_context_incorrect_questions() -> Dict[str, set]:
    """
    Load questions that were answered incorrectly in no-context mode for each model.
    
    Returns:
        Dict mapping model_name -> set of incorrect questions
    """
    base = get_base_path()
    no_context_dir = base / "src" / "response-jsonl-without-context"
    
    # Map no-context file patterns to display names
    file_to_model = {
        "bedrock_mistral.mistral-large-2402-v1:0": "Mistral Large",
        "claude-3-7-sonnet-20250219-v1:0-reasoning": "Claude 3.7 + Reasoning",
        "claude-3-7-sonnet-20250219-v1:0_reverified": "Claude 3.7 Sonnet",
        "deepseek.r1-v1:0-reasoning": "DeepSeek R1",
        "llama3-3-70b-instruct": "Llama 3.3 70B",
        "gpt-4o_reverified": "GPT-4o",
        "openai_gpt-5": "GPT-5",
        "anthropic_claude_sonnet_4_5": "Claude Sonnet 4.5",
        "gemini-2.5-pro": "Gemini 2.5 Pro",
        "grok-4-fast": "Grok 4 Fast",
        "glm-4.6": "GLM 4.6",
    }
    
    no_context_incorrect = {}
    
    for file_path in no_context_dir.glob("*.jsonl"):
        # Find matching model name by checking if any pattern is in the filename
        display_name = None
        
        for pattern, name in file_to_model.items():
            if pattern in file_path.name:
                display_name = name
                break
        
        if not display_name:
            continue
        
        incorrect_questions = set()
        
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    question = data.get('raw', {}).get('question', '') or data.get('question', '')
                    is_correct = data.get('is_correct', False)
                    
                    if not is_correct and question:
                        incorrect_questions.add(question)
                        
                except json.JSONDecodeError:
                    continue
        
        if incorrect_questions:
            no_context_incorrect[display_name] = incorrect_questions
            print(f"Loaded {len(incorrect_questions)} incorrect questions from {display_name} (no-context)")
    
    return no_context_incorrect


def load_model_data(no_context_incorrect: Dict[str, set]) -> Dict[str, Dict]:
    """
    Load failure mode data for all models, filtered to only questions wrong in no-context mode.
    
    Returns:
        Dict with structure: {model_name: {'questions': {question: {failure_flags, is_correct}}}}
    """
    base = get_base_path()
    hallucination_dir = base  / "data" / "results" / "failure_modes"
    
    # Map with-context file patterns to display names (must match no-context mapping)
    file_patterns = {
        "Mistral Large": "bedrock_mistral.mistral-large-2402-v1:0",
        "Claude 3.7 + Reasoning": "claude-3-7-sonnet-20250219-v1:0-reasoning",
        "Claude 3.7 Sonnet": ["claude-3-7-sonnet-20250219-v1:0_reverified", "claude-3-7-sonnet-20250219-v1:0_"],
        "DeepSeek R1": "deepseek.r1-v1:0-reasoning",
        "Llama 3.3 70B": "llama3-3-70b-instruct",
        "GPT-4o": "gpt-4o_reverified",
        "GPT-5": "gpt-5_reverified",
        "Claude Sonnet 4.5": "claude-sonnet-4.5",
        "Gemini 2.5 Pro": "gemini-2.5-pro",
        "Grok 4 Fast": "grok-4-fast",
        "GLM 4.6": "glm-4.6",
    }
    
    model_data = {}
    
    for model_name in no_context_incorrect.keys():
        print(f"Processing {model_name}...")
        
        # Get the file pattern for this model
        pattern = file_patterns.get(model_name)
        if not pattern:
            print(f"Warning: No pattern mapping for {model_name}, skipping...")
            continue
        
        # Get set of questions that were wrong in no-context mode
        target_questions = no_context_incorrect[model_name]
        
        questions = {}
        
        # First load coverage gap data to get is_correct values
        coverage_file = None
        for cov_file in hallucination_dir.glob("*coverage_gap_judgments.jsonl"):
            # Handle both single pattern string and list of patterns
            patterns = [pattern] if isinstance(pattern, str) else pattern
            if any(p in cov_file.name for p in patterns):
                coverage_file = cov_file
                break
        
        if coverage_file:
            with open(coverage_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        question = data.get('question', '')
                        if not question or question not in target_questions:
                            continue
                        
                        parsed = data.get('parsed_judgment', {})
                        retrieval_coverage = parsed.get('retrieval_coverage_gap', {})
                        has_gap = retrieval_coverage.get('has_gap', False)
                        
                        # Get is_correct from coverage file (this is with-context correctness)
                        is_correct = data.get('is_correct', False)
                        
                        if question not in questions:
                            questions[question] = {}
                        
                        questions[question]['has_coverage_gap'] = has_gap
                        questions[question]['is_correct'] = is_correct
                        
                    except json.JSONDecodeError:
                        continue
        
        # Load hallucination judgments (composition failure and overconfident)
        hallucination_file = None
        for hall_file in hallucination_dir.glob("*hallucination_judgment.jsonl"):
            patterns = [pattern] if isinstance(pattern, str) else pattern
            if any(p in hall_file.name for p in patterns):
                hallucination_file = hall_file
                break
        
        if hallucination_file:
            with open(hallucination_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        question = data.get('question', '')
                        if not question or question not in target_questions:
                            continue
                        
                        parsed = data.get('parsed_judgment', {})
                        conf_misc = parsed.get('confidence_miscalibration', {})
                        comp_faith = parsed.get('composition_and_faithfulness', {})
                        
                        is_overconfident = conf_misc.get('direction', '') == 'overconfident_finalize'
                        has_composition_failure = comp_faith.get('composition_failure', False)
                        
                        if question not in questions:
                            questions[question] = {}
                        
                        questions[question]['is_overconfident'] = is_overconfident
                        questions[question]['has_composition_failure'] = has_composition_failure
                        
                    except json.JSONDecodeError:
                        continue
        
        # Load distractor latch data (from quality judgments)
        quality_file = None
        for qual_file in hallucination_dir.glob("*quality_judement.jsonl"):
            patterns = [pattern] if isinstance(pattern, str) else pattern
            if any(p in qual_file.name for p in patterns):
                quality_file = qual_file
                break
        
        if quality_file:
            with open(quality_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        question = data.get('question', '')
                        if not question or question not in target_questions:
                            continue
                        
                        parsed = data.get('parsed_judgment', {})
                        
                        # Check for distractor latch (run level only)
                        run_level = parsed.get('run_level', {})
                        has_distractor = run_level.get('distractor_latch', False)
                        
                        if question not in questions:
                            questions[question] = {}
                        
                        questions[question]['has_distractor_latch'] = has_distractor
                        
                    except json.JSONDecodeError:
                        continue
        
        # Only add model if we have data
        if questions:
            model_data[model_name] = {'questions': questions}
            print(f"  Loaded {len(questions)} questions (filtered from {len(target_questions)} no-context incorrect)")
    
    return model_data


def calculate_failure_metrics(model_data: Dict[str, Dict]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate prevalence, impact, and damage index for each model and failure mode.
    
    Returns:
        (prevalence_df, impact_df, damage_df)
    """
    failure_modes = [
        'Coverage Gap',
        'Composition Failure',
        'Overconfident',
        'Distractor Latch'
    ]
    
    failure_keys = {
        'Coverage Gap': 'has_coverage_gap',
        'Composition Failure': 'has_composition_failure',
        'Overconfident': 'is_overconfident',
        'Distractor Latch': 'has_distractor_latch'
    }
    
    results = []
    
    for model_name, data in model_data.items():
        questions = data['questions']
        total_questions = len(questions)
        
        if total_questions == 0:
            continue
        
        prevalence_row = {'Model': model_name}
        impact_row = {'Model': model_name}
        damage_row = {'Model': model_name}
        
        for failure_mode in failure_modes:
            failure_key = failure_keys[failure_mode]
            
            # Count questions with/without this failure
            with_failure = []
            without_failure = []
            
            for question, attrs in questions.items():
                has_failure = attrs.get(failure_key, False)
                is_correct = attrs.get('is_correct', False)
                
                if has_failure:
                    with_failure.append(is_correct)
                else:
                    without_failure.append(is_correct)
            
            # Calculate prevalence
            n_with_failure = len(with_failure)
            prevalence = (n_with_failure / total_questions * 100) if total_questions > 0 else 0
            
            # Calculate impact (accuracy drop in percentage points)
            if len(with_failure) > 0 and len(without_failure) > 0:
                acc_without = sum(without_failure) / len(without_failure) * 100
                acc_with = sum(with_failure) / len(with_failure) * 100
                impact = acc_without - acc_with  # Positive means failure hurts performance
            else:
                impact = 0
            
            # Calculate damage index
            damage = (prevalence / 100) * impact
            
            prevalence_row[failure_mode] = prevalence
            impact_row[failure_mode] = impact
            damage_row[failure_mode] = damage
        
        results.append({
            'prevalence': prevalence_row,
            'impact': impact_row,
            'damage': damage_row
        })
    
    # Create DataFrames
    prevalence_data = [r['prevalence'] for r in results]
    impact_data = [r['impact'] for r in results]
    damage_data = [r['damage'] for r in results]
    
    prevalence_df = pd.DataFrame(prevalence_data).set_index('Model')
    impact_df = pd.DataFrame(impact_data).set_index('Model')
    damage_df = pd.DataFrame(damage_data).set_index('Model')
    
    return prevalence_df, impact_df, damage_df


def format_table(df: pd.DataFrame, value_format: str = ".1f") -> str:
    """Format DataFrame as markdown table."""
    lines = []
    
    # Header
    cols = ['Model'] + list(df.columns)
    lines.append('| ' + ' | '.join(cols) + ' |')
    lines.append('| ' + ' | '.join(['---:'] * len(cols)) + ' |')
    
    # Rows
    for idx, row in df.iterrows():
        values = [str(idx)] + [f"{v:{value_format}}" for v in row]
        lines.append('| ' + ' | '.join(values) + ' |')
    
    return '\n'.join(lines)


def main():
    """Main execution."""
    base = get_base_path()
    output_dir = base / "data" / "plots" / "general" / "failure_mode_tables_no_context_wrong"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("FAILURE MODE ANALYSIS - NO-CONTEXT WRONG QUESTIONS ONLY")
    print("=" * 80)
    print()
    
    # Load questions that were wrong in no-context mode
    print("Loading no-context incorrect questions...")
    no_context_incorrect = load_no_context_incorrect_questions()
    print(f"Loaded no-context data for {len(no_context_incorrect)} models\n")
    
    # Load failure mode data for those questions
    print("Loading data for all models...")
    model_data = load_model_data(no_context_incorrect)
    print(f"Loaded data for {len(model_data)} models\n")
    
    # Calculate metrics
    print("Calculating metrics...")
    prevalence_df, impact_df, damage_df = calculate_failure_metrics(model_data)
    
    # Save CSV files
    prevalence_df.to_csv(output_dir / "prevalence.csv")
    impact_df.to_csv(output_dir / "impact.csv")
    damage_df.to_csv(output_dir / "damage_index.csv")
    
    print(f"✅ CSV files saved to {output_dir}/")
    
    # Create markdown report
    markdown = f"""# Failure Mode Analysis - No-Context Wrong Questions Only

**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

**Scope**: This analysis includes ONLY questions that were answered incorrectly in no-context mode.
This shows how failure modes specifically affect questions that required context to answer correctly.

## Formulas

- **Prevalence**: p_{{m,f}} = (# runs with failure f) / (# all no-context-wrong runs for model m) × 100%
- **Impact**: Δ_{{m,f}} = Accuracy_m(¬f) - Accuracy_m(f) [in percentage points]
- **Damage Index**: d_{{m,f}} = p_{{m,f}} × Δ_{{m,f}}

## Tables

### Prevalence (%)

{format_table(prevalence_df, '.1f')}

### Impact (pp drop)

{format_table(impact_df, '.1f')}

### Damage Index (expected loss)

{format_table(damage_df, '.2f')}

## Summary Statistics

### Average Prevalence
"""
    
    for col in prevalence_df.columns:
        avg = prevalence_df[col].mean()
        markdown += f"- **{col}**: {avg:.1f}%\n"
    
    markdown += "\n### Average Impact\n"
    for col in impact_df.columns:
        avg = impact_df[col].mean()
        markdown += f"- **{col}**: {avg:.1f} pp\n"
    
    markdown += "\n### Average Damage Index\n"
    for col in damage_df.columns:
        avg = damage_df[col].mean()
        markdown += f"- **{col}**: {avg:.2f} pp\n"
    
    # Save markdown
    markdown_path = output_dir / "failure_mode_analysis_no_context_wrong.md"
    with open(markdown_path, 'w') as f:
        f.write(markdown)
    
    print(f"✅ Markdown report saved to {markdown_path}")
    
    # Display summary
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print("\n## Prevalence (%)\n")
    print(format_table(prevalence_df, '.1f'))
    
    print("\n## Impact (pp drop)\n")
    print(format_table(impact_df, '.1f'))
    
    print("\n## Damage Index (expected loss)\n")
    print(format_table(damage_df, '.2f'))
    
    print("\n" + "=" * 80)
    print(f"📊 All results saved to {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
