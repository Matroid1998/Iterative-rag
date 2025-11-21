#!/usr/bin/env python3
"""
Calculate failure mode analysis tables for all models.

For each model m and failure f ∈ {Coverage-Gap, Composition-Failure, Overconfident, Distractor-Latch}:

1. Prevalence: p_{m,f} = (#runs with f) / (#all runs for m)
2. Impact (pp drop): Δ_{m,f} = Acc_m(¬f) - Acc_m(f)
3. Damage index: d_{m,f} = p_{m,f} × Δ_{m,f}

Interpretable as: expected pp of accuracy lost per question due to f for model m.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, List, Any
import pandas as pd


def get_base_path() -> Path:
    """Get the base path for the project."""
    return Path(__file__).resolve().parents[2]


def normalize_model_name(model_key: str) -> str:
    """Normalize model name for display."""
    model_names = {
        "bedrock_mistral.mistral-large-2402-v1:0": "Mistral Large",
        "bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning": "Claude 3.7 + Reasoning",
        "bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0": "Claude 3.7 Sonnet",
        "bedrock_us.deepseek.r1-v1:0-reasoning": "DeepSeek R1",
        "bedrock_us.meta.llama3-3-70b-instruct-v1:0": "Llama 3.3 70B",
        "openai_gpt-4o": "GPT-4o",
        "openai_gpt-5": "GPT-5",
        "openrouter_anthropic__claude-sonnet-4.5": "Claude Sonnet 4.5",
        "openrouter_google__gemini-2.5-pro": "Gemini 2.5 Pro",
        "openrouter_x-ai__grok-4-fast": "Grok 4 Fast",
        "openrouter_z-ai__glm-4.6": "GLM 4.6",
    }
    
    for key, name in model_names.items():
        if key in model_key:
            return name
    
    return model_key


def load_model_data(base_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load all necessary data for each model.
    
    Returns:
        Dict[model_name -> {
            'questions': {question -> {
                'is_correct': bool,
                'has_coverage_gap': bool,
                'has_composition_failure': bool,
                'is_overconfident': bool,
                'has_distractor_latch': bool,
            }}
        }]
    """
    hallucination_dir = base_path / "src" / "rag_analysis" / "output"
    coverage_gap_dir = base_path / "src" / "rag_analysis" / "output"
    reverified_dir = base_path / "src" / "responses_reverified"
    
    model_data = {}
    
    # Process each model
    for hallucination_file in sorted(hallucination_dir.glob("*hallucination_judgment.jsonl")):
        if 'backup' in hallucination_file.name:
            continue
        
        # Extract model key
        stem = hallucination_file.stem
        if stem.endswith("_hallucination_judgment"):
            stem = stem[:-len("_hallucination_judgment")]
        if stem.startswith("2_"):
            stem = stem[2:]
        if stem.startswith("responses_"):
            stem = stem[len("responses_"):]
        if stem.endswith("_reverified"):
            stem = stem[:-len("_reverified")]
        
        model_key = stem
        model_name = normalize_model_name(model_key)
        
        print(f"Processing {model_name}...")
        
        # Initialize model data
        questions = {}
        
        # Load hallucination judgments (for overconfident and composition failure)
        with open(hallucination_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    question = data.get('question', '')
                    if not question:
                        continue
                    
                    parsed = data.get('parsed_judgment', {})
                    
                    # Overconfident
                    cm = parsed.get('confidence_miscalibration', {})
                    is_overconfident = cm.get('direction', '') == 'overconfident_finalize'
                    
                    # Composition Failure (use the explicit composition_failure field)
                    composition = parsed.get('composition_and_faithfulness', {})
                    has_composition_failure = composition.get('composition_failure', False)
                    
                    if question not in questions:
                        questions[question] = {}
                    
                    questions[question]['is_overconfident'] = is_overconfident
                    questions[question]['has_composition_failure'] = has_composition_failure
                    
                except json.JSONDecodeError:
                    continue
        
        # Load coverage gap data
        coverage_file = None
        for cov_file in coverage_gap_dir.glob("*coverage_gap_judgments.jsonl"):
            if model_key in cov_file.name:
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
                        if not question:
                            continue
                        
                        parsed = data.get('parsed_judgment', {})
                        retrieval_coverage = parsed.get('retrieval_coverage_gap', {})
                        has_gap = retrieval_coverage.get('has_gap', False)
                        
                        # Get is_correct from coverage file
                        is_correct = data.get('is_correct', False)
                        
                        if question not in questions:
                            questions[question] = {}
                        
                        questions[question]['has_coverage_gap'] = has_gap
                        questions[question]['is_correct'] = is_correct
                        
                    except json.JSONDecodeError:
                        continue
        
        # Load distractor latch data (from quality judgments)
        quality_file = None
        for qual_file in hallucination_dir.glob("*quality_judement.jsonl"):
            if model_key in qual_file.name:
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
                        if not question:
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
    
    prevalence_data = []
    impact_data = []
    damage_data = []
    
    for model_name in sorted(model_data.keys()):
        questions = model_data[model_name]['questions']
        
        prevalence_row = {'Model': model_name}
        impact_row = {'Model': model_name}
        damage_row = {'Model': model_name}
        
        for failure_mode in failure_modes:
            failure_key = failure_keys[failure_mode]
            
            # Count questions with and without failure
            with_failure = []
            without_failure = []
            
            for question, data in questions.items():
                has_failure = data.get(failure_key, False)
                is_correct = data.get('is_correct', False)
                
                if has_failure:
                    with_failure.append(is_correct)
                else:
                    without_failure.append(is_correct)
            
            # Calculate prevalence
            total_questions = len(with_failure) + len(without_failure)
            if total_questions > 0:
                prevalence = len(with_failure) / total_questions * 100
            else:
                prevalence = 0.0
            
            # Calculate impact (pp drop)
            if len(with_failure) > 0 and len(without_failure) > 0:
                acc_with_failure = sum(with_failure) / len(with_failure) * 100
                acc_without_failure = sum(without_failure) / len(without_failure) * 100
                impact = acc_without_failure - acc_with_failure
            elif len(without_failure) > 0:
                # If no failures, impact is 0
                impact = 0.0
            else:
                # If all have failures, can't compute impact
                impact = float('nan')
            
            # Calculate damage index
            if not (prevalence == 0 or pd.isna(impact)):
                damage = (prevalence / 100) * impact
            else:
                damage = 0.0
            
            prevalence_row[failure_mode] = prevalence
            impact_row[failure_mode] = impact
            damage_row[failure_mode] = damage
        
        prevalence_data.append(prevalence_row)
        impact_data.append(impact_row)
        damage_data.append(damage_row)
    
    prevalence_df = pd.DataFrame(prevalence_data)
    impact_df = pd.DataFrame(impact_data)
    damage_df = pd.DataFrame(damage_data)
    
    return prevalence_df, impact_df, damage_df


def format_table(df: pd.DataFrame, metric_name: str, format_str: str = '.1f') -> str:
    """Format dataframe as a markdown table."""
    output = []
    output.append(f"\n## {metric_name}\n")
    
    # Create header
    cols = df.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    separator = "|" + "|".join([" --- " if col == "Model" else " ---: " for col in cols]) + "|"
    
    output.append(header)
    output.append(separator)
    
    # Add rows
    for _, row in df.iterrows():
        row_str = f"| {row['Model']} |"
        for col in cols[1:]:
            value = row[col]
            if pd.isna(value):
                row_str += " N/A |"
            else:
                row_str += f" {value:{format_str}} |"
        output.append(row_str)
    
    return "\n".join(output)


def main():
    """Main execution function."""
    base = get_base_path()
    output_dir = base / "src" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("FAILURE MODE ANALYSIS")
    print("="*80)
    print()
    
    # Load data
    print("Loading data for all models...")
    model_data = load_model_data(base)
    print(f"Loaded data for {len(model_data)} models\n")
    
    # Calculate metrics
    print("Calculating metrics...")
    prevalence_df, impact_df, damage_df = calculate_failure_metrics(model_data)
    
    # Save to CSV
    output_csv_dir = output_dir / "failure_mode_tables"
    output_csv_dir.mkdir(exist_ok=True)
    
    prevalence_df.to_csv(output_csv_dir / "prevalence.csv", index=False)
    impact_df.to_csv(output_csv_dir / "impact.csv", index=False)
    damage_df.to_csv(output_csv_dir / "damage_index.csv", index=False)
    
    print(f"✅ CSV files saved to {output_csv_dir}/")
    print()
    
    # Create markdown report
    report = []
    report.append("# Failure Mode Analysis Tables")
    report.append("\nGenerated: November 20, 2025")
    report.append("\n## Definitions")
    report.append("\nFor each model *m* and failure *f* ∈ {Coverage-Gap, Composition-Failure, Overconfident, Distractor-Latch}:")
    report.append("\n### 1. Prevalence")
    report.append("```")
    report.append("p_{m,f} = (#runs with f) / (#all runs for m)")
    report.append("```")
    report.append("(reported as %)")
    report.append("\n### 2. Impact (pp drop)")
    report.append("```")
    report.append("Δ_{m,f} = Acc_m(¬f) - Acc_m(f)")
    report.append("```")
    report.append("(average accuracy difference in percentage points)")
    report.append("\n### 3. Damage Index (expected loss)")
    report.append("```")
    report.append("d_{m,f} = p_{m,f} × Δ_{m,f}")
    report.append("```")
    report.append("Interpretable as: *expected pp of accuracy lost per question due to f for model m*")
    
    # Add tables
    report.append(format_table(prevalence_df, "Prevalence (%)", ".1f"))
    report.append(format_table(impact_df, "Impact (pp drop)", ".1f"))
    report.append(format_table(damage_df, "Damage Index (expected loss)", ".2f"))
    
    # Add summary statistics
    report.append("\n## Summary Statistics")
    report.append("\n### Average Across All Models")
    
    summary_data = []
    for col in prevalence_df.columns[1:]:
        summary_data.append({
            'Failure Mode': col,
            'Avg Prevalence (%)': prevalence_df[col].mean(),
            'Avg Impact (pp)': impact_df[col].mean(),
            'Avg Damage Index': damage_df[col].mean()
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    report.append("\n| Failure Mode | Avg Prevalence (%) | Avg Impact (pp) | Avg Damage Index |")
    report.append("| --- | ---: | ---: | ---: |")
    for _, row in summary_df.iterrows():
        report.append(f"| {row['Failure Mode']} | {row['Avg Prevalence (%)']:.1f} | {row['Avg Impact (pp)']:.1f} | {row['Avg Damage Index']:.2f} |")
    
    # Save markdown report
    report_path = output_dir / "failure_mode_analysis.md"
    with open(report_path, 'w') as f:
        f.write("\n".join(report))
    
    print(f"✅ Markdown report saved to {report_path}")
    print()
    
    # Print to console
    print("="*80)
    print("RESULTS")
    print("="*80)
    print(format_table(prevalence_df, "Prevalence (%)", ".1f"))
    print(format_table(impact_df, "Impact (pp drop)", ".1f"))
    print(format_table(damage_df, "Damage Index (expected loss)", ".2f"))
    print()
    print("="*80)
    print(f"📊 All results saved to {output_dir}/")
    print("="*80)


if __name__ == "__main__":
    main()
