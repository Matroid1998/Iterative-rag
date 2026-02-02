#!/usr/bin/env python3
"""
Procedural Compliance Metric Analysis

This script measures how well models adhere to the iterative RAG prompt when they
already know the answer (correct in no-context mode).

Three compliance categories:
1. Compliant (Verified): Retrieved evidence covers the reasoning path (no coverage gap)
2. Compliant (Attempted but Failed): Multiple steps (>1) but still has coverage gap
3. Non-Compliant (Lazy): Minimum steps (=1) with coverage gap

Only considers:
- Questions answered correctly in no-context mode
- Questions with >1 hop (2, 3, or 4 hops)
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def get_base_path() -> Path:
    """Get the base path for the project."""
    return Path(__file__).resolve().parents[4]


def load_question_hops():
    """
    Load question hop counts from chemrxiv_qa.json.
    Returns dict: {question_text: num_hops}
    """
    base = get_base_path()
    qa_file = base.parent / "data" / "corpus" / "chemrxiv_qa.json"
    
    question_hops = {}
    
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    for item in qa_data:
        if 'q' in item and 'path' in item:
            question = item['q']
            num_hops = len(item['path'])
            question_hops[question] = num_hops
    
    print(f"✅ Loaded {len(question_hops)} questions with hop counts")
    return question_hops


def load_no_context_correct_questions():
    """
    Load questions that were answered correctly in no-context mode.
    Returns dict: {model_name: set of correct questions}
    """
    base = get_base_path()
    no_context_dir = base / "src" / "response-jsonl-without-context"
    
    model_correct = defaultdict(set)
    
    # Map file patterns to model names (matching the 11 models with coverage gap files)
    file_to_model = {
        "bedrock_mistral.mistral-large-2402-v1:0_reverified": "Mistral Large",
        "claude-3-7-sonnet-20250219-v1:0-reasoning_reverified": "Claude 3.7 + Reasoning",
        "claude-3-7-sonnet-20250219-v1:0_reverified": "Claude 3.7 Sonnet",
        "deepseek.r1-v1:0-reasoning_reverified": "DeepSeek R1",
        "llama3-3-70b-instruct-v1:0_reverified": "Llama 3.3 70B",
        "gpt-4o_reverified": "GPT-4o",
        "gpt-5": "GPT-5",
        "claude_sonnet_4_5_reasoning": "Claude Sonnet 4.5",
        "gemini-2.5-pro-reasoning": "Gemini 2.5 Pro",
        "grok-4-fast-reasoning": "Grok 4 Fast",
        "glm-4.6-reasoning_reverified": "GLM 4.6",
    }
    
    for jsonl_file in no_context_dir.glob("*.jsonl"):
        # Find matching model name
        model_name = None
        for pattern, name in file_to_model.items():
            if pattern in jsonl_file.name:
                model_name = name
                break
        
        if not model_name:
            print(f"Warning: Could not match {jsonl_file.name} to a model")
            continue
        
        # Load responses
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    
                    # Check if correct
                    if data.get('is_correct', False):
                        # Question might be at top level or in 'raw' object
                        question = data.get('question', '')
                        if not question and 'raw' in data:
                            question = data['raw'].get('question', '')
                        if question:
                            model_correct[model_name].add(question)
        
        print(f"  {model_name}: {len(model_correct[model_name])} correct questions")
    
    return model_correct


def load_with_context_data():
    """
    Load with-context RAG data including coverage gaps and number of steps.
    Returns dict: {model_name: list of run data}
    """
    base = get_base_path()
    coverage_dir = base  / "data" / "results" / "failure_modes"
    reverified_dir = base / "src" / "responses_reverified"
    
    model_data = defaultdict(list)
    
    # Map file patterns to model names (exact matches from coverage_gap_judgments files)
    file_to_model = {
        "bedrock_mistral.mistral-large-2402-v1:0_reverified": "Mistral Large",
        "claude-3-7-sonnet-20250219-v1:0-reasoning_reverified": "Claude 3.7 + Reasoning",
        "claude-3-7-sonnet-20250219-v1:0_reverified": "Claude 3.7 Sonnet",
        "deepseek.r1-v1:0-reasoning_reverified": "DeepSeek R1",
        "llama3-3-70b-instruct-v1:0_reverified": "Llama 3.3 70B",
        "gpt-4o_reverified": "GPT-4o",
        "gpt-5_reverified": "GPT-5",
        "claude-sonnet-4.5_reverified": "Claude Sonnet 4.5",
        "gemini-2.5-pro_reverified": "Gemini 2.5 Pro",
        "grok-4-fast_reverified": "Grok 4 Fast",
        "glm-4.6_reverified": "GLM 4.6",
    }
    
    for coverage_file in coverage_dir.glob("*coverage_gap_judgment*.jsonl"):
        # Find matching model name
        model_name = None
        reverified_pattern = None
        for pattern, name in file_to_model.items():
            if pattern in coverage_file.name:
                model_name = name
                reverified_pattern = pattern
                break
        
        if not model_name:
            print(f"Warning: Could not match {coverage_file.name} to a model")
            continue
        
        # Map to actual reverified filenames
        reverified_filename_map = {
            "bedrock_mistral.mistral-large-2402-v1:0_reverified": "responses_bedrock_mistral.mistral-large-2402-v1:0_reverified.jsonl",
            "claude-3-7-sonnet-20250219-v1:0-reasoning_reverified": "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning_reverified.jsonl",
            "claude-3-7-sonnet-20250219-v1:0_reverified": "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0_reverified.jsonl",
            "deepseek.r1-v1:0-reasoning_reverified": "responses_bedrock_us.deepseek.r1-v1:0-reasoning_reverified.jsonl",
            "llama3-3-70b-instruct-v1:0_reverified": "responses_bedrock_us.meta.llama3-3-70b-instruct-v1:0_reverified.jsonl",
            "gpt-4o_reverified": "responses_openai_gpt-4o_reverified.jsonl",
            "gpt-5_reverified": "responses_openai_gpt-5_reverified.jsonl",
            "claude-sonnet-4.5_reverified": "responses_openrouter_anthropic_claude_sonnet_4_5_reasoning.jsonl",
            "gemini-2.5-pro_reverified": "responses_openrouter_google__gemini-2.5-pro_reverified.jsonl",
            "grok-4-fast_reverified": "responses_openrouter_x-ai__grok-4-fast_reverified.jsonl",
            "glm-4.6_reverified": "responses_openrouter_z-ai__glm-4.6_reverified.jsonl",
        }
        
        if not reverified_pattern:
            print(f"Warning: No reverified pattern for {model_name}")
            continue
        
        reverified_filename = reverified_filename_map.get(reverified_pattern)
        if not reverified_filename:
            print(f"Warning: No reverified filename mapping for {model_name}")
            continue
        
        reverified_file = reverified_dir / reverified_filename
        if not reverified_file.exists():
            print(f"Warning: Could not find reverified file {reverified_filename} for {model_name}")
            continue
        
        # Load reverified data to get num_steps (max source_step per question)
        question_steps = {}
        with open(reverified_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"Warning: JSON error in {reverified_filename} line {line_num}: {e}")
                        continue
                    
                    # Get question from raw_response
                    question = data.get('raw_response', {}).get('question', '')
                    if not question:
                        continue
                    
                    # Get max source_step from evidence
                    evidence = data.get('raw_response', {}).get('evidence', [])
                    max_step = 0
                    for ev in evidence:
                        step = ev.get('source_step', 0)
                        if step > max_step:
                            max_step = step
                    
                    question_steps[question] = max_step
        
        # Load coverage gap judgment data
        with open(coverage_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    
                    question = data.get('question', '')
                    is_correct = data.get('is_correct', False)
                    
                    # Extract has_gap from parsed_judgment
                    parsed = data.get('parsed_judgment', {})
                    coverage_gap_info = parsed.get('retrieval_coverage_gap', {})
                    has_gap = coverage_gap_info.get('has_gap', False)
                    
                    # Get num_steps from reverified data
                    num_steps = question_steps.get(question, 0)
                    
                    model_data[model_name].append({
                        'question': question,
                        'has_gap': has_gap,
                        'num_steps': num_steps,
                        'is_correct': is_correct
                    })
        
        print(f"  {model_name}: {len(model_data[model_name])} runs loaded")
    
    return model_data


def calculate_compliance(model_correct, model_rag_data, question_hops):
    """
    Calculate compliance metrics for each model.
    
    Returns DataFrame with columns:
    - Model
    - Verified (count and %)
    - Attempted (count and %)
    - Lazy (count and %)
    - Total Known Questions
    """
    results = []
    
    for model_name in sorted(model_correct.keys()):
        correct_questions = model_correct[model_name]
        rag_runs = model_rag_data.get(model_name, [])
        
        # Filter to multi-hop questions (>1 hop)
        known_multihop = set()
        for q in correct_questions:
            hops = question_hops.get(q, 0)
            if hops > 1:
                known_multihop.add(q)
        
        # Categorize each known multi-hop question
        verified = 0
        attempted = 0
        lazy = 0
        
        for run in rag_runs:
            question = run['question']
            
            # Only consider known multi-hop questions
            if question not in known_multihop:
                continue
            
            has_gap = run['has_gap']
            num_steps = run['num_steps']
            
            # Categorize
            if not has_gap:
                # Compliant (Verified): No coverage gap
                verified += 1
            elif num_steps > 1:
                # Compliant (Attempted but Failed): Multiple steps but still has gap
                attempted += 1
            else:
                # Non-Compliant (Lazy): Single step with gap
                lazy += 1
        
        total = verified + attempted + lazy
        
        if total > 0:
            verified_pct = (verified / total) * 100
            attempted_pct = (attempted / total) * 100
            lazy_pct = (lazy / total) * 100
        else:
            verified_pct = attempted_pct = lazy_pct = 0
        
        results.append({
            'Model': model_name,
            'Verified': verified,
            'Verified_pct': verified_pct,
            'Attempted': attempted,
            'Attempted_pct': attempted_pct,
            'Lazy': lazy,
            'Lazy_pct': lazy_pct,
            'Total': total,
            'Compliance_Rate': verified_pct + attempted_pct  # Both are compliant
        })
        
        print(f"\n{model_name}:")
        print(f"  Total known multi-hop questions: {total}")
        print(f"  Verified: {verified} ({verified_pct:.1f}%)")
        print(f"  Attempted: {attempted} ({attempted_pct:.1f}%)")
        print(f"  Lazy: {lazy} ({lazy_pct:.1f}%)")
        print(f"  Overall Compliance: {verified_pct + attempted_pct:.1f}%")
    
    return pd.DataFrame(results)


def create_compliance_plot(df: pd.DataFrame, output_dir: Path):
    """Create stacked bar chart showing procedural compliance."""
    
    # Sort by compliance rate (descending)
    df_sorted = df.sort_values('Compliance_Rate', ascending=False)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    models = df_sorted['Model'].tolist()
    verified = df_sorted['Verified_pct'].tolist()
    attempted = df_sorted['Attempted_pct'].tolist()
    lazy = df_sorted['Lazy_pct'].tolist()
    
    # Create stacked bars
    x = range(len(models))
    width = 0.7
    
    # Stack: Green (Verified) at bottom, Yellow (Attempted) in middle, Red (Lazy) on top
    bars1 = ax.bar(x, verified, width, label='Compliant (Verified)', 
                   color='#2ecc71', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, attempted, width, bottom=verified,
                   label='Compliant (Attempted but Failed)', 
                   color='#f39c12', edgecolor='black', linewidth=0.5)
    
    # Calculate bottom for lazy bars
    bottom_lazy = [v + a for v, a in zip(verified, attempted)]
    bars3 = ax.bar(x, lazy, width, bottom=bottom_lazy,
                   label='Non-Compliant (Lazy)', 
                   color='#e74c3c', edgecolor='black', linewidth=0.5)
    
    # Add percentage labels on each segment
    for i, (v, a, l) in enumerate(zip(verified, attempted, lazy)):
        # Verified label
        if v > 3:  # Only show if segment is large enough
            ax.text(i, v/2, f'{v:.1f}%', ha='center', va='center',
                   fontsize=9, fontweight='bold', color='white')
        
        # Attempted label
        if a > 3:
            ax.text(i, v + a/2, f'{a:.1f}%', ha='center', va='center',
                   fontsize=9, fontweight='bold', color='white')
        
        # Lazy label
        if l > 3:
            ax.text(i, v + a + l/2, f'{l:.1f}%', ha='center', va='center',
                   fontsize=9, fontweight='bold', color='white')
    
    # Styling
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('Percentage of Known Multi-Hop Questions (%)', fontsize=13, fontweight='bold')
    ax.set_title('Procedural Compliance: Do Models Follow Iterative RAG When They Know the Answer?',
                fontsize=15, fontweight='bold', pad=20)
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Legend - placed outside the plot area on the right
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / "procedural_compliance_stacked.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")
    
    plt.close()


def save_compliance_table(df: pd.DataFrame, output_dir: Path):
    """Save compliance data as CSV."""
    
    # Create output dataframe with all metrics
    output_df = df[[
        'Model', 'Total',
        'Verified', 'Verified_pct',
        'Attempted', 'Attempted_pct',
        'Lazy', 'Lazy_pct',
        'Compliance_Rate'
    ]].copy()
    
    output_df = output_df.sort_values('Compliance_Rate', ascending=False)
    
    # Save CSV
    csv_path = output_dir / "procedural_compliance_metrics.csv"
    output_df.to_csv(csv_path, index=False, float_format='%.2f')
    print(f"✅ Table saved to: {csv_path}")


def save_pcr_table(df: pd.DataFrame, output_dir: Path):
    """
    Save PCR (Procedural Compliance Rate) table.
    
    PCR = |Q_known ∩ (Verified ∪ Attempted)| / |Q_known|
        = 1 - |Lazy| / |Q_known|
    
    Where Q_known = questions answered correctly in no-context mode with >1 hops
    """
    
    # Create PCR table
    pcr_data = []
    
    for _, row in df.iterrows():
        model = row['Model']
        q_known = row['Total']  # Total known multi-hop questions
        verified = row['Verified']
        attempted = row['Attempted']
        lazy = row['Lazy']
        
        # Calculate PCR using the formula
        if q_known > 0:
            pcr = ((verified + attempted) / q_known) * 100  # As percentage
            # Alternatively: pcr = (1 - (lazy / q_known)) * 100
        else:
            pcr = 0.0
        
        pcr_data.append({
            'Model': model,
            '|Q_known|': q_known,
            '|Verified|': verified,
            '|Attempted|': attempted,
            '|Lazy|': lazy,
            'PCR (%)': pcr
        })
    
    pcr_df = pd.DataFrame(pcr_data)
    pcr_df = pcr_df.sort_values('PCR (%)', ascending=False)
    
    # Save CSV
    csv_path = output_dir / "pcr_procedural_compliance_rate.csv"
    pcr_df.to_csv(csv_path, index=False, float_format='%.2f')
    print(f"✅ PCR table saved to: {csv_path}")
    
    # Print summary table
    print("\n" + "=" * 90)
    print("PCR (PROCEDURAL COMPLIANCE RATE)")
    print("Formula: PCR = |Q_known ∩ (Verified ∪ Attempted)| / |Q_known| = 1 - |Lazy| / |Q_known|")
    print("=" * 90)
    print(f"{'Model':<30} {'|Q_known|':<12} {'|Verified|':<12} {'|Attempted|':<12} {'|Lazy|':<10} {'PCR (%)':<10}")
    print("-" * 90)
    
    for _, row in pcr_df.iterrows():
        print(f"{row['Model']:<30} {row['|Q_known|']:<12.0f} {row['|Verified|']:<12.0f} "
              f"{row['|Attempted|']:<12.0f} {row['|Lazy|']:<10.0f} {row['PCR (%)']:>7.2f}%")
    
    print("=" * 90)


def main():
    """Main execution."""
    base = get_base_path()
    output_dir = base / "data" / "plots" / "general"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("PROCEDURAL COMPLIANCE ANALYSIS")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading question hop counts...")
    question_hops = load_question_hops()
    
    print("\nLoading no-context correct answers...")
    model_correct = load_no_context_correct_questions()
    
    print("\nLoading with-context RAG data...")
    model_rag_data = load_with_context_data()
    
    # Calculate compliance
    print("\n" + "=" * 80)
    print("CALCULATING COMPLIANCE METRICS")
    print("=" * 80)
    df = calculate_compliance(model_correct, model_rag_data, question_hops)
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    create_compliance_plot(df, output_dir)
    save_compliance_table(df, output_dir)
    
    # Generate PCR table
    print("\n" + "=" * 80)
    print("GENERATING PCR TABLE")
    print("=" * 80)
    save_pcr_table(df, output_dir)
    
    # Print detailed summary
    print("\n" + "=" * 100)
    print("PROCEDURAL COMPLIANCE SUMMARY")
    print("=" * 100)
    print(f"{'Model':<30} {'Total':<8} {'Verified':<12} {'Attempted':<12} {'Lazy':<12} {'Compliance':<12}")
    print("-" * 100)
    
    df_sorted = df.sort_values('Compliance_Rate', ascending=False)
    for _, row in df_sorted.iterrows():
        print(f"{row['Model']:<30} {row['Total']:<8.0f} "
              f"{row['Verified']:>5.0f} ({row['Verified_pct']:>5.1f}%) "
              f"{row['Attempted']:>5.0f} ({row['Attempted_pct']:>5.1f}%) "
              f"{row['Lazy']:>5.0f} ({row['Lazy_pct']:>5.1f}%) "
              f"{row['Compliance_Rate']:>6.1f}%")
    
    print("=" * 100)
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
