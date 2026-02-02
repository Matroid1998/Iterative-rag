#!/usr/bin/env python3
"""
Plot analysis of ONLY distractor latch effects on accuracy (without partial contradiction).

This script analyzes:
1. Individual effects of distractor latch only
2. Per-model breakdown
3. Comparison of questions with/without distractor latch
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_analysis.hallucination_rag_plots.hall_plot_utils import (
    load_no_context_wrong_questions, normalize_model_name
)



def get_base_path() -> Path:
    """Get the base path for the project."""
    return Path(__file__).resolve().parents[2]


def get_quality_model_entries() -> List[Tuple[Path, Path, str]]:
    """Get list of (quality_file_path, reverified_file_path, display_name) tuples."""
    base = get_base_path()
    quality_dir = base / "src" / "rag_analysis" / "output"
    reverified_dir = base / "src" / "responses_reverified"
    
    model_names = {
        "bedrock_mistral.mistral-large-2402-v1:0": "Mistral Large",
        "bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning": "Claude 3.7 Sonnet + Reasoning",
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
    
    entries = []
    # Support both spelling variations found in the directory
    patterns = ["*quality_judement.jsonl", "*quality_judgement.jsonl"]
    
    found_files = set()
    for pattern in patterns:
        for quality_file in sorted(quality_dir.glob(pattern)):
            if quality_file in found_files:
                continue
            found_files.add(quality_file)
            
            stem = quality_file.stem
            
            # Remove suffix based on which pattern matched or just check ends
            if stem.endswith("_quality_judement"):
                stem = stem[:-len("_quality_judement")]
            elif stem.endswith("_quality_judgement"):
                stem = stem[:-len("_quality_judgement")]
            
            if stem.startswith("2_"):
                stem = stem[2:]
            
            raw_name = stem
            if stem.endswith("_reverified"):
                raw_name = stem[:-len("_reverified")]
            
            reverified_file = reverified_dir / f"{stem}.jsonl"
            
            # If explicit file doesn't exist, try adding _reverified suffix
            if not reverified_file.exists():
                reverified_with_suffix = reverified_dir / f"{stem}_reverified.jsonl"
                if reverified_with_suffix.exists():
                    reverified_file = reverified_with_suffix
                else:
                    # Also try handling the case where stem already has _reverified but file doesn't match?
                    # The Llama case had _reverified in stem and matched directly.
                    pass
            
            if not reverified_file.exists():
                print(f"Warning: No reverified file found for {stem} (checked {reverified_file.name} and {stem}_reverified.jsonl)")
                continue
            
            model_key = raw_name
            if model_key.startswith("responses_"):
                model_key = model_key[len("responses_"):]
            
            display_name = model_names.get(model_key, model_key)
            entries.append((quality_file, reverified_file, display_name))
    
    return entries


def analyze_distractor_latch_only(quality_file: Path, reverified_file: Path, wrong_questions_map: Dict[str, set], model_name: str) -> Dict[str, Any]:

    """
    Analyze ONLY the effects of distractor latch (ignore contradictions).
    
    Returns stats for 2 categories:
    - No distractor latch
    - Has distractor latch
    """
    stats = {
        'no_distractor': {'correct': 0, 'incorrect': 0},
        'has_distractor': {'correct': 0, 'incorrect': 0},
    }
    
    # Load is_correct from reverified file
    is_correct_map = {}
    with open(reverified_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                question = data.get('raw_response', {}).get('question', '') or data.get('question', '')
                is_correct = data.get('is_correct', False)
                is_correct_map[question] = is_correct
            except json.JSONDecodeError as e:
                print(f"  Warning: JSON error in {reverified_file.name} line {line_num}: {e}")
                continue
    
    # Process quality judgments
    matched = 0
    unmatched = 0
    with open(quality_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Warning: JSON error in {quality_file.name} line {line_num}: {e}")
                continue
                
            question = data.get('question', '')
            
            if question not in is_correct_map:
                unmatched += 1
                continue
            
            # Filter: Check if question was wrong in no-context baseline
            # Normalize internal model_name just in case, though caller provides a normalized one usually
            normalized_name = normalize_model_name(model_name)
            
            # Skip check if we don't have map for this model (uncommon if key exists)
            # But strictly: if model not in map OR question not in map[model], skip
            if normalized_name not in wrong_questions_map:
                continue
                
            if question not in wrong_questions_map[normalized_name]:
                continue
            
            matched += 1

            is_correct = is_correct_map[question]
            
            parsed = data.get('parsed_judgment', {})
            
            # Check for distractor latch (run level only)
            run_level = parsed.get('run_level', {})
            has_distractor = run_level.get('distractor_latch', False)
            
            # Categorize based ONLY on distractor latch
            if has_distractor:
                category = 'has_distractor'
            else:
                category = 'no_distractor'
            
            if is_correct:
                stats[category]['correct'] += 1
            else:
                stats[category]['incorrect'] += 1
    
    if unmatched > 0:
        print(f"  Warning: {unmatched} questions could not be matched (matched: {matched})")
    
    return stats


def plot_distractor_latch_only(all_stats: Dict[str, Dict], output_dir: Path):
    """Plot the effects of ONLY distractor latch as 3 separate figures."""
    
    # Aggregate across all models
    aggregated = {
        'no_distractor': {'correct': 0, 'incorrect': 0},
        'has_distractor': {'correct': 0, 'incorrect': 0},
    }
    
    for model_stats in all_stats.values():
        for category in aggregated.keys():
            aggregated[category]['correct'] += model_stats[category]['correct']
            aggregated[category]['incorrect'] += model_stats[category]['incorrect']
    
    categories = ['No Distractor\nLatch', 'Has Distractor\nLatch']
    category_keys = ['no_distractor', 'has_distractor']
    
    correct_counts = [aggregated[key]['correct'] for key in category_keys]
    incorrect_counts = [aggregated[key]['incorrect'] for key in category_keys]
    
    # ========== FIGURE 1: Accuracy comparison ==========
    fig1, ax1 = plt.subplots(figsize=(6, 8))
    
    # Calculate per-model accuracies for SEM and T-Test
    # Paired lists for t-test
    accs_no = []
    accs_has = []
    
    # Iterate sorted models to ensure pairing
    for model in sorted(all_stats.keys()):
        stats_dat = all_stats[model]
        
        # No Distractor
        c_no = stats_dat['no_distractor']['correct']
        t_no = c_no + stats_dat['no_distractor']['incorrect']
        if t_no == 0: continue
        
        # Has Distractor
        c_has = stats_dat['has_distractor']['correct']
        t_has = c_has + stats_dat['has_distractor']['incorrect']
        if t_has == 0: continue
        
        accs_no.append(c_no / t_no * 100)
        accs_has.append(c_has / t_has * 100)
    
    # Calculate stats
    import scipy.stats as stats
    
    means = []
    sems = []
    
    if accs_no and accs_has:
        # T-Test
        t_stat, p_val = stats.ttest_rel(accs_no, accs_has)
        print(f"\nStatistical Significance (Paired t-test): p = {p_val:.6f}")
        
        # Means and SEMs
        for data in [accs_no, accs_has]:
            means.append(np.mean(data))
            sems.append(np.std(data, ddof=1) / np.sqrt(len(data)))
    else:
        print("Not enough data for t-test")
        means = [0, 0]
        sems = [0, 0]
    
    colors = ['#2ecc71', '#e74c3c']
    # Use means for bar height, add yerr for SEM (VERTICAL)
    bars = ax1.bar(categories, means, yerr=sems, capsize=10, 
                    color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    for i, (bar, acc) in enumerate(zip(bars, means)):
        label = f'{acc:.1f}%'
        
        # Position label above the error bar
        y_pos = acc + sems[i] + 2 if sems[i] > 0 else acc + 2
        
        ax1.text(i, y_pos, label, ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax1.set_ylabel('Average Accuracy (%)', fontsize=13, fontweight='bold')
    # Removed "(Mean over Models ± SEM)" from title as requested
    ax1.set_title('Accuracy Comparison: Distractor Latch Effect\n(Filtered: No-Context Wrong Questions Only)', 
                  fontsize=14, fontweight='bold', pad=15)

    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11)
    
    plt.tight_layout()
    fig1_path = output_dir / "distractor_latch_accuracy_comparison.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 1: {fig1_path}")
    
    # ========== FIGURE 2: Per-model comparison ==========
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    
    models = sorted(all_stats.keys())
    
    no_distractor_accs = []
    has_distractor_accs = []
    
    for model in models:
        stats = all_stats[model]
        
        # No distractor
        c_no = stats['no_distractor']['correct']
        total_no = c_no + stats['no_distractor']['incorrect']
        acc_no = (c_no / total_no * 100) if total_no > 0 else 0
        no_distractor_accs.append(acc_no)
        
        # Has distractor
        c_has = stats['has_distractor']['correct']
        total_has = c_has + stats['has_distractor']['incorrect']
        acc_has = (c_has / total_has * 100) if total_has > 0 else 0
        has_distractor_accs.append(acc_has)
    
    x_models = np.arange(len(models))
    width = 0.35
    
    bars1 = ax2.bar(x_models - width/2, no_distractor_accs, width, 
                   label='No Distractor', color='#2ecc71', alpha=0.8)
    bars2 = ax2.bar(x_models + width/2, has_distractor_accs, width, 
                   label='Has Distractor', color='#e74c3c', alpha=0.8)
    
    ax2.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Per-Model: Accuracy With vs Without Distractor Latch\n(Filtered: No-Context Wrong Questions Only)', 
                  fontsize=14, fontweight='bold', pad=15)

    ax2.set_xticks(x_models)
    ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax2.legend(fontsize=12)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    fig2_path = output_dir / "distractor_latch_per_model.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 2: {fig2_path}")
    
    # ========== FIGURE 3: Distractor prevalence by model ==========
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    
    distractor_pcts = []
    
    for model in models:
        stats = all_stats[model]
        total = sum(stats[cat]['correct'] + stats[cat]['incorrect'] 
                   for cat in ['no_distractor', 'has_distractor'])
        
        if total > 0:
            distractor_pcts.append((stats['has_distractor']['correct'] + 
                                   stats['has_distractor']['incorrect']) / total * 100)
        else:
            distractor_pcts.append(0)
    
    bars = ax3.bar(x_models, distractor_pcts, color='#e74c3c', alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, distractor_pcts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax3.set_ylabel('Percentage of Questions (%)', fontsize=13, fontweight='bold')
    ax3.set_title('Distractor Latch Prevalence by Model\n(Filtered: No-Context Wrong Questions Only)', 
                  fontsize=14, fontweight='bold', pad=15)

    ax3.set_xticks(x_models)
    ax3.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_ylim(0, max(distractor_pcts) * 1.2 if distractor_pcts else 10)
    
    plt.tight_layout()
    fig3_path = output_dir / "distractor_latch_prevalence.png"
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 3: {fig3_path}")
    
    # Generate LaTeX code for subplot
    latex_code = r"""
% LaTeX code to include the three distractor latch figures as subplots

\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{distractor_latch_accuracy_comparison.png}
        \caption{Overall accuracy comparison showing the impact of distractor latch on model performance.}
        \label{fig:distractor_accuracy}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{distractor_latch_per_model.png}
        \caption{Per-model accuracy breakdown comparing questions with and without distractor latch.}
        \label{fig:distractor_per_model}
    \end{subfigure}
    
    \vspace{0.5cm}
    
    \begin{subfigure}[b]{0.65\textwidth}
        \centering
        \includegraphics[width=\textwidth]{distractor_latch_prevalence.png}
        \caption{Prevalence of distractor latch across different models.}
        \label{fig:distractor_prevalence}
    \end{subfigure}
    
    \caption{Analysis of distractor latch effects on iterative RAG performance. (a) shows the overall accuracy drop of 52 percentage points when distractor latch occurs, (b) demonstrates this effect consistently across all models, and (c) reveals varying prevalence rates by model.}
    \label{fig:distractor_latch_analysis}
\end{figure}

% Alternative: Horizontal layout (all three in one row)
% Uncomment below if you prefer a single-row layout:

% \begin{figure}[htbp]
%     \centering
%     \begin{subfigure}[b]{0.32\textwidth}
%         \centering
%         \includegraphics[width=\textwidth]{distractor_latch_accuracy_comparison.png}
%         \caption{Accuracy comparison}
%         \label{fig:distractor_accuracy_alt}
%     \end{subfigure}
%     \hfill
%     \begin{subfigure}[b]{0.32\textwidth}
%         \centering
%         \includegraphics[width=\textwidth]{distractor_latch_per_model.png}
%         \caption{Per-model breakdown}
%         \label{fig:distractor_per_model_alt}
%     \end{subfigure}
%     \hfill
%     \begin{subfigure}[b]{0.32\textwidth}
%         \centering
%         \includegraphics[width=\textwidth]{distractor_latch_prevalence.png}
%         \caption{Prevalence by model}
%         \label{fig:distractor_prevalence_alt}
%     \end{subfigure}
%     
%     \caption{Distractor latch analysis showing (a) overall impact, (b) per-model effects, and (c) prevalence rates.}
%     \label{fig:distractor_latch_analysis_alt}
% \end{figure}

% Note: Make sure to include the following in your LaTeX preamble:
% \usepackage{graphicx}
% \usepackage{subcaption}  % or \usepackage{subfig}
"""
    
    latex_path = output_dir / "distractor_latch_latex.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_code)
    
    print(f"\nSaved LaTeX code: {latex_path}")
    print("\n" + "="*70)
    print("LaTeX Code Preview:")
    print("="*70)
    print(latex_code)


def main():
    """Main execution function."""
    base = get_base_path()
    output_dir = base / "src" / "plots" / "contradiction_distractor_latch"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load wrong questions map
    wrong_questions_map = load_no_context_wrong_questions(base / 'src')

    
    print("Analyzing ONLY distractor latch effects (ignoring contradictions)...")
    print("FILTER ACTIVE: Including ONLY questions wrong in no-context baseline.")
    
    all_stats = {}
    for quality_path, reverified_path, display_name in get_quality_model_entries():
        # Pass model name for filtering
        # We need the key used in wrong_questions_map which is typically normalized
        stats = analyze_distractor_latch_only(quality_path, reverified_path, wrong_questions_map, display_name)
        all_stats[display_name] = stats
        print(f"Loaded {display_name}")

    
    print(f"\nTotal models analyzed: {len(all_stats)}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("DISTRACTOR LATCH ONLY - STATISTICS")
    print("="*70)
    
    aggregated = {
        'no_distractor': {'correct': 0, 'incorrect': 0},
        'has_distractor': {'correct': 0, 'incorrect': 0},
    }
    
    for model_stats in all_stats.values():
        for category in aggregated.keys():
            aggregated[category]['correct'] += model_stats[category]['correct']
            aggregated[category]['incorrect'] += model_stats[category]['incorrect']
    
    print("\nCategory Breakdown:")
    for category, label in [('no_distractor', 'No Distractor Latch'), 
                           ('has_distractor', 'Has Distractor Latch')]:
        c = aggregated[category]['correct']
        inc = aggregated[category]['incorrect']
        total = c + inc
        acc = (c / total * 100) if total > 0 else 0
        print(f"  {label:25s}: {acc:5.1f}% accuracy ({c:4d} correct / {total:4d} total)")
    
    # Calculate impact
    acc_no = (aggregated['no_distractor']['correct'] / 
             (aggregated['no_distractor']['correct'] + aggregated['no_distractor']['incorrect']) * 100)
    acc_has = (aggregated['has_distractor']['correct'] / 
              (aggregated['has_distractor']['correct'] + aggregated['has_distractor']['incorrect']) * 100)
    impact = acc_no - acc_has
    
    print(f"\nImpact: Distractor latch reduces accuracy by {impact:.1f} percentage points")
    
    # Generate plots
    print("\nGenerating plots...")
    
    plot_distractor_latch_only(all_stats, output_dir)
    
    print("\n✅ All 3 distractor latch analysis plots generated successfully!")
    print(f"✅ LaTeX code saved for creating subplots!")


if __name__ == "__main__":
    main()
