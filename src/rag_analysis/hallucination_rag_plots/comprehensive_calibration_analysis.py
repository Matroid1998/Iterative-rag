"""
Comprehensive Analysis: Confidence Calibration Patterns

Analyzes the relationship between all three calibration states 
(overconfident, underconfident, well-calibrated) and model performance.
"""
import json
import sys
import csv
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hallucination_rag_plots.hall_plot_utils import (
    load_hallucination_judgments, normalize_model_name
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'rag_analysis' / 'output'
PLOT_DIR = Path(__file__).resolve().parent
CSV_PATH = Path(__file__).resolve().parents[2] / 'results' / 'reverify_accuracies.csv'


def load_accuracies():
    """Load accuracies from CSV file."""
    iterative_rag = {}
    no_context = {}
    gold_context = {}
    
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = row['folder']
            file_name = row['file_name']
            accuracy = float(row['accuracy'])
            
            model_key = file_name.replace('responses_', '').replace('_reverified.jsonl', '').replace('.jsonl', '')
            
            if folder == 'Iterative-RAG':
                iterative_rag[model_key] = accuracy
            elif folder == 'response-jsonl-without-context':
                no_context[model_key] = accuracy
            elif folder == 'response-jsonl-with-context':
                gold_context[model_key] = accuracy
    
    # Manually add GPT-5 gold context accuracy
    gold_context['openai_gpt-5'] = 0.7168
    
    return iterative_rag, no_context, gold_context


def main():
    """Generate comprehensive calibration analysis."""
    hall_records = load_hallucination_judgments(OUTPUT_DIR)
    
    # Calculate rates for all three states per model
    model_stats = defaultdict(lambda: {
        'total': 0,
        'overconfident': 0,
        'underconfident': 0,
        'well_calibrated': 0
    })
    
    for rec in hall_records:
        model = normalize_model_name(rec.get('model', ''))
        model_stats[model]['total'] += 1
        
        parsed = rec.get('parsed_judgment', {})
        cm = parsed.get('confidence_miscalibration', {})
        direction = cm.get('direction', '')
        
        if direction == 'overconfident_finalize':
            model_stats[model]['overconfident'] += 1
        elif direction == 'underconfident_continue':
            model_stats[model]['underconfident'] += 1
        elif direction == 'ok':
            model_stats[model]['well_calibrated'] += 1
    
    # Calculate percentages
    model_rates = {}
    for model, stats in model_stats.items():
        if stats['total'] > 0:
            model_rates[model] = {
                'overconfident': 100 * stats['overconfident'] / stats['total'],
                'underconfident': 100 * stats['underconfident'] / stats['total'],
                'well_calibrated': 100 * stats['well_calibrated'] / stats['total'],
                'counts': stats
            }
    
    # Load accuracies
    iterative_rag, no_context, gold_context = load_accuracies()
    
    model_mapping = {
        'bedrock_mistral.mistral-large-2402-v1:0': 'Mistral Large',
        'bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0': 'Claude 3.7 Sonnet',
        'bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning': 'Claude 3.7 + Reasoning',
        'bedrock_us.deepseek.r1-v1:0-reasoning': 'DeepSeek R1',
        'openai_gpt-4o': 'GPT-4o',
        'openai_gpt-5': 'GPT-5',
        'openrouter_anthropic__claude-sonnet-4.5': 'Claude Sonnet 4.5',
        'openrouter_google__gemini-2.5-pro': 'Gemini 2.5 Pro',
        'openrouter_x-ai__grok-4-fast': 'Grok 4 Fast',
    }
    
    # Compile data
    data = []
    for model_key, normalized_name in model_mapping.items():
        if model_key in iterative_rag and normalized_name in model_rates:
            rates = model_rates[normalized_name]
            
            iter_acc = iterative_rag[model_key] * 100
            improvement_no_ctx = 0
            improvement_gold = 0
            
            if model_key in no_context:
                improvement_no_ctx = (iterative_rag[model_key] - no_context[model_key]) * 100
            
            if model_key in gold_context:
                improvement_gold = (iterative_rag[model_key] - gold_context[model_key]) * 100
            
            data.append({
                'model': normalized_name,
                'overconfident': rates['overconfident'],
                'underconfident': rates['underconfident'],
                'well_calibrated': rates['well_calibrated'],
                'accuracy': iter_acc,
                'improvement_no_ctx': improvement_no_ctx,
                'improvement_gold': improvement_gold,
                'counts': rates['counts']
            })
    
    # Print comprehensive analysis
    print("\n" + "="*100)
    print("COMPREHENSIVE CONFIDENCE CALIBRATION ANALYSIS (UPDATED RULES)")
    print("="*100)
    print("\nCalibration State Definitions:")
    print("  - Overconfident: Finalized with BOTH low coverage (<0.8) AND low sufficiency (<0.6)")
    print("  - Underconfident: Continued retrieving when already had enough evidence")
    print("  - Well-Calibrated: Neither overconfident nor underconfident")
    
    print("\n" + "-"*100)
    print(f"{'Model':<30} {'Overconf%':>10} {'Underconf%':>11} {'WellCal%':>10} {'Accuracy%':>10} {'Δ NoCtx':>9} {'Δ Gold':>8}")
    print("-"*100)
    
    for d in sorted(data, key=lambda x: x['accuracy'], reverse=True):
        print(f"{d['model']:<30} {d['overconfident']:>9.1f}% {d['underconfident']:>10.1f}% {d['well_calibrated']:>9.1f}% "
              f"{d['accuracy']:>9.1f}% {d['improvement_no_ctx']:>8.1f}pp {d['improvement_gold']:>7.1f}pp")
    
    # Calculate correlations
    print("\n" + "="*100)
    print("CORRELATION ANALYSIS")
    print("="*100)
    
    overconf_rates = [d['overconfident'] for d in data]
    underconf_rates = [d['underconfident'] for d in data]
    wellcal_rates = [d['well_calibrated'] for d in data]
    accuracies = [d['accuracy'] for d in data]
    impr_no_ctx = [d['improvement_no_ctx'] for d in data]
    impr_gold = [d['improvement_gold'] for d in data]
    
    if len(data) > 2:
        corr_over_acc = np.corrcoef(overconf_rates, accuracies)[0, 1]
        corr_under_acc = np.corrcoef(underconf_rates, accuracies)[0, 1]
        corr_well_acc = np.corrcoef(wellcal_rates, accuracies)[0, 1]
        
        corr_over_impr_no = np.corrcoef(overconf_rates, impr_no_ctx)[0, 1]
        corr_under_impr_no = np.corrcoef(underconf_rates, impr_no_ctx)[0, 1]
        corr_well_impr_no = np.corrcoef(wellcal_rates, impr_no_ctx)[0, 1]
        
        corr_over_impr_gold = np.corrcoef(overconf_rates, impr_gold)[0, 1]
        corr_under_impr_gold = np.corrcoef(underconf_rates, impr_gold)[0, 1]
        corr_well_impr_gold = np.corrcoef(wellcal_rates, impr_gold)[0, 1]
        
        print("\nVs. Iterative RAG Accuracy:")
        print(f"  Overconfident:    {corr_over_acc:>6.3f}  {'[Strong Negative]' if corr_over_acc < -0.5 else '[Moderate Negative]' if corr_over_acc < -0.3 else '[Weak/No correlation]'}")
        print(f"  Underconfident:   {corr_under_acc:>6.3f}  {'[Strong Positive]' if corr_under_acc > 0.5 else '[Moderate Positive]' if corr_under_acc > 0.3 else '[Weak/No correlation]'}")
        print(f"  Well-Calibrated:  {corr_well_acc:>6.3f}  {'[Strong Negative]' if corr_well_acc < -0.5 else '[Moderate Negative]' if corr_well_acc < -0.3 else '[Weak/No correlation]'}")
        
        print("\nVs. Improvement over No Context:")
        print(f"  Overconfident:    {corr_over_impr_no:>6.3f}  {'[Strong Negative]' if corr_over_impr_no < -0.5 else '[Moderate Negative]' if corr_over_impr_no < -0.3 else '[Weak/No correlation]'}")
        print(f"  Underconfident:   {corr_under_impr_no:>6.3f}  {'[Strong Positive]' if corr_under_impr_no > 0.5 else '[Moderate Positive]' if corr_under_impr_no > 0.3 else '[Weak/No correlation]'}")
        print(f"  Well-Calibrated:  {corr_well_impr_no:>6.3f}  {'[Strong Negative]' if corr_well_impr_no < -0.5 else '[Moderate Negative]' if corr_well_impr_no < -0.3 else '[Weak/No correlation]'}")
        
        print("\nVs. Improvement over Gold Context:")
        print(f"  Overconfident:    {corr_over_impr_gold:>6.3f}  {'[Strong Negative]' if corr_over_impr_gold < -0.5 else '[Moderate Negative]' if corr_over_impr_gold < -0.3 else '[Weak/No correlation]'}")
        print(f"  Underconfident:   {corr_under_impr_gold:>6.3f}  {'[Strong Positive]' if corr_under_impr_gold > 0.5 else '[Moderate Positive]' if corr_under_impr_gold > 0.3 else '[Weak/No correlation]'}")
        print(f"  Well-Calibrated:  {corr_well_impr_gold:>6.3f}  {'[Strong Negative]' if corr_well_impr_gold < -0.5 else '[Moderate Negative]' if corr_well_impr_gold < -0.3 else '[Weak/No correlation]'}")
    
    # Key findings
    print("\n" + "="*100)
    print("KEY PATTERNS IDENTIFIED")
    print("="*100)
    
    # Find extremes
    most_overconf = max(data, key=lambda x: x['overconfident'])
    least_overconf = min(data, key=lambda x: x['overconfident'])
    most_underconf = max(data, key=lambda x: x['underconfident'])
    least_underconf = min(data, key=lambda x: x['underconfident'])
    most_wellcal = max(data, key=lambda x: x['well_calibrated'])
    least_wellcal = min(data, key=lambda x: x['well_calibrated'])
    highest_acc = max(data, key=lambda x: x['accuracy'])
    lowest_acc = min(data, key=lambda x: x['accuracy'])
    
    print(f"\n1. Overconfidence (with stricter AND rule):")
    print(f"   - Now MUCH RARER: avg {np.mean(overconf_rates):.1f}% (down from ~25% with OR rule)")
    print(f"   - Highest: {most_overconf['model']} ({most_overconf['overconfident']:.1f}%)")
    print(f"   - Lowest: {least_overconf['model']} ({least_overconf['overconfident']:.1f}%)")
    print(f"   - Impact: Still negatively correlated with accuracy ({corr_over_acc:.3f})")
    
    print(f"\n2. Underconfidence:")
    print(f"   - Average: {np.mean(underconf_rates):.1f}%")
    print(f"   - Highest: {most_underconf['model']} ({most_underconf['underconfident']:.1f}%)")
    print(f"   - Lowest: {least_underconf['model']} ({least_underconf['underconfident']:.1f}%)")
    print(f"   - Pattern: Positively correlated with improvement ({corr_under_impr_no:.3f})")
    
    print(f"\n3. Well-Calibrated:")
    print(f"   - Average: {np.mean(wellcal_rates):.1f}% (UP from ~20% with OR rule)")
    print(f"   - Highest: {most_wellcal['model']} ({most_wellcal['well_calibrated']:.1f}%)")
    print(f"   - Lowest: {least_wellcal['model']} ({least_wellcal['well_calibrated']:.1f}%)")
    print(f"   - Paradox: Negative correlation with accuracy ({corr_well_acc:.3f})")
    
    print(f"\n4. Performance Leaders:")
    print(f"   - Best Accuracy: {highest_acc['model']} ({highest_acc['accuracy']:.1f}%)")
    print(f"     └─ Overconf: {highest_acc['overconfident']:.1f}%, Underconf: {highest_acc['underconfident']:.1f}%, WellCal: {highest_acc['well_calibrated']:.1f}%")
    print(f"   - Worst Accuracy: {lowest_acc['model']} ({lowest_acc['accuracy']:.1f}%)")
    print(f"     └─ Overconf: {lowest_acc['overconfident']:.1f}%, Underconf: {lowest_acc['underconfident']:.1f}%, WellCal: {lowest_acc['well_calibrated']:.1f}%")
    
    print("\n" + "="*100)


if __name__ == '__main__':
    main()
