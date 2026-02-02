"""
Analyze hallucination judgment data to find optimal overconfident rule.

Goals:
1. Maximize accuracy difference between overconfident and well-calibrated
2. Ensure reasonable coverage (not too few cases flagged)
3. Balance across all models

Current rule:
  (finalize_step < number_of_hops AND hop_coverage_est < 0.8) OR sufficiency_score_est < 0.60

We'll test various threshold combinations and analyze their impact.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np


def get_max_source_step_from_unsupported_claims(parsed_judgment: Dict) -> int:
    """Get the maximum source_step from unsupported_claims."""
    max_step = 0
    composition = parsed_judgment.get('composition_and_faithfulness', {})
    unsupported_claims = composition.get('unsupported_claims', [])
    
    if isinstance(unsupported_claims, list):
        for claim in unsupported_claims:
            if isinstance(claim, dict):
                step = claim.get('source_step')
                if isinstance(step, (int, float)):
                    max_step = max(max_step, int(step))
    
    return max_step


def load_all_records() -> List[Dict]:
    """Load all hallucination judgment records."""
    base_dir = Path(__file__).resolve().parents[2]
    output_dir = base_dir / "data" / "results" / "failure_modes"
    
    all_records = []
    hallucination_files = list(output_dir.glob("*hallucination_judgment.jsonl"))
    
    for file_path in hallucination_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    parsed = record.get('parsed_judgment', {})
                    
                    # Get required values
                    number_of_hops = record.get('number_of_hops')
                    composition = parsed.get('composition_and_faithfulness', {})
                    sufficiency_score_est = composition.get('sufficiency_score_est')
                    
                    confidence_misc = parsed.get('confidence_miscalibration', {})
                    hop_coverage_est = confidence_misc.get('hop_coverage_est')
                    
                    finalize_step = get_max_source_step_from_unsupported_claims(parsed)
                    is_correct = record.get('is_correct', False)
                    
                    if (number_of_hops is not None and 
                        sufficiency_score_est is not None and 
                        hop_coverage_est is not None and 
                        finalize_step > 0):
                        
                        all_records.append({
                            'number_of_hops': number_of_hops,
                            'finalize_step': finalize_step,
                            'sufficiency_score_est': sufficiency_score_est,
                            'hop_coverage_est': hop_coverage_est,
                            'is_correct': is_correct
                        })
                except json.JSONDecodeError:
                    continue
    
    return all_records


def test_rule(records: List[Dict], 
              coverage_threshold: float,
              sufficiency_threshold: float,
              require_both_conditions: bool = False) -> Dict:
    """
    Test a specific rule configuration.
    
    Rule: (finalize_step < number_of_hops AND hop_coverage_est < coverage_threshold) 
          OR/AND sufficiency_score_est < sufficiency_threshold
    """
    overconfident_correct = 0
    overconfident_total = 0
    well_calibrated_correct = 0
    well_calibrated_total = 0
    
    for record in records:
        finalize_step = record['finalize_step']
        number_of_hops = record['number_of_hops']
        hop_coverage_est = record['hop_coverage_est']
        sufficiency_score_est = record['sufficiency_score_est']
        is_correct = record['is_correct']
        
        # Check conditions
        condition_1 = finalize_step < number_of_hops and hop_coverage_est < coverage_threshold
        condition_2 = sufficiency_score_est < sufficiency_threshold
        
        if require_both_conditions:
            is_overconfident = condition_1 and condition_2
        else:
            is_overconfident = condition_1 or condition_2
        
        if is_overconfident:
            overconfident_total += 1
            if is_correct:
                overconfident_correct += 1
        else:
            well_calibrated_total += 1
            if is_correct:
                well_calibrated_correct += 1
    
    # Calculate metrics
    overconfident_acc = (overconfident_correct / overconfident_total * 100) if overconfident_total > 0 else 0
    well_calibrated_acc = (well_calibrated_correct / well_calibrated_total * 100) if well_calibrated_total > 0 else 0
    
    impact = well_calibrated_acc - overconfident_acc
    coverage = overconfident_total / len(records) * 100
    
    return {
        'overconfident_acc': overconfident_acc,
        'well_calibrated_acc': well_calibrated_acc,
        'impact': impact,
        'coverage': coverage,
        'overconfident_count': overconfident_total,
        'well_calibrated_count': well_calibrated_total
    }


def analyze_distributions(records: List[Dict]):
    """Analyze the distribution of sufficiency and coverage scores."""
    print("\n" + "="*80)
    print("DATA DISTRIBUTION ANALYSIS")
    print("="*80)
    
    sufficiency_scores = [r['sufficiency_score_est'] for r in records]
    coverage_scores = [r['hop_coverage_est'] for r in records]
    
    print(f"\nTotal records: {len(records)}")
    print(f"\nSufficiency Score Distribution:")
    print(f"  Min: {min(sufficiency_scores):.3f}")
    print(f"  25th percentile: {np.percentile(sufficiency_scores, 25):.3f}")
    print(f"  Median: {np.median(sufficiency_scores):.3f}")
    print(f"  75th percentile: {np.percentile(sufficiency_scores, 75):.3f}")
    print(f"  Max: {max(sufficiency_scores):.3f}")
    
    print(f"\nHop Coverage Distribution:")
    print(f"  Min: {min(coverage_scores):.3f}")
    print(f"  25th percentile: {np.percentile(coverage_scores, 25):.3f}")
    print(f"  Median: {np.median(coverage_scores):.3f}")
    print(f"  75th percentile: {np.percentile(coverage_scores, 75):.3f}")
    print(f"  Max: {max(coverage_scores):.3f}")
    
    # Analyze by correctness
    correct_records = [r for r in records if r['is_correct']]
    incorrect_records = [r for r in records if not r['is_correct']]
    
    print(f"\n\nCorrect Answers (n={len(correct_records)}):")
    correct_sufficiency = [r['sufficiency_score_est'] for r in correct_records]
    print(f"  Sufficiency - Mean: {np.mean(correct_sufficiency):.3f}, Median: {np.median(correct_sufficiency):.3f}")
    
    print(f"\nIncorrect Answers (n={len(incorrect_records)}):")
    incorrect_sufficiency = [r['sufficiency_score_est'] for r in incorrect_records]
    print(f"  Sufficiency - Mean: {np.mean(incorrect_sufficiency):.3f}, Median: {np.median(incorrect_sufficiency):.3f}")
    
    # Find optimal separation point
    print(f"\n\nSufficiency Score Ranges vs Accuracy:")
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        below = [r for r in records if r['sufficiency_score_est'] < threshold]
        above = [r for r in records if r['sufficiency_score_est'] >= threshold]
        
        below_acc = sum(r['is_correct'] for r in below) / len(below) * 100 if below else 0
        above_acc = sum(r['is_correct'] for r in above) / len(above) * 100 if above else 0
        
        print(f"  < {threshold:.2f}: {below_acc:.1f}% (n={len(below):5d})  |  >= {threshold:.2f}: {above_acc:.1f}% (n={len(above):5d})  |  Diff: {above_acc - below_acc:+.1f}pp")


def grid_search(records: List[Dict]):
    """Perform grid search to find optimal thresholds."""
    print("\n" + "="*80)
    print("GRID SEARCH FOR OPTIMAL RULE")
    print("="*80)
    
    coverage_thresholds = [0.7, 0.75, 0.8, 0.85, 0.9]
    sufficiency_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    
    best_results = []
    
    print("\nTesting OR combinations (condition_1 OR condition_2):")
    print("-" * 80)
    
    for cov_thresh in coverage_thresholds:
        for suff_thresh in sufficiency_thresholds:
            result = test_rule(records, cov_thresh, suff_thresh, require_both_conditions=False)
            
            # Store results with score (weighted combination of impact and coverage)
            # We want high impact (difference) but also reasonable coverage (15-30%)
            coverage_penalty = 0
            if result['coverage'] < 15:
                coverage_penalty = (15 - result['coverage']) * 0.5  # Penalty for too low coverage
            elif result['coverage'] > 35:
                coverage_penalty = (result['coverage'] - 35) * 0.3  # Penalty for too high coverage
            
            score = result['impact'] - coverage_penalty
            
            best_results.append({
                'cov_thresh': cov_thresh,
                'suff_thresh': suff_thresh,
                'operator': 'OR',
                'score': score,
                **result
            })
    
    # Sort by score
    best_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Display top 15 results
    print(f"\n{'Rank':<5} {'Cov≤':<6} {'Suff≤':<7} {'Op':<4} {'Impact':<8} {'Coverage':<10} {'Over.Acc':<10} {'Well.Acc':<10} {'Score':<7}")
    print("-" * 100)
    
    for i, result in enumerate(best_results[:20], 1):
        print(f"{i:<5} {result['cov_thresh']:<6.2f} {result['suff_thresh']:<7.2f} {result['operator']:<4} "
              f"{result['impact']:>6.1f}pp  {result['coverage']:>7.1f}%  "
              f"{result['overconfident_acc']:>8.1f}%  {result['well_calibrated_acc']:>8.1f}%  "
              f"{result['score']:>6.1f}")
    
    # Show current rule for comparison
    print("\n" + "="*80)
    print("CURRENT RULE COMPARISON")
    print("="*80)
    current = test_rule(records, 0.8, 0.60, require_both_conditions=False)
    print(f"Current Rule (coverage≤0.8 OR sufficiency≤0.60):")
    print(f"  Impact: {current['impact']:.1f}pp")
    print(f"  Coverage: {current['coverage']:.1f}% ({current['overconfident_count']} questions)")
    print(f"  Overconfident Accuracy: {current['overconfident_acc']:.1f}%")
    print(f"  Well-Calibrated Accuracy: {current['well_calibrated_acc']:.1f}%")
    
    return best_results


def main():
    print("Loading hallucination judgment data...")
    records = load_all_records()
    print(f"Loaded {len(records)} records")
    
    # Analyze distributions
    analyze_distributions(records)
    
    # Grid search
    best_results = grid_search(records)
    
    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    top_3 = best_results[:3]
    
    print("\nTop 3 Rule Configurations:\n")
    for i, result in enumerate(top_3, 1):
        print(f"{i}. Coverage ≤ {result['cov_thresh']:.2f} OR Sufficiency ≤ {result['suff_thresh']:.2f}")
        print(f"   - Impact: {result['impact']:.1f}pp (Well-Calibrated {result['well_calibrated_acc']:.1f}% vs Overconfident {result['overconfident_acc']:.1f}%)")
        print(f"   - Coverage: {result['coverage']:.1f}% ({result['overconfident_count']} questions)")
        print(f"   - Balance Score: {result['score']:.1f}")
        print()
    
    print("\nRecommendation:")
    best = top_3[0]
    print(f"Use: (finalize_step < number_of_hops AND hop_coverage_est < {best['cov_thresh']:.2f}) OR sufficiency_score_est < {best['suff_thresh']:.2f}")
    print(f"\nThis rule provides:")
    print(f"  - Strong accuracy impact: {best['impact']:.1f}pp difference")
    print(f"  - Reasonable coverage: {best['coverage']:.1f}% of questions")
    print(f"  - Clear separation between overconfident ({best['overconfident_acc']:.1f}%) and well-calibrated ({best['well_calibrated_acc']:.1f}%)")


if __name__ == "__main__":
    main()
