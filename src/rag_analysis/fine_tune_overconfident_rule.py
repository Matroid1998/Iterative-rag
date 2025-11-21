"""
Detailed analysis focusing on optimal balance between impact and coverage.
Testing more granular thresholds around the promising ranges.
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
    """Load all hallucination judgment records with model names."""
    base_dir = Path(__file__).resolve().parents[2]
    output_dir = base_dir / "src" / "rag_analysis" / "output"
    
    all_records = []
    hallucination_files = list(output_dir.glob("*hallucination_judgment.jsonl"))
    
    for file_path in hallucination_files:
        # Extract model name from filename
        model_name = file_path.stem.replace('_hallucination_judgment', '').replace('responses_', '').replace('_reverified', '')
        
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
                            'model': model_name,
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
              sufficiency_threshold: float) -> Dict:
    """Test a specific rule configuration."""
    overconfident_correct = 0
    overconfident_total = 0
    well_calibrated_correct = 0
    well_calibrated_total = 0
    
    # Track per model
    model_stats = defaultdict(lambda: {'over_correct': 0, 'over_total': 0, 'well_correct': 0, 'well_total': 0})
    
    for record in records:
        finalize_step = record['finalize_step']
        number_of_hops = record['number_of_hops']
        hop_coverage_est = record['hop_coverage_est']
        sufficiency_score_est = record['sufficiency_score_est']
        is_correct = record['is_correct']
        model = record['model']
        
        # Check conditions
        condition_1 = finalize_step < number_of_hops and hop_coverage_est < coverage_threshold
        condition_2 = sufficiency_score_est < sufficiency_threshold
        
        is_overconfident = condition_1 or condition_2
        
        if is_overconfident:
            overconfident_total += 1
            model_stats[model]['over_total'] += 1
            if is_correct:
                overconfident_correct += 1
                model_stats[model]['over_correct'] += 1
        else:
            well_calibrated_total += 1
            model_stats[model]['well_total'] += 1
            if is_correct:
                well_calibrated_correct += 1
                model_stats[model]['well_correct'] += 1
    
    # Calculate metrics
    overconfident_acc = (overconfident_correct / overconfident_total * 100) if overconfident_total > 0 else 0
    well_calibrated_acc = (well_calibrated_correct / well_calibrated_total * 100) if well_calibrated_total > 0 else 0
    
    impact = well_calibrated_acc - overconfident_acc
    coverage = overconfident_total / len(records) * 100
    
    # Calculate per-model metrics
    model_impacts = []
    model_coverages = []
    
    for model, stats in model_stats.items():
        if stats['over_total'] > 0 and stats['well_total'] > 0:
            over_acc = stats['over_correct'] / stats['over_total'] * 100
            well_acc = stats['well_correct'] / stats['well_total'] * 100
            model_impact = well_acc - over_acc
            model_coverage = stats['over_total'] / (stats['over_total'] + stats['well_total']) * 100
            
            model_impacts.append(model_impact)
            model_coverages.append(model_coverage)
    
    return {
        'overconfident_acc': overconfident_acc,
        'well_calibrated_acc': well_calibrated_acc,
        'impact': impact,
        'coverage': coverage,
        'overconfident_count': overconfident_total,
        'well_calibrated_count': well_calibrated_total,
        'min_model_impact': min(model_impacts) if model_impacts else 0,
        'max_model_impact': max(model_impacts) if model_impacts else 0,
        'avg_model_impact': np.mean(model_impacts) if model_impacts else 0,
        'std_model_impact': np.std(model_impacts) if model_impacts else 0,
        'min_model_coverage': min(model_coverages) if model_coverages else 0,
        'max_model_coverage': max(model_coverages) if model_coverages else 0,
        'model_stats': model_stats
    }


def fine_grained_search(records: List[Dict]):
    """Fine-grained search around promising regions."""
    print("\n" + "="*80)
    print("FINE-GRAINED SEARCH (Targeting 18-25% coverage with max impact)")
    print("="*80)
    
    # Test combinations focusing on the sweet spot
    coverage_thresholds = [0.70, 0.75, 0.80, 0.85]
    sufficiency_thresholds = [0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68]
    
    results = []
    
    for cov_thresh in coverage_thresholds:
        for suff_thresh in sufficiency_thresholds:
            result = test_rule(records, cov_thresh, suff_thresh)
            
            # Score: prioritize impact, but penalize if coverage is too low (<15%) or too high (>30%)
            target_coverage = 20  # Sweet spot
            coverage_penalty = abs(result['coverage'] - target_coverage) * 0.2
            
            score = result['impact'] - coverage_penalty
            
            results.append({
                'cov_thresh': cov_thresh,
                'suff_thresh': suff_thresh,
                'score': score,
                **result
            })
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Display top results
    print(f"\n{'Rank':<5} {'Cov≤':<6} {'Suff≤':<7} {'Impact':<9} {'Coverage':<10} {'Over.Acc':<10} {'Well.Acc':<10} {'ModelImpact':<15} {'Score':<7}")
    print("-" * 115)
    
    for i, result in enumerate(results[:15], 1):
        print(f"{i:<5} {result['cov_thresh']:<6.2f} {result['suff_thresh']:<7.2f} "
              f"{result['impact']:>7.1f}pp  {result['coverage']:>7.1f}%  "
              f"{result['overconfident_acc']:>8.1f}%  {result['well_calibrated_acc']:>8.1f}%  "
              f"{result['avg_model_impact']:>5.1f}±{result['std_model_impact']:.1f}pp  "
              f"{result['score']:>6.1f}")
    
    return results


def main():
    print("Loading hallucination judgment data...")
    records = load_all_records()
    print(f"Loaded {len(records)} records")
    
    # Fine-grained search
    results = fine_grained_search(records)
    
    # Show top 5 in detail
    print("\n" + "="*80)
    print("TOP 5 CANDIDATES - DETAILED VIEW")
    print("="*80)
    
    for i, result in enumerate(results[:5], 1):
        print(f"\n{i}. Rule: (coverage ≤ {result['cov_thresh']:.2f}) OR (sufficiency ≤ {result['suff_thresh']:.2f})")
        print(f"   Overall Metrics:")
        print(f"     - Impact: {result['impact']:.1f}pp")
        print(f"     - Coverage: {result['coverage']:.1f}% ({result['overconfident_count']} questions)")
        print(f"     - Overconfident Accuracy: {result['overconfident_acc']:.1f}%")
        print(f"     - Well-Calibrated Accuracy: {result['well_calibrated_acc']:.1f}%")
        print(f"   Per-Model Consistency:")
        print(f"     - Avg Impact: {result['avg_model_impact']:.1f}pp (±{result['std_model_impact']:.1f}pp)")
        print(f"     - Impact Range: {result['min_model_impact']:.1f}pp to {result['max_model_impact']:.1f}pp")
        print(f"     - Coverage Range: {result['min_model_coverage']:.1f}% to {result['max_model_coverage']:.1f}%")
        print(f"   Balance Score: {result['score']:.1f}")
    
    # Recommendation
    print("\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)
    
    best = results[0]
    print(f"\n✨ Optimal Rule: (finalize_step < number_of_hops AND hop_coverage_est < {best['cov_thresh']:.2f})")
    print(f"                 OR sufficiency_score_est < {best['suff_thresh']:.2f}")
    print(f"\nWhy this rule is best:")
    print(f"  ✓ Strong accuracy impact: {best['impact']:.1f}pp difference")
    print(f"  ✓ Balanced coverage: {best['coverage']:.1f}% of questions (~{best['overconfident_count']} questions)")
    print(f"  ✓ Consistent across models: {best['avg_model_impact']:.1f}pp avg impact")
    print(f"  ✓ Overconfident cases have {best['overconfident_acc']:.1f}% accuracy (clearly worse)")
    print(f"  ✓ Well-calibrated cases have {best['well_calibrated_acc']:.1f}% accuracy")
    
    print(f"\nComparison to current rule (coverage≤0.8, sufficiency≤0.60):")
    current = test_rule(records, 0.8, 0.60)
    print(f"  Current: {current['impact']:.1f}pp impact, {current['coverage']:.1f}% coverage")
    print(f"  New:     {best['impact']:.1f}pp impact, {best['coverage']:.1f}% coverage")
    print(f"  Improvement: {best['impact'] - current['impact']:+.1f}pp better impact, {best['coverage'] - current['coverage']:+.1f}% coverage change")


if __name__ == "__main__":
    main()
