"""
Update Overconfidence Classification Rule

Changes the overconfidence detection rule from OR to AND:
Old: hop_coverage_est < 0.8 OR sufficiency_score_est < 0.60
New: hop_coverage_est < 0.8 AND sufficiency_score_est < 0.60

For records marked as "overconfident_finalize" that don't meet the new stricter criteria,
change them to "ok" with is_miscalibrated=false.
"""
import json
from pathlib import Path
from collections import defaultdict


def update_overconfidence_rule(input_dir: Path) -> None:
    """
    Update overconfidence classifications in all hallucination judgment files.
    
    Changes records from "overconfident_finalize" to "ok" if:
    - hop_coverage_est >= 0.8 OR sufficiency_score_est >= 0.60
    """
    
    # Find all hallucination judgment files
    pattern = "*_hallucination_judgment.jsonl"
    files = list(input_dir.glob(pattern))
    
    if not files:
        print(f"No files matching '{pattern}' found in {input_dir}")
        return
    
    print(f"Found {len(files)} hallucination judgment files to process")
    
    # Statistics tracking
    stats = defaultdict(lambda: {
        'total': 0,
        'overconfident_original': 0,
        'changed_to_ok': 0,
        'reasons': {
            'high_coverage': 0,  # hop_coverage_est >= 0.8
            'high_sufficiency': 0,  # sufficiency_score_est >= 0.60
            'both_high': 0  # both conditions met
        }
    })
    
    for file_path in files:
        print(f"\nProcessing: {file_path.name}")
        
        updated_records = []
        file_stats = stats[file_path.name]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    file_stats['total'] += 1
                    
                    # Check if this record has the structure we need
                    parsed = record.get('parsed_judgment', {})
                    cm = parsed.get('confidence_miscalibration', {})
                    
                    direction = cm.get('direction', '')
                    is_miscalibrated = cm.get('is_miscalibrated', False)
                    
                    # Only process records marked as overconfident_finalize
                    if direction == 'overconfident_finalize':
                        file_stats['overconfident_original'] += 1
                        
                        hop_coverage = cm.get('hop_coverage_est', 0)
                        sufficiency = parsed.get('composition_and_faithfulness', {}).get('sufficiency_score_est', 0)
                        
                        # Check if it should be changed to "ok" under new rule
                        # (if either threshold is met, it's not overconfident anymore)
                        high_coverage = hop_coverage >= 0.8
                        high_sufficiency = sufficiency >= 0.60
                        
                        if high_coverage or high_sufficiency:
                            # Update the record
                            cm['direction'] = 'ok'
                            cm['is_miscalibrated'] = False
                            
                            file_stats['changed_to_ok'] += 1
                            
                            # Track reason
                            if high_coverage and high_sufficiency:
                                file_stats['reasons']['both_high'] += 1
                            elif high_coverage:
                                file_stats['reasons']['high_coverage'] += 1
                            else:
                                file_stats['reasons']['high_sufficiency'] += 1
                            
                            print(f"  Line {line_num}: Changed to 'ok' "
                                  f"(cov={hop_coverage:.2f}, suff={sufficiency:.2f})")
                    
                    updated_records.append(record)
                
                except json.JSONDecodeError as e:
                    print(f"  WARNING: Failed to parse line {line_num}: {e}")
                    continue
        
        # Write updated records back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            for record in updated_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"  ✓ Updated {file_path.name}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY: Overconfidence Rule Update")
    print("="*80)
    print("\nNew Rule: overconfident_finalize requires BOTH:")
    print("  - hop_coverage_est < 0.8 AND")
    print("  - sufficiency_score_est < 0.60")
    print("\nOld Rule: overconfident_finalize if EITHER:")
    print("  - hop_coverage_est < 0.8 OR")
    print("  - sufficiency_score_est < 0.60")
    
    print("\n" + "-"*80)
    print("Per-File Statistics:")
    print("-"*80)
    
    total_all = 0
    total_overconfident = 0
    total_changed = 0
    total_reasons = defaultdict(int)
    
    for filename in sorted(stats.keys()):
        s = stats[filename]
        total_all += s['total']
        total_overconfident += s['overconfident_original']
        total_changed += s['changed_to_ok']
        
        for reason, count in s['reasons'].items():
            total_reasons[reason] += count
        
        print(f"\n{filename}:")
        print(f"  Total records: {s['total']}")
        print(f"  Originally overconfident: {s['overconfident_original']}")
        print(f"  Changed to 'ok': {s['changed_to_ok']}")
        
        if s['changed_to_ok'] > 0:
            print(f"  Reasons:")
            if s['reasons']['high_coverage'] > 0:
                print(f"    - High coverage only: {s['reasons']['high_coverage']}")
            if s['reasons']['high_sufficiency'] > 0:
                print(f"    - High sufficiency only: {s['reasons']['high_sufficiency']}")
            if s['reasons']['both_high'] > 0:
                print(f"    - Both high: {s['reasons']['both_high']}")
    
    print("\n" + "="*80)
    print("OVERALL TOTALS:")
    print("="*80)
    print(f"Total records processed: {total_all}")
    print(f"Originally marked as overconfident: {total_overconfident}")
    print(f"Changed to 'ok': {total_changed}")
    
    if total_changed > 0:
        pct = 100 * total_changed / total_overconfident if total_overconfident > 0 else 0
        print(f"Percentage changed: {pct:.1f}%")
        
        print(f"\nBreakdown of changes:")
        print(f"  - High coverage only (≥0.8): {total_reasons['high_coverage']}")
        print(f"  - High sufficiency only (≥0.6): {total_reasons['high_sufficiency']}")
        print(f"  - Both thresholds met: {total_reasons['both_high']}")
    
    print("\n" + "="*80)


def main():
    output_dir = Path(__file__).resolve().parent / 'output'
    
    print("Updating overconfidence classification rule...")
    print(f"Looking for files in: {output_dir}")
    
    if not output_dir.exists():
        print(f"ERROR: Directory does not exist: {output_dir}")
        return
    
    update_overconfidence_rule(output_dir)
    
    print("\n✓ Update complete!")


if __name__ == "__main__":
    main()
