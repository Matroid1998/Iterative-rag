"""
Update confidence_miscalibration.direction based on new overconfident rule.

New Rule for overconfident_finalize:
  (finalize_step < number_of_hops) AND (hop_coverage_est < 0.70 OR sufficiency_score_est < 0.60)

Where:
  - finalize_step = max(source_step) from evidence
  - number_of_hops = oracle hops for the question
  - hop_coverage_est and sufficiency_score_est are from parsed_judgment
  
Key difference: Model must have stopped early (finalize_step < number_of_hops)
AND have either insufficient coverage OR insufficient sufficiency.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import shutil
from datetime import datetime


def get_max_source_step_from_unsupported_claims(parsed_judgment: Dict[str, Any]) -> int:
    """
    Get the maximum source_step from unsupported_claims in parsed_judgment.
    This represents the finalize step (last step the model took).
    """
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


def determine_direction(
    finalize_step: int,
    number_of_hops: int,
    hop_coverage_est: float,
    sufficiency_score_est: float,
    has_unsupported_claims: bool
) -> str:
    """
    Determine confidence_miscalibration direction based on new rules.
    
    New overconfident_finalize rule:
      (finalize_step < number_of_hops) AND (hop_coverage_est < 0.70 OR sufficiency_score_est < 0.60)
    
    Key: Model must have stopped early (finalize_step < number_of_hops)
    AND have either insufficient coverage OR insufficient sufficiency.
    
    Underconfident_continue rule (unchanged):
      Check if some prior step had enough evidence
    
    Returns: "overconfident_finalize", "underconfident_continue", or "ok"
    """
    
    # New overconfident rule - stricter, requires early stopping AND quality issues
    stopped_early = finalize_step < number_of_hops
    quality_issue = hop_coverage_est < 0.70 or sufficiency_score_est < 0.60
    
    if stopped_early and quality_issue:
        return "overconfident_finalize"
    
    # For underconfident, we'll keep the existing determination from the file
    # since it requires analyzing prior steps which is more complex
    return "ok"  # Will keep existing if not overconfident


def process_hallucination_file(file_path: Path, dry_run: bool = False) -> Dict[str, int]:
    """
    Process a single hallucination judgment file and update confidence_miscalibration.
    
    Returns: Dictionary with counts of changes
    """
    print(f"\nProcessing: {file_path.name}")
    
    # Read all records
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"  Warning: Error decoding line {line_num}: {e}")
                continue
    
    print(f"  Loaded {len(records)} records")
    
    # Process each record
    changes = {
        'total': len(records),
        'already_overconfident': 0,
        'changed_to_overconfident': 0,
        'stayed_same': 0,
        'missing_data': 0
    }
    
    updated_records = []
    
    for record in records:
        # Get parsed judgment
        parsed = record.get('parsed_judgment', {})
        
        # Get required values
        number_of_hops = record.get('number_of_hops')
        
        # Get sufficiency_score_est and hop_coverage_est from parsed judgment
        composition = parsed.get('composition_and_faithfulness', {})
        sufficiency_score_est = composition.get('sufficiency_score_est')
        
        confidence_misc = parsed.get('confidence_miscalibration', {})
        hop_coverage_est = confidence_misc.get('hop_coverage_est')
        
        # Get finalize step (max source_step from unsupported_claims)
        finalize_step = get_max_source_step_from_unsupported_claims(parsed)
        
        # Get current direction
        current_direction = confidence_misc.get('direction', 'ok')
        
        # Check if we have all required data
        if (number_of_hops is None or 
            sufficiency_score_est is None or 
            hop_coverage_est is None or 
            finalize_step == 0):
            changes['missing_data'] += 1
            updated_records.append(record)
            continue
        
        # Check for unsupported claims
        unsupported_claims = parsed.get('unsupported_claims', [])
        has_unsupported = isinstance(unsupported_claims, list) and len(unsupported_claims) > 0
        
        # Determine new direction
        new_direction = determine_direction(
            finalize_step,
            number_of_hops,
            hop_coverage_est,
            sufficiency_score_est,
            has_unsupported
        )
        
        # Update if needed
        if new_direction == "overconfident_finalize":
            if current_direction == "overconfident_finalize":
                changes['already_overconfident'] += 1
            else:
                changes['changed_to_overconfident'] += 1
                # Update the direction
                if 'confidence_miscalibration' not in parsed:
                    parsed['confidence_miscalibration'] = {}
                parsed['confidence_miscalibration']['direction'] = "overconfident_finalize"
                record['parsed_judgment'] = parsed
        else:
            # Keep existing direction (don't change underconfident or ok)
            changes['stayed_same'] += 1
        
        updated_records.append(record)
    
    # Report changes
    print(f"  Summary:")
    print(f"    Total records: {changes['total']}")
    print(f"    Already overconfident: {changes['already_overconfident']}")
    print(f"    Changed to overconfident: {changes['changed_to_overconfident']}")
    print(f"    Stayed same: {changes['stayed_same']}")
    print(f"    Missing data: {changes['missing_data']}")
    
    # Write updated file if not dry run
    if not dry_run and changes['changed_to_overconfident'] > 0:
        # Backup original file
        backup_path = file_path.with_suffix(f'.jsonl.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        shutil.copy2(file_path, backup_path)
        print(f"  Backed up to: {backup_path.name}")
        
        # Write updated records
        with open(file_path, 'w', encoding='utf-8') as f:
            for record in updated_records:
                f.write(json.dumps(record) + '\n')
        print(f"  Updated file written")
    elif dry_run and changes['changed_to_overconfident'] > 0:
        print(f"  [DRY RUN] Would update {changes['changed_to_overconfident']} records")
    
    return changes


def main():
    # Get the output directory
    base_dir = Path(__file__).resolve().parents[2]  # Go up to project root
    output_dir = base_dir / "src" / "rag_analysis" / "output"
    
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        return
    
    # Find all hallucination judgment files
    hallucination_files = list(output_dir.glob("*hallucination_judgment.jsonl"))
    
    if not hallucination_files:
        print("No hallucination judgment files found!")
        return
    
    print(f"Found {len(hallucination_files)} hallucination judgment files")
    
    # Ask for confirmation
    print("\n" + "="*80)
    print("NEW OVERCONFIDENT RULE (STRICTER):")
    print("  (finalize_step < number_of_hops)")
    print("  AND")
    print("  (hop_coverage_est < 0.70 OR sufficiency_score_est < 0.60)")
    print("")
    print("  Key: Model must have stopped early AND have quality issues")
    print("="*80)
    
    # Dry run first
    print("\n" + "="*80)
    print("DRY RUN - Checking what would change...")
    print("="*80)
    
    total_changes = {
        'total': 0,
        'already_overconfident': 0,
        'changed_to_overconfident': 0,
        'stayed_same': 0,
        'missing_data': 0
    }
    
    for file_path in sorted(hallucination_files):
        changes = process_hallucination_file(file_path, dry_run=True)
        for key in total_changes:
            total_changes[key] += changes[key]
    
    # Summary
    print("\n" + "="*80)
    print("TOTAL SUMMARY (DRY RUN):")
    print("="*80)
    print(f"Total records: {total_changes['total']}")
    print(f"Already overconfident: {total_changes['already_overconfident']}")
    print(f"Would change to overconfident: {total_changes['changed_to_overconfident']}")
    print(f"Would stay same: {total_changes['stayed_same']}")
    print(f"Missing data: {total_changes['missing_data']}")
    
    if total_changes['changed_to_overconfident'] == 0:
        print("\nNo changes needed. All files already match the new rule.")
        return
    
    # Ask for confirmation
    print("\n" + "="*80)
    response = input(f"\nProceed with updating {len(hallucination_files)} files? (yes/no): ")
    
    if response.lower() != 'yes':
        print("Cancelled.")
        return
    
    # Actual run
    print("\n" + "="*80)
    print("UPDATING FILES...")
    print("="*80)
    
    for file_path in sorted(hallucination_files):
        process_hallucination_file(file_path, dry_run=False)
    
    print("\n" + "="*80)
    print("UPDATE COMPLETE!")
    print("="*80)
    print(f"\nBackup files created with .backup_* extension")
    print(f"Updated {total_changes['changed_to_overconfident']} records across {len(hallucination_files)} files")


if __name__ == "__main__":
    main()
