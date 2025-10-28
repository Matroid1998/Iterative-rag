"""
Fix overconfidence classifications based on sufficiency threshold.

Updates parsed_judgment (not raw_output) for records that should be 
classified as overconfident based on:
  - finalize_step < number_of_hops (early stopping), AND
  - sufficiency_score_est < 0.80
  
Coverage is NOT checked - only sufficiency matters.
"""

import json
from pathlib import Path
from collections import defaultdict

# Paths
OUTPUT_DIR = Path(__file__).resolve().parent / 'output'
RESPONSES_DIR = Path(__file__).resolve().parents[1] / 'responses_reverified'


def load_response_file(model_identifier: str) -> dict:
    """Load the corresponding response file to check finalize step."""
    # Try to find matching response file
    pattern = f"responses_{model_identifier}_reverified.jsonl"
    response_files = list(RESPONSES_DIR.glob(pattern))
    
    if not response_files:
        # Try without _reverified
        pattern = f"responses_{model_identifier}.jsonl"
        response_files = list(RESPONSES_DIR.glob(pattern))
    
    if not response_files:
        print(f"  WARNING: No response file found for {model_identifier}")
        return {}
    
    # Load all responses keyed by question
    responses = {}
    with open(response_files[0], 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                # Get question from raw or top-level
                raw = rec.get('raw', {})
                question = raw.get('question', '') or rec.get('question', '')
                if question:
                    responses[question.strip()] = rec
            except json.JSONDecodeError:
                continue
    
    return responses


def get_max_source_step(response_rec: dict) -> int:
    """Extract the maximum source_step from evidence in a response record."""
    raw_response = response_rec.get('raw_response', {})
    evidence = raw_response.get('evidence', [])
    
    if not evidence:
        return 0
    
    max_step = max((e.get('source_step', 0) for e in evidence), default=0)
    return max_step


def main():
    """Fix overconfidence classifications in parsed_judgment."""
    
    # Find all hallucination judgment files
    judgment_files = list(OUTPUT_DIR.glob('*hallucination_judgment.jsonl'))
    
    if not judgment_files:
        print("No hallucination judgment files found!")
        return
    
    print(f"Found {len(judgment_files)} hallucination judgment files")
    print("="*100)
    
    total_processed = 0
    total_fixed = 0
    stats_per_file = {}
    
    for judgment_file in sorted(judgment_files):
        model_name = judgment_file.stem.replace('responses_', '').replace('_reverified_hallucination_judgment', '').replace('_hallucination_judgment', '').replace('2_', '')
        print(f"\nProcessing: {model_name}")
        
        # Extract model identifier from filename
        filename = judgment_file.stem
        model_identifier = filename.replace('responses_', '').replace('_reverified_hallucination_judgment', '').replace('_hallucination_judgment', '')
        
        # Load corresponding response file
        responses = load_response_file(model_identifier)
        if not responses:
            print(f"  Skipping - no response file")
            continue
        
        print(f"  Loaded {len(responses)} responses")
        
        # Process judgments
        updated_records = []
        file_processed = 0
        file_fixed = 0
        
        with open(judgment_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                
                rec = json.loads(line)
                file_processed += 1
                
                question = rec.get('question', '').strip()
                number_of_hops = rec.get('number_of_hops', 0)
                
                # Get current classification from parsed_judgment
                parsed = rec.get('parsed_judgment', {})
                cm = parsed.get('confidence_miscalibration', {})
                cf = parsed.get('composition_and_faithfulness', {})
                
                sufficiency = cf.get('sufficiency_score_est', 1.0)
                current_direction = cm.get('direction', '')
                
                # Get finalize step from response file
                response_rec = responses.get(question)
                if response_rec:
                    finalize_step = get_max_source_step(response_rec)
                    
                    # Check if should be overconfident:
                    # 1. Early stopping: finalize_step < number_of_hops
                    # 2. Low sufficiency: sufficiency < 0.80
                    should_be_overconfident = (finalize_step < number_of_hops and 
                                              sufficiency < 0.80)
                    
                    if should_be_overconfident and current_direction != 'overconfident_finalize':
                        # Fix the parsed_judgment
                        rec['parsed_judgment']['confidence_miscalibration']['direction'] = 'overconfident_finalize'
                        rec['parsed_judgment']['confidence_miscalibration']['is_miscalibrated'] = True
                        file_fixed += 1
                        
                        if file_fixed <= 3:  # Show first 3 examples
                            print(f"  ✓ Fixed: finalize={finalize_step}, hops={number_of_hops}, suff={sufficiency:.2f} → overconfident")
                
                updated_records.append(rec)
        
        # Write updated records back to file
        with open(judgment_file, 'w') as f:
            for rec in updated_records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        
        print(f"  Processed: {file_processed}, Fixed: {file_fixed}")
        stats_per_file[model_name] = {
            'processed': file_processed,
            'fixed': file_fixed
        }
        
        total_processed += file_processed
        total_fixed += file_fixed
    
    # Print summary
    print("\n" + "="*100)
    print("SUMMARY: Fixed Overconfidence Classifications")
    print("="*100)
    print(f"Total records processed: {total_processed}")
    print(f"Total fixed: {total_fixed}")
    print(f"Percentage fixed: {100*total_fixed/total_processed:.2f}%")
    
    print("\nPer-file breakdown:")
    for model_name, stats in sorted(stats_per_file.items()):
        pct = 100 * stats['fixed'] / stats['processed'] if stats['processed'] > 0 else 0
        print(f"  {model_name:50s} {stats['fixed']:4d}/{stats['processed']:4d} ({pct:5.2f}%)")
    
    print("\n" + "="*100)
    print("Rule applied: finalize_step < number_of_hops AND sufficiency < 0.80")
    print("="*100)


if __name__ == '__main__':
    main()
