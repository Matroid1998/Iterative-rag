import json
import glob
import os
import sys
from pathlib import Path

# Fix python path to import hall_plot_utils
sys.path.insert(0, '/home/mehdi/Projects/Iterative-rag/src/rag_analysis/hallucination_rag_plots')
from hall_plot_utils import normalize_model_name

REVERIFIED_DIR = '/home/mehdi/Projects/Iterative-rag/src/responses_reverified'

def main():
    print("=== Checking Incorrect Answer Counts from src/responses_reverified ===")
    
    files = glob.glob(os.path.join(REVERIFIED_DIR, '*.jsonl'))
    
    for f in sorted(files):
        filename = os.path.basename(f)
        # Try to extract model name
        model_part = filename.replace('responses_', '').replace('_reverified.jsonl', '').replace('.jsonl', '')
        # Handle the weird sonnet 4.5 naming if needed, but normalized should catch it
        model_name = normalize_model_name(model_part)
        
        incorrect_count = 0
        total_count = 0
        
        try:
            with open(f, 'r') as fp:
                for line in fp:
                    if not line.strip(): continue
                    rec = json.loads(line)
                    total_count += 1
                    # Check is_correct
                    is_correct = rec.get('is_correct')
                    # Some files might use 'correctness' string or boolean
                    if is_correct is False:
                        incorrect_count += 1
                    elif str(is_correct).lower() == 'false':
                        incorrect_count += 1
                        
            print(f"File: {filename}")
            print(f"  Model: {model_name}")
            print(f"  Total: {total_count}")
            print(f"  Incorrect: {incorrect_count}")
            print("-" * 30)
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")

if __name__ == "__main__":
    main()
