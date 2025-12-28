
import json
from pathlib import Path

OUTPUT_DIR = Path('src/rag_analysis/output')

def check_coverage_accuracy():
    files = list(OUTPUT_DIR.glob('*coverage_gap_judgments.jsonl'))
    total_correct = 0
    total_count = 0
    
    for f in files:
        with open(f, 'r') as file:
            for line in file:
                if not line.strip(): continue
                rec = json.loads(line)
                if rec.get('is_correct'):
                    total_correct += 1
                total_count += 1
    
    print(f"Total Coverage Records: {total_count}")
    print(f"Total Correct: {total_correct}")
    print(f"Average Accuracy: {total_correct/total_count*100:.2f}%")

if __name__ == '__main__':
    check_coverage_accuracy()
