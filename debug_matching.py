import json
from pathlib import Path

def check_matching():
    base = Path("/home/mehdi/Projects/Iterative-rag")
    quality_file = base / "src/rag_analysis/output/responses_bedrock_mistral.mistral-large-2402-v1:0_quality_judgement.jsonl"
    reverified_file = base / "src/responses_reverified/responses_bedrock_mistral.mistral-large-2402-v1:0_reverified.jsonl"
    
    print(f"Checking matching for: {quality_file.name}")
    
    reverified_keys = set()
    with open(reverified_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                question = data.get('raw_response', {}).get('question', '') or data.get('question', '')
                reverified_keys.add(question)
            except:
                pass
                
    quality_keys = set()
    with open(quality_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                question = data.get('question', '')
                quality_keys.add(question)
            except:
                pass
                
    common = reverified_keys.intersection(quality_keys)
    print(f"Reverified keys: {len(reverified_keys)}")
    print(f"Quality keys: {len(quality_keys)}")
    print(f"Common keys: {len(common)}")
    
    if len(common) < len(quality_keys):
        print("\nSample mismatch:")
        diff = quality_keys - reverified_keys
        sample = list(diff)[0]
        print(f"In quality but not reverified: '{sample}'")
        print(f"Repr: {repr(sample)}")
        
        # Try to find a close match
        for k in list(reverified_keys)[:5]:
            print(f"Reverified sample: '{k}'")
            print(f"Repr: {repr(k)}")

if __name__ == "__main__":
    check_matching()
