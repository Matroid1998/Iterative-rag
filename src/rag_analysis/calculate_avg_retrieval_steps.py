#!/usr/bin/env python3
"""
Calculate the average retrieval steps used for each model.
Retrieval steps = max(source_step) from evidence array in each response.
"""

import json
import glob
from pathlib import Path
from collections import defaultdict

# Model name mapping
MODEL_NAME_MAP = {
    'responses_openai_gpt-4o_reverified': 'GPT-4o',
    'responses_openai_gpt-5_reverified': 'GPT-5',
    'responses_openrouter_google__gemini-2.5-pro_reverified': 'Gemini 2.5 Pro',
    'responses_openrouter_x-ai__grok-4-fast_reverified': 'Grok 4 Fast',
    'responses_openrouter_z-ai__glm-4.6_reverified': 'GLM 4.6',
    'responses_bedrock_us.deepseek.r1-v1:0-reasoning_reverified': 'DeepSeek R1',
    'responses_bedrock_mistral.mistral-large-2402-v1:0_reverified': 'Mistral Large',
    'responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0_reverified': 'Claude 3.7 Sonnet',
    'responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning_reverified': 'Claude 3.7 Sonnet + Reasoning',
    'responses_openrouter_anthropic__claude-sonnet-4.5_reverified': 'Claude Sonnet 4.5',
    '2_responses_bedrock_us.meta.llama3-3-70b-instruct-v1:0_reverified': 'Llama 3.3 70B',
    'responses_bedrock_us.meta.llama3-3-70b-instruct-v1:0_reverified': 'Llama 3.3 70B',
}

def get_max_source_step(evidence_list):
    """Extract the maximum source_step from evidence array."""
    if not evidence_list:
        return 0
    
    max_step = 0
    for evidence in evidence_list:
        step = evidence.get('source_step', 0)
        if step > max_step:
            max_step = step
    
    return max_step

def main():
    responses_dir = Path(__file__).parent.parent / 'responses_reverified'
    
    # Store results
    results = []
    
    # Process each response file
    for file_path in sorted(glob.glob(str(responses_dir / '*_reverified.jsonl'))):
        filename = Path(file_path).stem
        
        # Extract model name
        model_name = MODEL_NAME_MAP.get(filename, filename)
        
        # Calculate average max retrieval steps
        total_max_steps = 0
        count = 0
        
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        raw_response = record.get('raw_response', {})
                        evidence = raw_response.get('evidence', [])
                        
                        max_step = get_max_source_step(evidence)
                        total_max_steps += max_step
                        count += 1
                        
                    except (json.JSONDecodeError, KeyError):
                        continue
        
        if count > 0:
            avg_steps = total_max_steps / count
            results.append({
                'model': model_name,
                'total_questions': count,
                'total_max_steps': total_max_steps,
                'avg_steps': avg_steps
            })
    
    # Print results
    print("=" * 75)
    print("Average Retrieval Steps by Model")
    print("=" * 75)
    print(f"{'Model':<30} {'Questions':<12} {'Avg Steps':<12}")
    print("-" * 75)
    
    for result in sorted(results, key=lambda x: x['avg_steps'], reverse=True):
        print(f"{result['model']:<30} {result['total_questions']:<12} {result['avg_steps']:.2f}")
    
    print("-" * 75)
    
    # Calculate overall average
    total_questions = sum(r['total_questions'] for r in results)
    total_steps = sum(r['total_max_steps'] for r in results)
    overall_avg = total_steps / total_questions if total_questions > 0 else 0
    
    print(f"{'Overall Average:':<30} {total_questions:<12} {overall_avg:.2f}")
    print("=" * 75)
    
    # Also print as a simple table for easy copying
    print("\n" + "=" * 75)
    print("Simple Table Format")
    print("=" * 75)
    for result in sorted(results, key=lambda x: x['avg_steps'], reverse=True):
        print(f"{result['model']}: {result['avg_steps']:.2f}")
    print("=" * 75)

if __name__ == '__main__':
    main()
