#!/usr/bin/env python3
"""
Calculate the percentage of Coverage Gap for each model.
Coverage Gap = cases where retrieval_coverage_gap.has_gap is True
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
}

def main():
    output_dir = Path(__file__).parent / 'output'
    
    # Store results
    results = []
    
    # Process each coverage gap judgment file
    for file_path in sorted(glob.glob(str(output_dir / '*coverage_gap_judgments.jsonl'))):
        filename = Path(file_path).stem
        
        # Extract model name
        base_filename = filename.replace('_coverage_gap_judgments', '')
        model_name = MODEL_NAME_MAP.get(base_filename, base_filename)
        
        # Count total and coverage gaps
        total = 0
        coverage_gaps = 0
        
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        total += 1
                        
                        # Check if there's a coverage gap
                        parsed = record.get('parsed_judgment', {})
                        coverage = parsed.get('retrieval_coverage_gap', {})
                        
                        if coverage.get('has_gap', False):
                            coverage_gaps += 1
                            
                    except json.JSONDecodeError:
                        continue
        
        if total > 0:
            percentage = (coverage_gaps / total) * 100
            results.append({
                'model': model_name,
                'total': total,
                'coverage_gaps': coverage_gaps,
                'percentage': percentage
            })
    
    # Print results
    print("=" * 70)
    print("Coverage Gap Percentage by Model")
    print("=" * 70)
    print(f"{'Model':<30} {'Total':<8} {'Gaps':<8} {'Percentage':<10}")
    print("-" * 70)
    
    for result in sorted(results, key=lambda x: x['percentage'], reverse=True):
        print(f"{result['model']:<30} {result['total']:<8} {result['coverage_gaps']:<8} {result['percentage']:.2f}%")
    
    print("-" * 70)
    
    # Calculate average
    avg_percentage = sum(r['percentage'] for r in results) / len(results) if results else 0
    total_gaps = sum(r['coverage_gaps'] for r in results)
    total_questions = sum(r['total'] for r in results)
    overall_percentage = (total_gaps / total_questions * 100) if total_questions > 0 else 0
    
    print(f"{'Average across models:':<30} {'':<8} {'':<8} {avg_percentage:.2f}%")
    print(f"{'Overall (all questions):':<30} {total_questions:<8} {total_gaps:<8} {overall_percentage:.2f}%")
    print("=" * 70)

if __name__ == '__main__':
    main()
