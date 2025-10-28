#!/usr/bin/env python3
"""
Calculate how many times each model missed the FIRST retrieved hop.
In multi-hop retrieval: hop N→N-1→...→2→1 (reverse order)
So "first retrieved hop" = max(missed_hops) = the hop retrieved first
Shows both count and percentage across all questions.
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

def main():
    output_dir = Path(__file__).parent / 'output'
    
    # Store results
    results = []
    
    # Process each coverage gap judgment file
    for file_path in sorted(glob.glob(str(output_dir / '*coverage_gap_judgments.jsonl'))):
        filename = Path(file_path).stem
        base_filename = filename.replace('_coverage_gap_judgments', '')
        model_name = MODEL_NAME_MAP.get(base_filename, base_filename)
        
        total = 0
        total_with_gaps = 0
        missed_first_hop = 0
        
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        total += 1  # Count all questions
                        
                        parsed = record.get('parsed_judgment', {})
                        coverage = parsed.get('retrieval_coverage_gap', {})
                        
                        # Check questions with coverage gap
                        if coverage.get('has_gap'):
                            total_with_gaps += 1
                            
                            missed_hops = coverage.get('missed_hops', [])
                            
                            # First retrieved hop is max(missed_hops)
                            # E.g., in 4-hop: retrieve 4→3→2→1, so first = max([2,3,4]) = 4
                            if missed_hops:
                                first_retrieved_hop = max(missed_hops)
                                # Count if first hop was the one missed first
                                # (This means they failed from the very start of retrieval)
                                # We check if it's the max of ALL hops in question
                                late_hit = parsed.get('late_hit_per_hop', {})
                                per_hop = late_hit.get('per_hop', [])
                                if per_hop:
                                    max_hop_in_question = max(h.get('hop_index', 0) for h in per_hop)
                                    if first_retrieved_hop == max_hop_in_question:
                                        missed_first_hop += 1
                            
                    except json.JSONDecodeError:
                        continue
        
        if total > 0:
            percentage = (missed_first_hop / total) * 100
            gap_rate = (total_with_gaps / total) * 100
            results.append({
                'model': model_name,
                'total': total,
                'total_with_gaps': total_with_gaps,
                'missed_first_hop': missed_first_hop,
                'percentage': percentage,
                'gap_rate': gap_rate
            })
    
    # Print results
    print("=" * 90)
    print("First Retrieved Hop Miss Rate by Model")
    print("(Based on ALL questions)")
    print("=" * 90)
    print(f"{'Model':<30} {'Total':<8} {'w/Gaps':<8} {'1st Miss':<10} {'%':<8} {'Gap%':<8}")
    print("-" * 90)
    
    for result in sorted(results, key=lambda x: x['percentage'], reverse=True):
        print(f"{result['model']:<30} {result['total']:<8} {result['total_with_gaps']:<8} "
              f"{result['missed_first_hop']:<10} {result['percentage']:.2f}%   {result['gap_rate']:.2f}%")
    
    print("-" * 90)
    
    # Calculate average
    avg_percentage = sum(r['percentage'] for r in results) / len(results) if results else 0
    avg_gap_rate = sum(r['gap_rate'] for r in results) / len(results) if results else 0
    total_missed = sum(r['missed_first_hop'] for r in results)
    total_with_gaps = sum(r['total_with_gaps'] for r in results)
    total_questions = sum(r['total'] for r in results)
    overall_percentage = (total_missed / total_questions * 100) if total_questions > 0 else 0
    overall_gap_rate = (total_with_gaps / total_questions * 100) if total_questions > 0 else 0
    
    print(f"{'Average across models:':<30} {'':<8} {'':<8} {'':<10} {avg_percentage:.2f}%   {avg_gap_rate:.2f}%")
    print(f"{'Overall:':<30} {total_questions:<8} {total_with_gaps:<8} {total_missed:<10} "
          f"{overall_percentage:.2f}%   {overall_gap_rate:.2f}%")
    print("=" * 90)
    
    # Simple table format
    print("\n" + "=" * 90)
    print("Simple Table Format (% of ALL questions where first retrieved hop was missed)")
    print("=" * 90)
    for result in sorted(results, key=lambda x: x['percentage'], reverse=True):
        print(f"{result['model']}: {result['percentage']:.2f}% (Gap rate: {result['gap_rate']:.2f}%)")
    print("=" * 90)

if __name__ == '__main__':
    main()
