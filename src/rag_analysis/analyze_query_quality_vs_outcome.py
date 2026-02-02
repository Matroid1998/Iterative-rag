"""
Analyze the relationship between query quality and answer correctness.
Check if poor query quality correlates with incorrect answers.
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import numpy as np

def normalize_model_name(model_str):
    """Normalize model names for consistent display."""
    model_map = {
        'openai_gpt-5': 'GPT-5',
        'openai_gpt-4o': 'GPT-4o',
        'bedrock_us.deepseek.r1': 'DeepSeek R1',
        'bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning': 'Claude 3.7 Sonnet + Reasoning',
        'bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0': 'Claude 3.7 Sonnet',
        'bedrock_us.anthropic.claude-3-7-sonnet-reasoning': 'Claude 3.7 Sonnet + Reasoning',
        'bedrock_us.anthropic.claude-3-7-sonnet': 'Claude 3.7 Sonnet',
        'bedrock_mistral.mistral-large': 'Mistral Large'
    }
    
    if model_str in model_map:
        return model_map[model_str]
    
    for key, value in model_map.items():
        if key in model_str:
            return value
    return model_str


def load_correctness_map(output_dir):
    """Load is_correct information from coverage judgment files."""
    correctness = {}  # {(model, question): is_correct}
    
    for file_path in glob.glob(str(output_dir / '*coverage_gap_judgments.jsonl')):
        filename = Path(file_path).name
        model_from_file = filename.replace('responses_', '').replace('_reverified_coverage_gap_judgments.jsonl', '')
        model = normalize_model_name(model_from_file)
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    question = rec.get('question', '')
                    is_correct = rec.get('is_correct', False)
                    correctness[(model, question)] = is_correct
                except json.JSONDecodeError:
                    continue
    
    return correctness


def has_poor_query_quality(quality_judgment):
    """Check if any step has poor query quality."""
    if not quality_judgment:
        return False
    
    for step in quality_judgment.get('per_step', []):
        q = step.get('query_quality', {})
        if (q.get('vague') or q.get('over_broad') or 
            q.get('compound') or q.get('off_topic')):
            return True
    return False


def count_quality_flags(quality_judgment):
    """Count individual quality flags."""
    flags = {
        'vague': 0,
        'over_broad': 0,
        'compound': 0,
        'off_topic': 0
    }
    
    if not quality_judgment:
        return flags
    
    for step in quality_judgment.get('per_step', []):
        q = step.get('query_quality', {})
        if q.get('vague'):
            flags['vague'] += 1
        if q.get('over_broad'):
            flags['over_broad'] += 1
        if q.get('compound'):
            flags['compound'] += 1
        if q.get('off_topic'):
            flags['off_topic'] += 1
    
    return flags


def get_avg_scores(quality_judgment):
    """Get average specificity and on-topic scores."""
    if not quality_judgment:
        return None, None
    
    spec_scores = []
    topic_scores = []
    
    for step in quality_judgment.get('per_step', []):
        q = step.get('query_quality', {})
        spec = q.get('specificity_score')
        topic = q.get('on_topic_score')
        
        if spec is not None:
            spec_scores.append(spec)
        if topic is not None:
            topic_scores.append(topic)
    
    avg_spec = np.mean(spec_scores) if spec_scores else None
    avg_topic = np.mean(topic_scores) if topic_scores else None
    
    return avg_spec, avg_topic


def analyze_quality_vs_outcome(output_dir):
    """Analyze relationship between query quality and correctness."""
    correctness_map = load_correctness_map(output_dir)
    
    # Structure: {model: {is_correct: {has_poor_query: count}}}
    stats = defaultdict(lambda: {
        True: {'good_query': 0, 'poor_query': 0, 'specificity': [], 'on_topic': []},
        False: {'good_query': 0, 'poor_query': 0, 'specificity': [], 'on_topic': []}
    })
    
    # Track individual flags
    flag_stats = defaultdict(lambda: {
        True: {'vague': 0, 'over_broad': 0, 'compound': 0, 'off_topic': 0},
        False: {'vague': 0, 'over_broad': 0, 'compound': 0, 'off_topic': 0}
    })
    
    matched = 0
    unmatched = 0
    
    for file_path in glob.glob(str(output_dir / '*quality_judement.jsonl')):
        filename = Path(file_path).name
        model_from_file = filename.replace('responses_', '').replace('_reverified_quality_judement.jsonl', '')
        model = normalize_model_name(model_from_file)
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    question = data.get('question', '')
                    
                    # Look up correctness
                    is_correct = correctness_map.get((model, question), None)
                    if is_correct is None:
                        unmatched += 1
                        continue
                    
                    matched += 1
                    parsed = data.get('parsed_judgment', {})
                    
                    # Check query quality
                    poor_query = has_poor_query_quality(parsed)
                    
                    if poor_query:
                        stats[model][is_correct]['poor_query'] += 1
                    else:
                        stats[model][is_correct]['good_query'] += 1
                    
                    # Count individual flags
                    flags = count_quality_flags(parsed)
                    for flag_name, count in flags.items():
                        if count > 0:
                            flag_stats[model][is_correct][flag_name] += 1
                    
                    # Get scores
                    avg_spec, avg_topic = get_avg_scores(parsed)
                    if avg_spec is not None:
                        stats[model][is_correct]['specificity'].append(avg_spec)
                    if avg_topic is not None:
                        stats[model][is_correct]['on_topic'].append(avg_topic)
                
                except json.JSONDecodeError:
                    continue
    
    print(f"Matched: {matched}, Unmatched: {unmatched}")
    print("\n" + "="*100)
    print("QUERY QUALITY vs ANSWER CORRECTNESS")
    print("="*100)
    
    for model in sorted(stats.keys()):
        print(f"\n{'='*100}")
        print(f"{model}")
        print(f"{'='*100}")
        
        for is_correct in [True, False]:
            label = "CORRECT" if is_correct else "INCORRECT"
            data = stats[model][is_correct]
            
            total = data['good_query'] + data['poor_query']
            if total == 0:
                continue
            
            poor_pct = 100 * data['poor_query'] / total
            
            avg_spec = np.mean(data['specificity']) if data['specificity'] else 0
            avg_topic = np.mean(data['on_topic']) if data['on_topic'] else 0
            
            print(f"\n{label} (n={total}):")
            print(f"  Good query quality: {data['good_query']:4d} ({100*data['good_query']/total:.1f}%)")
            print(f"  Poor query quality: {data['poor_query']:4d} ({poor_pct:.1f}%)")
            print(f"  Avg specificity:    {avg_spec:.3f}")
            print(f"  Avg on-topic:       {avg_topic:.3f}")
            
            # Show individual flags
            flags = flag_stats[model][is_correct]
            if any(flags.values()):
                print(f"  Individual flags:")
                for flag_name in ['vague', 'over_broad', 'compound', 'off_topic']:
                    if flags[flag_name] > 0:
                        print(f"    {flag_name:12s}: {flags[flag_name]:4d} ({100*flags[flag_name]/total:.1f}%)")
        
        # Calculate correlation
        correct_data = stats[model][True]
        incorrect_data = stats[model][False]
        
        correct_total = correct_data['good_query'] + correct_data['poor_query']
        incorrect_total = incorrect_data['good_query'] + incorrect_data['poor_query']
        
        if correct_total > 0 and incorrect_total > 0:
            correct_poor_pct = 100 * correct_data['poor_query'] / correct_total
            incorrect_poor_pct = 100 * incorrect_data['poor_query'] / incorrect_total
            
            print(f"\n  SUMMARY:")
            print(f"    Poor query rate in CORRECT:   {correct_poor_pct:.1f}%")
            print(f"    Poor query rate in INCORRECT: {incorrect_poor_pct:.1f}%")
            print(f"    Difference:                   {incorrect_poor_pct - correct_poor_pct:+.1f}%")
            
            if incorrect_poor_pct > correct_poor_pct:
                print(f"    ⚠️  INCORRECT answers have {incorrect_poor_pct/correct_poor_pct:.2f}x more poor queries")
            else:
                print(f"    ⚠️  No significant difference or opposite trend")


def main():
    base_dir = Path(__file__).resolve().parents[2]
    output_dir = base_dir / "data" / "results" / "failure_modes"
    
    print("Analyzing query quality vs outcome...")
    analyze_quality_vs_outcome(output_dir)


if __name__ == "__main__":
    main()
