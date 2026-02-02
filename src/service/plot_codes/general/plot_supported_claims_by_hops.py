
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple

# Add project root to sys.path to enable imports
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from src.service.plot_codes.failure_modes.hallucination.hall_plot_utils import (
    load_hallucination_judgments,
    load_coverage_judgments,
    create_merged_dataset,
    normalize_model_name
)

def load_real_number_of_hops(output_dir: Path, models: list) -> Dict[Tuple[str, str], int]:
    """
    Load the actual number of hops (max_source_step) from the original reverified response files.
    Returns a dictionary mapping (model, question) -> max_source_step.
    """
    real_hops_map = {}
    reverified_dir = project_root / 'src' / 'responses_reverified'
    
    print("Loading real number of hops from source files...")
    
    unique_models = set(models)
    
    for model_name in unique_models:
        filename = f"responses_{model_name}_reverified.jsonl"
        filepath = reverified_dir / filename
        
        if not filepath.exists():
            print(f"Warning: Source file not found: {filepath}")
            continue
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        question = data.get('raw_response', {}).get('question')
                        if not question:
                             question = data.get('candidate', '')
                        
                        evidence = data.get('raw_response', {}).get('evidence', [])
                        max_step = 0
                        if evidence:
                            steps = [item.get('source_step', 0) for item in evidence if item.get('source_step') is not None]
                            if steps:
                                max_step = max(steps)
                        
                        if question:
                            real_hops_map[(model_name, question)] = max_step
                            
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            
    return real_hops_map

def plot_supported_claims_by_hops():
    output_dir = project_root  / 'data' / 'results' / 'failure_modes'
    plots_dir = project_root / 'data' / 'plots' / 'general'
    plots_dir.mkdir(exist_ok=True)

    print("Loading data...")
    hall_records = load_hallucination_judgments(output_dir)
    cov_records = load_coverage_judgments(output_dir)
    
    merged_data = create_merged_dataset(hall_records, cov_records, [])
    
    # Get list of unique models to load real hops for
    all_models = [entry.get('model') for entry in merged_data if entry.get('model')]
    
    # Load real hops mapping
    real_hops_map = load_real_number_of_hops(output_dir, all_models)
    
    # Aggregation: hops -> correct/incorrect -> list of scores
    scores_by_hop = {
        True: defaultdict(list),  # Correct
        False: defaultdict(list)  # Incorrect
    }

    print("Processing merged data...")
    count_missing_hops = 0
    count_processed = 0
    count_no_claims = 0
    
    for entry in merged_data:
        model = entry.get('model')
        question = entry.get('question')
        
        # Get real hops
        hops = real_hops_map.get((model, question))
        
        if hops is None:
            count_missing_hops += 1
            continue
            
        if not (1 <= hops <= 5):
            continue
            
        is_correct = entry.get('is_correct')
        if is_correct is None:
            continue
            
        hall_judgment = entry.get('hallucination', {})
        comp_faith = hall_judgment.get('composition_and_faithfulness', {})
        
        # Extract claims list
        # Note: The key is confusingly named 'unsupported_claims' but contains a list of claim objects
        # where each object has an 'is_supported' boolean.
        claims_list = comp_faith.get('unsupported_claims', [])
        
        if not claims_list:
            count_no_claims += 1
            continue
            
        # Calculate Supported Claims Score
        total_claims = len(claims_list)
        supported_count = sum(1 for claim in claims_list if claim.get('is_supported', True))
        
        supported_score = supported_count / total_claims
        
        scores_by_hop[is_correct][hops].append(supported_score)
        count_processed += 1

    print(f"Processed {count_processed} records.")
    print(f"Skipped {count_missing_hops} records missing hops info.")
    print(f"Skipped {count_no_claims} records with no claims listed.")

    # Calculate means
    means_correct = []
    means_incorrect = []
    hops_range = range(1, 6)
    
    for hop in hops_range:
        # Correct
        scores_c = scores_by_hop[True][hop]
        mean_c = np.mean(scores_c) if scores_c else 0.0
        # If no data, use None to break line or 0? 0 makes sense if we assume 0 supported.
        # But if no data at all, maybe None is better for plotting?
        # Let's stick to 0.0 or keep previous logic. Previous used 0.0.
        means_correct.append(mean_c)
        
        # Incorrect
        scores_i = scores_by_hop[False][hop]
        mean_i = np.mean(scores_i) if scores_i else 0.0
        means_incorrect.append(mean_i)

    print("Generating plot...")
    plt.figure(figsize=(10, 6))
    
    # Plot Correct lines
    plt.plot(hops_range, means_correct, marker='o', linestyle='-', color='green', label='Correct Answer', linewidth=2)
    
    # Plot Incorrect lines
    plt.plot(hops_range, means_incorrect, marker='o', linestyle='-', color='red', label='Incorrect Answer', linewidth=2)
    
    plt.xlabel('Retrieval Step', fontsize=12)
    plt.ylabel('Average Supported Claims Score', fontsize=12)
    plt.title('Average Supported Claims Score by Retrieval Step', fontsize=14)
    plt.xticks(hops_range)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    output_path = plots_dir / 'supported_claims_score.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_supported_claims_by_hops()
