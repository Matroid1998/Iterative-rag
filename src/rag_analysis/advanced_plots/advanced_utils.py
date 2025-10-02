"""
Utility functions for advanced analysis plots.
"""
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict


def load_all_judgments(output_dir: Path) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load all three types of judgments."""
    coverage_records = []
    quality_records = []
    hallucination_records = []
    
    # Load coverage judgments
    for f in output_dir.glob('*coverage_gap_judgments.jsonl'):
        with open(f, 'r', encoding='utf-8') as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    coverage_records.append(rec)
                except json.JSONDecodeError:
                    continue
    
    # Load quality judgments
    for f in output_dir.glob('*quality_judement.jsonl'):
        with open(f, 'r', encoding='utf-8') as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    quality_records.append(rec)
                except json.JSONDecodeError:
                    continue
    
    # Load hallucination judgments
    for f in output_dir.glob('*hallucination_judgment.jsonl'):
        with open(f, 'r', encoding='utf-8') as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    hallucination_records.append(rec)
                except json.JSONDecodeError:
                    continue
    
    return coverage_records, quality_records, hallucination_records


def create_merged_dataset(coverage_records, quality_records, hallucination_records) -> List[Dict[str, Any]]:
    """Merge all three judgment types by (model, question) key."""
    quality_index = {}
    for rec in quality_records:
        key = (rec.get('model', ''), rec.get('question', ''))
        quality_index[key] = rec
    
    hallucination_index = {}
    for rec in hallucination_records:
        key = (rec.get('model', ''), rec.get('question', ''))
        hallucination_index[key] = rec
    
    # Merge using coverage as base
    merged = []
    for cov_rec in coverage_records:
        key = (cov_rec.get('model', ''), cov_rec.get('question', ''))
        
        # Get raw hallucination record for top-level fields
        hall_rec = hallucination_index.get(key, {})
        
        entry = {
            'model': cov_rec.get('model', ''),
            'question': cov_rec.get('question', ''),
            'is_correct': cov_rec.get('is_correct', False),
            'coverage': cov_rec.get('parsed_judgment', {}),
            'number_of_hops': hall_rec.get('number_of_hops', 0),  # Get from hallucination record
        }
        
        if key in quality_index:
            entry['quality'] = quality_index[key].get('parsed_judgment', {})
        
        if key in hallucination_index:
            entry['hallucination'] = hallucination_index[key].get('parsed_judgment', {})
        
        merged.append(entry)
    
    return merged


def normalize_model_name(model: str) -> str:
    """Normalize model name for display."""
    if 'gpt-5' in model.lower() or 'openai-gpt-5' in model.lower():
        return 'GPT-5'
    elif 'gpt-4o' in model.lower():
        return 'GPT-4o'
    elif 'deepseek' in model.lower() and 'r1' in model.lower():
        return 'DeepSeek R1'
    elif 'claude-3-7' in model.lower() and 'reasoning' in model.lower():
        return 'Claude 3.7 + Reasoning'
    elif 'claude-3-7' in model.lower():
        return 'Claude 3.7 Sonnet'
    elif 'claude-3-5' in model.lower():
        return 'Claude 3.5 Sonnet'
    elif 'mistral' in model.lower():
        return 'Mistral Large'
    elif 'llama' in model.lower():
        return 'Llama 3.3 70B'
    return model


def load_accuracy_from_csv(csv_dir: Path) -> Dict[str, float]:
    """Load accuracy data from CSV files in results directory."""
    accuracy_map = {}
    
    for csv_file in csv_dir.glob('*.csv'):
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    model_name = row.get('Model', '')
                    
                    # Try multiple possible column names
                    accuracy = row.get('Accuracy (%)', '') or row.get('accuracy', '') or row.get('Accuracy', '')
                    
                    if model_name and accuracy:
                        try:
                            # Normalize model name
                            normalized = normalize_model_name(model_name)
                            # Accuracy is already a percentage in the CSV
                            accuracy_map[normalized] = float(accuracy)
                        except ValueError:
                            continue
        except Exception as e:
            continue
    
    return accuracy_map


def get_query_flags(step: Dict) -> Dict[str, bool]:
    """Extract all query quality flags from a step."""
    q = step.get('query_quality', {})
    return {
        'vague': q.get('vague', False),
        'over_broad': q.get('over_broad', False),
        'compound': q.get('compound', False),
        'off_topic': q.get('off_topic', False),
        'anchored': q.get('anchored', False),
    }


def get_quality_category(flags: Dict[str, bool]) -> str:
    """Categorize query quality based on flags."""
    if flags['off_topic']:
        return 'off_topic'
    elif flags['vague'] or flags['over_broad']:
        return 'poor'
    elif flags['compound']:
        return 'compound'
    elif flags['anchored']:
        return 'anchored'
    else:
        return 'clean'


def calculate_avg_retrieval_delay(coverage_judgment: Dict) -> float:
    """Calculate average delay between hop index and first hit step."""
    if not coverage_judgment:
        return 0.0
    
    per_hop = coverage_judgment.get('late_hit_per_hop', {}).get('per_hop', [])
    if not per_hop:
        return 0.0
    
    delays = []
    for hop in per_hop:
        hop_index = hop.get('hop_index', 0)
        first_hit_step = hop.get('first_hit_step')
        
        if first_hit_step is not None and first_hit_step > 0:
            delay = first_hit_step - hop_index
            delays.append(delay)
    
    return sum(delays) / len(delays) if delays else 0.0
