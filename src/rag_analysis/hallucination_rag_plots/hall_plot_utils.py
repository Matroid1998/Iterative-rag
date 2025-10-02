"""
Utility functions for hallucination analysis plots.
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict


def load_hallucination_judgments(output_dir: Path) -> List[Dict[str, Any]]:
    """Load all hallucination judgment records from output directory."""
    records = []
    for f in output_dir.glob('*hallucination_judgment.jsonl'):
        # Extract model name from filename
        filename = f.name
        # Remove 'responses_' prefix and '_reverified_hallucination_judgment.jsonl' or '_hallucination_judgment.jsonl' suffix
        model_from_file = filename.replace('responses_', '').replace('_reverified_hallucination_judgment.jsonl', '').replace('_hallucination_judgment.jsonl', '')
        
        with open(f, 'r', encoding='utf-8') as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    # Override model field with the one extracted from filename
                    rec['model'] = model_from_file
                    records.append(rec)
                except json.JSONDecodeError:
                    continue
    return records


def load_coverage_judgments(output_dir: Path) -> List[Dict[str, Any]]:
    """Load all coverage gap judgment records from output directory."""
    records = []
    for f in output_dir.glob('*coverage_gap_judgments.jsonl'):
        # Extract model name from filename
        filename = f.name
        # Remove 'responses_' prefix and '_reverified_coverage_gap_judgments.jsonl' or '_coverage_gap_judgments.jsonl' suffix
        model_from_file = filename.replace('responses_', '').replace('_reverified_coverage_gap_judgments.jsonl', '').replace('_coverage_gap_judgments.jsonl', '')
        
        with open(f, 'r', encoding='utf-8') as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    # Override model field with the one extracted from filename
                    rec['model'] = model_from_file
                    records.append(rec)
                except json.JSONDecodeError:
                    continue
    return records


def load_quality_judgments(output_dir: Path) -> List[Dict[str, Any]]:
    """Load all quality judgment records from output directory."""
    records = []
    for f in output_dir.glob('*quality_judement.jsonl'):
        # Extract model name from filename
        filename = f.name
        # Remove 'responses_' prefix and '_reverified_quality_judement.jsonl' or '_quality_judement.jsonl' suffix
        model_from_file = filename.replace('responses_', '').replace('_reverified_quality_judement.jsonl', '').replace('_quality_judement.jsonl', '')
        
        with open(f, 'r', encoding='utf-8') as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    # Override model field with the one extracted from filename
                    rec['model'] = model_from_file
                    records.append(rec)
                except json.JSONDecodeError:
                    continue
    return records


def create_merged_dataset(hall_records, cov_records, qual_records) -> List[Dict[str, Any]]:
    """Merge records by (model, question) key."""
    # Index coverage and quality by (model, question)
    cov_index = {}
    for rec in cov_records:
        key = (rec.get('model', ''), rec.get('question', ''))
        cov_index[key] = rec
    
    qual_index = {}
    for rec in qual_records:
        key = (rec.get('model', ''), rec.get('question', ''))
        qual_index[key] = rec
    
    # Merge hallucination records with coverage and quality
    merged = []
    for h_rec in hall_records:
        key = (h_rec.get('model', ''), h_rec.get('question', ''))
        
        entry = {
            'model': h_rec.get('model', ''),
            'question': h_rec.get('question', ''),
            'number_of_hops': h_rec.get('number_of_hops', 0),
            'hallucination': h_rec.get('parsed_judgment', {}),
        }
        
        if key in cov_index:
            entry['coverage'] = cov_index[key].get('parsed_judgment', {})
            entry['is_correct'] = cov_index[key].get('is_correct', False)
        
        if key in qual_index:
            entry['quality'] = qual_index[key].get('parsed_judgment', {})
        
        merged.append(entry)
    
    return merged


def count_unsupported_claims(hallucination_judgment: Dict[str, Any]) -> int:
    """Count unsupported claims from hallucination judgment."""
    cf = hallucination_judgment.get('composition_and_faithfulness', {})
    unsupported_count = 0
    for claim in cf.get('unsupported_claims', []):
        if not claim.get('is_supported', True):
            unsupported_count += 1
    return unsupported_count


def has_poor_query_quality(quality_judgment: Dict[str, Any]) -> bool:
    """Check if run has any poor query quality flags."""
    if not quality_judgment:
        return False
    
    for step in quality_judgment.get('per_step', []):
        q = step.get('query_quality', {})
        if (q.get('vague') or q.get('over_broad') or 
            q.get('compound') or q.get('off_topic')):
            return True
    return False


def normalize_model_name(model: str) -> str:
    """Normalize model name for display."""
    if 'gpt-5' in model.lower():
        return 'GPT-5'
    elif 'gpt-4o' in model.lower():
        return 'GPT-4o'
    elif 'deepseek' in model.lower() and 'r1' in model.lower():
        return 'DeepSeek R1'
    elif 'claude-3-7' in model.lower() and 'reasoning' in model.lower():
        return 'Claude 3.7 Sonnet + Reasoning'
    elif 'claude-3-7' in model.lower():
        return 'Claude 3.7 Sonnet'
    elif 'mistral' in model.lower():
        return 'Mistral Large'
    return model
