"""
Utility functions for cross-system analysis plots.
Joins data from coverage, quality, and hallucination judgments.
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict


def extract_model_from_filename(filename: str) -> str:
    """
    Extract model name from filename.
    Example: 'responses_openai_gpt-5_reverified_coverage_gap_judgments.jsonl' -> 'openai_gpt-5'
    """
    # Remove 'responses_' prefix and everything after the model name
    name = filename.replace('responses_', '')
    
    # Split by underscores and find where the judgment type starts
    # Try different suffixes
    for suffix in ['_reverified_', '_coverage_gap', '_quality', '_hallucination']:
        if suffix in name:
            return name.split(suffix)[0]
            
    # Fallback
    return name.split('_')[0] if '_' in name else name


def load_all_judgments(output_dir: Path) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load all three types of judgments, extracting model name from filename."""
    coverage_records = []
    quality_records = []
    hallucination_records = []
    
    # Load coverage judgments
    files = list(output_dir.glob('*coverage_gap_judgments.jsonl')) + list(output_dir.glob('*_coverage_gap.jsonl'))
    for f in files:
        model_name = extract_model_from_filename(f.name)
        with open(f, 'r', encoding='utf-8') as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    rec['eval_model'] = model_name  # Add the actual model being evaluated
                    coverage_records.append(rec)
                except json.JSONDecodeError:
                    continue
    
    # Load quality judgments
    files = list(output_dir.glob('*quality_judement.jsonl')) + list(output_dir.glob('*_quality.jsonl'))
    for f in files:
        model_name = extract_model_from_filename(f.name)
        with open(f, 'r', encoding='utf-8') as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    rec['eval_model'] = model_name  # Add the actual model being evaluated
                    quality_records.append(rec)
                except json.JSONDecodeError:
                    continue
    
    # Load hallucination judgments
    files = list(output_dir.glob('*hallucination_judgment.jsonl')) + list(output_dir.glob('*_hallucination.jsonl'))
    for f in files:
        model_name = extract_model_from_filename(f.name)
        with open(f, 'r', encoding='utf-8') as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    rec['eval_model'] = model_name  # Add the actual model being evaluated
                    hallucination_records.append(rec)
                except json.JSONDecodeError:
                    continue
    
    return coverage_records, quality_records, hallucination_records


def create_merged_dataset(coverage_records, quality_records, hallucination_records) -> List[Dict[str, Any]]:
    """
    Merge all three judgment types by (eval_model, question) key.
    Returns list of merged records with all available data.
    """
    # Index by (eval_model, question)
    quality_index = {}
    for rec in quality_records:
        key = (rec.get('eval_model', ''), rec.get('question', ''))
        quality_index[key] = rec
    
    hallucination_index = {}
    for rec in hallucination_records:
        key = (rec.get('eval_model', ''), rec.get('question', ''))
        hallucination_index[key] = rec
    
    # Merge using coverage as base
    merged = []
    for cov_rec in coverage_records:
        key = (cov_rec.get('eval_model', ''), cov_rec.get('question', ''))
        
        entry = {
            'model': cov_rec.get('eval_model', ''),  # Use eval_model as the main model field
            'question': cov_rec.get('question', ''),
            'is_correct': cov_rec.get('is_correct', False),
            'coverage': cov_rec.get('parsed_judgment', {}),
        }
        
        if key in quality_index:
            entry['quality'] = quality_index[key].get('parsed_judgment', {})
            entry['number_of_hops'] = quality_index[key].get('number_of_hops', 0)
        
        if key in hallucination_index:
            entry['hallucination'] = hallucination_index[key].get('parsed_judgment', {})
        
        merged.append(entry)
    
    return merged


def normalize_model_name(model: str) -> str:
    """Normalize model name for display."""
    if 'gpt-5.1' in model.lower():
        return 'GPT-5.1'
    elif 'gpt-5' in model.lower():
        return 'GPT-5'
    elif 'gpt-4o' in model.lower():
        return 'GPT-4o'
    elif 'deepseek' in model.lower() and 'r1' in model.lower():
        return 'DeepSeek R1'
    elif 'claude-3-7' in model.lower() and 'reasoning' in model.lower():
        return 'Claude 3.7 + Reasoning'
    elif 'claude-3-7' in model.lower():
        return 'Claude 3.7 Sonnet'
    elif 'claude-sonnet-4.5' in model.lower() or 'claude-4.5' in model.lower():
        return 'Claude Sonnet 4.5'
    elif 'claude-3-5' in model.lower():
        return 'Claude 3.5 Sonnet'
    elif 'gemini-3' in model.lower():
        return 'Gemini 3 Pro'
    elif 'gemini-2.5-pro' in model.lower() or 'gemini-2.5' in model.lower():
        return 'Gemini 2.5 Pro'
    elif 'grok-4' in model.lower():
        return 'Grok 4 Fast'
    elif 'glm-4.6' in model.lower() or 'glm-4' in model.lower():
        return 'GLM 4.6'
    elif 'mistral' in model.lower():
        return 'Mistral Large'
    elif 'llama' in model.lower():
        return 'Llama 3.3 70B'
    return model


def get_avg_steps(quality_judgment: Dict) -> float:
    """Calculate average number of steps from quality judgment."""
    if not quality_judgment:
        return 0.0
    steps = quality_judgment.get('per_step', [])
    return len(steps)


def get_avg_specificity(quality_judgment: Dict) -> float:
    """Calculate average specificity score from quality judgment."""
    if not quality_judgment:
        return 0.0
    steps = quality_judgment.get('per_step', [])
    if not steps:
        return 0.0
    
    scores = []
    for step in steps:
        q = step.get('query_quality', {})
        spec = q.get('specificity_score')
        if spec is not None:
            scores.append(float(spec))
    
    return sum(scores) / len(scores) if scores else 0.0


def has_coverage_gap(coverage_judgment: Dict) -> bool:
    """Check if run has coverage gap."""
    if not coverage_judgment:
        return False
    return coverage_judgment.get('retrieval_coverage_gap', {}).get('has_gap', False)


def has_carry_drop(coverage_judgment: Dict) -> bool:
    """Check if run has anchor carry-drop."""
    if not coverage_judgment:
        return False
    return coverage_judgment.get('anchor_carry_drop', {}).get('any_carry_drop', False)


def has_late_hit(coverage_judgment: Dict) -> bool:
    """Check if run has late hit."""
    if not coverage_judgment:
        return False
    return coverage_judgment.get('late_hit_per_hop', {}).get('any_late_hit', False)


def has_composition_failure(hallucination_judgment: Dict) -> bool:
    """Check if run has composition failure."""
    if not hallucination_judgment:
        return False
    return hallucination_judgment.get('composition_and_faithfulness', {}).get('composition_failure', False)


def is_miscalibrated(hallucination_judgment: Dict) -> bool:
    """Check if run is miscalibrated."""
    if not hallucination_judgment:
        return False
    return hallucination_judgment.get('confidence_miscalibration', {}).get('is_miscalibrated', False)


def is_overconfident(hallucination_judgment: Dict) -> bool:
    """Check if run is overconfident."""
    if not hallucination_judgment:
        return False
    direction = hallucination_judgment.get('confidence_miscalibration', {}).get('direction', 'ok')
    return direction == 'overconfident_finalize'


def has_poor_query_quality(quality_judgment: Dict) -> bool:
    """Check if run has any poor query quality flags."""
    if not quality_judgment:
        return False
    
    for step in quality_judgment.get('per_step', []):
        q = step.get('query_quality', {})
        if (q.get('vague') or q.get('over_broad') or 
            q.get('compound') or q.get('off_topic')):
            return True
    return False


def count_logical_hops(quality_judgment: Dict) -> Tuple[int, int]:
    """Count how many steps are next logical hop vs total steps."""
    if not quality_judgment:
        return 0, 0
    
    steps = quality_judgment.get('per_step', [])
    logical_count = sum(1 for s in steps if s.get('is_next_logical_hop', False))
    return logical_count, len(steps)


def get_step_carry_drop_flags(coverage_judgment: Dict) -> List[bool]:
    """Get per-step carry drop flags."""
    if not coverage_judgment:
        return []
    
    per_step = coverage_judgment.get('anchor_carry_drop', {}).get('per_step', [])
    return [s.get('carry_drop', False) for s in per_step]


def get_step_anchored_flags(quality_judgment: Dict) -> List[bool]:
    """Get per-step anchored flags."""
    if not quality_judgment:
        return []
    
    per_step = quality_judgment.get('per_step', [])
    return [s.get('query_quality', {}).get('anchored', False) for s in per_step]
