"""
Generate tables for each model showing:
1. Accuracy slope from step 1 to 5 (green line in the plot)
2. Distribution of oracle hops at each step (stacked bar breakdown)

For the plot: all_models_correctness_by_steps_no_context_wrong_no_coverage.png
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import numpy as np


def normalize_model_name(model: str) -> str:
    """Normalize model name for display."""
    if 'gpt-5' in model.lower() or 'openai-gpt-5' in model.lower() or 'openai_gpt-5' in model.lower():
        return 'GPT-5'
    elif 'gpt-4o' in model.lower():
        return 'GPT-4o'
    elif 'deepseek' in model.lower() and 'r1' in model.lower():
        return 'DeepSeek R1'
    elif 'claude-3-7' in model.lower() and 'reasoning' in model.lower():
        return 'Claude 3.7 + Reasoning'
    elif 'claude-3-7' in model.lower():
        return 'Claude 3.7 Sonnet'
    elif 'claude-sonnet-4.5' in model.lower() or 'claude_sonnet_4_5' in model.lower() or 'claude-4.5' in model.lower():
        return 'Claude Sonnet 4.5'
    elif 'claude-3-5' in model.lower():
        return 'Claude 3.5 Sonnet'
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


def load_qa_lookup(base_dir: Path) -> Dict[str, int]:
    """Load question -> hop count mapping from QA file."""
    qa_lookup = {}
    qa_path = base_dir.parent / "data" / "corpus" / "chemrxiv_qa.json"
    
    if not qa_path.exists():
        print(f"Warning: QA file not found: {qa_path}")
        return qa_lookup
    
    try:
        with qa_path.open("r", encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            question = entry.get("q")
            path_list = entry.get("path")
            if isinstance(question, str) and isinstance(path_list, list) and path_list:
                qa_lookup[question.strip()] = len(path_list)
    except json.JSONDecodeError as e:
        print(f"Error decoding QA file: {e}")
    
    return qa_lookup


def load_no_context_wrong_questions(base_dir: Path) -> Set[str]:
    """Load questions that were answered incorrectly in no-context baseline."""
    no_context_dir = base_dir / "src" / "response-jsonl-without-context"
    wrong_questions = set()
    
    if not no_context_dir.exists():
        print(f"Warning: No-context directory not found: {no_context_dir}")
        return wrong_questions
    
    for file_path in no_context_dir.glob("*.jsonl"):
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    
                    # Extract question
                    if 'raw' in data and isinstance(data['raw'], dict):
                        question = data['raw'].get('question', '')
                    else:
                        question = data.get('question', '')
                    
                    is_correct = data.get('is_correct', False)
                    
                    if not is_correct and question:
                        wrong_questions.add(question)
                
                except json.JSONDecodeError:
                    continue
    
    return wrong_questions


def extract_question(record: dict) -> str:
    """Extract question from response record."""
    question = record.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()
    for key in ("raw", "raw_response"):
        raw = record.get(key)
        if isinstance(raw, dict):
            q = raw.get("question")
            if isinstance(q, str) and q.strip():
                return q.strip()
    return ""


def extract_max_source_step(record: dict) -> int:
    """Return the maximum retrieval step (source_step) found in a record."""
    steps = []
    for key in ("raw_response", "raw"):
        raw = record.get(key)
        if not isinstance(raw, dict):
            continue
        evidence = raw.get("evidence")
        if not isinstance(evidence, list):
            continue
        for item in evidence:
            if not isinstance(item, dict):
                continue
            step = item.get("source_step")
            if isinstance(step, (int, float)):
                step_int = int(round(step))
                if step_int > 0:
                    steps.append(step_int)
    if steps:
        return max(steps)
    return 0


def load_model_data(
    base_dir: Path,
    qa_lookup: Dict[str, int],
    wrong_questions: Set[str]
) -> Dict[str, Tuple[List[Tuple[int, int, str]], List[Tuple[int, int, str]]]]:
    """
    Load model data filtered to no-context wrong questions.
    Returns: Dict[model_name, (correct_steps, incorrect_steps)]
    Each step tuple: (step, oracle_hops, question)
    """
    iterative_dir = base_dir / "src" / "responses_reverified"
    model_data = {}
    
    if not iterative_dir.exists():
        print(f"Warning: Iterative responses directory not found: {iterative_dir}")
        return model_data
    
    for file_path in iterative_dir.glob("*.jsonl"):
        model_name = file_path.stem
        if 'responses_' in model_name:
            model_name = model_name.replace('responses_', '')
        if '_reverified' in model_name:
            model_name = model_name.replace('_reverified', '')
        
        model_name = normalize_model_name(model_name)
        
        correct_steps = []
        incorrect_steps = []
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    question = extract_question(data)
                    
                    # Filter to no-context wrong questions
                    if question not in wrong_questions:
                        continue
                    
                    is_correct = data.get('is_correct', False)
                    max_step = extract_max_source_step(data)
                    
                    # Get oracle hops from QA lookup
                    oracle_hops = qa_lookup.get(question.strip(), 0)
                    
                    if max_step > 0 and oracle_hops > 0:
                        if is_correct:
                            correct_steps.append((max_step, oracle_hops, question))
                        else:
                            incorrect_steps.append((max_step, oracle_hops, question))
                
                except json.JSONDecodeError:
                    continue
        
        if correct_steps or incorrect_steps:
            model_data[model_name] = (correct_steps, incorrect_steps)
    
    return model_data


def calculate_model_statistics(
    correct_steps: List[Tuple[int, int, str]],
    incorrect_steps: List[Tuple[int, int, str]]
) -> Dict:
    """
    Calculate statistics for a single model:
    - Accuracy per step
    - Oracle hop distribution per step
    - Slope of accuracy line from step 1 to 5
    """
    all_steps = [s for s, _, _ in correct_steps] + [s for s, _, _ in incorrect_steps]
    if not all_steps:
        return None
    
    max_step = max(all_steps)
    
    stats = {
        'max_step': max_step,
        'steps': {}
    }
    
    # Calculate statistics per step
    for step in range(1, max_step + 1):
        # Count by hop and correctness
        hop_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        correct_count = 0
        incorrect_count = 0
        
        for s, hop, _ in correct_steps:
            if s == step:
                if hop in hop_counts:
                    hop_counts[hop] += 1
                correct_count += 1
        
        for s, hop, _ in incorrect_steps:
            if s == step:
                if hop in hop_counts:
                    hop_counts[hop] += 1
                incorrect_count += 1
        
        total = correct_count + incorrect_count
        accuracy = (correct_count / total * 100) if total > 0 else 0
        
        stats['steps'][step] = {
            'accuracy': accuracy,
            'total': total,
            'correct': correct_count,
            'incorrect': incorrect_count,
            'hop_distribution': hop_counts
        }
    
    # Calculate slope from step 1 to step 5 (or max step if less than 5)
    step_1_accuracy = stats['steps'].get(1, {}).get('accuracy', 0)
    end_step = min(5, max_step)
    step_end_accuracy = stats['steps'].get(end_step, {}).get('accuracy', 0)
    
    if end_step > 1:
        # Slope in percentage points per step
        slope = (step_end_accuracy - step_1_accuracy) / (end_step - 1)
        stats['slope_1_to_5'] = slope
        stats['step_1_accuracy'] = step_1_accuracy
        stats['step_5_accuracy'] = step_end_accuracy
        stats['slope_end_step'] = end_step
    else:
        stats['slope_1_to_5'] = 0
        stats['step_1_accuracy'] = step_1_accuracy
        stats['step_5_accuracy'] = None
        stats['slope_end_step'] = 1
    
    return stats


def print_model_table(model_name: str, stats: Dict) -> str:
    """Generate a markdown table for a single model."""
    if not stats:
        return f"\n## {model_name}\n\nNo data available.\n"
    
    output = []
    output.append(f"\n## {model_name}\n")
    
    # Summary statistics
    slope = stats.get('slope_1_to_5', 0)
    step_1_acc = stats.get('step_1_accuracy', 0)
    step_5_acc = stats.get('step_5_accuracy')
    end_step = stats.get('slope_end_step', 1)
    
    output.append(f"**Accuracy Slope (Step 1 to {end_step})**: {slope:+.2f} pp/step")
    if step_5_acc is not None:
        output.append(f" (from {step_1_acc:.1f}% to {step_5_acc:.1f}%)\n")
    else:
        output.append(f" (only step 1: {step_1_acc:.1f}%)\n")
    
    # Table header
    output.append("\n| Step | Accuracy | Total Qs | 1-hop | 2-hop | 3-hop | 4-hop |")
    output.append("\n|------|----------|----------|-------|-------|-------|-------|")
    
    # Table rows
    max_step = stats['max_step']
    for step in range(1, max_step + 1):
        step_data = stats['steps'].get(step, {})
        accuracy = step_data.get('accuracy', 0)
        total = step_data.get('total', 0)
        hop_dist = step_data.get('hop_distribution', {})
        
        row = f"| {step} | {accuracy:.1f}% | {total} | {hop_dist.get(1, 0)} | {hop_dist.get(2, 0)} | {hop_dist.get(3, 0)} | {hop_dist.get(4, 0)} |"
        output.append(f"\n{row}")
    
    output.append("\n")
    
    return ''.join(output)


def generate_summary_table(all_stats: Dict[str, Dict]) -> str:
    """Generate a summary table showing slopes for all models."""
    output = []
    output.append("\n# Summary: Accuracy Slopes Across All Models\n")
    output.append("\nSorted by slope (steepest improvement first):\n")
    
    # Sort models by slope
    model_slopes = []
    for model, stats in all_stats.items():
        if stats:
            slope = stats.get('slope_1_to_5', 0)
            step_1 = stats.get('step_1_accuracy', 0)
            step_5 = stats.get('step_5_accuracy', 0)
            end_step = stats.get('slope_end_step', 1)
            model_slopes.append((model, slope, step_1, step_5, end_step))
    
    model_slopes.sort(key=lambda x: x[1], reverse=True)
    
    output.append("\n| Rank | Model | Slope (pp/step) | Step 1 Acc | Step 5 Acc | Notes |")
    output.append("\n|------|-------|-----------------|------------|------------|-------|")
    
    for rank, (model, slope, step_1, step_5, end_step) in enumerate(model_slopes, 1):
        if step_5 is not None:
            row = f"| {rank} | {model} | {slope:+.2f} | {step_1:.1f}% | {step_5:.1f}% | Step 1→{end_step} |"
        else:
            row = f"| {rank} | {model} | {slope:+.2f} | {step_1:.1f}% | N/A | Only 1 step |"
        output.append(f"\n{row}")
    
    output.append("\n")
    
    return ''.join(output)


def main():
    base_dir = Path(__file__).resolve().parents[2]  # Go up to project root
    output_dir = base_dir / "src" / "plots"
    output_dir.mkdir(exist_ok=True)
    
    print("Loading QA lookup...")
    qa_lookup = load_qa_lookup(base_dir)
    print(f"Loaded {len(qa_lookup)} questions with oracle hop counts")
    
    print("\nLoading no-context wrong questions...")
    wrong_questions = load_no_context_wrong_questions(base_dir)
    print(f"Loaded {len(wrong_questions)} no-context wrong questions")
    
    print("\nLoading model data...")
    model_data = load_model_data(base_dir, qa_lookup, wrong_questions)
    print(f"Loaded data for {len(model_data)} models")
    
    print("\nCalculating statistics...")
    all_stats = {}
    for model_name, (correct_steps, incorrect_steps) in model_data.items():
        stats = calculate_model_statistics(correct_steps, incorrect_steps)
        all_stats[model_name] = stats
        print(f"  {model_name}: {len(correct_steps) + len(incorrect_steps)} questions")
    
    # Generate output
    print("\nGenerating tables...")
    output_lines = []
    
    # Title
    output_lines.append("# Model Performance Tables: No-Context Wrong Questions\n")
    output_lines.append("\nData from: `all_models_correctness_by_steps_no_context_wrong_no_coverage.png`\n")
    output_lines.append("\nThese tables show:\n")
    output_lines.append("- **Accuracy Slope**: Change in accuracy from step 1 to step 5 (or max step)\n")
    output_lines.append("- **Oracle Hop Distribution**: Number of questions at each step, broken down by the number of hops in the gold reasoning path\n")
    output_lines.append("\nNote: Questions are filtered to only include those that were answered **incorrectly** in the no-context baseline.\n")
    
    # Summary table
    output_lines.append(generate_summary_table(all_stats))
    
    output_lines.append("\n---\n")
    output_lines.append("\n# Detailed Tables by Model\n")
    
    # Individual model tables (sorted alphabetically)
    for model_name in sorted(all_stats.keys()):
        stats = all_stats[model_name]
        output_lines.append(print_model_table(model_name, stats))
    
    # Write to file
    output_file = output_dir / "model_tables_no_context_wrong.md"
    with open(output_file, 'w') as f:
        f.write(''.join(output_lines))
    
    print(f"\n✓ Tables saved to: {output_file}")
    
    # Also print summary to console
    print("\n" + "="*80)
    print(generate_summary_table(all_stats))
    print("="*80)


if __name__ == "__main__":
    main()
