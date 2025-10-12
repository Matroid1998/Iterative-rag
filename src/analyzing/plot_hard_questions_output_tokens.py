#!/usr/bin/env python3
"""Generate hard questions plot showing output tokens instead of question counts."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Set
import re
import numpy as np


def load_records(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
    return records


def iter_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError:
                continue


def extract_question(record: dict) -> str | None:
    question = record.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()
    
    # Check in raw_response
    raw_response = record.get("raw_response")
    if isinstance(raw_response, dict):
        q = raw_response.get("question")
        if isinstance(q, str) and q.strip():
            return q.strip()
    
    # Check in raw
    raw = record.get("raw")
    if isinstance(raw, dict):
        q = raw.get("question")
        if isinstance(q, str) and q.strip():
            return q.strip()
    
    return None


def extract_max_source_step(record: dict) -> int | None:
    """Return the maximum retrieval step (source_step) found in a record."""
    steps: List[int] = []
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
    return None


def extract_output_tokens(record: dict) -> int | None:
    """Extract output_tokens from the record."""
    output_tokens = record.get("output_tokens")
    if isinstance(output_tokens, (int, float)):
        return int(output_tokens)
    return None


def load_hard_questions_by_category(hard_questions_path: Path) -> Dict[str, Set[str]]:
    """Load hard questions grouped by category."""
    if not hard_questions_path.exists():
        return {}
    
    hard_questions_by_category = {}
    with hard_questions_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
        
    for category, questions in data.items():
        hard_questions_set = set()
        for q_data in questions:
            question = q_data.get("question")
            if question:
                hard_questions_set.add(question.strip())
        hard_questions_by_category[category] = hard_questions_set
    
    return hard_questions_by_category


def load_qa_hops(qa_path: Path) -> Dict[str, int]:
    """Load question-to-hops mapping from chemrxiv_qa.json."""
    qa_lookup: Dict[str, int] = {}
    if not qa_path.exists():
        return qa_lookup
        
    try:
        with qa_path.open("r", encoding="utf-8") as handle:
            entries = json.load(handle)
    except json.JSONDecodeError:
        return qa_lookup
        
    for entry in entries:
        question = entry.get("q")
        path_list = entry.get("path")
        if isinstance(question, str) and isinstance(path_list, list) and path_list:
            qa_lookup[question.strip()] = len(path_list)
    
    return qa_lookup


def load_iterative_summary(path: Path) -> Dict[str, dict]:
    """Load iterative RAG results and build question summary."""
    if not path.exists():
        return {}

    summary: Dict[str, dict] = {}
    for record in iter_records(path):
        question = extract_question(record)
        if not question:
            continue

        is_correct = bool(record.get("is_correct", False))
        raw_hops = record.get("number_of_hops")
        max_source_step = extract_max_source_step(record)
        output_tokens = extract_output_tokens(record)

        summary[question] = {
            "is_correct": is_correct,
            "raw_hops": raw_hops,
            "max_source_step": max_source_step,
            "output_tokens": output_tokens,
        }

    return summary


def prepare_hard_questions_tokens_by_category(
    records: List[dict],
    hard_questions_by_category: Dict[str, Set[str]],
    qa_hops: Dict[str, int],
    iterative_summary: Dict[str, dict],
) -> Dict[str, Tuple[List[int], Dict[int, List[int]], Dict[int, List[int]]]]:
    """
    Prepare token statistics for hard questions grouped by category.
    
    Returns:
        Dict mapping category -> (hop_values, correct_tokens_by_step, incorrect_tokens_by_step)
        where correct_tokens_by_step[step] = [list of output tokens for correct answers at that step]
    """
    result = {}
    
    for category, hard_questions in hard_questions_by_category.items():
        hop_values: List[int] = []
        correct_tokens_by_step: Dict[int, List[int]] = defaultdict(list)
        incorrect_tokens_by_step: Dict[int, List[int]] = defaultdict(list)
        
        seen_questions = set()
        
        for record in records:
            question = extract_question(record)
            if not question or question in seen_questions:
                continue
            seen_questions.add(question)
            
            # Only process if this is a hard question in this category
            if question not in hard_questions:
                continue
            
            # Get hop count from chemrxiv_qa.json
            hop_count = qa_hops.get(question)
            if hop_count:
                hop_values.append(min(4, hop_count))  # Cap at 4 for binning
            
            # Get correctness, step info, and tokens from iterative summary
            if question in iterative_summary:
                summary = iterative_summary[question]
                max_step = summary.get("max_source_step")
                is_correct = summary.get("is_correct", False)
                output_tokens = summary.get("output_tokens")
                
                if max_step is not None and output_tokens is not None:
                    if is_correct:
                        correct_tokens_by_step[max_step].append(output_tokens)
                    else:
                        incorrect_tokens_by_step[max_step].append(output_tokens)
        
        result[category] = (hop_values, dict(correct_tokens_by_step), dict(incorrect_tokens_by_step))
    
    return result


def plot_hard_questions_tokens_by_categories(
    model_data: Dict[str, Dict[str, Tuple[List[int], Dict[int, List[int]], Dict[int, List[int]]]]],
    qa_hops: Dict[str, int],
    output_path: Path,
) -> None:
    """Create a plot with 3 columns (categories 4,5,6) and 7 rows showing output tokens."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - external dependency
        raise SystemExit(
            "matplotlib is required for plotting. Install it with 'pip install matplotlib'."
        ) from exc

    categories = ["4", "5", "6"]
    models = list(model_data.keys())
    
    # Create 7x3 subplot layout (1 overview + 6 models)
    fig, axes = plt.subplots(7, 3, figsize=(15, 20))
    
    hop_bins = [1, 2, 3, 4]
    
    # First row: Overall hop distribution for each category
    combined_data = {}
    for category in categories:
        combined_hop_values = []
        for model in models:
            if category in model_data[model]:
                hop_values, _, _ = model_data[model][category]
                combined_hop_values.extend(hop_values)
        combined_data[category] = combined_hop_values
    
    for col, category in enumerate(categories):
        ax = axes[0, col]
        ax.set_title(f"{category} models wrong", fontweight='bold', fontsize=12)
        
        if col == 0:
            ax.set_ylabel("Hard questions", fontweight='bold')
        
        hop_values = combined_data[category]
        if hop_values:
            counts = Counter(hop_values)
            heights = [counts.get(bin_value, 0) for bin_value in hop_bins]
            bars = ax.bar(hop_bins, heights, color="#7f7f7f", alpha=0.8)
            ax.bar_label(bars, padding=2, fontsize=10, fontweight='bold')
        
        ax.set_xticks(hop_bins)
        ax.set_xlim(0.5, 4.5)
        ax.set_xlabel("Number of Hops")
        ax.grid(axis='y', alpha=0.3)
    
    # Rows 1-6: Individual models with correct/incorrect token totals by max source step
    for row, model in enumerate(models, start=1):
        for col, category in enumerate(categories):
            ax = axes[row, col]
            
            # Get data for this model and category
            if category in model_data[model]:
                hop_values, correct_tokens_by_step, incorrect_tokens_by_step = model_data[model][category]
            else:
                hop_values, correct_tokens_by_step, incorrect_tokens_by_step = [], {}, {}
            
            # Add model name as row label on the left side
            if col == 0:  # Only for the first column
                # Position the text to the left of the subplot
                fig.text(0.02, 0.85 - (row * 0.12), model, 
                        rotation=90, ha='center', va='center', 
                        fontweight='bold', fontsize=11)
            
            # Find the range of steps
            all_steps = list(correct_tokens_by_step.keys()) + list(incorrect_tokens_by_step.keys())
            if all_steps:
                max_step = max(all_steps)
                step_ticks = list(range(1, max_step + 1))
                x_positions = np.arange(len(step_ticks))
                bar_width = 0.35
                
                # Sum tokens for each step
                correct_token_sums = []
                incorrect_token_sums = []
                
                for step in step_ticks:
                    correct_tokens = correct_tokens_by_step.get(step, [])
                    incorrect_tokens = incorrect_tokens_by_step.get(step, [])
                    correct_token_sums.append(sum(correct_tokens))
                    incorrect_token_sums.append(sum(incorrect_tokens))
                
                # Plot bars
                bars_correct = ax.bar(
                    x_positions - bar_width / 2,
                    correct_token_sums,
                    bar_width,
                    color="#2ca02c",
                    label="Solved" if row == 1 and col == 0 else "",
                )
                bars_incorrect = ax.bar(
                    x_positions + bar_width / 2,
                    incorrect_token_sums,
                    bar_width,
                    color="#d62728",
                    label="Unresolved" if row == 1 and col == 0 else "",
                )
                
                # Add value labels
                ax.bar_label(bars_correct, padding=2, fontsize=8)
                ax.bar_label(bars_incorrect, padding=2, fontsize=8)
                
                ax.set_xticks(x_positions)
                ax.set_xticklabels(step_ticks)
                ax.set_xlim(-0.5, len(step_ticks) - 0.5)
                
                # Add legend only to first subplot
                if row == 1 and col == 0:
                    ax.legend(loc="upper right", fontsize=9)
            else:
                # No data for this model/category combination
                ax.text(0.5, 0.5, "No data", ha="center", va="center", 
                       transform=ax.transAxes, fontsize=10, color='gray')
                ax.set_xticks([])
                ax.set_yticks([])
            
            ax.set_xlabel("Max source step")
            if col == 0:
                ax.set_ylabel("Output tokens", fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle("Hard Questions: Output Tokens by Category and Model Performance", 
                 fontsize=16, fontweight='bold')
    # Adjust layout to make room for model labels on the left
    plt.tight_layout(rect=[0.13, 0.02, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    
    # Create output directory
    output_dir = base / "plots"
    output_dir.mkdir(exist_ok=True)
    
    # Load hard questions by category
    hard_questions_path = base / "results" / "unanswered_questions" / "hard_question_categories.json"
    hard_questions_by_category = load_hard_questions_by_category(hard_questions_path)
    
    total_hard_questions = sum(len(qs) for qs in hard_questions_by_category.values())
    print(f"Loaded hard questions by category:")
    for cat, questions in hard_questions_by_category.items():
        print(f"  Category {cat}: {len(questions)} questions")
    print(f"  Total: {total_hard_questions} hard questions")
    
    # Load QA hop data
    qa_path = base / "docs" / "chemrxiv_qa.json"
    qa_hops = load_qa_hops(qa_path)
    print(f"Loaded hop data for {len(qa_hops)} questions")
    
    # Iterative RAG directory
    iterative_dir = base / "responses_reverified"
    
    # Model files
    model_files = {
        "responses_bedrock_mistral.mistral-large-2402-v1:0_reverified.jsonl": "Mistral Large 2402",
        "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning_reverified.jsonl": "Claude 3.7 Sonnet Thinking",
        "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0_reverified.jsonl": "Claude 3.7 Sonnet",
        "responses_bedrock_us.deepseek.r1-v1:0-reasoning_reverified.jsonl": "DeepSeek R1",
        "responses_openai_gpt-4o_reverified.jsonl": "GPT-4o",
        "responses_openai_gpt-5_reverified.jsonl": "GPT-5",
    }
    
    # Collect data for each model
    model_data = {}
    
    for filename, display_name in model_files.items():
        iterative_path = iterative_dir / filename
        if not iterative_path.exists():
            print(f"Skipping {display_name}: {iterative_path} not found")
            continue
        
        iterative_summary = load_iterative_summary(iterative_path)
        all_records = load_records(iterative_path)
        
        # Get data for each category for this model
        model_category_data = prepare_hard_questions_tokens_by_category(
            all_records,
            hard_questions_by_category,
            qa_hops,
            iterative_summary,
        )
        
        model_data[display_name] = model_category_data
    
    # Create the plot
    output_path = output_dir / "hard_questions_output_tokens_by_categories.png"
    
    plot_hard_questions_tokens_by_categories(
        model_data,
        qa_hops,
        output_path,
    )
    
    print(f"Generated hard questions output tokens plot: {output_path}")
    
    # Print some statistics
    print(f"\nOutput token statistics by category:")
    for category in ["4", "5", "6"]:
        total_correct_tokens = 0
        total_incorrect_tokens = 0
        total_questions_correct = 0
        total_questions_incorrect = 0
        
        for model_name, category_data in model_data.items():
            if category in category_data:
                hop_values, correct_tokens_by_step, incorrect_tokens_by_step = category_data[category]
                
                for step, tokens_list in correct_tokens_by_step.items():
                    total_correct_tokens += sum(tokens_list)
                    total_questions_correct += len(tokens_list)
                
                for step, tokens_list in incorrect_tokens_by_step.items():
                    total_incorrect_tokens += sum(tokens_list)
                    total_questions_incorrect += len(tokens_list)
        
        print(f"Category {category} ({len(hard_questions_by_category.get(category, []))} unique questions):")
        print(f"  Correct answers: {total_questions_correct} attempts, {total_correct_tokens:,} total tokens")
        print(f"  Incorrect answers: {total_questions_incorrect} attempts, {total_incorrect_tokens:,} total tokens")
        if total_questions_correct > 0:
            avg_correct = total_correct_tokens / total_questions_correct
            print(f"  Average tokens per correct answer: {avg_correct:.1f}")
        if total_questions_incorrect > 0:
            avg_incorrect = total_incorrect_tokens / total_questions_incorrect
            print(f"  Average tokens per incorrect answer: {avg_incorrect:.1f}")


if __name__ == "__main__":
    main()