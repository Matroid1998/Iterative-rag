#!/usr/bin/env python3
"""Generate hop distribution plots for hard questions only."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Set
import re
import numpy as np

from config import get_iterative_model_entries


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


def load_hard_questions(hard_questions_path: Path) -> Set[str]:
    """Load the set of hard questions from hard_question_categories.json."""
    if not hard_questions_path.exists():
        return set()
    
    hard_questions = set()
    with hard_questions_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
        
    for category, questions in data.items():
        for q_data in questions:
            question = q_data.get("question")
            if question:
                hard_questions.add(question.strip())
    
    return hard_questions


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

        summary[question] = {
            "is_correct": is_correct,
            "raw_hops": raw_hops,
            "max_source_step": max_source_step,
        }

    return summary


def prepare_hard_questions_by_category(
    records: List[dict],
    hard_questions_by_category: Dict[str, Set[str]],
    qa_hops: Dict[str, int],
    iterative_summary: Dict[str, dict],
) -> Dict[str, Tuple[List[int], List[int], List[int]]]:
    """
    Prepare statistics for hard questions grouped by category.
    
    Returns:
        Dict mapping category -> (hop_values, correct_steps, incorrect_steps)
    """
    result = {}
    
    for category, hard_questions in hard_questions_by_category.items():
        hop_values: List[int] = []
        correct_steps: List[int] = []
        incorrect_steps: List[int] = []
        
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
            
            # Get correctness and step info from iterative summary
            if question in iterative_summary:
                summary = iterative_summary[question]
                max_step = summary.get("max_source_step")
                is_correct = summary.get("is_correct", False)
                
                if max_step is not None:
                    if is_correct:
                        correct_steps.append(max_step)
                    else:
                        incorrect_steps.append(max_step)
        
        result[category] = (hop_values, correct_steps, incorrect_steps)
    
    return result


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


def plot_hard_questions_by_categories(
    model_data: Dict[str, Dict[str, Tuple[List[int], List[int], List[int]]]],
    qa_hops: Dict[str, int],
    output_path: Path,
) -> None:
    """Create a plot with columns for categories (9,10,11) and one row per model."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - external dependency
        raise SystemExit(
            "matplotlib is required for plotting. Install it with 'pip install matplotlib'."
        ) from exc

    categories = ["9", "10", "11"]
    models = list(model_data.keys())

    if not categories or not models:
        print("No data available for hard question categories plot")
        return

    rows = len(models) + 1
    cols = len(categories)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows))

    hop_bins = [1, 2, 3, 4]

    combined_data = {}
    for category in categories:
        combined_hop_values: List[int] = []
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
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

        ax.set_xticks(hop_bins)
        ax.set_xlim(0.5, 4.5)
        ax.set_xlabel("Number of Hops")
        ax.grid(axis='y', alpha=0.3)

    for row, model in enumerate(models, start=1):
        for col, category in enumerate(categories):
            ax = axes[row, col]
            hop_values, correct_steps, incorrect_steps = model_data.get(model, {}).get(category, ([], [], []))

            if col == 0:
                ax.set_ylabel(model, fontweight='bold', fontsize=10)

            all_steps = correct_steps + incorrect_steps
            if all_steps:
                max_step = max(all_steps)
                step_ticks = list(range(1, max_step + 1))
                x_positions = np.arange(len(step_ticks))
                bar_width = 0.35

                correct_counts = Counter(correct_steps)
                incorrect_counts = Counter(incorrect_steps)
                correct_heights = [correct_counts.get(step, 0) for step in step_ticks]
                incorrect_heights = [incorrect_counts.get(step, 0) for step in step_ticks]

                bars_correct = ax.bar(
                    x_positions - bar_width / 2,
                    correct_heights,
                    bar_width,
                    color="#2ca02c",
                    label="Solved" if row == 1 and col == 0 else "",
                )
                bars_incorrect = ax.bar(
                    x_positions + bar_width / 2,
                    incorrect_heights,
                    bar_width,
                    color="#d62728",
                    label="Unresolved" if row == 1 and col == 0 else "",
                )

                ax.bar_label(bars_correct, padding=2, fontsize=8)
                ax.bar_label(bars_incorrect, padding=2, fontsize=8)
                ax.set_xticks(x_positions)
                ax.set_xticklabels(step_ticks)
                ax.set_xlim(-0.5, len(step_ticks) - 0.5)

                if row == 1 and col == 0:
                    ax.legend(loc="upper right", fontsize=9)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=10, color='gray')
                ax.set_xticks([])
                ax.set_yticks([])

            ax.set_xlabel("Max source step")
            ax.grid(axis='y', alpha=0.3)

    plt.suptitle(
        "Hard Questions: Hop Distribution and Model Performance by Category",
        fontsize=16,
        fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    base = Path(__file__).resolve().parents[3]
    
    # Create output directory
    output_dir = base.parent / "data" / "plots" / "general"
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
    qa_path = base.parent / "data" / "corpus" / "chemrxiv_qa.json"
    qa_hops = load_qa_hops(qa_path)
    print(f"Loaded hop data for {len(qa_hops)} questions")
    
    # Collect data for each model
    model_data = {}
    
    for iterative_path, display_name in get_iterative_model_entries():
        if not iterative_path.exists():
            print(f"Skipping {display_name}: {iterative_path} not found")
            continue
        
        iterative_summary = load_iterative_summary(iterative_path)
        all_records = load_records(iterative_path)
        
        # Get data for each category for this model
        model_category_data = prepare_hard_questions_by_category(
            all_records,
            hard_questions_by_category,
            qa_hops,
            iterative_summary,
        )
        
        model_data[display_name] = model_category_data
    
    # Create the plot
    output_path = output_dir / "hard_questions_categories_by_models.png"
    
    plot_hard_questions_by_categories(
        model_data,
        qa_hops,
        output_path,
    )
    
    print(f"Generated hard questions categories plot: {output_path}")
    
    # Print some statistics
    print(f"\nStatistics by category:")
    for category in ["9", "10", "11"]:
        total_attempts = 0
        total_correct = 0
        total_incorrect = 0
        total_hop_data = 0
        
        for model_name, category_data in model_data.items():
            if category in category_data:
                hop_values, correct_steps, incorrect_steps = category_data[category]
                total_attempts += len(correct_steps) + len(incorrect_steps)
                total_correct += len(correct_steps)
                total_incorrect += len(incorrect_steps)
                total_hop_data += len(hop_values)
        
        print(f"Category {category} ({len(hard_questions_by_category.get(category, []))} unique questions):")
        print(f"  Total attempts: {total_attempts}")
        print(f"  Correct: {total_correct}")
        print(f"  Incorrect: {total_incorrect}")
        print(f"  Questions with hop data: {total_hop_data}")
        if total_attempts > 0:
            accuracy = (total_correct / total_attempts) * 100
            print(f"  Accuracy: {accuracy:.1f}%")


if __name__ == "__main__":
    main()
