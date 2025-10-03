#!/usr/bin/env python3
"""Generate hop distribution plots for ALL questions (not just unanswered)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
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
    for key in ("raw", "raw_response"):
        raw = record.get(key)
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


def prepare_all_questions_stats(
    records: List[dict],
    question_hops_map: Dict[str, int],
    iterative_summary: Dict[str, dict],
) -> Tuple[List[int], List[int], List[int]]:
    """
    Prepare statistics for ALL questions (not just unanswered).
    
    Returns:
        hop_values: List of hop counts for all questions
        correct_steps: List of max source steps for correctly answered questions
        incorrect_steps: List of max source steps for incorrectly answered questions
    """
    hop_values: List[int] = []
    correct_steps: List[int] = []
    incorrect_steps: List[int] = []
    
    seen_questions = set()
    
    for record in records:
        question = extract_question(record)
        if not question or question in seen_questions:
            continue
        seen_questions.add(question)
        
        # Get hop count
        hop_count = question_hops_map.get(question)
        if hop_count:
            hop_values.append(hop_count)
        
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
    
    return hop_values, correct_steps, incorrect_steps


def plot_single_model_correctness(
    correct_steps: List[int],
    incorrect_steps: List[int],
    model_display_name: str,
    ax,
) -> None:
    """Plot correct vs incorrect by max source step for a single model."""
    all_step_values = correct_steps + incorrect_steps
    max_step = max(all_step_values) if all_step_values else 0
    step_ticks = list(range(1, max_step + 1)) if max_step else [1]
    
    x_positions = np.arange(len(step_ticks))
    bar_width = 0.35
    
    if not correct_steps and not incorrect_steps:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(model_display_name)
        return
    
    correct_counts = Counter(correct_steps)
    incorrect_counts = Counter(incorrect_steps)
    correct_heights = [correct_counts.get(step, 0) for step in step_ticks]
    incorrect_heights = [incorrect_counts.get(step, 0) for step in step_ticks]

    bars_correct = ax.bar(
        x_positions - bar_width / 2,
        correct_heights,
        bar_width,
        color="#2ca02c",
        label="Correct",
    )
    bars_incorrect = ax.bar(
        x_positions + bar_width / 2,
        incorrect_heights,
        bar_width,
        color="#d62728",
        label="Incorrect",
    )

    ax.bar_label(bars_correct, padding=3)
    ax.bar_label(bars_incorrect, padding=3)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(step_ticks)
    ax.set_xlim(-0.5, len(step_ticks) - 0.5)
    ax.set_xlabel("Max source step")
    ax.set_ylabel("Questions")
    ax.set_title(model_display_name)


def plot_combined_model_correctness(
    model_data: Dict[str, Tuple[List[int], List[int]]],
    output_path: Path,
) -> None:
    """Create a single plot with 6 subplots showing correctness by max source step for each model."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - external dependency
        raise SystemExit(
            "matplotlib is required for plotting. Install it with 'pip install matplotlib'."
        ) from exc

    # Create 2x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    model_names = list(model_data.keys())
    
    for idx, (model_name, (correct_steps, incorrect_steps)) in enumerate(model_data.items()):
        if idx < len(axes):
            plot_single_model_correctness(
                correct_steps,
                incorrect_steps,
                model_name,
                axes[idx]
            )
            
            # Add legend only to the first subplot
            if idx == 0:
                axes[idx].legend(loc="upper right")
    
    # Hide any unused subplots
    for idx in range(len(model_names), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle("Model Performance: Correct vs Incorrect by Max Source Step", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "model"


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    
    # Create a dedicated output directory for all-questions plots
    output_dir = base / "plots"
    output_dir.mkdir(exist_ok=True)

    # Source datasets - we'll use the full response files instead of unanswered subsets
    iterative_dir = base / "responses_reverified"

    # Load QA hop data
    qa_lookup: Dict[str, int] = {}
    qa_path = base / "docs" / "chemrxiv_qa.json"
    if qa_path.exists():
        try:
            with qa_path.open("r", encoding="utf-8") as handle:
                entries = json.load(handle)
        except json.JSONDecodeError:
            entries = []
        for entry in entries:
            question = entry.get("q")
            path_list = entry.get("path")
            if isinstance(question, str) and isinstance(path_list, list) and path_list:
                qa_lookup[question.strip()] = len(path_list)

    # Model files for iterative RAG
    model_files = {
        "responses_bedrock_mistral.mistral-large-2402-v1:0_reverified.jsonl": "Mistral Large 2402",
        "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning_reverified.jsonl": "Claude 3.7 Sonnet Thinking",
        "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0_reverified.jsonl": "Claude 3.7 Sonnet",
        "responses_bedrock_us.deepseek.r1-v1:0-reasoning_reverified.jsonl": "DeepSeek R1",
        "responses_openai_gpt-4o_reverified.jsonl": "GPT-4o",
        "responses_openai_gpt-5_reverified.jsonl": "GPT-5",
    }

    # Collect data for all models
    model_data: Dict[str, Tuple[List[int], List[int]]] = {}

    for filename, display_name in model_files.items():
        iterative_path = iterative_dir / filename
        if not iterative_path.exists():
            print(f"Skipping {display_name}: {iterative_path} not found")
            continue

        iterative_summary = load_iterative_summary(iterative_path)

        # Build question-to-hops mapping
        question_hops_map: Dict[str, int] = {}
        for question, data in iterative_summary.items():
            hop_value = data.get("raw_hops")
            if not isinstance(hop_value, int) or hop_value <= 0:
                hop_value = qa_lookup.get(question)
            if isinstance(hop_value, int) and hop_value > 0:
                question_hops_map[question] = max(1, min(4, hop_value))

        # Fall back entirely on QA hop data if needed
        if not question_hops_map and qa_lookup:
            question_hops_map = {k: max(1, min(4, v)) for k, v in qa_lookup.items()}

        # Load iterative RAG records
        all_records = load_records(iterative_path)
        
        # Get correctness data for this model
        hop_values, correct_steps, incorrect_steps = prepare_all_questions_stats(
            all_records,
            question_hops_map,
            iterative_summary,
        )
        
        model_data[display_name] = (correct_steps, incorrect_steps)

    # Generate the single combined plot with 6 subplots
    if model_data:
        output_path = output_dir / "all_models_correctness_by_steps.png"
        
        plot_combined_model_correctness(
            model_data,
            output_path,
        )

        print(f"Generated combined model correctness plot: {output_path}")
        
        # Remove the old individual plots
        old_patterns = [
            "all_questions_hop_distributions_*.png"
        ]
        
        import glob
        for pattern in old_patterns:
            for old_file in glob.glob(str(output_dir / pattern)):
                old_path = Path(old_file)
                if old_path.exists():
                    old_path.unlink()
                    print(f"Removed: {old_path}")
    else:
        print("No iterative response files found for plotting.")


if __name__ == "__main__":
    main()