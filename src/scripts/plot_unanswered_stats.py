#!/usr/bin/env python3
"""Generate plots summarizing unanswered questions and hop statistics."""

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


def build_reference_hops(folder: Path) -> Dict[str, List[int]]:
    lookup: Dict[str, List[int]] = {}
    for path in folder.glob("*.jsonl"):
        for record in iter_records(path):
            question = extract_question(record)
            if not question:
                continue
            hop_value = record.get("number_of_hops")
            if isinstance(hop_value, (int, float)):
                lookup.setdefault(question, []).append(int(round(hop_value)))
    return lookup


def build_folder_hops(folder: Path) -> Dict[str, List[int]]:
    lookup: Dict[str, List[int]] = {}
    for path in folder.glob("*.jsonl"):
        for record in iter_records(path):
            question = extract_question(record)
            if not question:
                continue
            hop_value = record.get("number_of_hops")
            if isinstance(hop_value, (int, float)):
                lookup.setdefault(question, []).append(int(round(hop_value)))
    return lookup


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


def load_iterative_summary(path: Path) -> Dict[str, Dict[str, object]]:
    """Load iterative RAG records keyed by question for quick lookup."""

    summary: Dict[str, Dict[str, object]] = {}
    if not path.exists():
        return summary

    for record in iter_records(path):
        question = extract_question(record)
        if not question:
            continue
        is_correct = bool(record.get("is_correct"))
        max_step = extract_max_source_step(record)
        raw_hops = None
        for key in ("raw", "raw_response"):
            raw_section = record.get(key)
            if isinstance(raw_section, dict):
                hop_value = raw_section.get("number_of_hops")
                if isinstance(hop_value, (int, float)):
                    raw_hops = int(round(hop_value))
                    break
        if raw_hops is None:
            hop_value = record.get("number_of_hops")
            if isinstance(hop_value, (int, float)):
                raw_hops = int(round(hop_value))
        existing = summary.get(question)
        if existing is None:
            summary[question] = {
                "is_correct": is_correct,
                "max_source_step": max_step,
                "raw_hops": raw_hops,
            }
            continue
        if is_correct and not existing.get("is_correct"):
            summary[question] = {
                "is_correct": True,
                "max_source_step": max_step,
                "raw_hops": raw_hops if raw_hops is not None else existing.get("raw_hops"),
            }
            continue
        if is_correct == existing.get("is_correct") and existing.get("max_source_step") is None:
            summary[question]["max_source_step"] = max_step
        if existing.get("raw_hops") is None and raw_hops is not None:
            summary[question]["raw_hops"] = raw_hops

    return summary


def prepare_category_stats(
    records: Iterable[dict],
    question_hops: Dict[str, int] | None,
    folder_lookup: Dict[str, List[int]] | None,
    fallback_lookup: Dict[str, List[int]] | None,
) -> Tuple[int, List[int]]:
    unanswered = 0
    hop_counts: List[int] = []
    for record in records:
        unanswered += 1
        hops: List[int] = []
        hop_map = record.get("number_of_hops")
        if isinstance(hop_map, dict):
            hops.extend(
                int(round(value))
                for value in hop_map.values()
                if isinstance(value, (int, float))
            )
        question = extract_question(record)
        if not hops and question and question_hops:
            hop_value = question_hops.get(question)
            if isinstance(hop_value, int):
                hops.append(hop_value)
        if not hops and folder_lookup and question:
            if question in folder_lookup:
                hops.extend(folder_lookup[question])
        if not hops:
            attempts = record.get("model_attempts")
            if isinstance(attempts, list):
                for attempt in attempts:
                    hop_value = attempt.get("number_of_hops")
                    if isinstance(hop_value, (int, float)):
                        hops.append(int(round(hop_value)))
        if not hops and fallback_lookup and question:
            if question in fallback_lookup:
                hops.extend(fallback_lookup[question])
        if hops:
            avg = sum(hops) / len(hops)
            hop_counts.append(max(1, min(4, int(round(avg)))))
    return unanswered, hop_counts


def plot_unanswered(categories: List[str], counts: List[int], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - external dependency
        raise SystemExit(
            "matplotlib is required for plotting. Install it with 'pip install matplotlib'."
        ) from exc

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(categories, counts, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title("Unanswered Questions per Category")
    ax.set_ylabel("Unanswered Questions")
    ax.bar_label(bars, padding=3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_hop_distributions(
    categories: List[str],
    hop_map: Dict[str, List[int]],
    solved_steps: Dict[str, List[int]],
    unsolved_steps: Dict[str, List[int]],
    display_titles: Dict[str, str],
    step_axis_label: str,
    model_display_name: str,
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - external dependency
        raise SystemExit(
            "matplotlib is required for plotting. Install it with 'pip install matplotlib'."
        ) from exc

    ncols = len(categories)
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 9), sharey=False)
    if ncols == 1:
        axes = axes.reshape(2, 1)

    hop_bins = [1, 2, 3, 4]
    all_step_values: List[int] = []
    for data_map in (solved_steps, unsolved_steps):
        for values in data_map.values():
            all_step_values.extend(values)
    max_step = max(all_step_values) if all_step_values else 0
    step_ticks = list(range(1, max_step + 1)) if max_step else [1]

    # First row – unanswered counts by hops (original view)
    for col, category in enumerate(categories):
        ax = axes[0, col]
        ax.set_title(display_titles.get(category, category))
        ax.set_xlabel("Number of Hops")
        if col == 0:
            ax.set_ylabel("Unanswered questions")
        values = hop_map.get(category, [])
        if not values:
            ax.text(0.5, 0.5, "No hop data", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        counts = Counter(values)
        heights = [counts.get(bin_value, 0) for bin_value in hop_bins]
        bars = ax.bar(hop_bins, heights, color="#1f77b4")
        ax.bar_label(bars, padding=3)
        ax.set_xticks(hop_bins)
        ax.set_xlim(0.5, 4.5)

    # Second row – questions recovered by iterative RAG
    x_positions = np.arange(len(step_ticks))
    bar_width = 0.35

    for col, category in enumerate(categories):
        ax = axes[1, col]
        if col == 0:
            ax.set_ylabel("Questions by source step")
        ax.set_xlabel(step_axis_label)
        solved_values = solved_steps.get(category, [])
        unsolved_values = unsolved_steps.get(category, [])
        if not solved_values and not unsolved_values:
            label = "Not applicable" if category == "Iterative RAG" else "No matching questions"
            ax.text(0.5, 0.5, label, ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        solved_counts = Counter(solved_values)
        unsolved_counts = Counter(unsolved_values)
        solved_heights = [solved_counts.get(step, 0) for step in step_ticks]
        unsolved_heights = [unsolved_counts.get(step, 0) for step in step_ticks]

        bars_solved = ax.bar(
            x_positions - bar_width / 2,
            solved_heights,
            bar_width,
            color="#2ca02c",
            label="Solved",
        )
        bars_unsolved = ax.bar(
            x_positions + bar_width / 2,
            unsolved_heights,
            bar_width,
            color="#d62728",
            label="Unresolved",
        )

        ax.bar_label(bars_solved, padding=3)
        ax.bar_label(bars_unsolved, padding=3)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(step_ticks)
        ax.set_xlim(-0.5, len(step_ticks) - 0.5)
        if col == 0:
            ax.legend(loc="upper right")

    plt.tight_layout(h_pad=2.0, w_pad=2.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "model"


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    report_root = base / "results" / "unanswered_questions"

    datasets: Dict[str, Path] = {
        "Without Context": report_root / "response-jsonl-without-context_unanswered.jsonl",
        "Gold Context": report_root / "response-jsonl-with-context_unanswered.jsonl",
        "Iterative RAG": report_root / "responses_reverified_unanswered.jsonl",
    }
    category_order = ["Without Context", "Gold Context", "Iterative RAG"]

    records_map: Dict[str, List[dict]] = {
        label: load_records(path) for label, path in datasets.items()
    }

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

    iterative_dir = base / "responses_reverified"
    model_files = {
        "responses_bedrock_mistral.mistral-large-2402-v1:0_reverified.jsonl": "Mistral Large 2402",
        "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning_reverified.jsonl": "Claude 3.7 Sonnet Thinking",
        "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0_reverified.jsonl": "Claude 3.7 Sonnet",
        "responses_bedrock_us.deepseek.r1-v1:0-reasoning_reverified.jsonl": "DeepSeek R1",
        "responses_openai_gpt-4o_reverified.jsonl": "GPT-4o",
        "responses_openai_gpt-5_reverified.jsonl": "GPT-5",
    }

    generated: List[Tuple[str, Path, Path]] = []

    for filename, display_name in model_files.items():
        iterative_path = iterative_dir / filename
        if not iterative_path.exists():
            continue

        slug = slugify(display_name)
        iterative_summary = load_iterative_summary(iterative_path)

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

        hop_values_map: Dict[str, List[int]] = {}
        unanswered_counts_map: Dict[str, int] = {}
        questions_by_category: Dict[str, List[str]] = {}

        for label in category_order:
            records = records_map.get(label, [])
            unanswered, hop_values = prepare_category_stats(
                records,
                question_hops_map,
                None,
                None,
            )
            hop_values_map[label] = hop_values
            unanswered_counts_map[label] = unanswered

            questions: List[str] = []
            seen: set[str] = set()
            for record in records:
                question = extract_question(record)
                if question and question not in seen:
                    seen.add(question)
                    questions.append(question)
            questions_by_category[label] = questions

        question_step_map: Dict[str, int] = {}
        for question, data in iterative_summary.items():
            max_step = data.get("max_source_step")
            if isinstance(max_step, int) and max_step > 0:
                question_step_map[question] = max_step

        solved_steps_map: Dict[str, List[int]] = {}
        unsolved_steps_map: Dict[str, List[int]] = {}

        for label, questions in questions_by_category.items():
            solved_steps: List[int] = []
            unsolved_steps: List[int] = []
            for question in questions:
                summary = iterative_summary.get(question)
                if not summary:
                    continue
                max_step = question_step_map.get(question)
                if summary.get("is_correct"):
                    if max_step is not None:
                        solved_steps.append(int(max_step))
                else:
                    if max_step is not None:
                        unsolved_steps.append(int(max_step))
            solved_steps_map[label] = solved_steps
            unsolved_steps_map[label] = unsolved_steps

        unanswered_counts = [unanswered_counts_map[label] for label in category_order]

        display_titles = {
            "Without Context": "Without Context",
            "Gold Context": "Gold Context",
            "Iterative RAG": f"Iterative RAG ({display_name})",
        }
        step_axis_label = f"Max source step ({display_name})"

        counts_path = report_root / f"unanswered_counts_{slug}.png"
        hops_path = report_root / f"hop_distributions_{slug}.png"

        plot_unanswered(category_order, unanswered_counts, counts_path)
        plot_hop_distributions(
            category_order,
            hop_values_map,
            solved_steps_map,
            unsolved_steps_map,
            display_titles,
            step_axis_label,
            display_name,
            hops_path,
        )

        generated.append((display_name, counts_path, hops_path))

    if generated:
        print("Generated plots:")
        for name, counts_path, hops_path in generated:
            print(f"- {name}: {counts_path}")
            print(f"- {name}: {hops_path}")
    else:
        print("No iterative response files found for plotting.")


if __name__ == "__main__":
    main()
