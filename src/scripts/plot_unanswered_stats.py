#!/usr/bin/env python3
"""Generate plots summarizing unanswered questions and hop statistics."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


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


def prepare_category_stats(
    records: Iterable[dict],
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
        if not hops and folder_lookup:
            question = extract_question(record)
            if question and question in folder_lookup:
                hops.extend(folder_lookup[question])
        if not hops:
            attempts = record.get("model_attempts")
            if isinstance(attempts, list):
                for attempt in attempts:
                    hop_value = attempt.get("number_of_hops")
                    if isinstance(hop_value, (int, float)):
                        hops.append(int(round(hop_value)))
        if not hops and fallback_lookup:
            question = extract_question(record)
            if question and question in fallback_lookup:
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
    categories: List[str], hop_values: List[List[int]], output_path: Path
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - external dependency
        raise SystemExit(
            "matplotlib is required for plotting. Install it with 'pip install matplotlib'."
        ) from exc

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    hop_bins = [1, 2, 3, 4]

    for ax, category, values in zip(axes, categories, hop_values):
        ax.set_title(category)
        ax.set_xlabel("Number of Hops")
        ax.set_ylabel("Unanswered Questions")
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
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    report_root = base / "results" / "unanswered_questions"

    datasets: Dict[str, Path] = {
        "Iterative RAG": report_root / "responses_reverified_unanswered.jsonl",
        "Without Context": report_root / "response-jsonl-without-context_unanswered.jsonl",
        "Gold Context": report_root / "response-jsonl-with-context_unanswered.jsonl",
    }

    # Preload reference data from the iterative RAG run to reuse hop counts
    folder_lookups = {
        label: build_folder_hops(path)
        for label, path in datasets.items()
    }

    raw_responses_dir = base / "responses"
    reference_hops = build_reference_hops(raw_responses_dir)

    reference_records = load_records(datasets["Iterative RAG"])

    categories: List[str] = []
    unanswered_counts: List[int] = []
    hop_distributions: List[List[int]] = []

    for label, path in datasets.items():
        if label == "Iterative RAG":
            records = reference_records
            folder_lookup = folder_lookups[label]
            fallback = None
        else:
            records = load_records(path)
            folder_lookup = folder_lookups[label]
            fallback = reference_hops
        unanswered, hop_values = prepare_category_stats(records, folder_lookup, fallback)
        categories.append(label)
        unanswered_counts.append(unanswered)
        hop_distributions.append(hop_values)

    plot_unanswered(
        categories,
        unanswered_counts,
        report_root / "unanswered_counts.png",
    )

    plot_hop_distributions(
        categories,
        hop_distributions,
        report_root / "hop_distributions.png",
    )

    print("Generated plots:")
    print(f"- {report_root / 'unanswered_counts.png'}")
    print(f"- {report_root / 'hop_distributions.png'}")


if __name__ == "__main__":
    main()
