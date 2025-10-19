"""Generate stacked bar plots showing unanswered questions broken down by number of hops."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


def load_records(path: Path) -> List[dict]:
    """Load JSONL records from a file."""
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
    """Iterate over JSONL records from a file."""
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
    """Extract the question text from a record."""
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


def extract_number_of_hops(record: dict) -> int | None:
    """Extract number of hops from a record, checking multiple locations."""
    # Check top-level number_of_hops
    hop_value = record.get("number_of_hops")
    if isinstance(hop_value, (int, float)):
        return int(round(hop_value))

    # Check raw.number_of_hops or raw_response.number_of_hops
    for key in ("raw", "raw_response"):
        raw = record.get(key)
        if isinstance(raw, dict):
            hop_value = raw.get("number_of_hops")
            if isinstance(hop_value, (int, float)):
                return int(round(hop_value))

    return None


def load_qa_hops(qa_path: Path) -> Dict[str, int]:
    """Load question to hop count mapping from chemrxiv_qa.json."""
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


def collect_hop_counts_by_category(
    records: List[dict],
    qa_lookup: Dict[str, int]
) -> Dict[int, int]:
    """Collect hop counts for unanswered questions in a category."""
    hop_counter: Dict[int, int] = defaultdict(int)

    for record in records:
        # Try to get hops from the record itself
        hops = extract_number_of_hops(record)

        # Fall back to QA lookup if not found in record
        if hops is None:
            question = extract_question(record)
            if question and question in qa_lookup:
                hops = qa_lookup[question]

        # Clamp to 1-4 range
        if hops is not None:
            hops = max(1, min(4, hops))
            hop_counter[hops] += 1

    return hop_counter


def plot_stacked_bar_by_hops(
    categories: List[str],
    hop_data: Dict[str, Dict[int, int]],
    output_path: Path
) -> None:
    """Generate a stacked bar plot showing unanswered questions by hop count."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "matplotlib and numpy are required for plotting. "
            "Install them with 'pip install matplotlib numpy'."
        ) from exc

    # Define hop bins and pastel colors
    hop_bins = [1, 2, 3, 4]
    colors = ['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA']  # Pastel red, green, blue, yellow

    # Prepare data for stacking
    data_by_hop: Dict[int, List[int]] = {hop: [] for hop in hop_bins}

    for category in categories:
        hop_counts = hop_data.get(category, {})
        for hop in hop_bins:
            data_by_hop[hop].append(hop_counts.get(hop, 0))

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))

    x = np.arange(len(categories))
    width = 0.6

    # Create stacked bars and add labels on each segment
    bottom = np.zeros(len(categories))
    bars = []

    for hop, color in zip(hop_bins, colors):
        values = data_by_hop[hop]
        bar = ax.bar(x, values, width, bottom=bottom,
                     label=f'{hop} hop{"s" if hop > 1 else ""}', color=color)
        bars.append(bar)

        # Add count labels on each hop segment
        for i, value in enumerate(values):
            if value > 0:
                y_pos = bottom[i] + value / 2
                ax.text(i, y_pos, str(int(value)), ha='center', va='center', fontsize=9)

        bottom += np.array(values)

    # Add total labels on top of each bar
    for i, total in enumerate(bottom):
        if total > 0:
            ax.text(i, total + 2, str(int(total)), ha='center',
                    va='bottom', fontsize=11, fontweight='bold')

    # Customize the plot
    ax.set_ylabel('Unanswered Questions (sqrt scale)', fontsize=12)
    ax.set_title('Distribution of Unanswered Questions by Number of Hops in Different Settings',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Use square root scale - compresses large values, expands small ones
    ax.set_yscale('function', functions=(lambda x: np.sqrt(x), lambda x: x**2))
    
    # Set y-axis limit
    max_total = max(bottom) if len(bottom) > 0 else 100
    ax.set_ylim(0, max_total * 1.05)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {output_path}")


def main() -> None:
    """Main function to generate stacked bar plots by hops."""
    base = Path(__file__).resolve().parents[1]
    report_root = base / "results" / "unanswered_questions"

    # Define the three categories with their file paths
    datasets: Dict[str, Path] = {
        "Without Context": report_root / "response-jsonl-without-context_unanswered.jsonl",
        "Gold Context": report_root / "response-jsonl-with-context_unanswered.jsonl",
        "Iterative RAG": report_root / "responses_reverified_unanswered.jsonl",
    }
    category_order = ["Without Context", "Gold Context", "Iterative RAG"]

    # Load QA hop data
    qa_path = base / "docs" / "chemrxiv_qa.json"
    qa_lookup = load_qa_hops(qa_path)

    # Collect hop data for each category
    hop_data: Dict[str, Dict[int, int]] = {}

    for label in category_order:
        records = load_records(datasets[label])
        hop_data[label] = collect_hop_counts_by_category(records, qa_lookup)
        print(f"{label}: {len(records)} unanswered questions")

    # Generate single plot
    output_path = report_root / "unanswered_by_hops_all_models.png"
    plot_stacked_bar_by_hops(category_order, hop_data, output_path)
    print(f"\nGenerated plot: {output_path}")


if __name__ == "__main__":
    main()