#!/usr/bin/env python3
"""
Plot showing step distribution sorted by number of recoveries.

This is similar to step_distribution_by_improvement.png but sorts models
by the count of questions recovered (gold wrong → iterative correct) rather
than accuracy improvement percentage points.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Set

import numpy as np

from config import (
    get_iterative_model_entries,
    get_display_name,
    normalize_model_key,
)


# -------- JSONL helpers ---------

def _iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                continue


def _extract_question(rec: dict) -> str | None:
    q = rec.get("question")
    if isinstance(q, str) and q.strip():
        return q.strip()
    for key in ("raw", "raw_response"):
        raw = rec.get(key)
        if isinstance(raw, dict):
            q2 = raw.get("question")
            if isinstance(q2, str) and q2.strip():
                return q2.strip()
    return None


def _extract_max_source_step(rec: dict) -> int | None:
    steps: List[int] = []
    for key in ("raw_response", "raw"):
        raw = rec.get(key)
        if not isinstance(raw, dict):
            continue
        evidence = raw.get("evidence")
        if not isinstance(evidence, list):
            continue
        for item in evidence:
            if not isinstance(item, dict):
                continue
            stp = item.get("source_step")
            if isinstance(stp, (int, float)):
                stp_i = int(round(stp))
                if stp_i > 0:
                    steps.append(stp_i)
    if steps:
        return max(steps)
    return None


def _dedup_latest(records: Iterable[dict]) -> Dict[str, dict]:
    by_q: Dict[str, dict] = {}
    for rec in records:
        q = _extract_question(rec)
        if not q:
            continue
        by_q[q] = rec
    return by_q


def _canon_key_from_stem(stem: str) -> str:
    import re
    key = normalize_model_key(stem)
    return re.sub(r"[^a-z0-9]+", "", key.lower())


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


def extract_question(record: dict) -> str | None:
    """Extract question text from a record."""
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


def calculate_recoveries(
    iterative_path: Path,
    gold_path: Path
) -> int:
    """
    Calculate number of recoveries (gold wrong → iterative correct).
    
    Returns:
        Count of questions wrong in gold context, correct in iterative RAG
    """
    # Load records
    gold_records = load_records(gold_path)
    iterative_records = load_records(iterative_path)
    
    # Build question → correctness maps
    gold_correctness: Dict[str, bool] = {}
    for record in gold_records:
        question = extract_question(record)
        if question:
            gold_correctness[question] = bool(record.get("is_correct", False))
    
    iterative_correctness: Dict[str, bool] = {}
    for record in iterative_records:
        question = extract_question(record)
        if question:
            iterative_correctness[question] = bool(record.get("is_correct", False))
    
    # Find common questions
    common_questions = set(gold_correctness.keys()) & set(iterative_correctness.keys())
    
    # Count recoveries
    recoveries = 0
    for question in common_questions:
        gold_correct = gold_correctness[question]
        iter_correct = iterative_correctness[question]
        
        if not gold_correct and iter_correct:
            recoveries += 1
    
    return recoveries


def build_gold_incorrect_sets_from_entries(
    gold_dir: Path, model_entries: List[Tuple[Path, str]]
) -> Dict[str, Set[str]]:
    iter_key_to_display: Dict[str, str] = {}
    for p, display in model_entries:
        iter_key_to_display[_canon_key_from_stem(p.stem)] = display

    result: Dict[str, Set[str]] = {display: set() for _, display in model_entries}

    for f in sorted(gold_dir.glob("*.jsonl")):
        gold_key = _canon_key_from_stem(f.stem)
        display = iter_key_to_display.get(gold_key)
        if not display:
            gold_key2 = gold_key.replace("reasoning", "")
            display = iter_key_to_display.get(gold_key2)
            if not display:
                continue
        by_q = _dedup_latest(_iter_jsonl(f))
        misses = {q for q, rec in by_q.items() if not bool(rec.get("is_correct", False))}
        if misses:
            result[display].update(misses)

    return result


def compute_iterative_recovered_counts_by_step(
    iterative_path: Path,
    gold_miss_questions: Set[str],
) -> Counter:
    by_q = _dedup_latest(_iter_jsonl(iterative_path))
    counts: Counter = Counter()
    for q, rec in by_q.items():
        if q not in gold_miss_questions:
            continue
        if not bool(rec.get("is_correct", False)):
            continue
        stp = _extract_max_source_step(rec)
        if not isinstance(stp, int) or stp <= 0:
            continue
        counts[stp] += 1
    return counts


def find_gold_file_for_model(
    gold_dir: Path, 
    model_entries: List[Tuple[Path, str]], 
    display_name: str
) -> Path | None:
    """Find the gold context file for a model by its display name."""
    iter_key_to_display: Dict[str, str] = {}
    for p, display in model_entries:
        iter_key_to_display[_canon_key_from_stem(p.stem)] = display
    
    for f in sorted(gold_dir.glob("*.jsonl")):
        gold_key = _canon_key_from_stem(f.stem)
        display = iter_key_to_display.get(gold_key)
        if not display:
            gold_key2 = gold_key.replace("reasoning", "")
            display = iter_key_to_display.get(gold_key2)
            if not display:
                continue
        if display == display_name:
            return f
    return None


def plot_step_distribution_by_recoveries(
    models: List[str],
    step_counts: Dict[str, Counter],
    recoveries_count: Dict[str, int],
    out_path: Path,
) -> None:
    """Bar chart showing step distribution sorted by number of recoveries."""
    plt = _require_matplotlib()
    
    # Sort models by number of recoveries (descending)
    sorted_models = sorted(models, key=lambda m: recoveries_count.get(m, 0), reverse=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    max_step = max(max(cnts.keys()) for cnts in step_counts.values() if cnts)
    steps = list(range(1, min(max_step + 1, 7)))
    
    x = np.arange(len(sorted_models))
    width = 0.12
    
    # Color palette for steps
    colors = ['#e74c3c', '#e67e22', '#f39c12', '#9b59b6', '#3498db', '#2ecc71']
    
    for step_idx, step in enumerate(steps):
        offsets = width * (step_idx - len(steps) / 2)
        counts = [step_counts.get(m, Counter()).get(step, 0) for m in sorted_models]
        
        color = colors[step_idx % len(colors)]
        bars = ax.bar(x + offsets, counts, width, label=f'Step {step}', 
                     alpha=0.8, color=color, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Models (sorted by recoveries)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Questions', fontsize=12, fontweight='bold')
    ax.set_title('Step Distribution per Model\n(Sorted by Questions Recovered: Gold Wrong → Iterative Correct)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({recoveries_count.get(m, 0)} recovered)" for m in sorted_models], 
                       rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required. Install with 'pip install matplotlib'.") from exc


def main() -> None:
    base = Path(__file__).resolve().parents[3]
    plots_dir = base / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    model_entries = get_iterative_model_entries(existing_only=True)
    model_order = [display for _, display in model_entries]

    dir_gold_ctx = base / "response-jsonl-with-context"
    dir_iterative = base / "responses_reverified"

    # Calculate recoveries for each model
    recoveries_count: Dict[str, int] = {}
    for iter_path, display in model_entries:
        gold_path = find_gold_file_for_model(dir_gold_ctx, model_entries, display)
        if gold_path and gold_path.exists():
            recoveries_count[display] = calculate_recoveries(iter_path, gold_path)
        else:
            print(f"Warning: No gold file found for {display}")
            recoveries_count[display] = 0

    # Build step counts for recovered questions
    gold_misses = build_gold_incorrect_sets_from_entries(dir_gold_ctx, model_entries)
    counts_by_model: Dict[str, Counter] = {}
    for p, display in model_entries:
        questions = gold_misses.get(display, set())
        cnts = compute_iterative_recovered_counts_by_step(p, questions) if questions else Counter()
        counts_by_model[display] = cnts

    # Create visualization
    out_path = plots_dir / "step_distribution_by_recoveries.png"
    plot_step_distribution_by_recoveries(model_order, counts_by_model, recoveries_count, out_path)
    print(f"✓ Created: {out_path}")

    # Print summary
    print("\nSummary (sorted by recoveries):")
    for m in sorted(model_order, key=lambda x: recoveries_count.get(x, 0), reverse=True):
        total_recovered = sum(counts_by_model.get(m, Counter()).values())
        print(f"{m:28s}: Recoveries={recoveries_count.get(m, 0):4d}, Recovered in Steps={total_recovered:4d}")


if __name__ == "__main__":
    main()
