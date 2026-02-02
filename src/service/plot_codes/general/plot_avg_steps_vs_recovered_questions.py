#!/usr/bin/env python3
"""
Scatter plot showing relationship between average steps and recovered questions.

X-axis: Average number of steps used by each model
Y-axis: Number of questions recovered by iterative RAG that were missed in Gold Context
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Set

import matplotlib.pyplot as plt
import numpy as np

from config import (
    get_iterative_model_entries,
    normalize_model_key,
)


# -------- JSONL helpers --------

def _iter_jsonl(path: Path) -> Iterable[dict]:
    """Iterate over JSONL records."""
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
    """Extract question text from a record."""
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
    """Extract the maximum source_step from evidence."""
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
    """Keep latest record for each question."""
    by_q: Dict[str, dict] = {}
    for rec in records:
        q = _extract_question(rec)
        if not q:
            continue
        by_q[q] = rec
    return by_q


def _canon_key_from_stem(stem: str) -> str:
    """Create canonical key from filename stem."""
    import re
    key = normalize_model_key(stem)
    return re.sub(r"[^a-z0-9]+", "", key.lower())


# -------- Analysis functions --------

def build_gold_incorrect_sets_from_entries(
    gold_dir: Path, model_entries: List[Tuple[Path, str]]
) -> Dict[str, Set[str]]:
    """Build sets of questions that were incorrect in gold context for each model."""
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


def count_recovered_questions(
    iterative_path: Path,
    gold_miss_questions: Set[str],
) -> int:
    """Count how many gold-missed questions were recovered in iterative RAG."""
    by_q = _dedup_latest(_iter_jsonl(iterative_path))
    recovered = 0
    for q, rec in by_q.items():
        if q not in gold_miss_questions:
            continue
        if bool(rec.get("is_correct", False)):
            recovered += 1
    return recovered


def get_avg_step_per_model(model_entries: List[Tuple[Path, str]]) -> Dict[str, float]:
    """For each model, compute average max step across all questions."""
    result = {}
    for path, display in model_entries:
        by_q = _dedup_latest(_iter_jsonl(path))
        steps = []
        for rec in by_q.values():
            stp = _extract_max_source_step(rec)
            if isinstance(stp, int) and stp > 0:
                steps.append(stp)
        result[display] = np.mean(steps) if steps else 1.0
    return result


# -------- Plotting function --------

def plot_avg_steps_vs_recovered(
    models: List[str],
    avg_steps: Dict[str, float],
    recovered_counts: Dict[str, int],
    output_path: Path,
) -> None:
    """Create scatter plot of average steps vs recovered questions."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data
    x_data = [avg_steps.get(m, 1.0) for m in models]
    y_data = [recovered_counts.get(m, 0) for m in models]
    
    # Color based on number of recovered questions
    colors = plt.cm.viridis(np.array(y_data) / max(y_data) if max(y_data) > 0 else [0] * len(y_data))
    
    # Create scatter plot
    scatter = ax.scatter(x_data, y_data, s=200, c=colors, alpha=0.7, 
                        edgecolors='black', linewidth=1.5)
    
    # Add model labels
    for i, model in enumerate(models):
        ax.annotate(model, (x_data[i], y_data[i]), 
                   fontsize=9, ha='left', va='bottom',
                   xytext=(5, 5), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Questions Recovered', pad=0.02)
    cbar.set_label('Questions Recovered', fontsize=11, fontweight='bold')
    
    # Styling
    ax.set_xlabel('Average Number of Steps Used', fontsize=13, fontweight='bold')
    ax.set_ylabel('Questions Recovered from Gold Context Misses', fontsize=13, fontweight='bold')
    ax.set_title('Average Steps vs Questions Recovered by Iterative RAG\n(Questions Missed in Gold Context but Recovered with Iterative Retrieval)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add reference lines for quartiles
    if y_data:
        median_y = np.median(y_data)
        ax.axhline(y=median_y, color='red', linestyle='--', linewidth=1.5, alpha=0.5, 
                  label=f'Median: {median_y:.0f} questions')
    
    if x_data:
        median_x = np.median(x_data)
        ax.axvline(x=median_x, color='blue', linestyle='--', linewidth=1.5, alpha=0.5,
                  label=f'Median: {median_x:.2f} steps')
    
    ax.legend(loc='upper left', fontsize=10)
    
    # Set reasonable limits
    if x_data:
        ax.set_xlim(min(x_data) * 0.9, max(x_data) * 1.1)
    if y_data:
        ax.set_ylim(-max(y_data) * 0.05, max(y_data) * 1.1)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plot to: {output_path}")


# -------- Main execution --------

def main() -> None:
    """Main execution function."""
    base = Path(__file__).resolve().parents[3]
    plots_dir = base / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Get model entries
    model_entries = get_iterative_model_entries(existing_only=True)
    model_order = [display for _, display in model_entries]

    # Directories
    dir_gold_ctx = base / "response-jsonl-with-context"
    
    print("Building gold context miss sets...")
    gold_misses = build_gold_incorrect_sets_from_entries(dir_gold_ctx, model_entries)
    
    print("Computing average steps per model...")
    avg_steps = get_avg_step_per_model(model_entries)
    
    print("Counting recovered questions per model...")
    recovered_counts: Dict[str, int] = {}
    for path, display in model_entries:
        questions = gold_misses.get(display, set())
        if questions:
            recovered = count_recovered_questions(path, questions)
            recovered_counts[display] = recovered
            print(f"  {display}: {recovered} recovered from {len(questions)} gold misses (avg steps: {avg_steps.get(display, 0):.2f})")
        else:
            recovered_counts[display] = 0
            print(f"  {display}: 0 recovered (no gold misses found)")
    
    print("\nCreating scatter plot...")
    output_path = plots_dir / "avg_steps_vs_recovered_questions.png"
    plot_avg_steps_vs_recovered(model_order, avg_steps, recovered_counts, output_path)
    
    print("\n" + "="*60)
    print("Summary (sorted by recovered questions):")
    print("="*60)
    for m in sorted(model_order, key=lambda x: recovered_counts.get(x, 0), reverse=True):
        print(f"{m:30s}: {recovered_counts.get(m, 0):3d} questions recovered (avg {avg_steps.get(m, 0):.2f} steps)")
    print("="*60)


if __name__ == "__main__":
    main()
