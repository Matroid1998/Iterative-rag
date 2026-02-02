#!/usr/bin/env python3
"""
Plot showing the relationship between number of steps and accuracy improvement.

This visualization shows how the iterative RAG improvement (vs gold context) 
relates to the maximum step used for each model.
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


# -------- JSONL helpers (reused from plot_iterative_by_step_on_gold_misses.py) ---------

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


def _compute_accuracy_for_file(path: Path) -> Tuple[int, int]:
    by_q = _dedup_latest(_iter_jsonl(path))
    total = len(by_q)
    correct = sum(1 for r in by_q.values() if bool(r.get("is_correct", False)))
    return total, correct


def scan_gold_accuracy_for_entries(
    gold_dir: Path, model_entries: List[Tuple[Path, str]]
) -> Dict[str, Tuple[int, int]]:
    iter_key_to_display: Dict[str, str] = {}
    for p, display in model_entries:
        iter_key_to_display[_canon_key_from_stem(p.stem)] = display

    best: Dict[str, Tuple[int, int]] = {display: (0, 0) for _, display in model_entries}

    for f in sorted(gold_dir.glob("*.jsonl")):
        key = _canon_key_from_stem(f.stem)
        display = iter_key_to_display.get(key)
        if not display:
            key2 = key.replace("reasoning", "")
            display = iter_key_to_display.get(key2)
            if not display:
                continue
        total, correct = _compute_accuracy_for_file(f)
        prev_total, prev_correct = best[display]
        if total > prev_total or (total == prev_total and correct > prev_correct):
            best[display] = (total, correct)
    return best


def get_max_step_used_per_model(model_entries: List[Tuple[Path, str]]) -> Dict[str, int]:
    """For each model, find the maximum step number used across all questions."""
    result = {}
    for path, display in model_entries:
        by_q = _dedup_latest(_iter_jsonl(path))
        max_step = 0
        for rec in by_q.values():
            stp = _extract_max_source_step(rec)
            if isinstance(stp, int) and stp > 0:
                max_step = max(max_step, stp)
        result[display] = max_step if max_step > 0 else 1
    return result


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


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required. Install with 'pip install matplotlib'.") from exc


def plot_steps_vs_improvement(
    models: List[str],
    max_steps: Dict[str, int],
    avg_steps: Dict[str, float],
    improvement_pp: Dict[str, float],
    out_path: Path,
) -> None:
    """Create visualization showing relationship between steps and improvement."""
    plt = _require_matplotlib()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Max Step vs Improvement
    x_max = [max_steps.get(m, 1) for m in models]
    y = [improvement_pp.get(m, 0) for m in models]
    
    colors = ['#d62728' if imp < 0 else '#2ca02c' for imp in y]
    scatter1 = ax1.scatter(x_max, y, s=150, c=colors, alpha=0.6, edgecolors='black', linewidth=1.5)
    
    # Add model labels
    for i, model in enumerate(models):
        ax1.annotate(model, (x_max[i], y[i]), fontsize=7, ha='right', va='bottom', 
                    xytext=(-5, 5), textcoords='offset points')
    
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Maximum Step Used', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy Improvement (pp)', fontsize=12, fontweight='bold')
    ax1.set_title('Max Step vs Accuracy Improvement\n(Iterative RAG vs Gold Context)', 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(range(1, max(x_max) + 1))
    
    # Plot 2: Average Step vs Improvement
    x_avg = [avg_steps.get(m, 1) for m in models]
    scatter2 = ax2.scatter(x_avg, y, s=150, c=colors, alpha=0.6, edgecolors='black', linewidth=1.5)
    
    # Add model labels
    for i, model in enumerate(models):
        ax2.annotate(model, (x_avg[i], y[i]), fontsize=7, ha='right', va='bottom',
                    xytext=(-5, 5), textcoords='offset points')
    
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Average Step Used', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy Improvement (pp)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Step vs Accuracy Improvement\n(Iterative RAG vs Gold Context)', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    fig.suptitle('Relationship Between Iterative Steps and Accuracy Improvement',
                 fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_step_distribution_with_improvement(
    models: List[str],
    step_counts: Dict[str, Counter],
    improvement_pp: Dict[str, float],
    out_path: Path,
) -> None:
    """Bar chart showing step distribution colored by improvement level."""
    plt = _require_matplotlib()
    
    # Sort models by improvement
    sorted_models = sorted(models, key=lambda m: improvement_pp.get(m, 0), reverse=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    max_step = max(max(cnts.keys()) for cnts in step_counts.values() if cnts)
    steps = list(range(1, min(max_step + 1, 7)))
    
    x = np.arange(len(sorted_models))
    width = 0.12
    
    # Normalize colormap based on improvement
    import matplotlib.cm as cm
    norm = plt.Normalize(vmin=min(improvement_pp.values()), vmax=max(improvement_pp.values()))
    cmap = cm.RdYlGn  # Red for negative, green for positive
    
    for step_idx, step in enumerate(steps):
        offsets = width * (step_idx - len(steps) / 2)
        counts = [step_counts.get(m, Counter()).get(step, 0) for m in sorted_models]
        
        bars = ax.bar(x + offsets, counts, width, label=f'Step {step}', alpha=0.8)
    
    ax.set_xlabel('Models (sorted by improvement)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Questions', fontsize=12, fontweight='bold')
    ax.set_title('Step Distribution per Model\n(Recovered Gold-Context Misses)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({improvement_pp.get(m, 0):+.1f}pp)" for m in sorted_models], 
                       rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


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


def main() -> None:
    base = Path(__file__).resolve().parents[3]
    plots_dir = base.parent / "data" / "plots" / "general"
    plots_dir.mkdir(parents=True, exist_ok=True)

    model_entries = get_iterative_model_entries(existing_only=True)
    model_order = [display for _, display in model_entries]

    dir_gold_ctx = base / "response-jsonl-with-context"
    dir_iterative = base / "responses_reverified"

    # Compute accuracies
    iterative_acc: Dict[str, Tuple[int, int]] = {}
    for p, display in model_entries:
        iterative_acc[display] = _compute_accuracy_for_file(p)

    gold_acc = scan_gold_accuracy_for_entries(dir_gold_ctx, model_entries)

    def pct(v: Tuple[int, int]) -> float:
        t, c = v
        return (c / t * 100.0) if t else 0.0

    improvement_pp = {
        m: (pct(iterative_acc.get(m, (0, 0))) - pct(gold_acc.get(m, (0, 0)))) 
        for m in model_order
    }

    # Get step information
    max_steps = get_max_step_used_per_model(model_entries)
    avg_steps = get_avg_step_per_model(model_entries)

    # Build step counts for recovered questions
    gold_misses = build_gold_incorrect_sets_from_entries(dir_gold_ctx, model_entries)
    counts_by_model: Dict[str, Counter] = {}
    for p, display in model_entries:
        questions = gold_misses.get(display, set())
        cnts = compute_iterative_recovered_counts_by_step(p, questions) if questions else Counter()
        counts_by_model[display] = cnts

    # Create visualizations
    out_path1 = plots_dir / "steps_vs_improvement_scatter.png"
    plot_steps_vs_improvement(model_order, max_steps, avg_steps, improvement_pp, out_path1)
    print(f"✓ Created: {out_path1}")

    out_path2 = plots_dir / "step_distribution_by_improvement.png"
    plot_step_distribution_with_improvement(model_order, counts_by_model, improvement_pp, out_path2)
    print(f"✓ Created: {out_path2}")

    # Print summary
    print("\nSummary:")
    for m in sorted(model_order, key=lambda x: improvement_pp.get(x, 0), reverse=True):
        print(f"{m:28s}: Max={max_steps.get(m, 0)}, Avg={avg_steps.get(m, 0):.2f}, Δ={improvement_pp.get(m, 0):+.1f}pp")


if __name__ == "__main__":
    main()
