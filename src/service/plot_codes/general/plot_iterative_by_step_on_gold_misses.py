#!/usr/bin/env python3
"""
Iterative RAG accuracy by max source step, restricted to questions each model
did NOT answer in its own Gold Context.

Reads:
- Gold Context JSONLs: src/response-jsonl-with-context/*.jsonl
- Iterative RAG JSONLs: src/responses_reverified/*.jsonl (from config entries)

Output:
- src/plots/iterative_rag_accuracy_by_max_source_step_gold_misses.png

Approach:
1) For each iterative model (display name from analyzing/config.py), find matching
   Gold Context file(s) mapped to the same display (robust mapping via get_display_name
   + canonicalization that strips 'reasoning').
2) Build set of questions that were incorrect in Gold Context for that model.
3) In that subset, compute Iterative RAG per-step accuracy using max source step.
4) Plot per-model bar charts (like iterative_rag_accuracy_by_max_source_step.png).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Set

import numpy as np

from config import (
    get_iterative_model_entries,
    get_iterative_display_names,
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


# -------- Mapping helpers ---------

def _canonize(name: str) -> str:
    import re
    s = re.sub(r"[^a-z0-9]+", "", name.lower())
    s = s.replace("reasoning", "")
    return s


def _canon_key_from_stem(stem: str) -> str:
    """Canonical key derived from a file stem using repo's normalization."""
    import re
    key = normalize_model_key(stem)
    return re.sub(r"[^a-z0-9]+", "", key.lower())


def build_gold_incorrect_sets_from_entries(
    gold_dir: Path, model_entries: List[Tuple[Path, str]]
) -> Dict[str, Set[str]]:
    """Map each model display name -> set of questions it missed in Gold Context.

    Matching is done by comparing canonical normalized keys of gold stems to those of
    the iterative entries (robust to provider prefixes and colon vs hyphen).
    """
    # Build lookup from canonical key -> display name based on iterative entries
    iter_key_to_display: Dict[str, str] = {}
    for p, display in model_entries:
        iter_key_to_display[_canon_key_from_stem(p.stem)] = display

    # Initialize results
    result: Dict[str, Set[str]] = {display: set() for _, display in model_entries}

    for f in sorted(gold_dir.glob("*.jsonl")):
        gold_key = _canon_key_from_stem(f.stem)
        display = iter_key_to_display.get(gold_key)
        if not display:
            # Fallback: some gold files append '-reasoning' where iterative uses none
            # Try removing 'reasoning' token from canonical key
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
    """Return counts of recovered (correct) questions by max source step, restricted
    to questions missed in Gold Context.
    """
    by_q = _dedup_latest(_iter_jsonl(iterative_path))
    counts: Counter = Counter()
    for q, rec in by_q.items():
        if q not in gold_miss_questions:
            continue
        if not bool(rec.get("is_correct", False)):
            continue  # only consider those solved by Iterative RAG
        stp = _extract_max_source_step(rec)
        if not isinstance(stp, int) or stp <= 0:
            continue
        counts[stp] += 1
    return counts


# -------- Plotting ---------

def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except ImportError as exc:  # pragma: no cover - external dependency
        raise SystemExit("matplotlib is required. Install with 'pip install matplotlib'.") from exc


def plot_iterative_recovered_counts_by_step(
    models: List[str],
    counts_by_model: Dict[str, Counter],
    out_path: Path,
    improvement_pp: Dict[str, float] | None = None,
) -> None:
    plt = _require_matplotlib()

    # Determine common step range across models, cap at 6
    max_step = 0
    for m in models:
        cnts = counts_by_model.get(m, Counter())
        if cnts:
            max_step = max(max_step, max(cnts.keys()))
    max_step = min(max_step if max_step > 0 else 5, 6)
    steps = list(range(1, max_step + 1))

    cols = 4 if len(models) > 6 else 3
    rows = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.2))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, model in enumerate(models):
        ax = axes[idx]
        cnts = counts_by_model.get(model, Counter())
        vals = [cnts.get(s, 0) for s in steps]
        bars = ax.bar(np.arange(len(steps)), vals, color="#2ca02c", width=0.55)
        # annotate with counts
        for i, b in enumerate(bars):
            if vals[i] > 0:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{vals[i]}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(np.arange(len(steps)))
        ax.set_xticklabels([str(s) for s in steps])
        # Title with optional improvement annotation
        if improvement_pp and model in improvement_pp:
            ax.set_title(f"{model}\n(Δ {improvement_pp[model]:+.1f} pp)", fontsize=10)
        else:
            ax.set_title(model, fontsize=10)
        if idx % cols == 0:
            ax.set_ylabel("Recovered questions (count)")
        ax.set_xlabel("Max source step (gold misses, iterative=correct)")
        ax.grid(axis="y", alpha=0.25, linestyle="--")

    for j in range(len(models), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Iterative RAG: Recovered Gold-Context Misses by Max Source Step",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    _require_matplotlib().close(fig)


# -------- Accuracy + improvement helpers ---------

def _compute_accuracy_for_file(path: Path) -> Tuple[int, int]:
    by_q = _dedup_latest(_iter_jsonl(path))
    total = len(by_q)
    correct = sum(1 for r in by_q.values() if bool(r.get("is_correct", False)))
    return total, correct


def scan_gold_accuracy_for_entries(
    gold_dir: Path, model_entries: List[Tuple[Path, str]]
) -> Dict[str, Tuple[int, int]]:
    """Return per-model (total, correct) for Gold Context files, matched to
    iterative entries via canonical stem normalization.
    """
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


def main() -> None:
    base = Path(__file__).resolve().parents[3]
    plots_dir = base / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Determine canonical model order from iterative entries we actually have
    model_entries = get_iterative_model_entries(existing_only=True)
    model_order = [display for _, display in model_entries]

    dir_gold_ctx = base / "response-jsonl-with-context"
    dir_iterative = base / "responses_reverified"

    # Build per-model gold-miss question sets (using robust key matching)
    gold_misses = build_gold_incorrect_sets_from_entries(dir_gold_ctx, model_entries)

    # For each model, compute counts of recovered (correct) gold-miss questions by step
    counts_by_model: Dict[str, Counter] = {}
    iterative_acc: Dict[str, Tuple[int, int]] = {}
    for p, display in model_entries:
        questions = gold_misses.get(display, set())
        cnts = compute_iterative_recovered_counts_by_step(p, questions) if questions else Counter()
        counts_by_model[display] = cnts
        iterative_acc[display] = _compute_accuracy_for_file(p)

    # Compute gold accuracies mapped to entries
    gold_acc = scan_gold_accuracy_for_entries(dir_gold_ctx, model_entries)

    # Improvement in percentage points per model
    def pct(v: Tuple[int, int]) -> float:
        t, c = v
        return (c / t * 100.0) if t else 0.0

    improvement_pp = {m: (pct(iterative_acc.get(m, (0, 0))) - pct(gold_acc.get(m, (0, 0)))) for m in model_order}

    # Plot counts by step, with improvement annotated
    out_path = plots_dir / "iterative_rag_recovered_by_max_source_step_gold_misses_improvement.png"
    plot_iterative_recovered_counts_by_step(model_order, counts_by_model, out_path, improvement_pp)

    # Summary to console
    for m in model_order:
        cnts = counts_by_model.get(m, Counter())
        total = sum(cnts.values())
        iter_t, iter_c = iterative_acc.get(m, (0, 0))
        gold_t, gold_c = gold_acc.get(m, (0, 0))+ (0,) if False else gold_acc.get(m, (0, 0))
        if total:
            parts = ", ".join(f"{s}:{cnts.get(s,0)}" for s in sorted(cnts))
            print(
                f"{m:28s}: recovered={total:4d}  per-step: {parts}  |  Δ={pct(iterative_acc[m]) - pct(gold_acc[m]):+.1f} pp"
            )
        else:
            print(f"{m:28s}: No recovered gold-miss questions  |  Δ={pct(iterative_acc[m]) - pct(gold_acc[m]):+.1f} pp")


if __name__ == "__main__":
    main()
