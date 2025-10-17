#!/usr/bin/env python3
"""
Compare model performance across three settings:
- No Context (baseline; files in src/response-jsonl-without-context)
- Gold Context (oracle context; files in src/response-jsonl-with-context)
- Iterative RAG (this repo's system; files in src/responses_reverified)

Outputs three plots to src/plots:
- context_benefit_accuracy_by_model.png
  Grouped bars per model: No Context vs Gold Context vs Iterative RAG accuracy (%).

- context_benefit_improvement_by_model.png
  Improvement bars per model: (Iterative − No Context) and (Iterative − Gold Context), in %-points.

- iterative_rag_accuracy_by_max_source_step.png
  Small multiples: per-model accuracy vs max source step in Iterative RAG.

Notes:
- Models are aligned across the three settings using the display names from
  analyzing/config.py (get_iterative_model_entries / get_iterative_display_names / get_display_name).
- Accuracy is computed per unique question (deduplicated by the "question" text).
- For iterative step plots, max source step is extracted from raw/raw_response -> evidence[].source_step.
"""

from __future__ import annotations

import json
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from config import (
    get_iterative_model_entries,
    get_iterative_display_names,
    get_display_name,
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
    """Deduplicate by question, keeping the last record encountered.

    Many files are single-pass; keeping last is a safe default when re-verification
    may have appended/overwritten lines.
    """
    by_q: Dict[str, dict] = {}
    for rec in records:
        q = _extract_question(rec)
        if not q:
            continue
        by_q[q] = rec
    return by_q


def compute_accuracy_for_file(path: Path) -> Tuple[int, int]:
    """Return (n_total, n_correct) for a single JSONL file.

    Deduplicates by question text.
    """
    by_q = _dedup_latest(_iter_jsonl(path))
    total = len(by_q)
    correct = sum(1 for r in by_q.values() if bool(r.get("is_correct", False)))
    return total, correct


def compute_iterative_step_stats(path: Path) -> Tuple[Counter, Counter]:
    """Return (total_by_step, correct_by_step) counters for iterative runs.

    Steps are the max source step per record; deduplicated per question.
    """
    by_q = _dedup_latest(_iter_jsonl(path))
    total_by_step: Counter = Counter()
    correct_by_step: Counter = Counter()
    for rec in by_q.values():
        stp = _extract_max_source_step(rec)
        if not isinstance(stp, int) or stp <= 0:
            continue
        total_by_step[stp] += 1
        if bool(rec.get("is_correct", False)):
            correct_by_step[stp] += 1
    return total_by_step, correct_by_step


def scan_context_dir(context_dir: Path, desired_models: List[str]) -> Dict[str, Tuple[int, int]]:
    """Aggregate (total, correct) by display name for a context directory.

    - Maps files to display names via get_display_name.
    - Filters to desired_models to ensure alignment with iterative models.
    """
    model_stats: Dict[str, Tuple[int, int]] = {m: (0, 0) for m in desired_models}
    for f in sorted(context_dir.glob("*.jsonl")):
        display = get_display_name(f.stem)
        if display not in model_stats:
            continue
        total, correct = compute_accuracy_for_file(f)
        prev_total, prev_correct = model_stats[display]
        # Some dirs might contain duplicate variants; keep the larger coverage
        if total > prev_total or (total == prev_total and correct > prev_correct):
            model_stats[display] = (total, correct)
    return model_stats


# -------- Plotting helpers ---------

def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except ImportError as exc:  # pragma: no cover - external dependency
        raise SystemExit("matplotlib is required. Install with 'pip install matplotlib'.") from exc


def plot_accuracy_grouped(
    models: List[str],
    no_ctx: Dict[str, Tuple[int, int]],
    gold_ctx: Dict[str, Tuple[int, int]],
    iterative: Dict[str, Tuple[int, int]],
    out_path: Path,
) -> None:
    plt = _require_matplotlib()

    def pct(v):
        t, c = v
        return (c / t * 100.0) if t else 0.0

    no_vals = [pct(no_ctx.get(m, (0, 0))) for m in models]
    gold_vals = [pct(gold_ctx.get(m, (0, 0))) for m in models]
    iter_vals = [pct(iterative.get(m, (0, 0))) for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(14, len(models) * 1.6), 6))
    bars1 = ax.bar(x - width, no_vals, width, label="No Context", color="#7f7f7f")
    bars2 = ax.bar(x, gold_vals, width, label="Gold Context", color="#1f77b4")
    bars3 = ax.bar(x + width, iter_vals, width, label="Iterative RAG", color="#2ca02c")

    # Annotate bars
    for bars in (bars1, bars2, bars3):
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x() + b.get_width() / 2, h + 0.6, f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Accuracy by Context")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_improvement_grouped(
    models: List[str],
    no_ctx: Dict[str, Tuple[int, int]],
    gold_ctx: Dict[str, Tuple[int, int]],
    iterative: Dict[str, Tuple[int, int]],
    out_path: Path,
) -> None:
    plt = _require_matplotlib()

    def pct(v):
        t, c = v
        return (c / t * 100.0) if t else 0.0

    imp_no = [pct(iterative.get(m, (0, 0))) - pct(no_ctx.get(m, (0, 0))) for m in models]
    imp_gold = [pct(iterative.get(m, (0, 0))) - pct(gold_ctx.get(m, (0, 0))) for m in models]

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(14, len(models) * 1.6), 5.5))
    bars1 = ax.bar(x - width / 2, imp_no, width, label="Iterative − No Context", color="#ff7f0e")
    bars2 = ax.bar(x + width / 2, imp_gold, width, label="Iterative − Gold Context", color="#9467bd")

    # Annotate bars
    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, (h + (1 if h >= 0 else -1)), f"{h:+.1f} pp",
                    ha="center", va="bottom" if h >= 0 else "top", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Improvement (percentage points)")
    ax.set_title("Benefit of Iterative RAG vs Baselines")
    ax.legend(loc="upper left")
    ax.axhline(0, color="#333333", linewidth=1)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_iterative_accuracy_by_step(
    models: List[str],
    step_stats: Dict[str, Tuple[Counter, Counter]],
    out_path: Path,
) -> None:
    plt = _require_matplotlib()

    # Determine common step range across models, cap at 6
    max_step = 0
    for m in models:
        totals, _ = step_stats.get(m, (Counter(), Counter()))
        if totals:
            max_step = max(max_step, max(totals.keys()))
    max_step = min(max_step if max_step > 0 else 5, 6)
    steps = list(range(1, max_step + 1))

    cols = 4 if len(models) > 6 else 3
    rows = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.2))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, model in enumerate(models):
        ax = axes[idx]
        totals, corrects = step_stats.get(model, (Counter(), Counter()))
        acc = []
        ns = []
        for s in steps:
            n = totals.get(s, 0)
            c = corrects.get(s, 0)
            ns.append(n)
            acc.append((c / n * 100.0) if n else 0.0)
        bars = ax.bar(np.arange(len(steps)), acc, color="#2ca02c", width=0.55)
        # annotate with n
        for i, b in enumerate(bars):
            if ns[i] > 0:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{ns[i]}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(np.arange(len(steps)))
        ax.set_xticklabels([str(s) for s in steps])
        ax.set_ylim(0, 100)
        ax.set_title(model, fontsize=10)
        if idx % cols == 0:
            ax.set_ylabel("Accuracy (%)")
        ax.set_xlabel("Max source step")
        ax.grid(axis="y", alpha=0.25, linestyle="--")

    for j in range(len(models), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Iterative RAG: Accuracy by Max Source Step (per model)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    plots_dir = base / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Determine canonical model order from iterative entries we actually have
    model_entries = get_iterative_model_entries(existing_only=True)
    model_order = [display for _, display in model_entries]

    # Paths
    dir_no_ctx = base / "response-jsonl-without-context"
    dir_gold_ctx = base / "response-jsonl-with-context"
    dir_iterative = base / "responses_reverified"

    # Aggregate baseline accuracies
    no_ctx_stats = scan_context_dir(dir_no_ctx, model_order)
    gold_ctx_stats = scan_context_dir(dir_gold_ctx, model_order)

    # Aggregate iterative accuracies and step stats
    iterative_stats: Dict[str, Tuple[int, int]] = {m: (0, 0) for m in model_order}
    step_stats: Dict[str, Tuple[Counter, Counter]] = {}
    for p, display in model_entries:
        tot, cor = compute_accuracy_for_file(p)
        iterative_stats[display] = (tot, cor)
        step_stats[display] = compute_iterative_step_stats(p)

    # Plot 1: grouped accuracy by context
    plot_accuracy_grouped(
        model_order,
        no_ctx_stats,
        gold_ctx_stats,
        iterative_stats,
        plots_dir / "context_benefit_accuracy_by_model.png",
    )

    # Plot 2: improvement of iterative vs baselines
    plot_improvement_grouped(
        model_order,
        no_ctx_stats,
        gold_ctx_stats,
        iterative_stats,
        plots_dir / "context_benefit_improvement_by_model.png",
    )

    # Plot 3: iterative accuracy vs max source step (per model)
    plot_iterative_accuracy_by_step(
        model_order,
        step_stats,
        plots_dir / "iterative_rag_accuracy_by_max_source_step.png",
    )

    # Console summary
    def to_pct_str(v):
        t, c = v
        return f"{c}/{t} ({(c/t*100.0 if t else 0.0):.1f}%)"

    print("\nSummary (accuracy):")
    for m in model_order:
        print(
            f"  {m:28s} | No Ctx: {to_pct_str(no_ctx_stats.get(m, (0,0))):>16s}  "
            f"Gold: {to_pct_str(gold_ctx_stats.get(m, (0,0))):>16s}  "
            f"Iter: {to_pct_str(iterative_stats.get(m, (0,0))):>16s}"
        )


if __name__ == "__main__":
    main()

