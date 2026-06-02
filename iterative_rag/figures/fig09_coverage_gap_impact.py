"""Figure 9 — Impact of retrieval coverage gaps on accuracy (No-Context-wrong questions).

(a) aggregate accuracy with vs without a coverage gap; (b) per-model accuracy-impact (pp).
"""

from __future__ import annotations

from pathlib import Path

from iterative_rag.figures import common as C


def _rows():
    cov = C.load_judgments("coverage")
    it = C.correct_by_question("iterative")
    nc = C.correct_by_question("no_context")
    rows = []  # (model, has_gap, is_correct)
    for m, judg in cov.items():
        if m not in it:
            continue
        for q, pj in judg.items():
            if m in nc and nc[m].get(q):   # restrict to No-Context-wrong
                continue
            if q not in it[m]:
                continue
            has_gap = bool((pj.get("retrieval_coverage_gap", {}) or {}).get("has_gap"))
            rows.append((m, has_gap, it[m][q]))
    return rows


def render(out_dir: Path) -> Path:
    import numpy as np
    import matplotlib.pyplot as plt

    C.use_style()
    rows = _rows()
    if not rows:
        raise RuntimeError("no coverage judgments available")

    def acc(filt):
        vals = [c for _, g, c in rows if filt(g)]
        return 100 * np.mean(vals) if vals else float("nan")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.4]})

    # (a) aggregate
    with_gap, without_gap = acc(lambda g: g), acc(lambda g: not g)
    ax1.bar(["With Coverage Gap", "Without Coverage Gap"], [with_gap, without_gap],
            color=["#d62728", "#2ca02c"], edgecolor="black")
    for i, v in enumerate([with_gap, without_gap]):
        ax1.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax1.set_ylabel("Average Accuracy (%)")
    ax1.set_ylim(0, 100)
    ax1.set_title("Impact of Coverage Gaps on Model Accuracy\n(No-Context Wrong Questions Only)")

    # (b) per-model impact
    impacts = []
    for m in C.PAPER_MODELS:
        mr = [(g, c) for mm, g, c in rows if mm == m]
        gap = [c for g, c in mr if g]
        nogap = [c for g, c in mr if not g]
        if gap and nogap:
            impacts.append((m, 100 * np.mean(nogap) - 100 * np.mean(gap)))
    impacts.sort(key=lambda t: t[1])
    models = [m for m, _ in impacts]
    vals = [v for _, v in impacts]

    def sev_color(v):
        return ("#2ca02c" if v < 20 else "#f4a259" if v < 25 else
                "#e76f51" if v < 30 else "#c1121f" if v < 35 else "#7a0a12")
    y = np.arange(len(models))
    ax2.barh(y, vals, color=[sev_color(v) for v in vals], edgecolor="black")
    for yi, v in zip(y, vals):
        ax2.text(v, yi, f" {v:.1f}pp", va="center", fontsize=8)
    if vals:
        ax2.axvline(float(np.mean(vals)), color="blue", ls="--", alpha=0.7,
                    label=f"Average: {np.mean(vals):.1f}pp")
        ax2.legend()
    ax2.set_yticks(y, models)
    ax2.set_xlabel("Accuracy Impact (percentage points)")
    ax2.set_title("Coverage Gap Impact by Model")
    fig.tight_layout()
    return C.save(fig, out_dir, "fig09_coverage_gap_impact")
