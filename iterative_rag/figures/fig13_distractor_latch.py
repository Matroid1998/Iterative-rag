"""Figure 13 — Distractor Latch effects (from query-quality judgments).

(a) accuracy with vs without a distractor latch (No-Context-wrong questions);
(b) per-model latch prevalence.
"""

from __future__ import annotations

from pathlib import Path

from iterative_rag.figures import common as C


def render(out_dir: Path) -> Path:
    import numpy as np
    import matplotlib.pyplot as plt

    C.use_style()
    qual = C.load_judgments("quality")
    it = C.correct_by_question("iterative")
    nc = C.correct_by_question("no_context")

    agg = []          # (has_latch, is_correct) over No-Context-wrong
    prevalence = []   # (model, % latch)
    for m in C.PAPER_MODELS:
        judg = qual.get(m)
        if not judg or m not in it:
            continue
        latch_flags = []
        for q, pj in judg.items():
            latch = bool((pj.get("run_level", {}) or {}).get("distractor_latch"))
            latch_flags.append(latch)
            if q in it[m] and not (m in nc and nc[m].get(q)):
                agg.append((latch, it[m][q]))
        if latch_flags:
            prevalence.append((m, 100 * np.mean(latch_flags)))

    if not agg:
        raise RuntimeError("no quality judgments available")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.6]})
    no_latch = [c for l, c in agg if not l]
    has_latch = [c for l, c in agg if l]
    a_no = 100 * np.mean(no_latch) if no_latch else 0
    a_has = 100 * np.mean(has_latch) if has_latch else 0
    ax1.bar(["No Distractor\nLatch", "Has Distractor\nLatch"], [a_no, a_has],
            color=["#2ca02c", "#d62728"], edgecolor="black")
    for i, v in enumerate([a_no, a_has]):
        ax1.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Average Accuracy (%)")
    ax1.set_title("(a) Distractor Latch Effect on Accuracy\n(No-Context Wrong Questions Only)")

    prevalence_ordered = [(m, v) for m in C.PAPER_MODELS for mm, v in prevalence if mm == m]
    models = [m for m, _ in prevalence_ordered]
    vals = [v for _, v in prevalence_ordered]
    x = np.arange(len(models))
    ax2.bar(x, vals, color="#8c9eb2", edgecolor="black")
    for xi, v in zip(x, vals):
        ax2.text(xi, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("Percentage of Questions (%)")
    ax2.set_title("(b) Distractor Latch Prevalence by Model")
    ax2.set_xticks(x, models, rotation=45, ha="right")
    fig.tight_layout()
    return C.save(fig, out_dir, "fig13_distractor_latch")
