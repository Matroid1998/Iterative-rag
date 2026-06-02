"""Figure 6 — Distribution of unanswered questions (wrong by ALL models) by hop count,
across the three settings (No Context, Gold Context, Iterative RAG). Log-scaled stacked bars.
"""

from __future__ import annotations

from pathlib import Path

from iterative_rag.figures import common as C

REGIMES = [("no_context", "Without Context"), ("gold", "Gold Context"), ("iterative", "Iterative RAG")]
HOP_COLORS = {1: "#d62728", 2: "#ff7f0e", 3: "#1f77b4", 4: "#9467bd"}


def render(out_dir: Path) -> Path:
    import numpy as np
    import matplotlib.pyplot as plt

    C.use_style()
    hops_map = C._qa_hops_map()

    # per regime: questions wrong by every model present
    per_regime_unanswered = {}
    for key, _ in REGIMES:
        cbq = C.correct_by_question(key)
        if not cbq:
            continue
        all_qs = set.union(*[set(d) for d in cbq.values()])
        unanswered = []
        for q in all_qs:
            answered_by_any = any(cbq[m].get(q, False) for m in cbq)
            if not answered_by_any:
                unanswered.append(q)
        per_regime_unanswered[key] = unanswered

    labels = [lab for key, lab in REGIMES if key in per_regime_unanswered]
    keys = [key for key, _ in REGIMES if key in per_regime_unanswered]
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(keys))
    bottoms = np.zeros(len(keys))
    for hop in [1, 2, 3, 4]:
        counts = []
        for key in keys:
            counts.append(sum(1 for q in per_regime_unanswered[key] if hops_map.get(q, 0) == hop))
        counts = np.array(counts, dtype=float)
        ax.bar(x, counts, bottom=bottoms, color=HOP_COLORS[hop], label=f"{hop} hop" + ("s" if hop > 1 else ""))
        for xi, c, b in zip(x, counts, bottoms):
            if c > 0:
                ax.text(xi, b + c / 2, int(c), ha="center", va="center", fontsize=8, color="white")
        bottoms += counts
    for xi, tot in zip(x, bottoms):
        ax.text(xi, tot, int(tot), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_yscale("log")
    ax.set_ylabel("Unanswered Questions (log scale)")
    ax.set_title("Distribution of Unanswered Questions by Number of Hops in Different Settings")
    ax.set_xticks(x, labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return C.save(fig, out_dir, "fig06_unanswered_by_hops")
