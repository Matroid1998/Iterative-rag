"""Figure 4 — Recoveries vs Regressions from Gold Context to Iterative RAG.

Recovery = Gold-incorrect -> Iterative-correct; Regression = Gold-correct -> Iterative-incorrect.
Top panel: green/red bars + net-gain line; bottom panel: net effect (recoveries - regressions).
"""

from __future__ import annotations

from pathlib import Path

from iterative_rag.figures import common as C


def render(out_dir: Path) -> Path:
    import numpy as np
    import matplotlib.pyplot as plt

    C.use_style()
    gc = C.correct_by_question("gold")
    it = C.correct_by_question("iterative")
    rows = []
    for m in C.PAPER_MODELS:
        if m not in gc or m not in it:
            continue
        qs = set(gc[m]) & set(it[m])
        rec = sum(1 for q in qs if (not gc[m][q]) and it[m][q])
        reg = sum(1 for q in qs if gc[m][q] and (not it[m][q]))
        rows.append((m, rec, reg))
    rows.sort(key=lambda r: r[1] - r[2], reverse=True)
    models = [r[0] for r in rows]
    recov = np.array([r[1] for r in rows])
    regr = np.array([r[2] for r in rows])
    net = recov - regr
    x = np.arange(len(models))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [3, 1]})
    ax1.bar(x, recov, color="#2ca02c", label="Recoveries (Gold Wrong → Iter Correct)")
    ax1.bar(x, -regr, color="#d62728", label="Regressions (Gold Correct → Iter Wrong)")
    ax1.axhline(0, color="black", lw=0.8)
    ax1b = ax1.twinx()
    ax1b.plot(x, net, "k-o", lw=2, label="Net Gain")
    ax1b.set_ylabel("Net Gain")
    ax1.set_ylabel("Number of Questions")
    ax1.set_title("Regressions vs. Recoveries: Gold Context → Iterative RAG\nQuantifying Iteration's Net Effect per Model")
    ax1.set_xticks(x, models, rotation=45, ha="right")
    ax1.legend(loc="upper right", fontsize=8)

    bars = ax2.bar(x, net, color="#2ca02c")
    for xi, n in zip(x, net):
        ax2.text(xi, n, f"+{n}", ha="center", va="bottom" if n >= 0 else "top", fontsize=8)
    ax2.set_ylabel("Net Gain")
    ax2.set_title("Net Effect (Recoveries - Regressions)")
    ax2.set_xticks(x, models, rotation=45, ha="right")
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return C.save(fig, out_dir, "fig04_recoveries_regressions")
