"""Figure 3 — Partition of Solvability heatmap.

Each correct/incorrect outcome is classified by the *necessary condition* for success:
Parametric Memory (correct with No Context), Optimum retrieval (Gold-dependent),
Synchronized retrieval and reasoning (Iterative-exclusive), and Not solved. Rows sum to 100%.
"""

from __future__ import annotations

from pathlib import Path

from iterative_rag.figures import common as C

COLS = ["Parametric\nMemory", "Optimum\nretrieval", "Synchronized retrieval\nand reasoning", "not\nsolved"]


def render(out_dir: Path) -> Path:
    import numpy as np
    import matplotlib.pyplot as plt

    C.use_style()
    nc = C.correct_by_question("no_context")
    gc = C.correct_by_question("gold")
    it = C.correct_by_question("iterative")
    models = [m for m in C.PAPER_MODELS if m in nc and m in gc and m in it]

    mat, rows = [], []
    for m in models:
        qs = set(nc[m]) & set(gc[m]) & set(it[m])
        if not qs:
            continue
        param = optimum = synced = unsolved = 0
        for q in qs:
            n, g, i = nc[m][q], gc[m][q], it[m][q]
            if n:
                param += 1
            elif g:
                optimum += 1
            elif i:
                synced += 1
            else:
                unsolved += 1
        tot = len(qs)
        mat.append([100 * param / tot, 100 * optimum / tot, 100 * synced / tot, 100 * unsolved / tot])
        rows.append(m)

    mat = np.array(mat)
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(rows) + 2))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(COLS)), COLS)
    ax.set_yticks(range(len(rows)), rows)
    for i in range(len(rows)):
        for j in range(len(COLS)):
            ax.text(j, i, f"{mat[i, j]:.1f}%", ha="center", va="center",
                    color="black", fontsize=9, fontweight="bold")
    ax.set_title("Questions correctly answered (%) by models in different settings")
    fig.colorbar(im, ax=ax, label="Percentage (%)", shrink=0.8)
    ax.grid(False)
    return C.save(fig, out_dir, "fig03_solvability_partition")
