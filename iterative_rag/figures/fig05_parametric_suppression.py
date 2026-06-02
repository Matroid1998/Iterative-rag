"""Figure 5 — Parametric Suppression Rate (PSR).

PSR = |No-Context correct AND Iterative incorrect| / |No-Context correct|, per model:
the fraction of originally-correct (parametric) answers suppressed by retrieval.
"""

from __future__ import annotations

from pathlib import Path

from iterative_rag.figures import common as C


def render(out_dir: Path) -> Path:
    import numpy as np
    import matplotlib.pyplot as plt

    C.use_style()
    nc = C.correct_by_question("no_context")
    it = C.correct_by_question("iterative")
    models, psr = [], []
    for m in C.PAPER_MODELS:
        if m not in nc or m not in it:
            continue
        qs = set(nc[m]) & set(it[m])
        correct_nc = [q for q in qs if nc[m][q]]
        if not correct_nc:
            continue
        suppressed = sum(1 for q in correct_nc if not it[m][q])
        models.append(m)
        psr.append(100.0 * suppressed / len(correct_nc))

    avg = float(np.mean(psr)) if psr else 0.0
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, psr, color="#9aa0a6", edgecolor="black")
    for xi, v in zip(x, psr):
        ax.text(xi, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.axhline(avg, color="red", ls="--", label=f"Average: {avg:.1f}%")
    ax.set_ylabel("Parametric Suppression Rate (%)")
    ax.set_xlabel("Models")
    ax.set_title("Parametric Suppression Rate (PSR): Proportion of No-Context Correct Answers Suppressed by Iterative RAG")
    ax.set_xticks(x, models, rotation=45, ha="right")
    ax.legend()
    return C.save(fig, out_dir, "fig05_parametric_suppression")
