"""Figure S14 — Gold-context token-length distribution.

Overlaid distributions for (i) questions answered correctly in Gold Context and (ii) questions
wrong in Gold Context but correct in Iterative RAG, over all model-question pairs.
"""

from __future__ import annotations

from pathlib import Path

from iterative_rag.figures import common as C


def render(out_dir: Path) -> Path:
    import numpy as np
    import matplotlib.pyplot as plt

    C.use_style()
    tok = C.gold_context_tokens_map()
    gc = C.correct_by_question("gold")
    it = C.correct_by_question("iterative")
    if not gc or not it:
        raise RuntimeError("need both gold and iterative responses")

    correct_gc, improved = [], []
    for m in C.PAPER_MODELS:
        if m not in gc or m not in it:
            continue
        for q in set(gc[m]) & set(it[m]):
            n = tok.get(q)
            if n is None:
                continue
            if gc[m][q]:
                correct_gc.append(n)
            elif it[m][q]:           # wrong in GC, correct in iterative
                improved.append(n)

    if not correct_gc and not improved:
        raise RuntimeError("no matching gold/iterative questions")
    hi = int(np.percentile(correct_gc + improved, 99)) if (correct_gc + improved) else 1300
    bins = np.linspace(0, max(hi, 100), 40)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(improved, bins=bins, density=True, alpha=0.6, color="#2ca02c",
            label="Wrong in GC → Correct in Iterative RAG")
    ax.hist(correct_gc, bins=bins, density=True, histtype="step", lw=2, color="black",
            label="Correct in GC")
    ax.set_xlabel("Gold Context Token Count")
    ax.set_ylabel("Density")
    ax.set_title("Token Length Distribution: Correct in GC vs Iterative RAG-Improved Questions")
    ax.legend()
    return C.save(fig, out_dir, "figS14_token_length_distribution")
