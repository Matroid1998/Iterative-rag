"""Figure S13 — Effect of Gold-Context length on Gold-Context accuracy.

Questions are binned by gold-context token length; each bar is the average Gold-Context
accuracy across models in that bin, shaded by sample size.
"""

from __future__ import annotations

from pathlib import Path

from iterative_rag.figures import common as C

BINS = [(0, 150), (150, 300), (300, 450), (450, 600), (600, 900), (900, 1200), (1200, 2000)]


def render(out_dir: Path) -> Path:
    import numpy as np
    import matplotlib.pyplot as plt

    C.use_style()
    tok = C.gold_context_tokens_map()
    gc = C.correct_by_question("gold")
    if not gc:
        raise RuntimeError("no gold-context responses available")

    # per question: average gold accuracy across models + token length
    q_acc = {}
    for m, d in gc.items():
        for q, ok in d.items():
            q_acc.setdefault(q, []).append(ok)

    bin_vals = [[] for _ in BINS]
    for q, oks in q_acc.items():
        n = tok.get(q)
        if n is None:
            continue
        for i, (lo, hi) in enumerate(BINS):
            if lo <= n < hi:
                bin_vals[i].append(100 * np.mean(oks))
                break

    means = [np.mean(v) if v else 0 for v in bin_vals]
    counts = [len(v) for v in bin_vals]
    labels = [f"{lo}-{hi}\nn={c}" for (lo, hi), c in zip(BINS, counts)]
    overall = np.mean([m for m, c in zip(means, counts) if c]) if any(counts) else 0

    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.cm.Blues
    norm = plt.Normalize(0, max(counts) or 1)
    x = np.arange(len(BINS))
    ax.bar(x, means, color=[cmap(0.3 + 0.7 * norm(c)) for c in counts], edgecolor="black")
    ax.axhline(overall, color="red", ls="--", label=f"Overall mean: {overall:.1f}%")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_xlabel("Gold Context Token Count")
    ax.set_ylabel("Average Accuracy Across Models (%)")
    ax.set_title("Effect of Gold Context Length on Accuracy")
    ax.legend()
    return C.save(fig, out_dir, "figS13_gold_context_length")
