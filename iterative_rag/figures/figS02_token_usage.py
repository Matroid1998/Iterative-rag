"""Figure S2 — Token utilization.

(a) average output tokens for correct vs incorrect answers per model;
(b) scatter of average output tokens vs iterative-RAG accuracy.
"""

from __future__ import annotations

from pathlib import Path

from iterative_rag.figures import common as C


def render(out_dir: Path) -> Path:
    import numpy as np
    import matplotlib.pyplot as plt

    C.use_style()
    df = C.iterative_dataframe()
    models = [m for m in C.PAPER_MODELS if m in df.model.unique()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # (a) correct vs incorrect output tokens
    corr = [df[(df.model == m) & (df.is_correct)]["output_tokens"].mean() for m in models]
    wrong = [df[(df.model == m) & (~df.is_correct)]["output_tokens"].mean() for m in models]
    x = np.arange(len(models))
    w = 0.4
    ax1.bar(x - w / 2, corr, w, label="Correct", color="#2ca02c", edgecolor="black")
    ax1.bar(x + w / 2, wrong, w, label="Incorrect", color="#d62728", edgecolor="black")
    ax1.set_ylabel("Average Output Tokens")
    ax1.set_title("(a) Average Output Tokens by Correctness")
    ax1.set_xticks(x, models, rotation=45, ha="right")
    ax1.legend()

    # (b) avg output tokens vs accuracy
    acc = C.accuracy_by_model("iterative")
    for m in models:
        avg_tok = df[df.model == m]["output_tokens"].mean()
        ax2.scatter(acc.get(m, np.nan), avg_tok, s=90, color=C.color_for(m), edgecolor="black", zorder=5)
        ax2.annotate(m, (acc.get(m, np.nan), avg_tok), fontsize=8,
                     xytext=(5, 5), textcoords="offset points")
    ax2.set_xlabel("Accuracy in Iterative RAG (%)")
    ax2.set_ylabel("Average Output Tokens")
    ax2.set_title("(b) Output Tokens vs Accuracy")
    fig.tight_layout()
    return C.save(fig, out_dir, "figS02_token_usage")
