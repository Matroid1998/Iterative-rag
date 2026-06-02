"""Figure 14 — Iterative RAG cost vs accuracy per model.

Cost is estimated from total input/output tokens times approximate per-model API prices
(USD per 1M tokens). Prices are best-effort public list prices and can be edited below.
"""

from __future__ import annotations

from pathlib import Path

from iterative_rag.figures import common as C

# Approximate USD per 1M tokens: (input, output). Best-effort; edit as needed.
PRICING = {
    "GPT-4o": (2.5, 10.0),
    "GPT-5": (1.25, 10.0),
    "Claude 3.7 Sonnet": (3.0, 15.0),
    "Claude 3.7 Sonnet Thinking": (3.0, 15.0),
    "Claude Sonnet 4.5": (3.0, 15.0),
    "DeepSeek R1": (0.55, 2.19),
    "Llama 3.3 70B Instruct": (0.72, 0.72),
    "Mistral Large 2402": (4.0, 12.0),
    "Gemini 2.5 Pro": (1.25, 10.0),
    "Grok 4 Fast": (0.20, 0.50),
    "GLM 4.6": (0.60, 2.20),
}


def render(out_dir: Path) -> Path:
    import numpy as np
    import matplotlib.pyplot as plt

    C.use_style()
    df = C.iterative_dataframe()
    acc = C.accuracy_by_model("iterative")
    fig, ax = plt.subplots(figsize=(11, 7))
    for m in C.PAPER_MODELS:
        if m not in PRICING or m not in acc:
            continue
        sub = df[df.model == m]
        pin, pout = PRICING[m]
        cost = sub["input_tokens"].sum() / 1e6 * pin + sub["output_tokens"].sum() / 1e6 * pout
        ax.scatter(cost, acc[m], s=120, color=C.color_for(m), edgecolor="black", zorder=5)
        ax.annotate(m, (cost, acc[m]), fontsize=9, xytext=(6, 4), textcoords="offset points")
    ax.set_xlabel("Total Cost ($, estimated)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Cost vs Accuracy")
    return C.save(fig, out_dir, "fig14_cost_vs_accuracy")
