"""Figure 15 — Adaptivity vs Predictability.

x: Token Scaling Factor S = mean output tokens(hard) / mean output tokens(easy).
y: Token Usage Consistency = mean coefficient of variation (%) of output tokens across
   the easy/medium/hard difficulty buckets. One point per model.
"""

from __future__ import annotations

from pathlib import Path

from iterative_rag.figures import common as C


def render(out_dir: Path) -> Path:
    import numpy as np
    import matplotlib.pyplot as plt

    C.use_style()
    df = C.iterative_dataframe()
    diff = C.difficulty_map(df)
    df = df.assign(difficulty=df["question"].map(diff))

    fig, ax = plt.subplots(figsize=(11, 7))
    for m in C.PAPER_MODELS:
        sub = df[df.model == m]
        if sub.empty:
            continue
        by = {d: sub[sub.difficulty == d]["output_tokens"] for d in ("easy", "medium", "hard")}
        if by["easy"].mean() and by["hard"].mean():
            S = by["hard"].mean() / by["easy"].mean()
        else:
            continue
        cvs = []
        for d in ("easy", "medium", "hard"):
            vals = by[d]
            if len(vals) and vals.mean():
                cvs.append(100 * vals.std() / vals.mean())
        cv = float(np.mean(cvs)) if cvs else 0.0
        ax.scatter(S, cv, s=120, color=C.color_for(m), edgecolor="black", zorder=5)
        ax.annotate(m, (S, cv), fontsize=9, xytext=(6, 4), textcoords="offset points")

    ax.set_xlabel("Token Scaling Factor (Hard/Easy Multiplier)\n(Higher = More Adaptive Effort)")
    ax.set_ylabel("Token Usage Consistency (Average CV %)\n(Lower = More Predictable)")
    ax.set_title("Adaptivity vs. Predictability")
    return C.save(fig, out_dir, "fig15_adaptivity_predictability")
