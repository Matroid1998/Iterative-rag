"""Figure 2 — Gaussian distributions of accuracy across No-Context / Gold-Context /
Iterative RAG, with per-model scatter and pairwise t-test significance bars."""

from __future__ import annotations

from pathlib import Path

from iterative_rag.figures import common as C

REGIMES = [("no_context", "No Context", "blue"),
           ("gold", "Gold Context", "red"),
           ("iterative", "Iterative RAG", "green")]


def render(out_dir: Path) -> Path:
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    C.use_style()
    series = {key: [v for v in C.accuracy_by_model(key).values()] for key, _, _ in REGIMES}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    xs = np.linspace(0, 100, 400)
    stats_by = {}
    for key, label, color in REGIMES:
        vals = np.array(series[key], dtype=float)
        if len(vals) == 0:
            continue
        mu, sd = float(vals.mean()), float(vals.std(ddof=0) or 1.0)
        stats_by[key] = vals
        ax.plot(xs, stats.norm.pdf(xs, mu, sd), color=color, lw=2,
                label=f"{label} (μ={mu:.2f}, σ={sd:.2f})")
        ax.axvline(mu, color=color, ls="--", alpha=0.5)
        ax.scatter(vals, np.full_like(vals, 0.001) + np.random.uniform(0, 0.004, len(vals)),
                   color=color, alpha=0.6, s=25, zorder=5)

    # pairwise t-tests
    order = [k for k, _, _ in REGIMES if k in stats_by]
    y0 = ax.get_ylim()[1]
    pairs = [(order[i], order[j]) for i in range(len(order)) for j in range(i + 1, len(order))]
    for n, (a, b) in enumerate(pairs):
        t, p = stats.ttest_ind(stats_by[a], stats_by[b], equal_var=False)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ya = y0 * (1.03 + 0.06 * n)
        xa, xb = stats_by[a].mean(), stats_by[b].mean()
        ax.plot([xa, xb], [ya, ya], color="black", lw=1)
        ax.text((xa + xb) / 2, ya, sig, ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Gaussian Distributions with Significance Testing")
    ax.legend(loc="upper left", fontsize=9)
    return C.save(fig, out_dir, "fig02_accuracy_distributions")
