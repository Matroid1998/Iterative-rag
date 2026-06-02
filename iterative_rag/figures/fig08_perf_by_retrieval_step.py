"""Figure 8 — Model performance by finalized retrieval step (questions failed by Gold Context).

Per-model grid: stacked bars = #questions finalized at each step, colored by oracle hop depth
(right axis); green line = Iterative-RAG accuracy on exactly those questions (left axis).
"""

from __future__ import annotations

from pathlib import Path

from iterative_rag.figures import common as C

HOP_COLORS = {1: "#e15759", 2: "#f1a340", 3: "#4e79a7", 4: "#9467bd"}
MAX_STEP = 5


def _grid(out_dir, name, title, subset_correct_map, second_line=None, second_label=None):
    import numpy as np
    import matplotlib.pyplot as plt

    C.use_style()
    df = C.iterative_dataframe()
    models = [m for m in C.PAPER_MODELS if m in df.model.unique()]
    ncol = 4
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for idx, m in enumerate(models):
        ax = axes[idx]
        sub = df[df.model == m]
        # restrict to the subset (e.g., questions where Gold/No-context was wrong)
        keep = subset_correct_map.get(m, {})
        sub = sub[sub["question"].isin([q for q, ok in keep.items() if not ok])]
        steps = list(range(1, MAX_STEP + 1))
        bottoms = np.zeros(len(steps))
        axb = ax.twinx()
        for hop in [1, 2, 3, 4]:
            counts = [int(((sub.steps == s) & (sub.hops == hop)).sum()) for s in steps]
            axb.bar(steps, counts, bottom=bottoms, color=HOP_COLORS[hop], width=0.8,
                    label=f"{hop} hop" + ("s" if hop > 1 else ""))
            bottoms += np.array(counts)
        # iterative accuracy line on the subset
        acc = []
        for s in steps:
            ss = sub[sub.steps == s]
            acc.append(100 * ss["is_correct"].mean() if len(ss) else np.nan)
        ax.plot(steps, acc, "g-o", lw=2, zorder=5)
        if second_line is not None:
            sl = [second_line(m, s, sub) for s in steps]
            ax.plot(steps, sl, color="#ff7f0e", ls="--", marker="s", lw=1.6, zorder=5)
        ax.set_zorder(axb.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.set_title(m, fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_xticks(steps)
        ax.set_ylabel("Accuracy (%)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(False)
    for j in range(len(models), len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=HOP_COLORS[h]) for h in [1, 2, 3, 4]]
    labels = [f"{h} hop" + ("s" if h > 1 else "") for h in [1, 2, 3, 4]]
    if second_label:
        import matplotlib.lines as mlines
        handles += [mlines.Line2D([], [], color="g", marker="o", label="Iterative RAG"),
                    mlines.Line2D([], [], color="#ff7f0e", ls="--", marker="s", label=second_label)]
        labels += ["Iterative RAG", second_label]
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return C.save(fig, out_dir, name)


def render(out_dir: Path) -> Path:
    gc = C.correct_by_question("gold")
    return _grid(out_dir, "fig08_perf_by_retrieval_step",
                 "Model Performance by Retrieval Step (questions failed by Gold Context)", gc)
