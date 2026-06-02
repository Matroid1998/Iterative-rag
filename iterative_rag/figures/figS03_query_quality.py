"""Figure S3 — Query Quality analysis (from query-quality judgments).

(a) accuracy with vs without each query flag (Vague, Over-Broad, Off-Topic, Fusion);
(b) per-model prevalence of each flag across steps.
"""

from __future__ import annotations

from pathlib import Path

from iterative_rag.figures import common as C

FLAGS = [("vague", "Vague"), ("over_broad", "Over-Broad"), ("off_topic", "Off-Topic"), ("fusion", "Fusion")]
FLAG_COLORS = {"Fusion": "#4e79a7", "Over-Broad": "#e15759", "Vague": "#f1a340", "Off-Topic": "#59a14f"}


def render(out_dir: Path) -> Path:
    import numpy as np
    import matplotlib.pyplot as plt

    C.use_style()
    qual = C.load_judgments("quality")
    it = C.correct_by_question("iterative")

    # question-level flag presence (any step) + per-model step prevalence
    agg = {f: {"with": [], "without": []} for f, _ in FLAGS}
    prevalence = {m: {f: (0, 0) for f, _ in FLAGS} for m in C.PAPER_MODELS}
    for m, judg in qual.items():
        if m not in it:
            continue
        for q, pj in judg.items():
            steps = pj.get("per_step", []) or []
            if not steps or q not in it[m]:
                continue
            for f, _ in FLAGS:
                present = False
                cnt = 0
                for st in steps:
                    val = st.get("fusion") if f == "fusion" else (st.get("query_quality", {}) or {}).get(f)
                    if val:
                        present = True
                        cnt += 1
                agg[f]["with" if present else "without"].append(it[m][q])
                tot, c = prevalence[m][f]
                prevalence[m][f] = (tot + len(steps), c + cnt)

    fig = plt.figure(figsize=(16, 11))
    # (a) 2x2 flag impact
    gsA = fig.add_gridspec(2, 2, left=0.06, right=0.5, top=0.92, bottom=0.1, hspace=0.5, wspace=0.4)
    for i, (f, label) in enumerate(FLAGS):
        ax = fig.add_subplot(gsA[i // 2, i % 2])
        w = agg[f]["with"]; wo = agg[f]["without"]
        aw = 100 * np.mean(w) if w else 0
        awo = 100 * np.mean(wo) if wo else 0
        ax.bar([f"Without\n{label}", f"With\n{label}"], [awo, aw],
               color=["#2ca02c", "#d62728"], edgecolor="black")
        for j, v in enumerate([awo, aw]):
            ax.text(j, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.set_title(f"{label} Query Impact", fontsize=9)
        ax.tick_params(labelsize=7)

    # (b) per-model prevalence grouped bars
    axB = fig.add_axes([0.57, 0.1, 0.4, 0.82])
    models = [m for m in C.PAPER_MODELS if any(prevalence[m][f][0] for f, _ in FLAGS)]
    x = np.arange(len(models))
    nb = len(FLAGS)
    width = 0.8 / nb
    for k, (f, label) in enumerate(FLAGS):
        vals = [100 * prevalence[m][f][1] / prevalence[m][f][0] if prevalence[m][f][0] else 0 for m in models]
        axB.bar(x + (k - nb / 2) * width + width / 2, vals, width, color=FLAG_COLORS[label], label=label)
    axB.set_xticks(x, models, rotation=45, ha="right", fontsize=8)
    axB.set_ylabel("Percentage of Steps (%)")
    axB.set_title("Query Characteristics by Model")
    axB.legend(fontsize=8)
    fig.suptitle("Query Quality Analysis", fontweight="bold")
    return C.save(fig, out_dir, "figS03_query_quality")
