"""Figure 11 — Miscalibration analysis (No-Context-wrong questions).

(a) average accuracy by calibration state; (b) per-model stacked calibration-state
distribution by hop count.
"""

from __future__ import annotations

from pathlib import Path

from iterative_rag.figures import common as C

STATE_OF = {"overconfident_finalize": "Overconfident",
            "underconfident_continue": "Underconfident",
            "ok": "Well-Calibrated"}
STATE_COLORS = {"Well-Calibrated": "#2ca02c", "Underconfident": "#1f77b4", "Overconfident": "#d62728"}


def _rows():
    hal = C.load_judgments("hallucination")
    it = C.correct_by_question("iterative")
    nc = C.correct_by_question("no_context")
    hops_map = C._qa_hops_map()
    rows = []  # (model, state, hop, is_correct)
    for m, judg in hal.items():
        if m not in it:
            continue
        for q, pj in judg.items():
            if m in nc and nc[m].get(q):
                continue
            if q not in it[m]:
                continue
            direction = (pj.get("confidence_miscalibration", {}) or {}).get("direction", "ok")
            state = STATE_OF.get(direction, "Well-Calibrated")
            rows.append((m, state, hops_map.get(q, 0), it[m][q]))
    return rows


def render(out_dir: Path) -> Path:
    import numpy as np
    import matplotlib.pyplot as plt

    C.use_style()
    rows = _rows()
    if not rows:
        raise RuntimeError("no hallucination judgments available")

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.05, 2.4])
    axA = fig.add_subplot(gs[0, 0])

    states = ["Well-Calibrated", "Overconfident", "Underconfident"]
    accs = []
    for st in states:
        vals = [c for _, s, _, c in rows if s == st]
        accs.append(100 * np.mean(vals) if vals else 0)
    axA.bar(states, accs, color=[STATE_COLORS[s] for s in states], edgecolor="black")
    for i, v in enumerate(accs):
        axA.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold")
    axA.set_ylabel("Average Accuracy (%)")
    axA.set_ylim(0, 100)
    axA.set_title("(a) Accuracy by Calibration State")
    axA.tick_params(axis="x", rotation=20)

    # (b) per-model grid
    models = [m for m in C.PAPER_MODELS if any(r[0] == m for r in rows)]
    gridspec = gs[0, 2].subgridspec(3, 4, hspace=0.6, wspace=0.3)
    for idx, m in enumerate(models):
        ax = fig.add_subplot(gridspec[idx // 4, idx % 4])
        bottoms = np.zeros(4)
        for st in states:
            fracs = []
            for hop in [1, 2, 3, 4]:
                tot = sum(1 for mm, s, h, _ in rows if mm == m and h == hop)
                cnt = sum(1 for mm, s, h, _ in rows if mm == m and h == hop and s == st)
                fracs.append(100 * cnt / tot if tot else 0)
            ax.bar([1, 2, 3, 4], fracs, bottom=bottoms, color=STATE_COLORS[st], width=0.8)
            bottoms += np.array(fracs)
        ax.set_title(m, fontsize=8)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_ylim(0, 100)
        ax.tick_params(labelsize=7)
        ax.grid(False)
    fig.suptitle("Miscalibration Analysis  |  (b) Calibration State by Hop Depth (per model)", fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=STATE_COLORS[s]) for s in states]
    fig.legend(handles, states, loc="lower center", ncol=3)
    return C.save(fig, out_dir, "fig11_miscalibration")
