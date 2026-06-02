"""Figure 7 — Anchor Carry-Drop Rate by step and model (from coverage-gap judgments)."""

from __future__ import annotations

from pathlib import Path

from iterative_rag.figures import common as C


def render(out_dir: Path) -> Path:
    import numpy as np
    import matplotlib.pyplot as plt

    C.use_style()
    cov = C.load_judgments("coverage")
    fig, ax = plt.subplots(figsize=(11, 6))
    max_step = 5
    plotted = 0
    for model in C.PAPER_MODELS:
        judg = cov.get(model)
        if not judg:
            continue
        # accumulate carry_drop true/total per step
        num = np.zeros(max_step + 1)
        den = np.zeros(max_step + 1)
        for pj in judg.values():
            for entry in (pj.get("anchor_carry_drop", {}) or {}).get("per_step", []) or []:
                s = entry.get("step")
                if isinstance(s, int) and 1 <= s <= max_step:
                    den[s] += 1
                    if entry.get("carry_drop"):
                        num[s] += 1
        steps = [s for s in range(1, max_step + 1) if den[s] > 0]
        if not steps:
            continue
        rates = [100 * num[s] / den[s] for s in steps]
        ax.plot(steps, rates, marker="o", color=C.color_for(model), label=model, lw=1.8)
        plotted += 1

    if plotted == 0:
        raise RuntimeError("no coverage judgments with anchor_carry_drop available")
    ax.set_xlabel("Step Number")
    ax.set_ylabel("Anchor Carry-Drop Rate (%)")
    ax.set_title("Anchor Carry-Drop Rate by Step and Model\n(Higher % = More anchor loss)")
    ax.set_xticks(range(1, max_step + 1))
    ax.legend(fontsize=7, ncol=2)
    return C.save(fig, out_dir, "fig07_anchor_carry_drop")
