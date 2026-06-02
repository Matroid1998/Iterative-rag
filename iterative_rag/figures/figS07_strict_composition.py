"""Figure S7 — Strict composition failure rate (excluding coverage gaps).

Among incorrect answers where ALL oracle hops were retrieved (no coverage gap), the fraction
that still fail composition — isolating synthesis failures from retrieval misses.
"""

from __future__ import annotations

from pathlib import Path

from iterative_rag.figures import common as C


def render(out_dir: Path) -> Path:
    import numpy as np
    import matplotlib.pyplot as plt

    C.use_style()
    hal = C.load_judgments("hallucination")
    cov = C.load_judgments("coverage")
    it = C.correct_by_question("iterative")

    models, rates, labels = [], [], []
    for m in C.PAPER_MODELS:
        hj, cj = hal.get(m), cov.get(m)
        if not hj or not cj or m not in it:
            continue
        denom = numer = 0
        for q, pj in hj.items():
            if q not in it[m] or it[m][q]:           # incorrect only
                continue
            cgap = (cov.get(m, {}).get(q, {}).get("retrieval_coverage_gap", {}) or {}).get("has_gap")
            if cgap:                                 # exclude coverage gaps
                continue
            denom += 1
            if (pj.get("composition_and_faithfulness", {}) or {}).get("composition_failure"):
                numer += 1
        if denom:
            models.append(m)
            rates.append(100 * numer / denom)
            labels.append(f"{numer}/{denom}")

    if not models:
        raise RuntimeError("no joined hallucination+coverage judgments available")
    avg = float(np.mean(rates))
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, rates, color="#8c9eb2", edgecolor="black")
    for xi, v, lab in zip(x, rates, labels):
        ax.text(xi, v, f"{v:.1f}%\n({lab})", ha="center", va="bottom", fontsize=7)
    ax.axhline(avg, color="red", ls="--", label=f"Average: {avg:.1f}%")
    ax.set_ylabel("Composition Failure Rate (%)")
    ax.set_xlabel("Model")
    ax.set_title("Strict Composition Failure Rate by Model\n(% of incorrect answers with composition failure, excluding coverage gaps)")
    ax.set_xticks(x, models, rotation=45, ha="right")
    ax.legend()
    return C.save(fig, out_dir, "figS07_strict_composition")
