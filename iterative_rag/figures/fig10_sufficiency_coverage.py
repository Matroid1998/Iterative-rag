"""Figure 10 — Sufficiency-Coverage interaction heatmap.

Bins sufficiency_score_est (<0.4, 0.4-0.6, >=0.6) x hop_coverage_est (<0.8, >=0.8);
each cell is the average accuracy across all model-question runs in that cell.
"""

from __future__ import annotations

from pathlib import Path

from iterative_rag.figures import common as C

SUFF_BINS = [(-0.01, 0.4, "< 0.4"), (0.4, 0.6, "0.4-0.6"), (0.6, 1.01, ">= 0.6")]
COV_BINS = [(0.8, 1.01, ">= 0.8"), (-0.01, 0.8, "< 0.8")]  # top row = high coverage


def render(out_dir: Path) -> Path:
    import numpy as np
    import matplotlib.pyplot as plt

    C.use_style()
    hal = C.load_judgments("hallucination")
    it = C.correct_by_question("iterative")

    cells = {(r, c): [] for r in range(len(COV_BINS)) for c in range(len(SUFF_BINS))}
    for m, judg in hal.items():
        if m not in it:
            continue
        for q, pj in judg.items():
            if q not in it[m]:
                continue
            suff = (pj.get("composition_and_faithfulness", {}) or {}).get("sufficiency_score_est")
            cov = (pj.get("confidence_miscalibration", {}) or {}).get("hop_coverage_est")
            if suff is None or cov is None:
                continue
            ci = next((i for i, (lo, hi, _) in enumerate(COV_BINS) if lo < cov <= hi), None)
            si = next((i for i, (lo, hi, _) in enumerate(SUFF_BINS) if lo < suff <= hi), None)
            if ci is None or si is None:
                continue
            cells[(ci, si)].append(it[m][q])

    mat = np.full((len(COV_BINS), len(SUFF_BINS)), np.nan)
    counts = np.zeros_like(mat)
    for (r, c), vals in cells.items():
        if vals:
            mat[r, c] = 100 * np.mean(vals)
            counts[r, c] = len(vals)
    if np.all(np.isnan(mat)):
        raise RuntimeError("no hallucination judgments available")

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(SUFF_BINS)), [b[2] for b in SUFF_BINS])
    ax.set_yticks(range(len(COV_BINS)), [b[2] for b in COV_BINS])
    ax.set_xlabel("Sufficiency Score")
    ax.set_ylabel("Hop Coverage")
    for r in range(len(COV_BINS)):
        for c in range(len(SUFF_BINS)):
            if not np.isnan(mat[r, c]):
                ax.text(c, r, f"{mat[r, c]:.1f}%\nn={int(counts[r, c])}", ha="center", va="center",
                        fontsize=10, fontweight="bold")
    ax.set_title("Sufficiency-Coverage Interaction")
    fig.colorbar(im, ax=ax, label="Accuracy (%)")
    ax.grid(False)
    return C.save(fig, out_dir, "fig10_sufficiency_coverage")
