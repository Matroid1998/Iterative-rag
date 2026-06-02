"""Figure 12 — Composition Failure Rate by model.

Among incorrect iterative answers, the fraction where the correct evidence was retrieved
but not synthesized correctly (composition_failure), from the hallucination judgments.
"""

from __future__ import annotations

from pathlib import Path

from iterative_rag.figures import common as C


def render(out_dir: Path) -> Path:
    import numpy as np
    import matplotlib.pyplot as plt

    C.use_style()
    hal = C.load_judgments("hallucination")
    it = C.correct_by_question("iterative")
    models, rates, labels = [], [], []
    for m in C.PAPER_MODELS:
        judg = hal.get(m)
        if not judg or m not in it:
            continue
        incorrect = [q for q in judg if q in it[m] and not it[m][q]]
        if not incorrect:
            continue
        fails = sum(1 for q in incorrect
                    if (judg[q].get("composition_and_faithfulness", {}) or {}).get("composition_failure"))
        models.append(m)
        rates.append(100 * fails / len(incorrect))
        labels.append(f"{fails}/{len(incorrect)}")

    if not models:
        raise RuntimeError("no hallucination judgments available")
    avg = float(np.mean(rates))
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, rates, color="#8c9eb2", edgecolor="black")
    for xi, v, lab in zip(x, rates, labels):
        ax.text(xi, v, f"{v:.1f}%\n({lab})", ha="center", va="bottom", fontsize=7)
    ax.axhline(avg, color="red", ls="--", label=f"Average: {avg:.1f}%")
    ax.set_ylabel("Composition Failure Rate (%)")
    ax.set_xlabel("Model")
    ax.set_title("Composition Failure Rate by Model\n(% of incorrect answers with composition failure)")
    ax.set_xticks(x, models, rotation=45, ha="right")
    ax.legend()
    return C.save(fig, out_dir, "fig12_composition_failure")
