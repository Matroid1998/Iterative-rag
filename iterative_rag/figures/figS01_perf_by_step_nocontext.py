"""Figure S1 — Performance by retrieval step (No-Context-wrong questions).

Same per-model grid as Figure 8 but restricted to questions the model got wrong with No
Context, with a second (orange dashed) line for Gold-Context accuracy on those questions.
"""

from __future__ import annotations

from pathlib import Path

from iterative_rag.figures import common as C
from iterative_rag.figures.fig08_perf_by_retrieval_step import _grid


def render(out_dir: Path) -> Path:
    nc = C.correct_by_question("no_context")
    gc = C.correct_by_question("gold")

    def gold_line(model, step, sub):
        import numpy as np
        qs = sub[sub.steps == step]["question"].tolist()
        gmap = gc.get(model, {})
        vals = [gmap[q] for q in qs if q in gmap]
        return 100 * np.mean(vals) if vals else np.nan

    return _grid(out_dir, "figS01_perf_by_step_nocontext",
                 "Model Performance by Retrieval Step (No-Context wrong)", nc,
                 second_line=gold_line, second_label="Gold Context")
