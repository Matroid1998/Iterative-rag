"""Render all paper figures into the output directory.

Each figure lives in its own module exposing ``render(out_dir) -> Path``. ``FIGURES`` is the
ordered registry (module suffix -> import path). ``run_all`` imports and renders each,
catching errors so one missing input doesn't abort the rest.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import List, Optional

from iterative_rag import config

# Registry: short key -> module name under iterative_rag.figures
FIGURES = {
    "fig02": "fig02_accuracy_distributions",
    "fig03": "fig03_solvability_partition",
    "fig04": "fig04_recoveries_regressions",
    "fig05": "fig05_parametric_suppression",
    "fig06": "fig06_unanswered_by_hops",
    "fig07": "fig07_anchor_carry_drop",
    "fig08": "fig08_perf_by_retrieval_step",
    "fig09": "fig09_coverage_gap_impact",
    "fig10": "fig10_sufficiency_coverage",
    "fig11": "fig11_miscalibration",
    "fig12": "fig12_composition_failure",
    "fig13": "fig13_distractor_latch",
    "fig14": "fig14_cost_vs_accuracy",
    "fig15": "fig15_adaptivity_predictability",
    "figS01": "figS01_perf_by_step_nocontext",
    "figS02": "figS02_token_usage",
    "figS03": "figS03_query_quality",
    "figS07": "figS07_strict_composition",
    "figS13": "figS13_gold_context_length",
    "figS14": "figS14_token_length_distribution",
}


def run_all(out_dir: Optional[Path] = None, only: Optional[List[str]] = None) -> List[Path]:
    out_dir = Path(out_dir) if out_dir else config.FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = only or list(FIGURES)
    written: List[Path] = []
    for key in keys:
        mod_name = FIGURES.get(key)
        if not mod_name:
            print(f"  [skip] unknown figure key: {key}")
            continue
        try:
            mod = importlib.import_module(f"iterative_rag.figures.{mod_name}")
            path = mod.render(out_dir)
            written.append(Path(path))
            print(f"  [ok]   {key} -> {Path(path).name}")
        except Exception as e:  # keep going; report the failure
            print(f"  [fail] {key}: {type(e).__name__}: {e}")
    print(f"\nWrote {len(written)} figure(s) to {out_dir}")
    return written
