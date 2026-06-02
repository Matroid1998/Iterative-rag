#!/usr/bin/env python3
"""Endpoint 5 — generate LLM-as-judge diagnostics for the failure-mode figures.

Runs the diagnostic auditors (paper Figures S8/S9/S10) over an iterative-RAG response
JSONL and writes the judgment JSONL files into ``diagnostics_output/`` that the failure-mode
figures (coverage gap, sufficiency/calibration, query quality, distractor latch, ...) consume.

The judge LLM defaults to ``iterative_rag.config.JUDGE_MODEL`` and needs its provider's API
key in the environment.

Examples
--------
    # All three diagnostics for one model's responses
    irag-diagnose --responses responses/responses_reverified/responses_openai_gpt-4o_reverified.jsonl

    # Just coverage-gap, first 2 records (smoke test)
    irag-diagnose --responses responses/.../<file>.jsonl --kind coverage --limit 2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from iterative_rag import config
from iterative_rag.diagnostics.judge import run_diagnostics
from iterative_rag.diagnostics.prompts import DIAGNOSTICS


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate LLM-judge diagnostics for the figures.")
    ap.add_argument("--responses", required=True, help="Path to an iterative-RAG response JSONL.")
    ap.add_argument("--kind", choices=list(DIAGNOSTICS) + ["all"], default="all",
                    help="Which diagnostic(s) to produce.")
    ap.add_argument("--judge-provider", default=config.JUDGE_PROVIDER, help="Judge LLM provider.")
    ap.add_argument("--judge-model", default=config.JUDGE_MODEL, help="Judge LLM model id.")
    ap.add_argument("--out-dir", default=str(config.DIAGNOSTICS_DIR), help="Output directory.")
    ap.add_argument("--limit", type=int, default=None, help="Judge only N records (debug).")
    ap.add_argument("--workers", type=int, default=2, help="Parallel judge workers.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    kinds = list(DIAGNOSTICS) if args.kind == "all" else [args.kind]
    for kind in kinds:
        run_diagnostics(
            Path(args.responses),
            kind,
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
            out_dir=Path(args.out_dir),
            limit=args.limit,
            workers=args.workers,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
