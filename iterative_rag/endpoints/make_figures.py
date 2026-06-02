#!/usr/bin/env python3
"""Endpoint 4 — regenerate the paper figures (only).

Reads the shipped response JSONL (``responses/``), diagnostic judgments
(``diagnostics_output/``), and the QA dataset, and renders every figure that appears in the
paper into ``paper_figures/``. No LLM calls are made, so this runs fully offline.

Examples
--------
    irag-figures                       # render all paper figures
    irag-figures --only fig07,figS03   # render a subset
    irag-figures --out-dir /tmp/figs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from iterative_rag import config
from iterative_rag.figures.run_all import run_all, FIGURES


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Regenerate the paper figures.")
    ap.add_argument("--out-dir", default=str(config.FIGURES_DIR), help="Output directory for figures.")
    ap.add_argument("--only", default=None,
                    help="Comma-separated figure keys to render (e.g. fig07,figS03). Default: all. "
                         f"Available: {','.join(FIGURES)}")
    args = ap.parse_args(list(argv) if argv is not None else None)

    only = [k.strip() for k in args.only.split(",")] if args.only else None
    run_all(out_dir=Path(args.out_dir), only=only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
