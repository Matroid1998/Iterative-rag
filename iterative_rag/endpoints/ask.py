#!/usr/bin/env python3
"""Endpoint 2 — run the iterative RAG system on a single question.

Builds the RagTextService against an existing Chroma collection (created by ``irag-index``)
and runs the planner/retrieve/compose loop, printing the answer, citations, and the
action trace. The planner/composer LLM is taken from ``iterative_rag.config`` (provider +
``OPENAI_API_KEY`` from the environment); if no LLM is available it falls back to the
rule-based planner.

Examples
--------
    irag-ask --question "Which (1,4)-linked unit is the building block of cyclodextrins?"
    irag-ask -q "What is the pKa of formic acid?" --collection chemrxiv_graph --json
"""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from iterative_rag import config


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Ask the iterative RAG system a single question.")
    ap.add_argument("--question", "-q", required=True, help="The question to answer.")
    ap.add_argument("--persist", default=str(config.CHROMA_DIR), help="Chroma persist directory.")
    ap.add_argument("--collection", default=config.DEFAULT_COLLECTION, help="Chroma collection name.")
    ap.add_argument("--model-name", default=config.DEFAULT_EMBED_MODEL, help="SentenceTransformer embedding model.")
    ap.add_argument("--device", default=config.DEFAULT_DEVICE, help="Embedding device.")
    ap.add_argument("--k", type=int, default=8, help="Top-k passages per retrieval.")
    ap.add_argument("--max-steps", type=int, default=6, help="Maximum retrieval steps.")
    ap.add_argument("--json", action="store_true", help="Print the full result as JSON.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    # Imported here so `--help` works without the heavy ML stack installed.
    from iterative_rag.indexing.embedding_config import EmbedderConfig
    from iterative_rag.system.service import RagTextService

    svc = RagTextService(
        persist_path=args.persist,
        collection_name=args.collection,
        embedder_cfg=EmbedderConfig(model_name=args.model_name, device=args.device),
        planner=None,      # built from iterative_rag.config (JSON planner or rule-based fallback)
        composer=None,
        max_steps=args.max_steps,
    )

    result = svc.answer(args.question, k_default=args.k, trace=True)

    if args.json:
        print(json.dumps(result, default=str, indent=2, ensure_ascii=False))
        return 0

    print(f"\nQuestion: {args.question}")
    print(f"\nAnswer:\n{result.get('answer')}")
    print(f"\nSteps: {result.get('steps')} | stop_reason: {result.get('stop_reason')}")
    citations = result.get("citations") or []
    if citations:
        print("\nCitations:")
        for c in citations:
            print(" -", c)
    actions = result.get("actions_trace") or []
    if actions:
        print("\nActions trace:")
        for a in actions:
            print(" -", a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
