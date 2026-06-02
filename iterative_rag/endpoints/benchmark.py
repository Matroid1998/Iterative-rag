#!/usr/bin/env python3
"""Endpoint 3 — run the iterative RAG system over the whole dataset and score it.

Loads the ChemKGMultiHopQA questions, runs the iterative RAG pipeline on each one with
the chosen QA model, verifies each answer with the LLM entity-equivalence judge, writes
per-question results to ``responses/responses_<provider>_<model>.jsonl`` and an aggregate
accuracy / latency / token CSV to ``results/``.

Only the **Iterative RAG** regime is run here (the paper's No-Context / Gold-Context
baselines are out of scope for this endpoint).

Requires an API key for the chosen provider in the environment (e.g. ``OPENAI_API_KEY``,
``OPENROUTER_API_KEY``, or AWS credentials for Bedrock). The corpus must already be indexed
with ``irag-index`` (default collection ``chemrxiv_graph``).

Examples
--------
    irag-benchmark --provider openai --model gpt-4o
    irag-benchmark --provider openai --model gpt-4o --limit 50 --workers 4
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence

from iterative_rag import config
from iterative_rag.benchmark.dataset import load_records


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark iterative RAG over the full dataset.")
    ap.add_argument("--provider", required=True, help="LLM provider: openai | bedrock | openrouter | ollama | nvidia.")
    ap.add_argument("--model", required=True, help="Model id for the provider (e.g. gpt-4o).")
    ap.add_argument("--dataset", default=str(config.QA_DATASET), help="Path to the QA dataset JSON.")
    ap.add_argument("--limit", type=int, default=None, help="Evaluate only N (sampled) questions.")
    ap.add_argument("--workers", type=int, default=2, help="Number of parallel workers.")
    ap.add_argument("--collection", default=config.DEFAULT_COLLECTION, help="Chroma collection name.")
    ap.add_argument("--responses-dir", default=str(config.RESPONSES_DIR), help="Where to write the response JSONL.")
    ap.add_argument("--results-dir", default=str(config.RESULTS_DIR), help="Where to write the metrics CSV.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    # Imported here so `--help` works without the heavy ML stack installed.
    from iterative_rag.benchmark.evaluator import BenchmarkRunner, Provider

    try:
        provider = Provider(args.provider)
    except ValueError:
        raise SystemExit(
            f"Invalid provider '{args.provider}'. Choose one of: {[p.value for p in Provider]}"
        )

    # Number of workers is read from EVAL_WORKERS inside the engine.
    os.environ["EVAL_WORKERS"] = str(args.workers)
    # The engine resolves the chroma collection per-domain; override the default.
    os.environ["IRAG_COLLECTION"] = args.collection

    os.makedirs(args.responses_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    selection_file = None
    if args.limit:
        slug = f"{args.provider}_{args.model}".replace("/", "__").replace(":", "_")
        selection_file = os.path.join(args.responses_dir, f"selection_{slug}_limit{args.limit}.json")
    records = load_records(args.dataset, limit=args.limit, selection_file=selection_file)

    model_slug = args.model.replace("/", "__").replace(":", "_")
    results_file = os.path.join(args.results_dir, f"results_{args.provider}_{model_slug}.csv")

    print(f"Benchmarking iterative RAG | provider={args.provider} model={args.model} "
          f"| {len(records)} questions | collection={args.collection}")

    runner = BenchmarkRunner(
        records=records,
        responses_dir=args.responses_dir,
        results_file=results_file,
        domain="chemistry",
    )
    runner.run_benchmark_for_model(provider, args.model)
    print(f"Done. Responses in {args.responses_dir}; metrics in {results_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
