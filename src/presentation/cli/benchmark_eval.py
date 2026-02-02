#!/usr/bin/env python3
"""Presentation entrypoint for benchmark evaluation."""

from __future__ import annotations

import argparse
import os
import runpy


def _apply_env(args: argparse.Namespace) -> None:
    if args.domain:
        os.environ["EVAL_DOMAIN"] = args.domain
    if args.provider:
        os.environ["EVAL_PROVIDER"] = args.provider
    if args.model:
        os.environ["EVAL_MODEL"] = args.model
    if args.limit is not None:
        os.environ["EVAL_LIMIT"] = str(args.limit)
    if args.workers is not None:
        os.environ["EVAL_WORKERS"] = str(args.workers)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation (wrapper for src/benchmark/evaluator.py)."
    )
    parser.add_argument(
        "--domain",
        choices=["chemistry", "legal"],
        help="Evaluation domain (default: chemistry)",
    )
    parser.add_argument("--provider", help="Provider name (e.g., openai, bedrock)")
    parser.add_argument("--model", help="Model id (provider-specific)")
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of questions (default: 5, evaluator respects EVAL_LIMIT)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of worker threads for evaluation (EVAL_WORKERS)",
    )

    args = parser.parse_args()
    _apply_env(args)

    runpy.run_module("benchmark.evaluator", run_name="__main__")


if __name__ == "__main__":
    main()
