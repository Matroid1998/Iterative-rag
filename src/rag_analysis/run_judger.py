"""
Run LLM-based diagnostics over a reverified responses JSONL using gpt-5-mini.

For each record in the input JSONL, this tool:
  1) Builds the DATA_JSON payload via the same builder as prepare_llm_inputs.py
  2) Composes the judging prompt (system + user) using the embedded schema
  3) Calls OpenAI gpt-5-mini and parses the returned JSON diagnostics
  4) Writes one JSON line per record with: {question, expected_answer, diagnostics, model, date, error?}

Usage:
  python3 -u src/rag_analysis/run_judger.py \
    --jsonl src/responses_reverified/<file>.jsonl \
    --out   src/rag_analysis/judged_<file>_gpt5mini.jsonl \
    --workers 4

Env:
  OPENAI_API_KEY must be set.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Reuse the payload builder and prompt constants
# Ensure 'src' is on sys.path for package imports when running as a script
import sys as _sys
from pathlib import Path as _Path
_SRC_BASE = _Path(__file__).resolve().parents[1]
if str(_SRC_BASE) not in _sys.path:
    _sys.path.insert(0, str(_SRC_BASE))

from rag_analysis.prepare_llm_inputs import (  # type: ignore
    load_ground_truth_map,
    build_llm_input_payload,
    PROMPT_SYSTEM,
    PROMPT_INPUT_SCHEMA,
    PROMPT_OUTPUT_SHAPE,
)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from a text blob.
    Conservative regex that finds the first {...} balanced segment.
    """
    try:
        # Quick attempt: parse as-is
        return json.loads(text)
    except Exception:
        pass
    # Fallback: find first {...}
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        return None


def _call_openai(system: str, user: str, model: str = "gpt-5-mini", max_tokens: int = 1200) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _compose_user_prompt(payload: Dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"{PROMPT_INPUT_SCHEMA}DATA_JSON:\n{data_json}\n\n{PROMPT_OUTPUT_SHAPE}"


def _iter_records(jsonl_path: Path):
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _judge_one(rec: Dict[str, Any], gt_map: Dict[str, Dict[str, Any]], model: str) -> Dict[str, Any]:
    # Build DATA_JSON payload
    try:
        payload = build_llm_input_payload(rec, gt_map)
    except Exception as e:
        return {
            "question": (rec.get("raw") or {}).get("question") or rec.get("question"),
            "expected_answer": (rec.get("raw") or {}).get("expected") or rec.get("expected"),
            "diagnostics": None,
            "model": model,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "error": f"payload_build_failed: {e}",
        }

    system = PROMPT_SYSTEM
    user = _compose_user_prompt(payload)

    try:
        raw = _call_openai(system, user, model=model)
        parsed = _extract_json_object(raw)
        return {
            "question": payload.get("question"),
            "expected_answer": payload.get("expected_answer"),
            "diagnostics": parsed,
            "model": model,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "error": None if parsed is not None else "parse_failed",
        }
    except Exception as e:
        return {
            "question": payload.get("question"),
            "expected_answer": payload.get("expected_answer"),
            "diagnostics": None,
            "model": model,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "error": f"llm_failed: {e}",
        }


def run(
    jsonl_path: Path,
    out_path: Path,
    workers: int = 4,
    model: str = "gpt-5-mini",
    incorrect_only: bool = False,
    subset_out: Optional[Path] = None,
) -> None:
    gt_map = load_ground_truth_map()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Optionally prefilter incorrect records and persist the subset
    records = list(_iter_records(jsonl_path))
    if incorrect_only:
        records = [r for r in records if r.get("is_correct") is False]
        if subset_out is not None:
            subset_out.parent.mkdir(parents=True, exist_ok=True)
            with subset_out.open("w", encoding="utf-8") as sub:
                for r in records:
                    sub.write(json.dumps(r, ensure_ascii=False) + "\n")

    with out_path.open("w", encoding="utf-8") as dst:
        with cf.ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
            futs = [ex.submit(_judge_one, rec, gt_map, model) for rec in records]
            for fut in cf.as_completed(futs):
                res = fut.result()
                dst.write(json.dumps(res, ensure_ascii=False) + "\n")
                dst.flush()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run gpt-5-mini diagnostics over reverified JSONL")
    ap.add_argument("--jsonl", type=Path, required=True, help="Path to *_reverified.jsonl")
    ap.add_argument("--out", type=Path, default=Path("src/rag_analysis/judged_results_gpt5mini.jsonl"))
    ap.add_argument("--workers", type=int, default=4)
    # Default: run only incorrect answers. Use --all to override.
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--incorrect-only", dest="incorrect_only", action="store_true", help="Judge only records with is_correct == false (default)")
    g.add_argument("--all", dest="incorrect_only", action="store_false", help="Judge all records (override default)")
    ap.set_defaults(incorrect_only=True)
    ap.add_argument("--subset-out", type=Path, default=Path("src/rag_analysis/incorrect_subset.jsonl"), help="Where to write the filtered incorrect-only subset (when filtering)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.jsonl,
        args.out,
        workers=args.workers,
        incorrect_only=args.incorrect_only,
        subset_out=args.subset_out if args.incorrect_only else None,
    )
