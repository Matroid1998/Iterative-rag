"""
Prepare the JSON payload required by the query-audit prompt.

This script mirrors the data shape expected by the new auditing prompt but does NOT
call any LLM. It simply selects one record from a *_reverified.jsonl file, enriches it
with the oracle hop metadata, and writes the payload to disk for inspection/testing.

Usage:
  python -m rag_analysis.query_audit_inputs \
      --jsonl src/responses_reverified/<file>.jsonl \
      [--question-substr "substring"] \
      [--output path.json]

If --question-substr is omitted, the first record is used. The resulting JSON file is
exactly what should be pasted into the DATA_JSON field of the auditing prompt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


SRC_BASE = Path(__file__).resolve().parents[1]
QA_PATH = SRC_BASE.parent / "data" / "corpus" / "chemrxiv_qa.json"
OUT_DEFAULT = SRC_BASE / "rag_analysis" / "query_audit_input_example.json"


def load_ground_truth_map() -> Dict[str, Dict[str, Any]]:
    data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    for r in data:
        q = r.get("q") or r.get("question")
        if q:
            out[q] = r
    return out


def select_record(jsonl_path: Path, substr: Optional[str]) -> Dict[str, Any]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = (rec.get("raw") or {}).get("question") or rec.get("question") or ""
            if substr is None or substr.lower() in q.lower():
                return rec
    raise FileNotFoundError("No matching record found in JSONL")


def build_payload(rec: Dict[str, Any], gt_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    raw = rec.get("raw") or {}
    question = raw.get("question") or rec.get("question") or ""
    expected = raw.get("expected") or rec.get("expected") or rec.get("a") or ""

    gt = gt_map.get(question)
    if not gt:
        gt = next((v for k, v in gt_map.items() if (k or "").strip() == (question or "").strip()), None)
    if not gt:
        raise KeyError("Ground-truth question not found in chemrxiv_qa.json")

    path_entries = []
    for i, hop in enumerate(gt.get("path") or [], 1):
        path_entries.append(
            {
                "hop_index": i,
                "hop_subq": hop.get("q") or "",
                "answer_subq": hop.get("a") or "",
                "text": hop.get("text") or "",
            }
        )

    run = rec.get("raw_response") or {}
    evidence = run.get("evidence") or []

    grouped: Dict[int, Dict[str, Any]] = {}
    for h in evidence:
        step = int(h.get("source_step") or 0)
        bucket = grouped.setdefault(
            step,
            {
                "source_step": step,
                "source_query": h.get("source_query"),
                "text": [],
                "partial_answer": h.get("partial_answer") or "",
                "proposed_answer": h.get("proposed_answer"),
            },
        )
        if not bucket.get("source_query") and h.get("source_query"):
            bucket["source_query"] = h.get("source_query")
        txt = h.get("text")
        if txt:
            bucket.setdefault("text", []).append(txt)

    evidence_entries = []
    for step in sorted(grouped):
        entry = grouped[step]
        entry["text"] = entry.get("text") or []
        evidence_entries.append(entry)

    payload = {
        "question": question,
        "expected_answer": expected,
        "number_of_hops": len(path_entries),
        "path": path_entries,
        "run": {"evidence": evidence_entries},
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare query-audit INPUT payload (no LLM call)")
    parser.add_argument("--jsonl", type=Path, required=True, help="Path to *_reverified.jsonl")
    parser.add_argument("--question-substr", type=str, default=None, help="Substring to match a specific question")
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT, help="Where to write the prepared JSON")
    args = parser.parse_args()

    rec = select_record(args.jsonl, args.question_substr)
    payload = build_payload(rec, load_ground_truth_map())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved payload to {args.output}")


if __name__ == "__main__":
    main()
