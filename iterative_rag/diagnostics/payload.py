"""Build the judge-input payload from an iterative-RAG response record.

Each response record (a line of a ``responses_reverified`` JSONL) is combined with the
oracle hop path from the QA dataset into a compact payload the LLM auditors consume:

    {question, expected_answer, number_of_hops, path[...], run{candidate, evidence[...]}}

This payload is shared by all three diagnostics (coverage gap, hallucination/calibration,
query quality); the auditors differ only in their system prompt + output schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from iterative_rag import config


def load_ground_truth_map(qa_path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Map question string -> raw QA record (with the oracle ``path``)."""
    qa_path = Path(qa_path) if qa_path else config.QA_DATASET
    data = json.loads(Path(qa_path).read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    for r in data:
        q = r.get("q") or r.get("question")
        if q:
            out[q] = r
    return out


def iter_records(jsonl_path: Path) -> Iterator[Dict[str, Any]]:
    """Yield response records from a JSONL file."""
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _to_hops(gt: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, h in enumerate(gt.get("path") or [], 1):
        out.append({
            "hop_index": i,
            "hop_subq": h.get("q") or "",
            "answer_subq": h.get("a") or "",
            "text": h.get("text") or "",
        })
    return out


def _index_actions_by_step(actions: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    by_step: Dict[int, Dict[str, Any]] = {}
    step = 0
    for a in actions or []:
        act = a.get("action") or {}
        if not isinstance(act, dict):
            continue
        step += 1
        if act.get("action") == "retrieve_text":
            by_step[step] = {"query": act.get("query"), "partial_answer": act.get("partial_answer")}
        elif act.get("action") == "propose_answer":
            by_step[step] = {"propose_answer": act.get("answer")}
        else:
            by_step[step] = {"action": act.get("action")}
    return by_step


def build_payload(rec: Dict[str, Any], gt_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Assemble the per-record payload (raises KeyError if the question is unknown)."""
    raw = rec.get("raw") or {}
    question = raw.get("question") or rec.get("question") or ""
    expected = raw.get("expected") or rec.get("expected") or rec.get("a") or ""

    gt = gt_map.get(question)
    if not gt:
        gt = next((v for k, v in gt_map.items() if (k or "").strip() == question.strip()), None)
    if not gt:
        raise KeyError("Ground-truth question not found in QA dataset")

    hops = _to_hops(gt)
    run = rec.get("raw_response") or {}
    evidence = run.get("evidence") or []
    actions_trace = run.get("actions_trace") or []

    act_by_step = _index_actions_by_step(actions_trace)
    llm_calls = list(rec.get("llm_calls") or [])

    ev_by_step: Dict[int, List[Dict[str, Any]]] = {}
    for h in evidence:
        ev_by_step.setdefault(int(h.get("source_step") or 0), []).append(h)

    grouped: List[Dict[str, Any]] = []
    for s in sorted(ev_by_step):
        step_hits = ev_by_step[s]
        step_query = act_by_step.get(s, {}).get("query")
        if not step_query and step_hits:
            step_query = step_hits[0].get("source_query")
        call_curr = llm_calls[s] if s < len(llm_calls) else {}
        proposed = call_curr.get("proposed_answer") if isinstance(call_curr, dict) else None
        partial = ""
        if isinstance(call_curr, dict) and isinstance(call_curr.get("partial_answer"), str):
            partial = call_curr["partial_answer"]
        if proposed:
            partial = ""
        texts = [h.get("text") for h in step_hits if h.get("text")]
        grouped.append({
            "source_step": s,
            "source_query": step_query,
            "text": texts,
            "partial_answer": partial,
            "proposed_answer": proposed,
        })

    return {
        "question": question,
        "expected_answer": expected,
        "number_of_hops": len(hops),
        "path": hops,
        "run": {"candidate": rec.get("candidate") or "", "evidence": grouped},
    }
