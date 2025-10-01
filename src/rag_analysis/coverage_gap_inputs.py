"""
Prepare structured inputs for LLM-based judgments on iterative RAG runs.

Builds a compact payload combining:
- Ground truth question + hop path from src/docs/chemrxiv_qa.json
- One run entry from a *_reverified.jsonl file in src/responses_reverified/
  (containing evidence, actions_trace, and per-call ledger llm_calls)

The output is designed to support downstream judgments for:
  (1) Retrieval Coverage Gap (missed-hop)
  (2) Anchor Carry-Drop
  (3) Late-Hit per Hop

We DO NOT call any model here. This just prepares the input payload that
can be fed to a judge model later (e.g., gpt-5-mini) with your prompt.

Usage:
  python3 -u src/rag_analysis/prepare_llm_inputs.py \
    --jsonl src/responses_reverified/<file>.jsonl \
    [--question-substr "silica gel matrix"] \
    [--output src/rag_analysis/preview_inputs.json]

If --question-substr is omitted, the first record in the JSONL is used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SRC_BASE = Path(__file__).resolve().parents[1]
QA_PATH = SRC_BASE / "docs" / "chemrxiv_qa.json"


# ------------------------------ Judging prompt -------------------------------

PROMPT_SYSTEM = (
    "SYSTEM (role: instructions)\n"
    "You are an exacting QA auditor for an iterative retrieval–planning–composition RAG system.\n"
    "Given one question, its oracle hop path, the system’s per-step queries(queries generated in each step)/partial answers(partial_answer generate for each query), and the retrieved snippets (tagged by step),\n"
    "you must return concise JSON labels for FOUR diagnostics only:\n\n"
    "(1) Retrieval Coverage Gap (missed-hop)\n"
    "Definition: For any oracle hop k, across ALL steps, NONE of the retrieved snippets are about that hop’s key entity/relationship.\n"
    "In other words, the system never fetches the document(s) needed for one of the hops.\n"
    "Output: list of missed_hops (by hop_index) + overall boolean.\n\n"
    "(2) Anchor Carry-Drop\n"
    "Definition: If at step t>1 the previous partial answer (step t-1) names a key entity/anchor (surface form), the query at step t SHOULD carry at least one of those anchors.\n"
    "If it carries none, that step is a carry-drop. Only judge when a previous partial exists and clearly names at least one anchor.\n"
    "Output: per-step true/false and an overall boolean (true if any).\n\n"
    "(3) Late-Hit per Hop\n"
    "Definition: For oracle hop k, find the FIRST step where any retrieved snippet (for that step) contains/targets that hop’s key entity (using surface forms and obvious aliases).\n"
    "If first_hit_step_for_hop_k > hop_index, mark late_hit=true for that hop.\n"
    "Output: list of {hop_index, first_hit_step, late_hit} + overall boolean (true if any hop is late).\n\n"
    "Rules & heuristics:\n"
    "- Work only with the supplied text. No world knowledge beyond common-sense aliasing (e.g., “H2” ≡ “hydrogen gas”).\n"
    "- “Snippet mentions hop entity” means the snippet text or the step’s query text includes a clear surface form of a hop’s key entity that is central to that hop’s relation.\n"
    "  Treat the hop’s “answer_subq” and salient named entities in “hop_subq” as the hop’s key entity/anchor.\n"
    "- Anchor detection for (2): anchors = salient named entity strings explicitly present in the previous partial (proper names, formulae, distinctive class labels).\n"
    "  Ignore generic words (e.g., “compound”, “reaction”, “catalyst”).\n"
    "- If a previous partial is empty or non-specific (no salient anchors), label carry-drop=false for that step due to N/A.\n"
    "- Be conservative: prefer false over true when ambiguous.\n"
    "- Return ONLY the JSON object described below. No prose.\n"
)

PROMPT_INPUT_SCHEMA = (
    "USER (role: data)\n"
    "INPUT SCHEMA (paste your data as DATA_JSON below):\n"
    "{\n"
    "  \"question\": \"<string> — Full multi-hop question string.\",\n"
    "  \"expected_answer\": \"<string> — Gold final answer for the full question (root answer).\",\n"
    "  \"number_of_hops\": <int> — Count of oracle hops in `path`,\n"
    "  \"path\": [\n"
    "    {\n"
    "      \"hop_index\": <int> — 1-based hop position in the oracle chain (1, 2, …),\n"
    "      \"hop_subq\": \"<string> — Atomic sub-question for this hop (the oracle’s sub-question).\",\n"
    "      \"answer_subq\": \"<string> — Gold answer to this hop’s sub-question; treat as the hop’s key entity/anchor (with obvious aliases).\"\n"
    "    }\n"
    "    // … one object per hop in order\n"
    "  ],\n"
    "  \"run\": {\n"
    "    \"evidence\": [\n"
    "      {\n"
    "        \"source_step\": <int> — 1-based planner step number that issued this step’s query and retrieved these snippets,\n"
    "        \"source_query\": \"<string> — Exact query string used at this step (judge anchor carry and coverage against this).\",\n"
    "        \"text\": [\"<string>\", \"...\"] — Array of snippet texts retrieved at this step (judge coverage/late-hits against these contents),\n"
    "        \"partial_answer\": \"<string> or \"\" — Planner’s partial hypothesis at this step. It's the answer to the current source_query based on the text evidences. Empty string if absent or if a proposal is made at this step.\",\n"
    "        \"proposed_answer\": \"<string> or null — Final proposed answer if the planner proposes at this step(last step), otherwise null.\"\n"
    "      }\n"
    "      // … one object per retrieval step\n"
    "    ]\n"
    "  }\n"
    "}\n\n"
)

PROMPT_OUTPUT_SHAPE = (
    "REQUIRED OUTPUT JSON SHAPE:\n"
    "{\n"
    "  \"retrieval_coverage_gap\": {\n"
    "    \"missed_hops\": [ <int>, ... ],\n"
    "    \"has_gap\": <true|false>\n"
    "  },\n"
    "  \"anchor_carry_drop\": {\n"
    "    \"per_step\": [\n"
    "      {\"step\": <int>, \"carry_drop\": <true|false>}\n"
    "    ],\n"
    "    \"any_carry_drop\": <true|false>\n"
    "  },\n"
    "  \"late_hit_per_hop\": {\n"
    "    \"per_hop\": [\n"
    "      {\"hop_index\": <int>, \"first_hit_step\": <int|null>, \"late_hit\": <true|false>}\n"
    "    ],\n"
    "    \"any_late_hit\": <true|false>\n"
    "  }\n"
    "}\n"
    "Return ONLY the JSON object above. It's your turn to answer based on the provided DATA_JSON.\n"
)


def build_judging_prompt(data_obj: Dict[str, Any]) -> str:
    data_json = json.dumps(data_obj, ensure_ascii=False, indent=2)
    parts = [PROMPT_SYSTEM, PROMPT_INPUT_SCHEMA, "DATA_JSON:\n", data_json, "\n\n", PROMPT_OUTPUT_SHAPE]
    return "".join(parts)


def load_ground_truth_map() -> Dict[str, Dict[str, Any]]:
    with QA_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, Dict[str, Any]] = {}
    for r in data:
        q = r.get("q") or r.get("question")
        if q:
            out[q] = r
    return out


def find_record_in_jsonl(jsonl_path: Path, question_substr: Optional[str] = None) -> Dict[str, Any]:
    """
    Return the first record in the reverified JSONL matching question_substr (if provided),
    else the first valid JSON line.
    """
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # prefer rec["raw"]["question"], else rec["question"]
            q = None
            raw = rec.get("raw") or {}
            if isinstance(raw, dict):
                q = raw.get("question")
            if not q:
                q = rec.get("question")

            if question_substr:
                hay = (q or "") + "\n" + (rec.get("candidate") or "")
                if question_substr.lower() in hay.lower():
                    return rec
            else:
                return rec
    raise FileNotFoundError("No matching record found in JSONL.")


def to_hops(gt: Dict[str, Any]) -> List[Dict[str, Any]]:
    path = gt.get("path") or []
    out: List[Dict[str, Any]] = []
    for i, h in enumerate(path, 1):
        out.append(
            {
                "hop_index": i,
                "hop_subq": h.get("q") or "",
                "answer_subq": h.get("a") or "",
            }
        )
    return out


def index_actions_by_step(actions: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    by_step: Dict[int, Dict[str, Any]] = {}
    if not actions:
        return by_step
    # actions_trace entries come in chronological order; each contains {action: {...}, added, k}
    step = 0
    for a in actions:
        act = a.get("action") or {}
        if not isinstance(act, dict):
            continue
        # steps are implicit by loop order
        step += 1
        if act.get("action") == "retrieve_text":
            by_step[step] = {
                "query": act.get("query"),
                "partial_answer": act.get("partial_answer"),
                "k": act.get("k"),
            }
        elif act.get("action") == "propose_answer":
            by_step[step] = {
                "propose_answer": act.get("answer"),
                "needs_citations": act.get("needs_citations", True),
            }
        else:
            by_step[step] = {"action": act.get("action")}
    return by_step


def group_evidence_by_step(evidence: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    steps: Dict[int, List[Dict[str, Any]]] = {}
    for h in evidence or []:
        s = int(h.get("source_step") or 0)
        steps.setdefault(s, []).append(h)
    return steps


def _llm_calls_zero_based(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the raw llm_calls list as-is (zero-based indexing).

    Per user spec:
      partial_answer should come from llm_calls[source_step]
      proposed_answer should come from llm_calls[source_step] (when present)

    Note: source_step is 1-based from the orchestrator; mapping here uses
    llm_calls[source_step] (zero-based), exactly as requested.
    """
    return list(rec.get("llm_calls") or [])


def build_llm_input_payload(rec: Dict[str, Any], gt_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    raw = rec.get("raw") or {}
    question = raw.get("question") or rec.get("question") or ""
    expected = raw.get("expected") or rec.get("expected") or rec.get("a") or ""

    gt = gt_map.get(question)
    if not gt:
        # Fallback: try loose matching by trimming spaces
        gt = next((v for k, v in gt_map.items() if (k or "").strip() == question.strip()), None)
    if not gt:
        raise KeyError("Ground-truth question not found in chemrxiv_qa.json")

    hops = to_hops(gt)
    run = rec.get("raw_response") or {}
    evidence = run.get("evidence") or []
    actions_trace = run.get("actions_trace") or []
    steps = int(run.get("steps") or 0)

    # per-step planner info (query from actions; partial/proposed from llm_calls using zero-based index = source_step)
    act_by_step = index_actions_by_step(actions_trace)
    llm_calls = _llm_calls_zero_based(rec)
    # per-step evidence grouping
    ev_by_step = group_evidence_by_step(evidence)

    # Prepare grouped evidence entries per step to avoid repeating shared fields
    grouped_evidence: List[Dict[str, Any]] = []
    for s in sorted(ev_by_step.keys()):
        step_hits = ev_by_step[s]
        step_info = act_by_step.get(s, {})
        step_query = step_info.get("query")
        if not step_query and step_hits:
            step_query = step_hits[0].get("source_query")

        # Partial/proposed answers come from llm_calls[source_step] (zero-based)
        call_curr = llm_calls[s] if s < len(llm_calls) else {}
        proposed_answer = call_curr.get("proposed_answer") if isinstance(call_curr, dict) else None
        partial_answer = ""
        if isinstance(call_curr, dict):
            pa = call_curr.get("partial_answer")
            if isinstance(pa, str):
                partial_answer = pa
        if proposed_answer:
            partial_answer = ""

        texts = []
        for h in step_hits:
            t = h.get("text") or ""
            if t:
                texts.append(t)

        grouped_evidence.append(
            {
                "source_step": s,
                "source_query": step_query,
                "text": texts,
                "partial_answer": partial_answer,
                "proposed_answer": proposed_answer,
            }
        )

    # Assemble payload (only the fields requested)
    payload = {
        "question": question,
        "expected_answer": expected,
        "number_of_hops": len(hops),
        "path": hops,
        "run": {
            "evidence": grouped_evidence,
        },
    }
    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare LLM inputs for RAG analysis")
    ap.add_argument("--jsonl", type=Path, required=True, help="Path to *_reverified.jsonl")
    ap.add_argument("--question-substr", type=str, default=None, help="Substring to match a specific question")
    ap.add_argument("--output", type=Path, default=SRC_BASE / "rag_analysis" / "preview_inputs.json", help="Output JSON path")
    ap.add_argument("--emit-prompt", action="store_true", help="Emit the full judging prompt with DATA_JSON embedded")
    ap.add_argument("--prompt-out", type=Path, default=SRC_BASE / "rag_analysis" / "judging_prompt.txt", help="File to save the judging prompt")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    gt_map = load_ground_truth_map()
    rec = find_record_in_jsonl(args.jsonl, args.question_substr)
    payload = build_llm_input_payload(rec, gt_map)

    # Pretty-print to stdout and save to file
    out_text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(out_text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(out_text, encoding="utf-8")
    print(f"\nSaved prepared inputs to: {args.output}")

    if args.emit_prompt:
        prompt_text = build_judging_prompt(payload)
        print("\n\n===== JUDGING PROMPT =====\n")
        print(prompt_text)
        args.prompt_out.parent.mkdir(parents=True, exist_ok=True)
        args.prompt_out.write_text(prompt_text, encoding="utf-8")
        print(f"\nSaved judging prompt to: {args.prompt_out}")


if __name__ == "__main__":
    main()
