"""
Prepare input data for the hallucination audit LLM prompt.

This script builds structured input payloads that can be fed to an LLM 
with the hallucination audit system prompt. It extracts data from reverified JSONL files
and ground truth, then formats it according to the required input schema.

The output is designed to be used with the hallucination audit prompt that judges:
- Composition/Answer Synthesis Failure
- Unsupported Claims (Faithfulness) 
- Confidence Miscalibration

Usage:
  python3 -u src/service/failure_modes_analysis/hallucination_inputs.py \
    --jsonl src/responses_reverified/<file>.jsonl \
    [--question-substr "silica gel matrix"] \
    [--output data/results/failure_modes/hallucination_inputs.json]

If --question-substr is omitted, the first record in the JSONL is used.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
QA_PATH = REPO_ROOT / "data" / "corpus" / "chemrxiv_qa.json"


def load_ground_truth_map() -> Dict[str, Dict[str, Any]]:
    """Load the ground truth QA data."""
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
    """Convert ground truth path to hop format."""
    path = gt.get("path") or []
    out: List[Dict[str, Any]] = []
    for i, h in enumerate(path, 1):
        out.append(
            {
                "hop_index": i,
                "hop_subq": h.get("q") or "",
                "answer_subq": h.get("a") or "",
                "text": h.get("text") or "",
            }
        )
    return out


def index_actions_by_step(actions: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Index actions by step number."""
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
    """Group evidence by source step."""
    steps: Dict[int, List[Dict[str, Any]]] = {}
    for h in evidence or []:
        s = int(h.get("source_step") or 0)
        steps.setdefault(s, []).append(h)
    return steps


def _llm_calls_zero_based(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the raw llm_calls list as-is (zero-based indexing)."""
    return list(rec.get("llm_calls") or [])


def build_payload(rec: Dict[str, Any], gt_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Build the input payload for hallucination detection."""
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
            "candidate": rec.get("candidate") or "",
            "evidence": grouped_evidence,
        },
    }
    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare LLM inputs for hallucination analysis")
    ap.add_argument("--jsonl", type=Path, required=True, help="Path to *_reverified.jsonl")
    ap.add_argument("--question-substr", type=str, default=None, help="Substring to match a specific question")
    ap.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "results" / "failure_modes" / "hallucination_inputs.json",
        help="Output JSON path",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    gt_map = load_ground_truth_map()
    rec = find_record_in_jsonl(args.jsonl, args.question_substr)
    payload = build_payload(rec, gt_map)

    # Build the complete prompt with system instructions and actual input
    system_prompt = """SYSTEM (role: instructions)
You are an exacting auditor of an iterative retrieval–augmented QA system.
Your job: judge the FINAL ANSWER for faithfulness to the provided evidence, detect composition failure, and diagnose confidence miscalibration.
Use ONLY the supplied text. No outside knowledge. Be conservative when unsure.
Return EXACT JSON in the schema below. No prose outside JSON.

SCOPE OF THIS JUDGMENT (RUN-LEVEL, FINAL ANSWER FOCUS)

(1) Composition / Answer Synthesis Failure
- true if the correct entity/claim is present in the evidence but the final candidate ("candidate" key in the input json) either:
  (a) selects a different entity, or
  (b) paraphrases without clearly naming the correct entity, or
  (c) muddles/merges entities so the core answer is wrong or unclear.
expected_answer is oracle answer for full question.  

(2) Unsupported Claim (Faithfulness)
- For each atomic sentence in the partial answers, decide if at least one evidence text supports it. You should look for previous and current evidence texts in source steps.
- "Support" = directly stated or a tight paraphrase; speculation is unsupported.

(3) Confidence Miscalibration
- Purpose: detect when the system (i) answered confidently with weak/insufficient evidence, or (ii) kept retrieving despite already having enough evidence.
- Inputs you may use: source_step, maximum number of source_step, number_of_hops, partial_answers, source_query in each step, hop_subq, answer_subq, your own estimated support sufficiency (see below), and simple hop-coverage approximation (see below).
- You must compute two internal estimates:
  * sufficiency_score_est ∈ [0,1]: fraction of partial answers sentences that are supported by ≥1 snippet.
  * hop_coverage_est ∈ [0,1]: fraction of oracle hops(hop_subq and answer_subq) whose key surface entity or relation appears anywhere in the partial answers OR in any provided evidence snippet's text. (Use surface tokens only; simple case-insensitive matches; hyphen/space variants OK.)
- Decision rules:
  * Overconfident finalize ("overconfident_finalize"): Under thinking (early stopping)
    - Trigger if ANY hold:
      (i) finalize step(maximum number of source_step) < number_of_hops AND ( hop_coverage_est < 0.7, OR
      (ii) sufficiency_score_est < 0.60), 
  * Underconfident continue ("underconfident_continue"): Overthinking
    - Trigger if ANY hold:
      some prior step t < finalize_step (maximum number of source_step) likely had "enough": This means final answer (expected answer) can be supported by evidences before the finalize step ( last source_step )

INPUT (from user):
{
"question": "<string> — Full multi-hop question string.",
"expected_answer": "<string> — Gold final answer for the full question (root answer).",
"number_of_hops": <int> — Count of oracle hops in path. The number of hops for the original question.",
"path": [
{
"hop_index": <int> — 1-based hop position in the oracle chain (1, 2, …),
"hop_subq": "<string> — Atomic sub-question for this hop (the oracle's sub-question).",
"answer_subq": "<string> — Gold answer to this hop's sub-question; treat as the hop's key entity/anchor (with obvious aliases)."
"text": The text that this hop subq is generated from.
}
// … one object per hop in order
],
"run": {
"candidate": "<final candidate answer by the model>",
"evidence": [ //evidences retrieved at each iteration. texts are coming from retriever based on the source_query. partial answers are answer in this step based on the texts provided and previouse partial answers and queires. 
{
"source_step": <int> — 1-based planner step number that issued this step's query and retrieved these snippets. This is the i-th call in iterative rag system.,
"source_query": "<string> — Exact query string used at this step (judge anchor carry and coverage against this. it is generated from previouse queries and partial answers and original question to find what we should address.",
"text": ["<string>", "..."] — Array of snippet texts retrieved at this step (judge coverage/late-hits against these contents),
"partial_answer": "<string> or "" — Planner's partial hypothesis at this step. It's the answer to the current source_query based on the text evidences. Empty string if absent or if a proposal is made at this step.",
"proposed_answer": "<string> or null — Final proposed answer if the planner proposes at this step(last step), otherwise null."
}
// … one object per retrieval step
]

}

REQUIRED OUTPUT JSON SHAPE:
{
  "composition_and_faithfulness": {
    "composition_failure": <true|false>,               // (1)  
    "unsupported_claims": [                               // (2)
      {
        "source_step": <int>,          // step number that the evidence doesn't support partial answer
        "is_supported": <true|false>
      }
    ],
    "sufficiency_score_est": <number 0..1>                // fraction of supported sentences
  },
  "confidence_miscalibration": {                           // (3)
"hop_coverage_est": <number 0..1>,    
"is_miscalibrated": <true|false>,
    "direction": "overconfident_finalize" | "underconfident_continue" | "ok",
  }
}

Return ONLY the JSON, nothing else."""

    # Build the complete prompt with input data at the end
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    full_prompt = f"{system_prompt}\n\n{payload_json}"
    
    # Pretty-print to stdout and save to file
    print("\n" + "="*60)
    print("GENERATED FULL PROMPT FOR HALLUCINATION AUDIT LLM:")
    print("="*60)
    print(full_prompt)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(full_prompt, encoding="utf-8")
    print(f"\nSaved full hallucination audit prompt to: {args.output}")


if __name__ == "__main__":
    main()
