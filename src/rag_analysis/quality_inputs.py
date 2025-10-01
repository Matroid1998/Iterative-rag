"""
Prepare input data for the query audit LLM prompt.

This script builds structured input payloads that can be fed to an LLM 
with the query audit system prompt. It extracts data from reverified JSONL files
and ground truth, then formats it according to the required input schema.

The output is designed to be used with the query audit prompt that judges:
- Next-Logical-Hop (Hop Intent)
- Query Quality assessment
- Partial answer contradictions  
- Run-level distractor latch detection

Usage:
  python3 -u src/rag_analysis/prepare_query_audit_inputs.py \
    --jsonl src/responses_reverified/<file>.jsonl \
    [--question-substr "silica gel matrix"] \
    [--output src/rag_analysis/query_audit_input.json]

If --question-substr is omitted, the first record in the JSONL is used.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


SRC_BASE = Path(__file__).resolve().parents[1]
QA_PATH = SRC_BASE / "docs" / "chemrxiv_qa.json"


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


def to_hops_with_text(gt: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert ground truth path to hop format with text included."""
    path = gt.get("path") or []
    out: List[Dict[str, Any]] = []
    for i, h in enumerate(path, 1):
        out.append({
            "hop_index": i,
            "hop_subq": h.get("q") or "",
            "answer_subq": h.get("a") or "",
            "text": h.get("text") or ""
        })
    return out


def index_actions_by_step(actions: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Index actions by step number."""
    by_step: Dict[int, Dict[str, Any]] = {}
    if not actions:
        return by_step
    
    step = 0
    for a in actions:
        act = a.get("action") or {}
        if not isinstance(act, dict):
            continue
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


def _llm_calls_zero_based(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the raw llm_calls list as-is (zero-based indexing)."""
    return list(rec.get("llm_calls") or [])


def group_evidence_by_step(evidence: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """Group evidence entries by source step."""
    steps: Dict[int, List[Dict[str, Any]]] = {}
    for h in evidence or []:
        s = int(h.get("source_step") or 0)
        steps.setdefault(s, []).append(h)
    return steps


def build_run_evidence_entries(
    run: Dict[str, Any],
    actions_by_step: Dict[int, Dict[str, Any]],
    llm_calls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build evidence entries with partial_answer and proposed_answer.
    
    Since llm_calls may not be available in all files, we'll extract partial_answer
    from the actions_trace and handle proposed_answer for the final step.
    """
    evidence = run.get("evidence") or []
    ev_by_step = group_evidence_by_step(evidence)
    
    entries = []
    all_steps = sorted(set(ev_by_step.keys()) | set(actions_by_step.keys()))
    max_step = max(all_steps) if all_steps else 0
    
    for step in all_steps:
        step_evidence = ev_by_step.get(step, [])
        step_action = actions_by_step.get(step, {})
        
        # Get query from action or first evidence entry
        query = step_action.get("query")
        if not query and step_evidence:
            query = step_evidence[0].get("source_query", "")
        
        # Collect all text snippets for this step
        snippets = []
        for ev in step_evidence:
            txt = ev.get("text")
            if isinstance(txt, list):
                snippets.extend([t for t in txt if isinstance(t, str) and t])
            elif isinstance(txt, str) and txt:
                snippets.append(txt)
        
        # Extract partial/proposed answers from llm_calls (if available) or actions
        partial_value: Optional[str] = ""
        proposed_value: Optional[str] = None
        
        # Try to get from llm_calls first - mapping step to correct llm_calls index
        if llm_calls:
            # For non-final steps, get partial_answer from llm_calls[step] (step index, not step-1)
            if step < max_step and step < len(llm_calls):
                call = llm_calls[step]
                if isinstance(call, dict):
                    partial_value = call.get("partial_answer", "")
            
            # For final step, get proposed_answer from llm_calls[step] 
            if step == max_step and step < len(llm_calls):
                call = llm_calls[step]
                if isinstance(call, dict):
                    proposed_value = call.get("proposed_answer", None)
                    # Final step should have null partial_answer
                    partial_value = None
        
        # Fallback: get from action if not found in llm_calls
        if not partial_value and step != max_step:
            partial_value = step_action.get("partial_answer", "")
        if not proposed_value and step == max_step:
            if step_action.get("propose_answer"):
                proposed_value = step_action.get("propose_answer")
        
        # Ensure final answer is captured
        if step == max_step and run.get("answer") and not proposed_value:
                proposed_value = run.get("answer")
        
        entry = {
            "source_step": step,
            "source_query": query or "",
            "text": snippets,
            "partial_answer": partial_value if partial_value is not None else "",
            "proposed_answer": proposed_value,
        }
        entries.append(entry)
    
    entries.sort(key=lambda x: x["source_step"])
    return entries


def build_payload(rec: Dict[str, Any], gt_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Build the complete input payload for the query audit LLM."""
    raw = rec.get("raw") or {}
    question = raw.get("question") or rec.get("question") or ""
    expected = raw.get("expected") or rec.get("expected") or rec.get("a") or ""

    gt = gt_map.get(question)
    if not gt:
        gt = next((v for k, v in gt_map.items() if (k or "").strip() == question.strip()), None)
    if not gt:
        raise KeyError("Ground-truth question not found in chemrxiv_qa.json")

    path_entries = to_hops_with_text(gt)
    
    run = rec.get("raw_response") or {}
    actions_by_step = index_actions_by_step(run.get("actions_trace") or [])
    llm_calls = _llm_calls_zero_based(rec)
    evidence_entries = build_run_evidence_entries(run, actions_by_step, llm_calls)

    payload = {
        "question": question,
        "expected_answer": expected,
        "number_of_hops": len(path_entries),
        "path": path_entries,
        "run": {
            "candidate": rec.get("candidate") or "",
            "evidence": evidence_entries
        }
    }
    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare query audit inputs for LLM")
    ap.add_argument("--jsonl", type=Path, required=True, help="Path to *_reverified.jsonl")
    ap.add_argument("--question-substr", type=str, default=None, help="Substring to match a specific question")
    ap.add_argument("--output", type=Path, default=SRC_BASE / "rag_analysis" / "query_audit_input.json", help="Output JSON path")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    
    print(f"Loading ground truth from: {QA_PATH}")
    gt_map = load_ground_truth_map()
    
    print(f"Finding record in: {args.jsonl}")
    if args.question_substr:
        print(f"Looking for question containing: '{args.question_substr}'")
    
    rec = find_record_in_jsonl(args.jsonl, args.question_substr)
    
    print("Building payload...")
    payload = build_payload(rec, gt_map)
    
    # Build the complete prompt with system instructions and actual input
    system_prompt = """You are an exacting auditor of an iterative retrieval–planning RAG system.
For EACH step, judge the step's intended hop and the quality of its query.
Also detect partial-answer contradictions across steps, and a run-level "distractor latch".

You must use ONLY the provided text. No outside knowledge.
Return EXACT JSON in the required schema. No prose outside JSON.

JUDGMENTS TO MAKE

(1) Next-Logical-Hop (Hop Intent)

predicted_hop: Which oracle hop the step's query primarily aims to solve (1-based). Use surface-form matching
against the hop entities/relations; if unclear, set null.
is_next_logical_hop: true iff predicted_hop == (resolved_hops + 1). Otherwise false.
fusion_or_skip: true if the query tries to solve multiple hops at once (compound across hops) or skips ahead.
(2) Query Quality

vague: true if the query lacks concrete targets (e.g., "learn more about HAT").
over_broad: true if scope is too wide or mixes unrelated facets for the needed hop.
compound: true if it bundles multiple sub-questions/entities with AND/OR or comma lists.
off_topic: true if it targets a subject not required by any oracle hop.
anchored: true if the query includes at least one salient anchor from the immediately preceding partial_answer.
(Salient anchors = distinctive surface forms like "metal-oxo", "carbaisophlorinoid", "Fe(IV)=O", "H2". Ignore generic words.)
specificity_score: float in [0,1] (0 = extremely vague; 1 = tightly targeted to the needed sub-fact).
on_topic_score: float in [0,1] (0 = mostly irrelevant; 1 = well-aligned with the intended hop).
justification: short phrase citing the key tokens/phrases that drove your labels (≤140 chars).
(10) Partial Contradiction

For step t≥2: partial_contradiction_with_prev is true if partial_answer_t conflicts with partial_answer_(t-1).
Conflict = mutually exclusive claims or incompatible classes (LLM NLI-style judgment), based ONLY on given strings.
If true, set contradicts_prior_step = (t-1); otherwise null.
If partial_answer is null at either step, set partial_contradiction_with_prev = false.
(4) Distractor Latch (Scaffold Trap) — RUN LEVEL

True if the run's retrieved evidence appears locked onto a chemically similar but irrelevant scaffold/family
compared to the gold target family implied by the oracle path (e.g., "phenyl/benzylic" vs needed "phenoxyl").
Use simple family/entity-pattern matching over snippet texts vs. oracle hop entities (and optional gold_hop_entities).
Output only a single boolean for the whole run (no per-step flag).
Be conservative: if unclear, return false.
OPERATIONAL RULES

Use only the provided text in INPUT.
predicted_hop can be null if you cannot tell; then set is_next_logical_hop=false.
Multiple query-quality flags may be true simultaneously.
Anchored=false at step 1 (no prior partial).
Keep judgments conservative when ambiguous.
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
"candidate\": \"<final candidate answer by the model>\",\n"
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

}"""

    required_output_schema = """

REQUIRED OUTPUT JSON SHAPE:
{
"per_step": [
{
"step": <int>,
"predicted_hop": <int|null>,
"is_next_logical_hop": <true|false>,
"fusion_or_skip": <true|false>,
"query_quality": {
"vague": <true|false>,
"over_broad": <true|false>,
"compound": <true|false>,
"off_topic": <true|false>,
"anchored": <true|false>,
"specificity_score": <number 0..1>,
"on_topic_score": <number 0..1>,
"justification": "<≤140 chars>"
},
"partial_contradiction_with_prev": <true|false>,
"contradicts_prior_step": <int|null>
}
// one object per step in order
],
"run_level": {
"distractor_latch": <true|false>
}
}

Return ONLY the JSON, nothing else. Now, it's your turn to answer based on the data."""

    # Build the complete prompt with input data at the end
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    full_prompt = f"{system_prompt}{required_output_schema}\n\n{payload_json}"
    
    # Pretty-print to stdout and save to file
    print("\n" + "="*60)
    print("GENERATED FULL PROMPT FOR QUERY AUDIT LLM:")
    print("="*60)
    print(full_prompt)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(full_prompt, encoding="utf-8")
    print(f"\nSaved full audit prompt to: {args.output}")
    
    # Show some stats
    evidence_count = len(payload["run"]["evidence"])
    hop_count = payload["number_of_hops"]
    print(f"\nSummary:")
    print(f"- Question: {payload['question'][:80]}...")
    print(f"- Expected answer: {payload['expected_answer']}")
    print(f"- Number of hops: {hop_count}")
    print(f"- Evidence steps: {evidence_count}")


if __name__ == "__main__":
    main()