"""Run GPT-5-mini hallucination judgments on RAG responses using the hallucination audit prompt."""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set

from openai import OpenAI

# Ensure package imports work when running as a script via python path/to/file.py
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

from rag_analysis.hallucination_inputs import (  # noqa: E402
    build_payload,
    find_record_in_jsonl,
    load_ground_truth_map,
)


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Iterate over records in a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping line {idx}: cannot decode JSON ({exc})", file=sys.stderr)
                continue
            yield obj


def load_question_whitelist(path: Path) -> Set[str]:
    """Load question whitelist from a JSON array."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Question list must be a JSON array: {path}")

    return {str(item).strip() for item in data if isinstance(item, str) and item.strip()}


def build_hallucination_audit_prompt(payload: Dict[str, Any]) -> str:
    """Build the complete hallucination audit prompt with system instructions, schemas, and input data."""
    system_prompt = """SYSTEM (role: instructions)
You are an exacting auditor of an iterative retrieval–augmented QA system.
Your job: judge the FINAL ANSWER for faithfulness to the provided evidence, detect composition failure, and diagnose confidence miscalibration.
Use ONLY the supplied text. No outside knowledge. Be conservative when unsure.
Return EXACT JSON in the schema below. No prose outside JSON.

SCOPE OF THIS JUDGMENT (RUN-LEVEL, FINAL ANSWER FOCUS)

(1) Composition / Answer Synthesis Failure
- true if the correct entity/claim is present in the evidence but the final candidate either:
  (a) selects a different entity, or
  (b) paraphrases without clearly naming the correct entity, or
  (c) muddles/merges entities so the core answer is wrong or unclear.

(2) Unsupported Claim (Faithfulness)
- For each atomic sentence in the partial answers, decide if at least one evidence text supports it. You should look for previous and current evidence texts in source steps.
- "Support" = directly stated or a tight paraphrase; speculation is unsupported.

(3) Confidence Miscalibration
- Purpose: detect when the system (i) answered confidently with weak/insufficient evidence, or (ii) kept retrieving despite already having enough evidence.
- Inputs you may use: source_step, final number of source_step, number_of_hops, partial_answers, source_query in each step, hop_subq, answer_subq, your own estimated support sufficiency (see below), and simple hop-coverage approximation (see below).
- You must compute two internal estimates:
  * sufficiency_score_est ∈ [0,1]: fraction of partial answers sentences that are supported by ≥1 snippet.
  * hop_coverage_est ∈ [0,1]: fraction of oracle hops(hop_subq and answer_subq) whose key surface entity or relation appears anywhere in the partial answers OR in any provided evidence snippet's text. (Use surface tokens only; simple case-insensitive matches; hyphen/space variants OK.)
- Decision rules:
  * Overconfident finalize ("overconfident_finalize"): Under thinking (early stopping)
    - Trigger if ANY hold:
      (i) finalize step(last step number) < number_of_hops AND hop_coverage_est < 0.8, AND
      (ii) sufficiency_score_est < 0.60, 
  * Underconfident continue ("underconfident_continue"): Overthinking
    - Trigger if ANY hold:
      some prior step t < finalize_step likely had "enough": This means final answer (expected answer) can be supported by evidences before the finalize step ( last source_step )

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

    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    full_prompt = f"{system_prompt}\n\n{payload_json}"
    
    return full_prompt


def call_judge(client: Optional[OpenAI], prompt: str, model: str, dry_run: bool = False) -> Dict[str, Any]:
    """Call the judge model with the prompt."""

    
    if client is None:
        raise ValueError("OpenAI client is required for non-dry-run mode")
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    output_text = response.choices[0].message.content or ""
    return {"response": response, "text": output_text}


def parse_output(text: str) -> Optional[Dict[str, Any]]:
    """Parse the model output as JSON."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def build_cli() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    ap = argparse.ArgumentParser(description="Run GPT-5-mini hallucination judgments on RAG responses")
    default_jsonl = (
        CURRENT_DIR.parent
        / "responses_reverified"
        / "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning_reverified.jsonl"
    )
    ap.add_argument("--jsonl", type=Path, default=default_jsonl, help="Path to *_reverified.jsonl")
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to store the JSONL judgments (defaults beside the input with system suffix)",
    )
    ap.add_argument("--model", type=str, default="gpt-5-mini", help="Judge model to use")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of records to process")
    ap.add_argument("--print-output", action="store_true", help="Print the full model output for each processed record")
    ap.add_argument("--question-substr", type=str, default=None, help="Filter for questions containing this substring")
    ap.add_argument("--num-workers", type=int, default=6, help="Number of worker threads for parallel processing")
    ap.add_argument(
        "--question-list",
        type=Path,
        default=CURRENT_DIR / "chemrxiv_judgement_questions.json",
        help="Path to JSON array of questions to include",
    )
    ap.add_argument(
        "--save-prompts",
        type=Path,
        default=None,
        help="Optional JSONL path to record the prompts sent to the judge model",
    )
    ap.add_argument(
        "--append-output",
        action="store_true",
        help="Append new judgments to the output file instead of overwriting",
    )
    ap.add_argument("--dry-run", action="store_true", help="Test prompt generation without calling OpenAI API")
    return ap


def extract_question_from_record(rec: Dict[str, Any]) -> str:
    """Extract question from a record, checking both raw and top-level."""
    raw = rec.get("raw") or {}
    if isinstance(raw, dict):
        question = raw.get("question")
        if question:
            return question
    return rec.get("question", "")


def process_single_record(
    rec: Dict[str, Any],
    gt_map: Dict[str, Dict[str, Any]],
    client: Optional[OpenAI],
    model: str,
    print_output: bool,
    dry_run: bool = False,
    prompt_recorder: Optional[Callable[[Dict[str, Any], str], None]] = None,
) -> Optional[Dict[str, Any]]:
    """Process a single record and return the judgment."""
    try:
        payload = build_payload(rec, gt_map)
        prompt = build_hallucination_audit_prompt(payload)
        if prompt_recorder is not None:
            prompt_recorder(payload, prompt)

        call_data = call_judge(client, prompt, model, dry_run)
        output_text = call_data["text"]
        parsed = parse_output(output_text)

        if print_output:
            print(f"\n===== PROCESSING QUESTION =====\nQuestion: {payload.get('question', '')[:100]}...")
            print("===== GPT RAW OUTPUT =====")
            print(output_text)
            print("===== END OUTPUT =====\n")

        entry: Dict[str, Any] = {
            "question": payload.get("question"),
            "expected_answer": payload.get("expected_answer"),
            "number_of_hops": payload.get("number_of_hops"),
            "model": model,
            "raw_output": output_text,
            "dry_run": dry_run,
        }
        if parsed is not None:
            entry["parsed_judgment"] = parsed
        else:
            entry["parse_error"] = "Failed to parse output as JSON"

        return entry

    except Exception as exc:
        print(f"Error processing record: {exc}", file=sys.stderr)
        return None


def main() -> None:
    """Main entry point with multi-threading support."""
    args = build_cli().parse_args()

    if not args.jsonl.exists():
        raise FileNotFoundError(f"JSONL file not found: {args.jsonl}")

    if not args.question_list.exists():
        raise FileNotFoundError(f"Question list not found: {args.question_list}")

    question_whitelist = load_question_whitelist(args.question_list)
    if not question_whitelist:
        raise ValueError(f"Question list is empty: {args.question_list}")

    if args.output is None:
        derived_name = f"{args.jsonl.stem}_hallucination_judgment.jsonl"
        args.output = CURRENT_DIR / "output" / f"2_{derived_name}"

    gt_map = load_ground_truth_map()

    # Collect all records to process
    records = list(iter_jsonl(args.jsonl))

    # Filter records by question whitelist
    filtered_records: List[Dict[str, Any]] = []
    for rec in records:
        question = extract_question_from_record(rec).strip()
        if question and question in question_whitelist:
            filtered_records.append(rec)
    records = filtered_records

    # Filter records if question substring is provided
    if args.question_substr:
        filtered_records = []
        for rec in records:
            question = extract_question_from_record(rec)
            candidate = rec.get("candidate", "")
            combined_text = f"{question}\n{candidate}".lower()
            if args.question_substr.lower() in combined_text:
                filtered_records.append(rec)
        records = filtered_records
        print(f"Filtered to {len(records)} records matching '{args.question_substr}'")
    
    if args.limit:
        records = records[:args.limit]

    if not records:
        print("No records matched the provided question list.", file=sys.stderr)
        return

    prompt_recorder: Optional[Callable[[Dict[str, Any], str], None]] = None
    prompt_path = args.save_prompts
    if prompt_path is not None:
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text("", encoding="utf-8")
        prompt_lock = threading.Lock()

        def prompt_recorder(payload: Dict[str, Any], prompt: str) -> None:
            entry = {
                "question": payload.get("question"),
                "expected_answer": payload.get("expected_answer"),
                "payload": payload,
                "prompt_text": prompt,
            }
            with prompt_lock:
                with prompt_path.open("a", encoding="utf-8") as fh:
                    json.dump(entry, fh, ensure_ascii=False)
                    fh.write("\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    existing_questions = set()
    if args.append_output and args.output.exists():
        with args.output.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                question = obj.get("question")
                if question:
                    existing_questions.add(question)

    print(f"Processing {len(records)} records with {args.num_workers} workers...")
    
    if args.dry_run:
        print("🔥 DRY RUN MODE: Using mock responses instead of calling OpenAI API")
    
    processed = 0
    write_lock = threading.Lock()

    # Initialize OpenAI client only if not in dry-run mode
    client_factory = lambda: OpenAI() if not args.dry_run else None

    # Open output file for writing
    mode = "a" if args.append_output else "w"

    with args.output.open(mode, encoding="utf-8") as out_f:
        # Process records in parallel
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all tasks
            future_to_record = {
                executor.submit(
                    process_single_record, 
                    rec, 
                    gt_map, 
                    client_factory(),  # Each thread gets its own client or None
                    args.model, 
                    args.print_output,
                    args.dry_run,
                    prompt_recorder,
                ): rec for rec in records
            }
            
            # Process completed futures
            for future in as_completed(future_to_record):
                rec = future_to_record[future]
                try:
                    result = future.result()
                    if result is not None:
                        # Thread-safe writing
                        with write_lock:
                            question = result.get("question")
                            if args.append_output and question in existing_questions:
                                continue
                            json.dump(result, out_f, ensure_ascii=False)
                            out_f.write("\n")
                            out_f.flush()
                            processed += 1
                            if question:
                                existing_questions.add(question)
                            
                        if processed % 10 == 0:
                            print(f"Processed {processed}/{len(records)} records...")
                    
                except Exception as exc:
                    question = extract_question_from_record(rec)
                    print(f"Error processing record with question '{question[:50]}...': {exc}", file=sys.stderr)

    print(f"Completed! Processed {processed} records. Results saved to {args.output}")


if __name__ == "__main__":
    main()
