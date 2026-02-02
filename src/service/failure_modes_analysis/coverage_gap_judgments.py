"""Run GPT-5 judge on incorrect RAG responses using the prepared prompt."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

from openai import OpenAI

# Ensure package imports work when running as a script via python path/to/file.py
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

REPO_ROOT = CURRENT_DIR.parents[3]

from failure_modes_analysis.coverage_gap_inputs import (  # noqa: E402
    build_judging_prompt,
    build_llm_input_payload,
    load_ground_truth_map,
)


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
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
    """Load question whitelist from JSON array."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Question list must be a JSON array: {path}")

    return {str(item).strip() for item in data if isinstance(item, str) and item.strip()}


def extract_question(rec: Dict[str, Any]) -> str:
    """Extract question text from a record."""
    raw = rec.get("raw") or {}
    if isinstance(raw, dict):
        question = raw.get("question")
        if question:
            return question
    return rec.get("question") or ""


def call_judge(client: OpenAI, prompt: str, model: str) -> Dict[str, Any]:
    response = client.responses.create(model=model, input=prompt)
    # response.output_text provides the concatenated text segments
    output_text = getattr(response, "output_text", None)
    if output_text is None:
        # Fallback: assemble from content items
        chunks: List[str] = []
        for item in getattr(response, "output", []) or []:
            if item.get("type") == "output_text":
                chunks.append(item.get("text", ""))
        output_text = "".join(chunks)
    return {"response": response, "text": output_text or ""}


def parse_output(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run GPT-5-mini coverage-gap judgments on RAG responses")
    default_jsonl = (
        REPO_ROOT
        / "src"
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
    ap.add_argument(
        "--question-list",
        type=Path,
        default=REPO_ROOT / "data" / "results" / "failure_modes" / "chemrxiv_judgement_questions.json",
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
        help="Append to the output JSONL file instead of overwriting",
    )
    return ap


def main() -> None:
    args = build_cli().parse_args()

    if not args.jsonl.exists():
        raise FileNotFoundError(f"JSONL file not found: {args.jsonl}")

    if not args.question_list.exists():
        raise FileNotFoundError(f"Question list not found: {args.question_list}")

    question_whitelist = load_question_whitelist(args.question_list)
    if not question_whitelist:
        raise ValueError(f"Question list is empty: {args.question_list}")

    if args.output is None:
        derived_name = f"{args.jsonl.stem}_coverage_gap_judgments.jsonl"
        args.output = REPO_ROOT / "data" / "results" / "failure_modes" / f"2_{derived_name}"

    gt_map = load_ground_truth_map()
    records: List[Dict[str, Any]] = []
    for rec in iter_jsonl(args.jsonl):
        question = extract_question(rec).strip()
        if question and question in question_whitelist:
            records.append(rec)

    if args.limit:
        records = records[:args.limit]

    if not records:
        print("No records matched the provided question list.", file=sys.stderr)
        return

    prompt_path = args.save_prompts
    if prompt_path is not None:
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text("", encoding="utf-8")

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

    client = OpenAI()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    processed = 0

    mode = "a" if args.append_output else "w"

    with args.output.open(mode, encoding="utf-8") as out_f:
        for rec in records:
            payload = build_llm_input_payload(rec, gt_map)
            prompt = build_judging_prompt(payload)
            question = payload.get("question")
            if args.append_output and question in existing_questions:
                continue
            if prompt_path is not None:
                entry = {
                    "question": payload.get("question"),
                    "expected_answer": payload.get("expected_answer"),
                    "payload": payload,
                    "prompt_text": prompt,
                }
                with prompt_path.open("a", encoding="utf-8") as fh:
                    json.dump(entry, fh, ensure_ascii=False)
                    fh.write("\n")
            call_data = call_judge(client, prompt, args.model)
            output_text = call_data["text"]
            parsed = parse_output(output_text)

            if args.print_output:
                print("\n===== GPT RAW OUTPUT =====")
                print(output_text)
                print("===== END OUTPUT =====\n")

            entry: Dict[str, Any] = {
                "question": payload.get("question"),
                "is_correct": rec.get("is_correct"),
                "model": args.model,
                "raw_output": output_text,
            }
            if parsed is not None:
                entry["parsed_judgment"] = parsed
            else:
                entry["parse_error"] = "Failed to parse output as JSON"

            out_f.write(json.dumps(entry, ensure_ascii=False))
            out_f.write("\n")
            processed += 1
            if question:
                existing_questions.add(question)

    print(f"Processed {processed} records. Results saved to {args.output}")


if __name__ == "__main__":
    main()
