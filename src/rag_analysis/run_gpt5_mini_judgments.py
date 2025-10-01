"""Run GPT-5 judge on incorrect RAG responses using the prepared prompt."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from openai import OpenAI

# Ensure package imports work when running as a script via python path/to/file.py
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

from rag_analysis.prepare_llm_inputs import (  # noqa: E402
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


def filter_incorrect(records: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    for rec in records:
        if rec.get("is_correct") is False:
            yield rec


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
    ap = argparse.ArgumentParser(description="Run GPT-5-mini judgments on incorrect responses")
    default_jsonl = (
        CURRENT_DIR.parent
        / "responses_reverified"
        / "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning_reverified.jsonl"
    )
    default_output = CURRENT_DIR / "gpt5_mini_judgments.jsonl"
    ap.add_argument("--jsonl", type=Path, default=default_jsonl, help="Path to *_reverified.jsonl")
    ap.add_argument("--output", type=Path, default=default_output, help="Where to store the JSONL judgments")
    ap.add_argument("--model", type=str, default="gpt-5-mini", help="Judge model to use")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of records to process")
    ap.add_argument("--print-output", action="store_true", help="Print the full model output for each processed record")
    return ap


def main() -> None:
    args = build_cli().parse_args()

    if not args.jsonl.exists():
        raise FileNotFoundError(f"JSONL file not found: {args.jsonl}")

    gt_map = load_ground_truth_map()
    client = OpenAI()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    processed = 0

    with args.output.open("w", encoding="utf-8") as out_f:
        for rec in filter_incorrect(iter_jsonl(args.jsonl)):
            payload = build_llm_input_payload(rec, gt_map)
            prompt = build_judging_prompt(payload)
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

            if args.limit and processed >= args.limit:
                break

    print(f"Processed {processed} records. Results saved to {args.output}")


if __name__ == "__main__":
    main()
