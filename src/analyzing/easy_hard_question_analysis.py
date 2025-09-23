#!/usr/bin/env python3
"""Identify easiest and hardest questions and examine token usage."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

MODEL_NAME_MAP: Dict[str, str] = {
    "responses_bedrock_mistral.mistral-large-2402-v1:0": "Mistral Large",
    "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0": "Claude 3.7 Sonnet",
    "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning": "Claude 3.7 Sonnet (reasoning)",
    "responses_bedrock_us.deepseek.r1-v1:0": "DeepSeek R1",
    "responses_bedrock_us.deepseek.r1-v1:0-reasoning": "DeepSeek R1 (reasoning)",
    "responses_openai_gpt-4o_reverified": "GPT-4o (reverified)",
    "responses_openai_gpt-5": "GPT-5",
}


@dataclass
class ResponseRecord:
    model: str
    is_correct: bool
    reasoning_tokens: Optional[float]
    output_tokens: Optional[float]
    number_of_hops: Optional[float]


QuestionMap = Dict[str, List[ResponseRecord]]

MAX_QUESTION_DETAILS = 10


def load_responses(responses_dir: Path) -> Tuple[QuestionMap, List[str]]:
    """Load response JSONL files and group records by question text."""
    question_records: QuestionMap = defaultdict(list)
    models: List[str] = []

    for path in sorted(responses_dir.glob("*.jsonl")):
        model_key = path.stem
        models.append(model_key)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as error:
                    print(f"Skipping {path.name}:{line_number} (invalid JSON: {error})")
                    continue

                question = entry.get("raw_response", {}).get("question")
                if not question:
                    continue

                record = ResponseRecord(
                    model=model_key,
                    is_correct=bool(entry.get("is_correct")),
                    reasoning_tokens=_safe_float(entry.get("reasoning_tokens")),
                    output_tokens=_safe_float(entry.get("output_tokens")),
                    number_of_hops=_safe_float(entry.get("number_of_hops")),
                )
                question_records[question].append(record)

    return question_records, models


def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def identify_easy_questions(question_records: QuestionMap, models: List[str]) -> Dict[str, List[ResponseRecord]]:
    """Questions answered correctly by every model present."""
    total_models = len(models)
    model_set = set(models)
    easy_questions: Dict[str, List[ResponseRecord]] = {}

    for question, records in question_records.items():
        if len(records) < total_models:
            continue
        models_answered = {record.model for record in records}
        if models_answered != model_set:
            continue
        if all(record.is_correct for record in records):
            easy_questions[question] = records
    return easy_questions


def identify_hard_questions(question_records: QuestionMap, *, min_wrong: int | None = None) -> Dict[str, List[ResponseRecord]]:
    """Questions with the maximum number of wrong answers or above a threshold."""
    hardest: Dict[str, List[ResponseRecord]] = {}
    max_wrong = 0

    for question, records in question_records.items():
        wrong_count = sum(1 for record in records if not record.is_correct)
        if wrong_count == 0:
            continue

        if min_wrong is not None:
            if wrong_count >= min_wrong:
                hardest[question] = records
            continue

        if wrong_count > max_wrong:
            max_wrong = wrong_count
            hardest = {question: records}
        elif wrong_count == max_wrong:
            hardest[question] = records
    return hardest


def summarise_token_usage(question_map: Dict[str, List[ResponseRecord]], only_wrong: bool = False) -> Dict[str, Dict[str, float]]:
    """Return average reasoning/output tokens per model for the given questions."""
    per_model_tokens: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"reasoning": [], "output": []})
    for records in question_map.values():
        for record in records:
            if only_wrong and record.is_correct:
                continue
            if record.reasoning_tokens is not None:
                per_model_tokens[record.model]["reasoning"].append(record.reasoning_tokens)
            if record.output_tokens is not None:
                per_model_tokens[record.model]["output"].append(record.output_tokens)

    summaries: Dict[str, Dict[str, float]] = {}
    for model, token_lists in per_model_tokens.items():
        summaries[model] = {
            "reasoning_avg": mean(token_lists["reasoning"]) if token_lists["reasoning"] else float("nan"),
            "reasoning_samples": len(token_lists["reasoning"]),
            "output_avg": mean(token_lists["output"]) if token_lists["output"] else float("nan"),
            "output_samples": len(token_lists["output"]),
        }
    return summaries


def print_summary(title: str, question_map: Dict[str, List[ResponseRecord]], token_summary: Dict[str, Dict[str, float]], models: List[str]) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")
    print(f"Question count: {len(question_map)}")

    if not question_map:
        return

    for idx, (question, records) in enumerate(question_map.items(), 1):
        if idx > MAX_QUESTION_DETAILS:
            remaining = len(question_map) - MAX_QUESTION_DETAILS
            if remaining > 0:
                print(f"\n... (skipping {remaining} additional questions)")
            break
        wrong_count = sum(1 for record in records if not record.is_correct)
        hops = sorted({record.number_of_hops for record in records if record.number_of_hops is not None})
        print(f"\n[{idx}] {question}")
        print(f"  Wrong answers: {wrong_count} / {len(records)} | Hop counts: {hops if hops else 'n/a'}")
        for record in records:
            status = 'correct' if record.is_correct else 'wrong'
            display_name = MODEL_NAME_MAP.get(record.model, record.model)
            rt = f"{record.reasoning_tokens:.1f}" if record.reasoning_tokens is not None else 'n/a'
            ot = f"{record.output_tokens:.1f}" if record.output_tokens is not None else 'n/a'
            print(f"    - {display_name:<30} {status:>7} | reasoning={rt} | output={ot}")

    print("\nAverage token usage per model:")
    header = f"{'Model':<35} {'Reasoning n':>12} {'Reasoning avg':>16} {'Output n':>10} {'Output avg':>12}"
    print(header)
    print("-" * len(header))
    for model in models:
        display_name = MODEL_NAME_MAP.get(model, model)
        summary = token_summary.get(model)
        if summary is None:
            print(f"{display_name:<35} {'-':>12} {'-':>16} {'-':>10} {'-':>12}")
            continue
        reasoning_avg = summary.get('reasoning_avg', float('nan'))
        output_avg = summary.get('output_avg', float('nan'))
        reasoning_count = summary.get('reasoning_samples', 0)
        output_count = summary.get('output_samples', 0)
        reasoning_str = f"{reasoning_avg:.1f}" if reasoning_avg == reasoning_avg else "n/a"
        output_str = f"{output_avg:.1f}" if output_avg == output_avg else "n/a"
        print(f"{display_name:<35} {reasoning_count:>12} {reasoning_str:>16} {output_count:>10} {output_str:>12}")




def serialize_question_map(question_map: Dict[str, List[ResponseRecord]]) -> List[Dict[str, object]]:
    """Convert a question map into a list of serialisable dictionaries."""
    serialised: List[Dict[str, object]] = []
    for question, records in question_map.items():
        wrong_count = sum(1 for record in records if not record.is_correct)
        hop_values = sorted({record.number_of_hops for record in records if record.number_of_hops is not None})
        serialised.append({
            "question": question,
            "wrong_count": wrong_count,
            "total_answers": len(records),
            "hop_counts": hop_values,
            "records": [
                {
                    "model": record.model,
                    "is_correct": record.is_correct,
                    "reasoning_tokens": record.reasoning_tokens,
                    "output_tokens": record.output_tokens,
                    "number_of_hops": record.number_of_hops,
                }
                for record in records
            ],
        })
    return serialised


def write_question_map_jsonl(path: Path, question_map: Dict[str, List[ResponseRecord]]) -> None:
    """Write the question map to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = serialize_question_map(question_map)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_question_map_jsonl(path: Path) -> Dict[str, List[ResponseRecord]]:
    """Load question data from a JSONL file into a question map."""
    question_map: Dict[str, List[ResponseRecord]] = defaultdict(list)
    if not path.exists():
        return question_map
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            question = entry.get("question")
            if not question:
                continue
            records_data = entry.get("records") or []
            for record_data in records_data:
                question_map[question].append(
                    ResponseRecord(
                        model=record_data.get("model", ""),
                        is_correct=bool(record_data.get("is_correct")),
                        reasoning_tokens=_safe_float(record_data.get("reasoning_tokens")),
                        output_tokens=_safe_float(record_data.get("output_tokens")),
                        number_of_hops=_safe_float(record_data.get("number_of_hops")),
                    )
                )
    return question_map
