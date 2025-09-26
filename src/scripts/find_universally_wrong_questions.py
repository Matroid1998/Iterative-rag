#!/usr/bin/env python3
"""Identify unanswered questions per response folder and write JSONL reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-folder JSONL reports listing questions that were never "
            "answered correctly."
        )
    )
    parser.add_argument(
        "folders",
        nargs="*",
        type=Path,
        help="Folders to scan for *.jsonl files (defaults to the project response directories).",
    )
    parser.add_argument(
        "--show-warnings",
        action="store_true",
        help="Display detailed parse warnings for malformed JSON lines.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print each unanswered question instead of only the summary count.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory where per-folder JSONL reports are written "
            "(default: results/unanswered_questions)."
        ),
    )
    return parser.parse_args()


def iter_jsonl_files(folder: Path) -> Iterable[Path]:
    yield from sorted(p for p in folder.glob("*.jsonl") if p.is_file())


def build_context_lookup(folder: Path) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    if not folder.exists():
        return lookup
    for path in iter_jsonl_files(folder):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                question = extract_question(record)
                if not question:
                    continue
                context = None
                raw_section = record.get("raw")
                if isinstance(raw_section, dict):
                    context = raw_section.get("context")
                if context is None:
                    raw_response = record.get("raw_response")
                    if isinstance(raw_response, dict):
                        context = raw_response.get("context")
                if isinstance(context, str) and context.strip():
                    lookup.setdefault(question, context)
    return lookup


def extract_question(record: dict) -> Optional[str]:
    for key in ("raw", "raw_response"):
        raw = record.get(key)
        if isinstance(raw, dict):
            question = raw.get("question")
            if isinstance(question, str) and question.strip():
                return question.strip()
    question = record.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()
    return None


def extract_expected(record: dict) -> Optional[str]:
    for key in ("raw", "raw_response"):
        raw = record.get(key)
        if isinstance(raw, dict):
            expected = raw.get("expected") or raw.get("answer")
            if isinstance(expected, str) and expected.strip():
                return expected.strip()
    expected = record.get("expected")
    if isinstance(expected, str) and expected.strip():
        return expected.strip()
    return None


def parse_is_correct(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def main() -> None:
    args = parse_args()
    base = Path(__file__).resolve().parents[1]
    if args.folders:
        folders = [folder.resolve() for folder in args.folders]
    else:
        folders = [
            base / "responses_reverified",
            base / "response-jsonl-without-context",
            base / "response-jsonl-with-context",
        ]

    context_lookup: Dict[str, str] = build_context_lookup(base / "response-jsonl-with-context")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else base / "results" / "unanswered_questions"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    folder_stats: Dict[Path, Dict[str, Dict[str, object]]] = {}
    unanswered_by_folder: Dict[Path, list] = {}
    errors = []

    for folder in folders:
        if not folder.exists():
            errors.append(f"Folder not found: {folder}")
            continue
        per_stats: Dict[str, Dict[str, object]] = {}
        folder_stats[folder] = per_stats
        for path in iter_jsonl_files(folder):
            with path.open("r", encoding="utf-8") as handle:
                for idx, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        record = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        errors.append(f"Failed to parse {path}:{idx}: {exc}")
                        continue
                    question = extract_question(record)
                    if not question:
                        errors.append(f"Missing question in {path}:{idx}")
                        continue
                    expected = extract_expected(record)
                    is_correct = parse_is_correct(record.get("is_correct"))

                    entry = per_stats.setdefault(
                        question,
                        {
                            "attempts": [],
                            "correct": 0,
                            "expected": set(),
                            "context": None,
                            "hops": {},
                        },
                    )
                    if expected:
                        entry["expected"].add(expected)
                    if is_correct:
                        entry["correct"] += 1
                    candidate = record.get("candidate")
                    if not candidate:
                        raw_response = record.get("raw_response")
                        if isinstance(raw_response, dict):
                            candidate = raw_response.get("answer") or raw_response.get(
                                "candidate"
                            )
                    if not candidate:
                        raw_entry = record.get("raw")
                        if isinstance(raw_entry, dict):
                            candidate = raw_entry.get("answer")
                    if candidate is None:
                        candidate = ""
                    context = None
                    raw_section = record.get("raw")
                    if isinstance(raw_section, dict):
                        context = raw_section.get("context")
                    if context is None:
                        raw_response = record.get("raw_response")
                        if isinstance(raw_response, dict):
                            context = raw_response.get("context")
                    if not context and question in context_lookup:
                        context = context_lookup[question]
                    if isinstance(context, str) and context.strip():
                        context_lookup.setdefault(question, context)
                        if not entry["context"]:
                            entry["context"] = context
                    hop_count = record.get("number_of_hops")
                    if hop_count is not None:
                        entry["hops"].setdefault(path.name, hop_count)
                    attempt_data = {
                        "file": path.name,
                        "answer": candidate,
                        "number_of_hops": hop_count,
                        "is_correct": bool(is_correct),
                    }
                    entry["attempts"].append(attempt_data)

    for folder, per_stats in folder_stats.items():
        unanswered_records = []
        for question, data in per_stats.items():
            total_attempts = len(data["attempts"])
            if total_attempts and not data["correct"]:
                unanswered_records.append(
                    {
                        "source_folder": folder.name,
                        "question": question,
                        "expected_answers": sorted(data["expected"]),
                        "attempts": total_attempts,
                        "context": data.get("context"),
                        "number_of_hops": data.get("hops"),
                        "model_attempts": data["attempts"],
                    }
                )

        unanswered_records.sort(key=lambda item: item["question"])

        output_path = output_dir / f"{folder.name}_unanswered.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for record in unanswered_records:
                json.dump(record, handle, ensure_ascii=False)
                handle.write("\n")

        try:
            relative_output = output_path.relative_to(base)
        except ValueError:
            relative_output = output_path

        print(
            f"{folder.name}: {len(unanswered_records)} unanswered "
            f"(written to {relative_output})"
        )

        unanswered_by_folder[folder] = unanswered_records

        if args.list and unanswered_records:
            for record in unanswered_records:
                expected_display = (
                    record["expected_answers"][0]
                    if record["expected_answers"]
                    else "(no expected answer recorded)"
                )
                print("\nQuestion:")
                print(record["question"])
                print(f"Expected answer: {expected_display}")
                print(f"Attempts tracked: {record['attempts']}")
                if record.get("context"):
                    print("Context:")
                    print(record["context"])
                hops_map = record.get("number_of_hops") or {}
                if hops_map:
                    print("Number of hops:")
                    for file_name, hops in hops_map.items():
                        print(f"- {file_name}: {hops}")
                for attempt in record["model_attempts"]:
                    status = "correct" if attempt.get("is_correct") else "incorrect"
                    print(
                        f"- {attempt['file']}: answer={attempt['answer']} ({status})"
                    )

    if folders:
        base_folder = folders[0]
        base_unanswered = unanswered_by_folder.get(base_folder, [])
        answered_elsewhere_records = []

        for record in base_unanswered:
            question = record["question"]
            answered_in = []
            for other_folder, per_stats in folder_stats.items():
                if other_folder == base_folder:
                    continue
                data = per_stats.get(question)
                if not data:
                    continue
                correct_attempts = [
                    attempt
                    for attempt in data["attempts"]
                    if attempt.get("is_correct")
                ]
                if correct_attempts:
                    answered_in.append(
                        {
                            "source_folder": other_folder.name,
                            "attempts": correct_attempts,
                        }
                    )

            if answered_in:
                answered_elsewhere_records.append(
                    {
                        "question": question,
                        "expected_answers": record["expected_answers"],
                        "answered_in": answered_in,
                    }
                )

        if answered_elsewhere_records:
            cross_output = (
                output_dir / f"{base_folder.name}_answered_elsewhere.jsonl"
            )
            with cross_output.open("w", encoding="utf-8") as handle:
                for cross_record in answered_elsewhere_records:
                    json.dump(cross_record, handle, ensure_ascii=False)
                    handle.write("\n")

            try:
                relative_cross = cross_output.relative_to(base)
            except ValueError:
                relative_cross = cross_output

            print(
                f"{base_folder.name}: {len(answered_elsewhere_records)} unanswered here but answered elsewhere "
                f"(written to {relative_cross})"
            )

    if errors:
        if args.show_warnings:
            print("\nWarnings:")
            for message in errors:
                print(f"- {message}")
        else:
            print(
                f"\nSkipped {len(errors)} malformed records (use --show-warnings for details)."
            )


if __name__ == "__main__":
    main()
