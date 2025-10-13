#!/usr/bin/env python3

"""
Prune JSONL response files so that only questions present in a reference file remain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence, Set, Tuple


def extract_question(obj: object) -> Optional[str]:
    """
    Extract the question string from the JSON record, handling multiple schemas.
    """
    if not isinstance(obj, dict):
        return None

    raw = obj.get("raw")
    if isinstance(raw, dict):
        question = raw.get("question")
        if isinstance(question, str):
            return question

    raw_response = obj.get("raw_response")
    if isinstance(raw_response, dict):
        question = raw_response.get("question")
        if isinstance(question, str):
            return question
    elif isinstance(raw_response, str):
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            question = parsed.get("question")
            if isinstance(question, str):
                return question

    question = obj.get("question")
    if isinstance(question, str):
        return question

    return None


def load_allowed_questions(reference_path: Path) -> Set[str]:
    allowed: Set[str] = set()
    with reference_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            question = extract_question(record)
            if question:
                allowed.add(question)
    return allowed


def prune_file(path: Path, allowed_questions: Set[str]) -> Tuple[int, int, int]:
    """
    Return the tuple (total_records, kept_records, missing_question_records).
    """
    with path.open("r", encoding="utf-8") as fh:
        original_lines = fh.readlines()

    total_records = 0
    kept_records = 0
    missing_question_records = 0
    pruned_lines = []

    for line in original_lines:
        stripped = line.strip()
        if not stripped:
            pruned_lines.append(line)
            continue
        try:
            record = json.loads(line)
            total_records += 1
        except json.JSONDecodeError:
            pruned_lines.append(line)
            continue

        question = extract_question(record)
        if question is None:
            missing_question_records += 1
            pruned_lines.append(line)
            continue

        if question in allowed_questions:
            kept_records += 1
            pruned_lines.append(line)

    if pruned_lines != original_lines:
        with path.open("w", encoding="utf-8") as fh:
            fh.writelines(pruned_lines)

    return total_records, kept_records, missing_question_records


def iter_jsonl_files(paths: Sequence[Path]) -> Iterable[Path]:
    for root in paths:
        if root.is_file() and root.suffix == ".jsonl":
            yield root
        elif root.is_dir():
            for file_path in sorted(root.glob("*.jsonl")):
                if file_path.is_file():
                    yield file_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
        help="Reference JSONL file from responses_reverified.",
    )
    parser.add_argument(
        "targets",
        nargs="+",
        type=Path,
        help="Directories or files to prune.",
    )
    args = parser.parse_args()

    reference_path = args.reference
    if not reference_path.exists():
        raise SystemExit(f"Reference file not found: {reference_path}")

    allowed_questions = load_allowed_questions(reference_path)
    if not allowed_questions:
        raise SystemExit(
            f"No questions extracted from reference file: {reference_path}"
        )

    any_processed = False
    for jsonl_file in iter_jsonl_files(args.targets):
        any_processed = True
        total, kept, missing = prune_file(jsonl_file, allowed_questions)
        removed = total - kept
        print(
            f"{jsonl_file}: kept {kept}/{total}, removed {removed}, missing_question {missing}"
        )

    if not any_processed:
        raise SystemExit("No JSONL files found in provided targets.")


if __name__ == "__main__":
    main()
