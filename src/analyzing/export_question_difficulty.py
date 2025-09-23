#!/usr/bin/env python3
"""Extract easiest and hardest questions and store them as JSONL files."""

from __future__ import annotations

from pathlib import Path

from easy_hard_question_analysis import (
    identify_easy_questions,
    identify_hard_questions,
    load_responses,
    serialize_question_map,
    write_question_map_jsonl,
)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    responses_dir = repo_root / "src" / "responses"
    output_dir = script_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    question_records, models = load_responses(responses_dir)

    easy_questions = identify_easy_questions(question_records, models)
    hard_questions = identify_hard_questions(question_records, min_wrong=4)

    easy_path = output_dir / "easy_questions.jsonl"
    hard_path = output_dir / "hard_questions_min4.jsonl"

    write_question_map_jsonl(easy_path, easy_questions)
    write_question_map_jsonl(hard_path, hard_questions)

    print(f"Wrote {len(serialize_question_map(easy_questions))} easy questions to {easy_path}")
    print(f"Wrote {len(serialize_question_map(hard_questions))} hard questions to {hard_path}")


if __name__ == "__main__":
    main()
