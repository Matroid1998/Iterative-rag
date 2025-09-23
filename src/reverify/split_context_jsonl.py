"""Split response JSONL files by whether context was used."""
from __future__ import annotations

import json
from pathlib import Path


def split_jsonl_files() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    source_dir = base_dir / "responses-jsonl-old"
    with_context_dir = base_dir / "response-jsonl-with-context"
    without_context_dir = base_dir / "response-jsonl-without-context"

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    with_context_dir.mkdir(parents=True, exist_ok=True)
    without_context_dir.mkdir(parents=True, exist_ok=True)

    for jsonl_path in sorted(source_dir.glob("*.jsonl")):
        with_context_path = with_context_dir / jsonl_path.name
        without_context_path = without_context_dir / jsonl_path.name

        with (
            jsonl_path.open("r", encoding="utf-8") as src,
            with_context_path.open("w", encoding="utf-8") as with_dst,
            without_context_path.open("w", encoding="utf-8") as without_dst,
        ):
            for raw_line in src:
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    msg = f"Failed to parse JSON in {jsonl_path.name}: {exc}"
                    raise ValueError(msg) from exc

                target = without_dst if payload.get("context_used") is False else with_dst
                target.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")


if __name__ == "__main__":
    split_jsonl_files()
