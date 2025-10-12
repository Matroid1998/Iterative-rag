import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from evaluator import Answer, Evaluate, Provider, StructuredLLM


RESPONSES_DIR = Path(__file__).resolve().parent.parent / "responses_reverified"

# Files explicitly called out for re-run
TARGET_FILES = [
    RESPONSES_DIR / "responses_openrouter_anthropic__claude-sonnet-4.5_reverified.jsonl",
    RESPONSES_DIR / "responses_openrouter_google__gemini-2.5-pro_reverified.jsonl",
    RESPONSES_DIR / "responses_openrouter_x-ai__grok-4-fast_reverified.jsonl",
    RESPONSES_DIR / "responses_openrouter_z-ai__glm-4.6_reverified.jsonl",
]


def _load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: List[Dict]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as handle:
        for record in records:
            handle.write(json.dumps(record, default=str))
            handle.write("\n")
    os.replace(tmp_path, path)


def _parse_provider_model(path: Path) -> Tuple[Provider, str]:
    stem = path.stem  # e.g. responses_openrouter_anthropic__claude-sonnet-4.5_reverified
    if not stem.startswith("responses_"):
        raise ValueError(f"Unexpected filename format: {path.name}")
    remainder = stem[len("responses_") :]
    suffix = "_reverified"
    if remainder.endswith(suffix):
        remainder = remainder[: -len(suffix)]
    provider_value, _, model_slug = remainder.partition("_")
    if not provider_value or not model_slug:
        raise ValueError(f"Cannot extract provider/model from {path.name}")
    provider = Provider(provider_value)
    model_id = model_slug.replace("__", "/")
    return provider, model_id


def _make_records(entries: List[Dict]) -> List[Dict]:
    prepared: List[Dict] = []
    for entry in entries:
        raw = entry.get("raw") or {}
        question = raw.get("question")
        expected = raw.get("expected")
        number_of_hops = raw.get("number_of_hops", 0)
        if not question:
            raise ValueError("Missing question in raw entry")
        prepared.append(
            {
                "question": question,
                "expected": expected,
                "number_of_hops": number_of_hops,
            }
        )
    return prepared


def rerun_file(path: Path) -> None:
    provider, model_id = _parse_provider_model(path)
    print(f"\nProcessing {path.name} ({provider.value} / {model_id})")

    records = _load_jsonl(path)
    missing = [(idx, rec) for idx, rec in enumerate(records) if not rec.get("candidate")]
    if not missing:
        print("No empty candidates found; skipping.")
        return

    total_missing = len(missing)
    print(f"Found {total_missing} empty candidate entries; regenerating...")
    qa_llm = StructuredLLM(
        provider=provider,
        model_id=model_id,
        output_format=Answer,
    )

    records_payload = _make_records([rec for _, rec in missing])
    evaluator = Evaluate(
        qa_llm=qa_llm,
        records=records_payload,
        use_context=True,
        responses_save_path=None,
        num_workers=1,
    )
    verifier_llm = StructuredLLM(**evaluator.verifier_llm_params)
    rag_service, llm_client = evaluator._create_worker_rag_service()
    worker_ctx = (verifier_llm, rag_service, llm_client)

    for (idx, original_entry), record_payload in zip(missing, records_payload):
        print(f"- Re-running question: {record_payload['question']}")
        updated = evaluator._process_record_iterative_rag(
            record_payload, worker_ctx=worker_ctx
        )
        records[idx] = updated

    _write_jsonl(path, records)
    print(f"Recovered {total_missing} entries in {path.name}")


def main() -> None:
    summary: Dict[str, int] = {}
    for target in TARGET_FILES:
        if not target.exists():
            print(f"Warning: {target} does not exist; skipping.")
            continue
        before = len([True for _ in open(target) if '"candidate": ""' in _])
        rerun_file(target)
        after = len([True for _ in open(target) if '"candidate": ""' in _])
        recovered = before - after
        summary[target.name] = max(recovered, 0)

    if summary:
        print("\nRecovery summary:")
        for name, count in summary.items():
            print(f"- {name}: {count} question(s) recovered")


if __name__ == "__main__":
    main()
