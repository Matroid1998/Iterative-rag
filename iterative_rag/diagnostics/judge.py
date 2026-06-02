"""Batch LLM-as-judge runner that regenerates the diagnostic JSONL files.

For each record in a response JSONL it builds the payload, asks the judge model to audit
it under the chosen system prompt, parses the JSON verdict, and appends a record matching
the schema of the shipped ``diagnostics_output/*`` files::

    {question, expected_answer, number_of_hops, model, raw_output, parsed_judgment}
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from iterative_rag import config
from iterative_rag.diagnostics.payload import build_payload, iter_records, load_ground_truth_map
from iterative_rag.diagnostics.prompts import DIAGNOSTICS


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort parse of a JSON object out of an LLM response."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):]
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(t[start:end + 1])
    except json.JSONDecodeError:
        return None


def run_diagnostics(
    responses_path: Path,
    kind: str,
    *,
    judge_provider: Optional[str] = None,
    judge_model: Optional[str] = None,
    out_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    workers: int = 2,
) -> Path:
    """Judge ``responses_path`` for one diagnostic ``kind`` and write the JSONL.

    Returns the path of the written judgment file.
    """
    if kind not in DIAGNOSTICS:
        raise ValueError(f"Unknown diagnostic kind '{kind}'. Choose from {list(DIAGNOSTICS)}.")
    system_prompt, suffix = DIAGNOSTICS[kind]

    responses_path = Path(responses_path)
    out_dir = Path(out_dir) if out_dir else config.DIAGNOSTICS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{responses_path.stem}{suffix}"

    gt_map = load_ground_truth_map()
    judge_provider = judge_provider or config.JUDGE_PROVIDER
    judge_model = judge_model or config.JUDGE_MODEL

    # Imported lazily so the module imports without the LLM stack.
    from iterative_rag.system.structured_llm import StructuredLLMClient
    client = StructuredLLMClient(provider=judge_provider, model=judge_model, temperature=0.0)

    records = list(iter_records(responses_path))
    if limit:
        records = records[:limit]

    def judge_one(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            payload = build_payload(rec, gt_map)
        except KeyError:
            return None
        raw_output = client.complete(system_prompt, json.dumps(payload, ensure_ascii=False))
        return {
            "question": payload["question"],
            "expected_answer": payload["expected_answer"],
            "number_of_hops": payload["number_of_hops"],
            "is_correct": rec.get("is_correct"),
            "model": judge_model,
            "raw_output": raw_output,
            "parsed_judgment": _extract_json(raw_output),
        }

    results: List[Dict[str, Any]] = []
    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = [r for r in ex.map(judge_one, records) if r is not None]
    else:
        results = [r for r in (judge_one(r) for r in records) if r is not None]

    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[{kind}] judged {len(results)}/{len(records)} records -> {out_path}")
    return out_path
