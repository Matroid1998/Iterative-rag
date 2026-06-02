"""Load and normalize the ChemKGMultiHopQA benchmark dataset.

The raw dataset (``data/docs/chemrxiv_qa.json``) is a list of records shaped like::

    {"q": "...", "a": "...", "path": [{"entity1": ..., "relation": ..., "text": ...}, ...]}

We normalize each record to ``{"question", "expected", "number_of_hops"}`` where the
number of hops is the length of the oracle reasoning ``path``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from iterative_rag import config


def normalize_records(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize raw QA records to {question, expected, number_of_hops}."""
    out: List[Dict[str, Any]] = []
    for r in recs:
        q = r.get("question") or r.get("q")
        a = r.get("expected") or r.get("a")
        path = r.get("path")
        num_hops = len(path) if isinstance(path, list) else 0
        if not q:
            continue
        out.append({"question": q, "expected": a, "number_of_hops": num_hops})
    return out


def load_records(
    path: Optional[Path] = None,
    *,
    limit: Optional[int] = None,
    selection_file: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Load + normalize the QA dataset.

    Args:
        path: dataset JSON path (defaults to ``config.QA_DATASET``).
        limit: if set and smaller than the dataset, sample this many questions.
        selection_file: if given, persist/reuse the sampled question set here so
            repeated runs with the same ``limit`` evaluate the same questions.
    """
    path = Path(path) if path else config.QA_DATASET
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    records = normalize_records(raw if isinstance(raw, list) else [])

    if not limit or limit <= 0 or len(records) <= limit:
        return records

    selected_questions: List[str] = []
    if selection_file and Path(selection_file).exists():
        try:
            selected_questions = json.loads(Path(selection_file).read_text(encoding="utf-8")) or []
        except Exception:
            selected_questions = []

    if not selected_questions:
        sampled = random.sample(records, k=limit)
        selected_questions = [r["question"] for r in sampled if r.get("question")]
        if selection_file:
            Path(selection_file).parent.mkdir(parents=True, exist_ok=True)
            Path(selection_file).write_text(
                json.dumps(selected_questions, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    sel = set(selected_questions)
    order = {q: i for i, q in enumerate(selected_questions)}
    records = [r for r in records if r.get("question") in sel]
    records.sort(key=lambda r: order.get(r.get("question"), 1_000_000))
    return records
