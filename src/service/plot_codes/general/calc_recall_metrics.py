#!/usr/bin/env python3
"""
Compute Recall@K metrics for Iterative RAG runs.

The script derives gold supporting passages from `chemrxiv_qa.json`,
matches them to passage files in `chemrxiv_graph_v2_texts`, and then
evaluates each model jsonl in `src/responses_reverified/`.

Outputs:
    - Overall macro Recall@K per model.
    - Cumulative Recall@K by retrieval step.
    - Distribution of the first step where any supporting passage appears.

Usage:
    python src/presentation/analysis/general/calc_recall_metrics.py

Adjust constants `K_VALUES` or paths below to customise behaviour.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Configuration ----------------------------------------------------------------

K_VALUES: Tuple[int, ...] = (1, 3, 5, 10)
DOC_ROOT = Path("data/corpus/chemrxiv_graph_v2_texts")
QA_PATH = Path("data/corpus/chemrxiv_qa.json")
RESPONSES_DIR = Path("src/responses_reverified")


# Utilities --------------------------------------------------------------------

_WHITESPACE = re.compile(r"\s+")


def normalise_text(text: str) -> str:
    """Collapse whitespace so passage comparisons are robust to formatting."""
    return _WHITESPACE.sub(" ", text.strip())


def read_text_payload(path: Path) -> str:
    """Extract the main text payload from a passage file."""
    raw = path.read_text(encoding="utf-8")
    if "Text:\n" in raw:
        return raw.split("Text:\n", 1)[1]
    return raw


# Data loading -----------------------------------------------------------------

def build_passage_lookup(root: Path) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Returns:
        doc_text_map: {doc_id: normalised_text}
        exact_lookup: {normalised_text: [doc_id, ...]}
    """
    doc_text_map: Dict[str, str] = {}
    exact_lookup: Dict[str, List[str]] = defaultdict(list)

    for file_path in sorted(root.rglob("*.txt")):
        doc_id = str(file_path.relative_to(root))
        norm_text = normalise_text(read_text_payload(file_path))
        doc_text_map[doc_id] = norm_text
        exact_lookup[norm_text].append(doc_id)
    return doc_text_map, exact_lookup


def map_gold_passages(
    qa_path: Path,
    doc_text_map: Mapping[str, str],
    exact_lookup: Mapping[str, Sequence[str]],
) -> Dict[str, List[str]]:
    """
    Build question -> supporting doc ids index.

    Falls back to substring search when an exact passage match is not found.
    """
    questions = json.loads(qa_path.read_text())
    gold_index: Dict[str, List[str]] = {}

    missing_snippets: List[Tuple[str, str]] = []

    for item in questions:
        question = item["q"].strip()
        gold_docs: List[str] = []
        for hop in item.get("path", []):
            norm_text = normalise_text(hop["text"])

            matches = exact_lookup.get(norm_text)
            if matches:
                gold_docs.extend(matches)
                continue

            # Fallback: locate passages that contain the snippet.
            candidates = [
                doc_id
                for doc_id, doc_text in doc_text_map.items()
                if norm_text and norm_text in doc_text
            ]
            if candidates:
                gold_docs.extend(candidates)
            else:
                missing_snippets.append((question, hop["text"]))

        # Deduplicate while preserving order.
        seen: set[str] = set()
        ordered_docs: List[str] = []
        for doc_id in gold_docs:
            if doc_id not in seen:
                ordered_docs.append(doc_id)
                seen.add(doc_id)
        if ordered_docs:
            gold_index[question] = ordered_docs

    if missing_snippets:
        print(f"[WARN] Unmapped supporting passages: {len(missing_snippets)}", flush=True)
    return gold_index
    return gold_index


def load_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                records.append(json.loads(raw_line))
            except json.JSONDecodeError:
                continue
    return records


# Evidence parsing -------------------------------------------------------------

def extract_question(record: dict) -> Optional[str]:
    question = record.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()
    for key in ("raw", "raw_response"):
        value = record.get(key)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                continue
        if isinstance(value, dict):
            q = value.get("question")
            if isinstance(q, str) and q.strip():
                return q.strip()
    return None


@dataclass
class Evidence:
    step: int
    doc_id: str
    text: Optional[str]
    order: int  # original appearance order


def parse_evidence(record: dict) -> List[Evidence]:
    evidences: List[Evidence] = []

    def iter_sources() -> Iterable[dict]:
        if isinstance(record.get("evidence"), list):
            yield from record["evidence"]

        for key in ("raw_response", "raw"):
            value = record.get(key)
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    continue
            if isinstance(value, dict):
                ev = value.get("evidence")
                if isinstance(ev, list):
                    yield from ev

    for idx, ev in enumerate(iter_sources()):
        if not isinstance(ev, dict):
            continue
        doc_id = ev.get("doc_id")
        if not doc_id:
            metadata = ev.get("metadata")
            if isinstance(metadata, dict):
                doc_id = metadata.get("doc_id")
            if not doc_id:
                raw_id = ev.get("id")
                if isinstance(raw_id, str) and "::" in raw_id:
                    doc_id = raw_id.split("::", 1)[0]
        if not doc_id:
            continue

        step = ev.get("source_step")
        if isinstance(step, str) and step.isdigit():
            step = int(step)
        elif isinstance(step, (int, float)):
            step = int(step)
        else:
            step = 1

        evidences.append(
            Evidence(
                step=step,
                doc_id=str(doc_id),
                text=ev.get("text") if isinstance(ev.get("text"), str) else None,
                order=idx,
            )
        )

    evidences.sort(key=lambda e: (e.step, e.order))
    return evidences


# Metrics ----------------------------------------------------------------------

def recall_at_k(
    gold_docs: Sequence[str],
    retrieved_docs: Sequence[str],
    k: int,
) -> float:
    if not gold_docs or not retrieved_docs:
        return 0.0

    top_k = retrieved_docs[:k]
    hits = sum(1 for doc in gold_docs if doc in top_k)
    return hits / len(gold_docs)


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


@dataclass
class QuestionMetrics:
    question: str
    gold_docs: List[str]
    overall_recalls: Dict[int, float]
    step_recalls: Dict[int, Dict[int, float]]
    first_hit_step: Optional[int]
    total_steps: int


def evaluate_record(
    question: str,
    gold_docs: Sequence[str],
    evidences: Sequence[Evidence],
) -> QuestionMetrics:
    step_map: Dict[int, List[str]] = defaultdict(list)
    for ev in evidences:
        step_map[ev.step].append(ev.doc_id)

    steps_sorted = sorted(step_map)
    cumulative_docs: List[str] = []
    cumulative_by_step: Dict[int, List[str]] = {}
    for step in steps_sorted:
        cumulative_docs.extend(step_map[step])
        cumulative_docs = dedupe_preserve_order(cumulative_docs)
        cumulative_by_step[step] = cumulative_docs.copy()

    all_docs = cumulative_docs
    overall_recalls = {
        k: recall_at_k(gold_docs, all_docs, k)
        for k in K_VALUES
    }

    step_recalls: Dict[int, Dict[int, float]] = {}
    first_hit_step: Optional[int] = None
    for step in steps_sorted:
        recalls_at_step: Dict[int, float] = {}
        for k in K_VALUES:
            recall_value = recall_at_k(gold_docs, cumulative_by_step[step], k)
            recalls_at_step[k] = recall_value
        step_recalls[step] = recalls_at_step

        if first_hit_step is None and recalls_at_step[max(K_VALUES)] > 0:
            first_hit_step = step

    return QuestionMetrics(
        question=question,
        gold_docs=list(gold_docs),
        overall_recalls=overall_recalls,
        step_recalls=step_recalls,
        first_hit_step=first_hit_step,
        total_steps=len(steps_sorted),
    )


@dataclass
class AggregateMetrics:
    num_questions: int
    overall_by_k: Dict[int, float]
    step_by_k: Dict[int, Dict[int, float]]
    first_hit_distribution: Dict[str, int]
    gold_doc_counts: Counter


def aggregate_metrics(metrics: Sequence[QuestionMetrics]) -> AggregateMetrics:
    if not metrics:
        return AggregateMetrics(
            num_questions=0,
            overall_by_k={k: 0.0 for k in K_VALUES},
            step_by_k={},
            first_hit_distribution={},
            gold_doc_counts=Counter(),
        )

    overall_by_k: Dict[int, List[float]] = {k: [] for k in K_VALUES}
    step_temp: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: {k: [] for k in K_VALUES})
    first_hit: Counter = Counter()
    gold_doc_counts: Counter = Counter()

    for qm in metrics:
        gold_doc_counts[len(qm.gold_docs)] += 1
        for k, val in qm.overall_recalls.items():
            overall_by_k[k].append(val)
        for step, recalls in qm.step_recalls.items():
            for k, val in recalls.items():
                step_temp[step][k].append(val)

        if qm.first_hit_step is None:
            first_hit["never"] += 1
        else:
            first_hit[str(qm.first_hit_step)] += 1

    # Average
    overall_avg = {k: sum(vals) / len(vals) if vals else 0.0 for k, vals in overall_by_k.items()}
    step_avg: Dict[int, Dict[int, float]] = {}
    for step, by_k in step_temp.items():
        step_avg[step] = {
            k: sum(vals) / len(vals) if vals else 0.0
            for k, vals in by_k.items()
        }

    return AggregateMetrics(
        num_questions=len(metrics),
        overall_by_k=overall_avg,
        step_by_k=dict(sorted(step_avg.items())),
        first_hit_distribution=dict(
            sorted(
                first_hit.items(),
                key=lambda kv: (kv[0] == "never", int(kv[0]) if kv[0].isdigit() else math.inf),
            )
        ),
        gold_doc_counts=gold_doc_counts,
    )


# Reporting --------------------------------------------------------------------

def format_percentage(value: float) -> str:
    return f"{value * 100:5.1f}%"


def render_report(model: str, aggregate: AggregateMetrics) -> str:
    lines = []
    lines.append(f"=== {model} ===")
    lines.append(f"Questions evaluated: {aggregate.num_questions}")

    lines.append("Overall Recall@K:")
    lines.append(
        "  "
        + ", ".join(
            f"K={k}: {format_percentage(score)}"
            for k, score in aggregate.overall_by_k.items()
        )
    )

    lines.append("Cumulative Recall@K by step:")
    if not aggregate.step_by_k:
        lines.append("  (no retrieval evidence)")
    else:
        for step, by_k in aggregate.step_by_k.items():
            lines.append(
                f"  Step {step}: "
                + ", ".join(
                    f"K={k}: {format_percentage(score)}"
                    for k, score in by_k.items()
                )
            )

    total = max(sum(aggregate.first_hit_distribution.values()), 1)
    lines.append("First supporting passage step (using K=max):")
    for label, count in aggregate.first_hit_distribution.items():
        pct = count / total
        lines.append(f"  {label}: {format_percentage(pct)} ({count})")

    lines.append("Gold passage count distribution:")
    for num_docs, count in sorted(aggregate.gold_doc_counts.items()):
        pct = count / aggregate.num_questions if aggregate.num_questions else 0.0
        lines.append(f"  {num_docs} docs: {format_percentage(pct)} ({count})")

    return "\n".join(lines)


def aggregate_to_dict(aggregate: AggregateMetrics) -> dict:
    return {
        "num_questions": aggregate.num_questions,
        "overall_by_k": aggregate.overall_by_k,
        "step_by_k": {str(step): vals for step, vals in aggregate.step_by_k.items()},
        "first_hit_distribution": aggregate.first_hit_distribution,
        "gold_doc_counts": {str(k): v for k, v in aggregate.gold_doc_counts.items()},
    }


# Main -------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Recall@K metrics for Iterative RAG runs.")
    parser.add_argument("--responses-dir", type=Path, default=RESPONSES_DIR, help="Directory containing model response jsonl files.")
    parser.add_argument("--qa-path", type=Path, default=QA_PATH, help="Path to chemrxiv QA dataset.")
    parser.add_argument("--doc-root", type=Path, default=DOC_ROOT, help="Root directory of passage texts.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of questions per model (for debugging).")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to dump metrics as JSON.")
    args = parser.parse_args()

    doc_text_map, exact_lookup = build_passage_lookup(args.doc_root)
    gold_index = map_gold_passages(args.qa_path, doc_text_map, exact_lookup)

    results_dump: Dict[str, dict] = {}

    for jsonl_path in sorted(args.responses_dir.glob("*.jsonl")):
        model_name = jsonl_path.stem
        records = load_jsonl(jsonl_path)

        question_metrics: List[QuestionMetrics] = []
        for record in records:
            question = extract_question(record)
            if not question:
                continue
            gold_docs = gold_index.get(question)
            if not gold_docs:
                continue
            evidences = parse_evidence(record)
            qm = evaluate_record(question, gold_docs, evidences)
            question_metrics.append(qm)

            if args.limit and len(question_metrics) >= args.limit:
                break

        aggregate = aggregate_metrics(question_metrics)
        report = render_report(model_name, aggregate)
        print(report)
        print()

        results_dump[model_name] = aggregate_to_dict(aggregate)

    if args.json_out:
        args.json_out.write_text(json.dumps(results_dump, indent=2))


if __name__ == "__main__":
    main()
