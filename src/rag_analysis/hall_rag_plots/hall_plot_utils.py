"""Utility helpers for hallucination analysis plots."""
from __future__ import annotations

import json
import glob
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Allow importing quality utilities for accuracy and query flag aggregation
CURRENT_DIR = Path(__file__).resolve().parent
QUAL_DIR = CURRENT_DIR.parent / "qual_rag_plots"
if str(QUAL_DIR) not in sys.path:
    sys.path.append(str(QUAL_DIR))

try:
    from quality_plot_utils import (  # type: ignore
        load_model_accuracy,
        match_accuracy,
        load_quality_data,
    )
except ImportError:
    load_model_accuracy = match_accuracy = load_quality_data = None  # type: ignore


@dataclass
class HallucinationRecord:
    system: str
    question: str
    number_of_hops: int
    composition_failure: bool
    unsupported_claims_count: int
    sufficiency_score_est: Optional[float]
    hop_coverage_est: Optional[float]
    is_miscalibrated: Optional[bool]
    direction: str
    coverage_gap: Optional[bool]
    carry_drop: Optional[bool]
    late_hit: Optional[bool]
    poor_query_quality: Optional[bool]
    accuracy: Optional[bool]


PALETTE = {
    "overconfident_finalize": "#c44e52",
    "underconfident_continue": "#55a868",
    "ok": "#4c72b0",
    "unknown": "#8172b2",
}


def _base_dirs() -> Tuple[Path, Path]:
    root = Path(__file__).resolve().parents[2]
    output_dir = root / "rag_analysis" / "output"
    csv_dir = root / "results" / "new_results_csv"
    return output_dir, csv_dir


def _system_slug(path: Path, suffix: str) -> str:
    name = path.name
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    if name.startswith("responses_"):
        name = name[len("responses_"):]
    return name


def _list_hall_files(output_dir: Path) -> List[Path]:
    pattern = str(output_dir / "*hallucination_judgment.jsonl")
    return [Path(p) for p in glob.glob(pattern)]


def _load_coverage_features(output_dir: Path) -> Dict[Tuple[str, str], Dict[str, bool]]:
    features: Dict[Tuple[str, str], Dict[str, bool]] = {}
    for path in output_dir.glob("*coverage_gap_judgments.jsonl"):
        system = _system_slug(path, "_coverage_gap_judgments.jsonl")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                question = entry.get("question")
                if question is None:
                    continue
                parsed = entry.get("parsed_judgment")
                if not isinstance(parsed, dict):
                    try:
                        parsed = json.loads(entry.get("raw_output", ""))
                    except json.JSONDecodeError:
                        parsed = {}
                coverage_gap = bool(parsed.get("retrieval_coverage_gap", {}).get("has_gap")) if isinstance(parsed, dict) else None
                carry_drop = bool(parsed.get("anchor_carry_drop", {}).get("any_carry_drop")) if isinstance(parsed, dict) else None
                late_hit = bool(parsed.get("late_hit_per_hop", {}).get("any_late_hit")) if isinstance(parsed, dict) else None
                features[(system, question)] = {
                    "coverage_gap": coverage_gap,
                    "carry_drop": carry_drop,
                    "late_hit": late_hit,
                    "is_correct": entry.get("is_correct") if entry.get("is_correct") is not None else None,
                }
    return features


def _load_quality_flags() -> Dict[Tuple[str, str], bool]:
    if load_quality_data is None:
        return {}
    step_df, _run_df, _output_dir, _csv_dir = load_quality_data()
    if step_df.empty:
        return {}
    flags = step_df[[
        "system",
        "question",
        "query_vague",
        "query_over_broad",
        "query_compound",
        "query_off_topic",
    ]].copy()
    for col in ["query_vague", "query_over_broad", "query_compound", "query_off_topic"]:
        flags[col] = flags[col].fillna(False).astype(bool)
    flags["poor_query"] = flags[["query_vague", "query_over_broad", "query_compound", "query_off_topic"]].any(axis=1)
    grouped = flags.groupby(["system", "question"])["poor_query"].any()
    return grouped.to_dict()


def _ensure_parsed(entry: Dict[str, object]) -> Dict[str, object]:
    parsed = entry.get("parsed_judgment")
    if isinstance(parsed, dict):
        return parsed
    raw = entry.get("raw_output")
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    raise ValueError("Unable to parse hallucination judgment")


def load_hallucination_data() -> Tuple[pd.DataFrame, Path, Path]:
    output_dir, csv_dir = _base_dirs()
    hallucination_files = _list_hall_files(output_dir)
    coverage_features = _load_coverage_features(output_dir)
    quality_flags = _load_quality_flags()
    accuracy_table = load_model_accuracy(csv_dir) if load_model_accuracy else {}

    records: List[HallucinationRecord] = []

    for hall_file in hallucination_files:
        system = _system_slug(hall_file, "_hallucination_judgment.jsonl")
        with hall_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    parsed = _ensure_parsed(entry)
                except ValueError:
                    continue

                question = entry.get("question")
                num_hops = entry.get("number_of_hops")
                if question is None or num_hops is None:
                    continue
                num_hops = int(num_hops)

                comp = parsed.get("composition_and_faithfulness", {}) if isinstance(parsed, dict) else {}
                mis = parsed.get("confidence_miscalibration", {}) if isinstance(parsed, dict) else {}

                unsupported = comp.get("unsupported_claims", [])
                unsupported_count = 0
                if isinstance(unsupported, list):
                    for item in unsupported:
                        if isinstance(item, dict) and not item.get("is_supported", True):
                            unsupported_count += 1

                direction = mis.get("direction") or "unknown"

                coverage_info = coverage_features.get((system, question), {})
                poor_query = quality_flags.get((system, question)) if quality_flags else None

                accuracy_val = coverage_info.get("is_correct") if coverage_info else None
                if accuracy_val is not None:
                    accuracy_val = bool(accuracy_val)
                elif accuracy_table and match_accuracy:
                    model_acc = match_accuracy(system, accuracy_table)
                    accuracy_val = None if model_acc is None else model_acc / 100.0

                records.append(
                    HallucinationRecord(
                        system=system,
                        question=question,
                        number_of_hops=num_hops,
                        composition_failure=bool(comp.get("composition_failure")),
                        unsupported_claims_count=unsupported_count,
                        sufficiency_score_est=_maybe_float(comp.get("sufficiency_score_est")),
                        hop_coverage_est=_maybe_float(mis.get("hop_coverage_est")),
                        is_miscalibrated=_maybe_bool(mis.get("is_miscalibrated")),
                        direction=str(direction),
                        coverage_gap=_maybe_bool(coverage_info.get("coverage_gap")),
                        carry_drop=_maybe_bool(coverage_info.get("carry_drop")),
                        late_hit=_maybe_bool(coverage_info.get("late_hit")),
                        poor_query_quality=poor_query,
                        accuracy=_maybe_bool(entry.get("is_correct")) if entry.get("is_correct") is not None else accuracy_val,
                    )
                )

    df = pd.DataFrame([r.__dict__ for r in records])
    return df, output_dir, csv_dir


def _maybe_bool(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    return None


def _maybe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (ValueError, TypeError):
        return None
