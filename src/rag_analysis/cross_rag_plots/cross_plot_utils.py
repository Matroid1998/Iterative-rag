"""Utilities for cross-system RAG analysis plots."""
from __future__ import annotations

import json
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Reuse helpers from quality and hallucination utilities
QUAL_DIR = Path(__file__).resolve().parents[1] / "qual_rag_plots"
HALL_DIR = Path(__file__).resolve().parents[1] / "hall_rag_plots"

import sys
for extra in (QUAL_DIR, HALL_DIR):
    if str(extra) not in sys.path:
        sys.path.append(str(extra))

from quality_plot_utils import load_quality_data  # type: ignore
from hall_plot_utils import load_hallucination_data  # type: ignore


@dataclass
class CoverageRecord:
    system: str
    question: str
    number_of_hops: Optional[int]
    has_gap: Optional[bool]
    any_late_hit: Optional[bool]
    any_carry_drop: Optional[bool]
    is_correct: Optional[bool]


@dataclass
class CoverageStepRecord:
    system: str
    question: str
    step: int
    carry_drop: Optional[bool]


@dataclass
class LateHitRecord:
    system: str
    question: str
    hop_index: int
    first_hit_step: int


def _base_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _output_dir() -> Path:
    return _base_dir() / "rag_analysis" / "output"


def _slug_from_filename(path: Path, suffix: str) -> str:
    name = path.name
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    if name.startswith("responses_"):
        name = name[len("responses_"):]
    return name


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
    return {}


def load_coverage_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir = _output_dir()
    coverage_files = glob.glob(str(output_dir / "*coverage_gap_judgments.jsonl"))

    run_records: List[CoverageRecord] = []
    step_records: List[CoverageStepRecord] = []
    late_records: List[LateHitRecord] = []

    for file_path in coverage_files:
        path = Path(file_path)
        system = _slug_from_filename(path, "_coverage_gap_judgments.jsonl")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                parsed = _ensure_parsed(entry)
                question = entry.get("question")
                if question is None:
                    continue
                num_hops = entry.get("number_of_hops")
                has_gap = _maybe_bool(parsed.get("retrieval_coverage_gap", {}).get("has_gap")) if isinstance(parsed, dict) else None
                any_late_hit = _maybe_bool(parsed.get("late_hit_per_hop", {}).get("any_late_hit")) if isinstance(parsed, dict) else None
                carry = parsed.get("anchor_carry_drop", {}) if isinstance(parsed, dict) else {}
                any_carry_drop = _maybe_bool(carry.get("any_carry_drop"))
                is_correct = entry.get("is_correct")
                run_records.append(
                    CoverageRecord(
                        system=system,
                        question=question,
                        number_of_hops=int(num_hops) if isinstance(num_hops, int) else None,
                        has_gap=has_gap,
                        any_late_hit=any_late_hit,
                        any_carry_drop=any_carry_drop,
                        is_correct=_maybe_bool(is_correct),
                    )
                )
                per_step = carry.get("per_step") if isinstance(carry, dict) else []
                if isinstance(per_step, list):
                    for item in per_step:
                        step = item.get("step")
                        if step is None:
                            continue
                        step_records.append(
                            CoverageStepRecord(
                                system=system,
                                question=question,
                                step=int(step),
                                carry_drop=_maybe_bool(item.get("carry_drop")),
                            )
                        )
                late_hops = parsed.get("late_hit_per_hop", {}).get("per_hop") if isinstance(parsed, dict) else []
                if isinstance(late_hops, list):
                    for hop_entry in late_hops:
                        hop_idx = hop_entry.get("hop_index")
                        first_hit = hop_entry.get("first_hit_step")
                        if hop_idx is None or first_hit is None:
                            continue
                        late_records.append(
                            LateHitRecord(
                                system=system,
                                question=question,
                                hop_index=int(hop_idx),
                                first_hit_step=int(first_hit),
                            )
                        )
    coverage_df = pd.DataFrame([r.__dict__ for r in run_records])
    coverage_step_df = pd.DataFrame([s.__dict__ for s in step_records])
    late_hit_df = pd.DataFrame([l.__dict__ for l in late_records])
    return coverage_df, coverage_step_df, late_hit_df


def _maybe_bool(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    return None


def load_cross_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coverage_df, coverage_step_df, late_hit_df = load_coverage_data()
    quality_step_df, quality_run_df, _output_dir, _csv_dir = load_quality_data()
    hall_df, _output_dir, _csv_dir = load_hallucination_data()

    # Aggregate query issues from quality step data
    if not quality_step_df.empty:
        flag_cols = ["query_vague", "query_over_broad", "query_compound", "query_off_topic"]
        for col in flag_cols:
            quality_step_df[col] = quality_step_df[col].fillna(False).astype(bool)
        query_flags = (
            quality_step_df.groupby(["system", "question"])[flag_cols]
            .any()
            .reset_index()
        )
        query_flags["any_query_issue"] = query_flags[flag_cols].any(axis=1)
    else:
        query_flags = pd.DataFrame(columns=["system", "question", "any_query_issue"])

    if not quality_run_df.empty:
        quality_run = quality_run_df[["system", "question", "number_of_hops", "has_fusion", "any_partial_contradiction"]].copy()
    else:
        quality_run = pd.DataFrame(columns=["system", "question", "number_of_hops", "has_fusion", "any_partial_contradiction"])

    # Merge quality run with flags
    quality_features = quality_run.merge(query_flags[["system", "question", "any_query_issue"]], on=["system", "question"], how="left")

    run_df = coverage_df.merge(quality_features, on=["system", "question"], how="outer", suffixes=("_cov", "_qual"))
    run_df = run_df.merge(hall_df, on=["system", "question"], how="outer", suffixes=("", "_hall"))

    # Reconcile hop counts
    hop_cols = [col for col in run_df.columns if "number_of_hops" in col]
    hop_series = None
    for col in hop_cols:
        series = run_df[col]
        hop_series = series if hop_series is None else hop_series.fillna(series)
    if hop_series is not None:
        run_df["number_of_hops"] = hop_series
    for col in hop_cols:
        if col != "number_of_hops":
            run_df.drop(columns=col, inplace=True)

    if "is_correct_cov" in run_df.columns:
        run_df["is_correct"] = run_df["is_correct_cov"]
        run_df.drop(columns=["is_correct_cov"], inplace=True)
    run_df["is_correct"] = run_df.get("is_correct", np.nan)

    run_df["any_query_issue"] = run_df.get("any_query_issue", False).fillna(False)

    run_df["hallucination_issue"] = (
        run_df.get("composition_failure", False).fillna(False).astype(bool)
        | run_df.get("is_miscalibrated", False).fillna(False).astype(bool)
    )

    bool_cols = [
        "has_gap",
        "any_late_hit",
        "any_carry_drop",
        "any_query_issue",
        "composition_failure",
        "is_miscalibrated",
        "hallucination_issue",
    ]
    for col in bool_cols:
        if col in run_df.columns:
            run_df[col] = run_df[col].fillna(False).astype(bool)

    run_df["direction"] = run_df.get("direction", "unknown").fillna("unknown")
    if "unsupported_claims_count" in run_df.columns:
        run_df["unsupported_claims_count"] = run_df["unsupported_claims_count"].fillna(0).astype(int)

    return run_df, coverage_df, coverage_step_df, quality_step_df, hall_df, late_hit_df
