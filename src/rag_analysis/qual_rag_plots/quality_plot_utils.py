"""Utility helpers for Quality (Query Audit) analysis plots."""
from __future__ import annotations

import json
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class QualityRecord:
    system: str
    question: str
    number_of_hops: int
    is_correct: Optional[bool]
    distractor_latch: Optional[bool]
    has_fusion: bool
    any_partial_contradiction: bool


@dataclass
class QualityStep:
    system: str
    question: str
    number_of_hops: int
    step: int
    predicted_hop: Optional[int]
    is_next_logical_hop: Optional[bool]
    fusion_or_skip: Optional[bool]
    partial_contradiction: Optional[bool]
    query_vague: Optional[bool]
    query_over_broad: Optional[bool]
    query_compound: Optional[bool]
    query_off_topic: Optional[bool]
    query_anchored: Optional[bool]
    specificity_score: Optional[float]
    on_topic_score: Optional[float]


def _base_dirs() -> Tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[2]
    output_dir = root / "rag_analysis" / "output"
    csv_dir = root / "results" / "new_results_csv"
    plot_dir = Path(__file__).resolve().parent
    return output_dir, csv_dir, plot_dir


def list_quality_files(output_dir: Path) -> List[Path]:
    pattern = str(output_dir / "*quality_judement.jsonl")
    return [Path(p) for p in glob.glob(pattern)]


def list_coverage_files(output_dir: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in output_dir.glob("*coverage_gap_judgments.jsonl"):
        prefix = path.name.replace("_coverage_gap_judgments.jsonl", "")
        mapping[prefix] = path
    return mapping


def derive_system_name(quality_path: Path) -> str:
    prefix = quality_path.name.replace("_quality_judement.jsonl", "")
    if prefix.startswith("responses_"):
        prefix = prefix[len("responses_"):]
    return prefix


def load_coverage_accuracy(output_dir: Path) -> Dict[Tuple[str, str], bool]:
    records: Dict[Tuple[str, str], bool] = {}
    for coverage_path in output_dir.glob("*coverage_gap_judgments.jsonl"):
        system = derive_system_name(coverage_path.with_name(coverage_path.name.replace("_coverage_gap_judgments.jsonl", "_quality_judement.jsonl")))
        with coverage_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                question = data.get("question")
                is_correct = data.get("is_correct")
                if question is None or is_correct is None:
                    continue
                records[(system, question)] = bool(is_correct)
    return records


def ensure_parsed(entry: Dict[str, object]) -> Dict[str, object]:
    parsed = entry.get("parsed_judgment")
    if isinstance(parsed, dict):
        return parsed
    raw = entry.get("raw_output")
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    raise ValueError("Unable to parse quality judgment")


def load_quality_data() -> Tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    output_dir, csv_dir, plot_dir = _base_dirs()
    quality_files = list_quality_files(output_dir)
    coverage_accuracy = load_coverage_accuracy(output_dir)

    run_records: List[QualityRecord] = []
    step_records: List[QualityStep] = []

    for quality_file in quality_files:
        system = derive_system_name(quality_file)
        with quality_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    parsed = ensure_parsed(entry)
                except ValueError:
                    continue

                question = entry.get("question")
                num_hops = entry.get("number_of_hops")
                if question is None or num_hops is None:
                    continue
                num_hops = int(num_hops)

                per_step = parsed.get("per_step", []) if isinstance(parsed, dict) else []
                run_level = parsed.get("run_level", {}) if isinstance(parsed, dict) else {}

                has_fusion = False
                has_partial_contradiction = False

                for step in per_step:
                    step_index = step.get("step")
                    if not step_index:
                        continue
                    step_index = int(step_index)
                    fusion = bool(step.get("fusion_or_skip"))
                    partial = bool(step.get("partial_contradiction_with_prev"))
                    has_fusion = has_fusion or fusion
                    has_partial_contradiction = has_partial_contradiction or partial

                    q = step.get("query_quality", {}) if isinstance(step, dict) else {}
                    step_records.append(
                        QualityStep(
                            system=system,
                            question=question,
                            number_of_hops=num_hops,
                            step=step_index,
                            predicted_hop=_maybe_int(step.get("predicted_hop")),
                            is_next_logical_hop=_maybe_bool(step.get("is_next_logical_hop")),
                            fusion_or_skip=fusion,
                            partial_contradiction=partial,
                            query_vague=_maybe_bool(q.get("vague")),
                            query_over_broad=_maybe_bool(q.get("over_broad")),
                            query_compound=_maybe_bool(q.get("compound")),
                            query_off_topic=_maybe_bool(q.get("off_topic")),
                            query_anchored=_maybe_bool(q.get("anchored")),
                            specificity_score=_maybe_float(q.get("specificity_score")),
                            on_topic_score=_maybe_float(q.get("on_topic_score")),
                        )
                    )

                run_records.append(
                    QualityRecord(
                        system=system,
                        question=question,
                        number_of_hops=num_hops,
                        is_correct=coverage_accuracy.get((system, question)),
                        distractor_latch=_maybe_bool(run_level.get("distractor_latch")) if isinstance(run_level, dict) else None,
                        has_fusion=has_fusion,
                        any_partial_contradiction=has_partial_contradiction,
                    )
                )

    step_df = pd.DataFrame([s.__dict__ for s in step_records])
    run_df = pd.DataFrame([r.__dict__ for r in run_records])
    return step_df, run_df, output_dir, csv_dir


def _maybe_int(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (ValueError, TypeError):
        return None


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


def load_model_accuracy(csv_dir: Path) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    for csv_path in csv_dir.glob("results_*.csv"):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "Accuracy (%)" not in df.columns:
            continue
        accuracy = df.iloc[0]["Accuracy (%)"]
        identifier = csv_path.stem.replace("results_", "")
        mapping[identifier] = float(accuracy)
    return mapping


def match_accuracy(system: str, accuracy_table: Dict[str, float]) -> Optional[float]:
    slug = system.replace(":", "_")
    for key, value in accuracy_table.items():
        if key in slug:
            return value
    return None
