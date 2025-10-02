"""Generate lightweight SVG plots for Quality (Query Audit) judgments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


SVG_WIDTH = 800
SVG_HEIGHT = 480
MARGIN_LEFT = 80
MARGIN_RIGHT = 40
MARGIN_TOP = 60
MARGIN_BOTTOM = 60
BAR_COLOR = "#4c72b0"
ALT_BAR_COLOR = "#55a868"
LINE_COLOR = "#c44e52"
MAX_PLANNER_STEP = 5


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def ensure_parsed(entry: Dict[str, Any]) -> Dict[str, Any]:
    parsed = entry.get("parsed_judgment")
    if isinstance(parsed, dict):
        return parsed
    raw = entry.get("raw_output")
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    raise ValueError("Unable to parse judgment for record")


def load_quality_steps(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in iter_jsonl(path):
        try:
            parsed = ensure_parsed(entry)
        except ValueError:
            continue
        question = entry.get("question")
        model = entry.get("model")
        num_hops = entry.get("number_of_hops")
        for step in parsed.get("per_step", []):
            step_index = step.get("step")
            if not step_index:
                continue
            step_index = int(step_index)
            if step_index > MAX_PLANNER_STEP:
                continue
            q = step.get("query_quality", {})
            rows.append(
                {
                    "question": question,
                    "model": model,
                    "number_of_hops": num_hops,
                    "step": step_index,
                    "is_next_logical_hop": bool(step.get("is_next_logical_hop")),
                    "fusion_or_skip": bool(step.get("fusion_or_skip")),
                    "partial_contradiction": bool(step.get("partial_contradiction_with_prev")),
                    "query_vague": bool(q.get("vague")),
                    "query_over_broad": bool(q.get("over_broad")),
                    "query_compound": bool(q.get("compound")),
                    "query_off_topic": bool(q.get("off_topic")),
                    "query_anchored": bool(q.get("anchored")),
                    "specificity_score": _safe_float(q.get("specificity_score")),
                    "on_topic_score": _safe_float(q.get("on_topic_score")),
                }
            )
    return rows


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (ValueError, TypeError):
        return None


def percentage(values: Sequence[bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if v) / len(values)


def average(values: Sequence[float | None]) -> float:
    clean = [v for v in values if isinstance(v, (int, float))]
    if not clean:
        return 0.0
    return sum(clean) / len(clean)


def aggregate_step_metric(rows: List[Dict[str, Any]], key: str) -> List[Tuple[int, float]]:
    by_step: Dict[int, List[Any]] = {}
    for row in rows:
        step = row.get("step")
        if step is None:
            continue
        by_step.setdefault(int(step), []).append(row.get(key))
    result: List[Tuple[int, float]] = []
    for step in range(1, MAX_PLANNER_STEP + 1):
        values = by_step.get(step, [])
        if key in {"specificity_score", "on_topic_score"}:
            result.append((step, average(values) if values else 0.0))
        else:
            result.append((step, percentage(values) if values else 0.0))
    return result


def aggregate_flag_rates(rows: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
    flags = [
        ("vague", "query_vague"),
        ("over_broad", "query_over_broad"),
        ("compound", "query_compound"),
        ("off_topic", "query_off_topic"),
        ("anchored", "query_anchored"),
    ]
    result = []
    for label, key in flags:
        values = [row[key] for row in rows if key in row]
        result.append((label, percentage(values)))
    return result


def aggregate_score_means(rows: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
    metrics = ["specificity_score", "on_topic_score"]
    return [(metric, average([row.get(metric) for row in rows])) for metric in metrics]


def svg_bar_chart(
    data: Sequence[Tuple[str, float]],
    title: str,
    y_label: str,
    output_path: Path,
    max_value: float = 1.0,
    bar_color: str = BAR_COLOR,
) -> None:
    if not data:
        return
    labels = [item[0] for item in data]
    values = [min(max(item[1], 0.0), max_value) for item in data]
    n = len(values)
    chart_width = SVG_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    chart_height = SVG_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    bar_spacing = chart_width / max(n, 1)
    bar_width = bar_spacing * 0.6
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}">',
        f'<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{SVG_WIDTH/2}" y="{MARGIN_TOP/2}" text-anchor="middle" font-size="20" font-family="Arial">{title}</text>',
        f'<line x1="{MARGIN_LEFT}" y1="{SVG_HEIGHT-MARGIN_BOTTOM}" x2="{SVG_WIDTH-MARGIN_RIGHT}" y2="{SVG_HEIGHT-MARGIN_BOTTOM}" stroke="black"/>',
        f'<line x1="{MARGIN_LEFT}" y1="{MARGIN_TOP}" x2="{MARGIN_LEFT}" y2="{SVG_HEIGHT-MARGIN_BOTTOM}" stroke="black"/>',
        f'<text x="{MARGIN_LEFT/2}" y="{(SVG_HEIGHT)/2}" text-anchor="middle" transform="rotate(-90 {MARGIN_LEFT/2},{(SVG_HEIGHT)/2})" font-size="14" font-family="Arial">{y_label}</text>',
    ]
    for idx, (label, value) in enumerate(zip(labels, values)):
        x = MARGIN_LEFT + idx * bar_spacing + (bar_spacing - bar_width) / 2
        bar_height = value / max_value * chart_height
        y = SVG_HEIGHT - MARGIN_BOTTOM - bar_height
        svg_parts.append(
            f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="{bar_color}" opacity="0.85"/>'
        )
        svg_parts.append(
            f'<text x="{x + bar_width/2}" y="{SVG_HEIGHT-MARGIN_BOTTOM + 20}" text-anchor="middle" font-size="12" font-family="Arial">{label}</text>'
        )
        svg_parts.append(
            f'<text x="{x + bar_width/2}" y="{y - 5}" text-anchor="middle" font-size="11" font-family="Arial">{value:.2f}</text>'
        )
    svg_parts.append('</svg>')
    output_path.write_text("\n".join(svg_parts), encoding="utf-8")


def svg_line_chart(
    data: Sequence[Tuple[int, float]],
    title: str,
    y_label: str,
    output_path: Path,
    max_value: float = 1.0,
    line_color: str = LINE_COLOR,
) -> None:
    if not data:
        return
    steps = [item[0] for item in data]
    values = [min(max(item[1], 0.0), max_value) for item in data]
    n = len(values)
    chart_width = SVG_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    chart_height = SVG_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    if n == 1:
        spacing = chart_width
    else:
        spacing = chart_width / (n - 1)
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}">',
        f'<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{SVG_WIDTH/2}" y="{MARGIN_TOP/2}" text-anchor="middle" font-size="20" font-family="Arial">{title}</text>',
        f'<line x1="{MARGIN_LEFT}" y1="{SVG_HEIGHT-MARGIN_BOTTOM}" x2="{SVG_WIDTH-MARGIN_RIGHT}" y2="{SVG_HEIGHT-MARGIN_BOTTOM}" stroke="black"/>',
        f'<line x1="{MARGIN_LEFT}" y1="{MARGIN_TOP}" x2="{MARGIN_LEFT}" y2="{SVG_HEIGHT-MARGIN_BOTTOM}" stroke="black"/>',
        f'<text x="{MARGIN_LEFT/2}" y="{(SVG_HEIGHT)/2}" text-anchor="middle" transform="rotate(-90 {MARGIN_LEFT/2},{(SVG_HEIGHT)/2})" font-size="14" font-family="Arial">{y_label}</text>',
    ]
    points: List[str] = []
    for idx, (step, value) in enumerate(zip(steps, values)):
        x = MARGIN_LEFT + idx * spacing
        bar_height = value / max_value * chart_height
        y = SVG_HEIGHT - MARGIN_BOTTOM - bar_height
        points.append(f"{x},{y}")
        svg_parts.append(
            f'<circle cx="{x}" cy="{y}" r="5" fill="{line_color}" opacity="0.9"/>'
        )
        svg_parts.append(
            f'<text x="{x}" y="{SVG_HEIGHT-MARGIN_BOTTOM + 20}" text-anchor="middle" font-size="12" font-family="Arial">{step}</text>'
        )
        svg_parts.append(
            f'<text x="{x}" y="{y - 8}" text-anchor="middle" font-size="11" font-family="Arial">{value:.2f}</text>'
        )
    svg_parts.append(
        f'<polyline points="{" ".join(points)}" fill="none" stroke="{line_color}" stroke-width="2"/>'
    )
    svg_parts.append('</svg>')
    output_path.write_text("\n".join(svg_parts), encoding="utf-8")


def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate Quality (Query Audit) SVG plots")
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("src/rag_analysis/output/responses_openai_gpt-5_reverified_quality_judement.jsonl"),
        help="Path to quality judgment JSONL",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/rag_analysis/rag_plots"),
        help="Directory to store generated plots",
    )
    return ap


def main() -> None:
    args = build_cli().parse_args()
    rows = load_quality_steps(args.input)
    if not rows:
        raise SystemExit("No rows loaded from input")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    alignment = aggregate_step_metric(rows, "is_next_logical_hop")
    svg_bar_chart(
        [(str(step), value) for step, value in alignment],
        "Step alignment with oracle hops",
        "Proportion of steps",
        args.output_dir / "quality_step_alignment.svg",
    )

    flag_rates = aggregate_flag_rates(rows)
    svg_bar_chart(
        flag_rates,
        "Query quality diagnostics",
        "Proportion of steps",
        args.output_dir / "quality_query_flags.svg",
        bar_color=ALT_BAR_COLOR,
    )

    fusion_rates = aggregate_step_metric(rows, "fusion_or_skip")
    svg_line_chart(
        fusion_rates,
        "Fusion/skip rate by planner step",
        "Proportion of steps",
        args.output_dir / "quality_fusion_rates.svg",
    )

    contradictions = aggregate_step_metric(rows, "partial_contradiction")
    svg_bar_chart(
        [(str(step), value) for step, value in contradictions],
        "Partial contradictions between steps",
        "Proportion of steps",
        args.output_dir / "quality_partial_contradictions.svg",
    )

    score_means = aggregate_score_means(rows)
    svg_bar_chart(
        score_means,
        "Average query scores",
        "Average score",
        args.output_dir / "quality_average_scores.svg",
    )


if __name__ == "__main__":
    main()
