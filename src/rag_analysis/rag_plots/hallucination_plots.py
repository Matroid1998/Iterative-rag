"""Generate lightweight SVG plots for Hallucination judgments."""
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
BAR_COLOR = "#c44e52"
ALT_BAR_COLOR = "#4c72b0"
SCATTER_COLORS = {
    "overconfident_finalize": "#c44e52",
    "underconfident_continue": "#55a868",
    "ok": "#4c72b0",
    "unknown": "#8172b2",
}


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
    raise ValueError("Unable to parse hallucination record")


def load_hallucination_runs(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in iter_jsonl(path):
        try:
            parsed = ensure_parsed(entry)
        except ValueError:
            continue
        caf = parsed.get("composition_and_faithfulness", {})
        mis = parsed.get("confidence_miscalibration", {})
        unsupported = caf.get("unsupported_claims", [])
        unsupported_count = 0
        if isinstance(unsupported, list):
            unsupported_count = sum(1 for item in unsupported if not item.get("is_supported"))
        rows.append(
            {
                "model": entry.get("model"),
                "composition_failure": bool(caf.get("composition_failure")),
                "sufficiency_score_est": _safe_float(caf.get("sufficiency_score_est")),
                "hop_coverage_est": _safe_float(mis.get("hop_coverage_est")),
                "is_miscalibrated": bool(mis.get("is_miscalibrated")),
                "direction": mis.get("direction") or "unknown",
                "unsupported_claims_count": unsupported_count,
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


def svg_bar_chart(
    data: Sequence[Tuple[str, float]],
    title: str,
    y_label: str,
    output_path: Path,
    max_value: float,
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
        bar_height = 0 if max_value == 0 else value / max_value * chart_height
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


def svg_histogram(
    values: Sequence[float],
    bins: int,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    max_value: float,
    bar_color: str = ALT_BAR_COLOR,
) -> None:
    if not values:
        return
    min_val = 0.0
    bin_size = (max_value - min_val) / bins
    counts = [0] * bins
    for value in values:
        if value is None:
            continue
        idx = int((value - min_val) / bin_size) if bin_size > 0 else 0
        if idx >= bins:
            idx = bins - 1
        if idx < 0:
            idx = 0
        counts[idx] += 1
    labels = [f"{(min_val + i*bin_size):.2f}" for i in range(bins)]
    max_count = max(counts) if counts else 1
    svg_bar_chart(
        list(zip(labels, [c / max_count if max_count else 0 for c in counts])),
        title,
        y_label,
        output_path,
        max_value=1.0,
        bar_color=bar_color,
    )


def svg_scatter(
    points: Sequence[Tuple[float, float, str]],
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    if not points:
        return
    chart_width = SVG_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    chart_height = SVG_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}">',
        f'<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{SVG_WIDTH/2}" y="{MARGIN_TOP/2}" text-anchor="middle" font-size="20" font-family="Arial">{title}</text>',
        f'<line x1="{MARGIN_LEFT}" y1="{SVG_HEIGHT-MARGIN_BOTTOM}" x2="{SVG_WIDTH-MARGIN_RIGHT}" y2="{SVG_HEIGHT-MARGIN_BOTTOM}" stroke="black"/>',
        f'<line x1="{MARGIN_LEFT}" y1="{MARGIN_TOP}" x2="{MARGIN_LEFT}" y2="{SVG_HEIGHT-MARGIN_BOTTOM}" stroke="black"/>',
        f'<text x="{SVG_WIDTH/2}" y="{SVG_HEIGHT-10}" text-anchor="middle" font-size="14" font-family="Arial">{x_label}</text>',
        f'<text x="{MARGIN_LEFT/2}" y="{(SVG_HEIGHT)/2}" text-anchor="middle" transform="rotate(-90 {MARGIN_LEFT/2},{(SVG_HEIGHT)/2})" font-size="14" font-family="Arial">{y_label}</text>',
    ]
    legend_items = {}
    for x_value, y_value, direction in points:
        color = SCATTER_COLORS.get(direction, "#000000")
        x = MARGIN_LEFT + x_value * chart_width
        y = SVG_HEIGHT - MARGIN_BOTTOM - y_value * chart_height
        svg_parts.append(
            f'<circle cx="{x}" cy="{y}" r="6" fill="{color}" opacity="0.75" stroke="black" stroke-width="0.5"/>'
        )
        legend_items.setdefault(direction, color)
    # Legend
    legend_x = SVG_WIDTH - MARGIN_RIGHT - 150
    legend_y = MARGIN_TOP + 20
    svg_parts.append(f'<rect x="{legend_x - 10}" y="{legend_y - 20}" width="140" height="{20*len(legend_items)+10}" fill="#f5f5f5" stroke="#cccccc"/>')
    for idx, (direction, color) in enumerate(sorted(legend_items.items())):
        y_pos = legend_y + idx * 20
        svg_parts.append(f'<rect x="{legend_x}" y="{y_pos - 10}" width="12" height="12" fill="{color}" stroke="black" stroke-width="0.4"/>'
        )
        svg_parts.append(f'<text x="{legend_x + 18}" y="{y_pos}" font-size="12" font-family="Arial">{direction}</text>')
    svg_parts.append('</svg>')
    output_path.write_text("\n".join(svg_parts), encoding="utf-8")


def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate Hallucination SVG plots")
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("src/rag_analysis/output/responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0_reverified_hallucination_judgment.jsonl"),
        help="Path to hallucination judgment JSONL",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/rag_analysis/rag_plots"),
        help="Directory to store plots",
    )
    return ap


def main() -> None:
    args = build_cli().parse_args()
    rows = load_hallucination_runs(args.input)
    if not rows:
        raise SystemExit("No hallucination rows parsed")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Composition failures per model
    by_model: Dict[str, List[bool]] = {}
    for row in rows:
        by_model.setdefault(row.get("model") or "unknown", []).append(row.get("composition_failure", False))
    comp_data = [(model, percentage(values)) for model, values in sorted(by_model.items())]
    svg_bar_chart(
        comp_data,
        "Composition failure rate by model",
        "Proportion of runs",
        args.output_dir / "hallucination_composition_failure.svg",
        max_value=1.0,
    )

    # Sufficiency histogram (normalized counts)
    sufficiency_values = [row.get("sufficiency_score_est") for row in rows if row.get("sufficiency_score_est") is not None]
    svg_histogram(
        sufficiency_values,
        bins=10,
        title="Evidence sufficiency distribution",
        x_label="Sufficiency score",
        y_label="Relative frequency",
        output_path=args.output_dir / "hallucination_sufficiency_distribution.svg",
        max_value=1.0,
    )

    # Miscalibration direction counts
    direction_counts: Dict[str, int] = {}
    for row in rows:
        direction = row.get("direction") or "unknown"
        direction_counts[direction] = direction_counts.get(direction, 0) + 1
    total = max(direction_counts.values()) if direction_counts else 1
    miscalibration_data = [(direction, count / total) for direction, count in sorted(direction_counts.items())]
    svg_bar_chart(
        miscalibration_data,
        "Confidence calibration outcomes",
        "Relative frequency",
        args.output_dir / "hallucination_miscalibration_directions.svg",
        max_value=1.0,
        bar_color=ALT_BAR_COLOR,
    )

    # Support vs coverage scatter
    scatter_points = []
    for row in rows:
        cov = row.get("hop_coverage_est")
        suf = row.get("sufficiency_score_est")
        if cov is None or suf is None:
            continue
        direction = row.get("direction") or "unknown"
        scatter_points.append((max(0.0, min(1.0, cov)), max(0.0, min(1.0, suf)), direction))
    svg_scatter(
        scatter_points,
        "Sufficiency vs hop coverage",
        "Hop coverage estimate",
        "Sufficiency score estimate",
        args.output_dir / "hallucination_support_vs_coverage.svg",
    )

    # Unsupported claims histogram
    unsupported_values = [row.get("unsupported_claims_count", 0) for row in rows]
    max_claims = max(unsupported_values) if unsupported_values else 0
    if max_claims > 0:
        counts = [(str(i), unsupported_values.count(i) / len(unsupported_values)) for i in range(max_claims + 1)]
        svg_bar_chart(
            counts,
            "Unsupported claims per run",
            "Relative frequency",
            args.output_dir / "hallucination_unsupported_claims.svg",
            max_value=1.0,
            bar_color="#8172b2",
        )


if __name__ == "__main__":
    main()
