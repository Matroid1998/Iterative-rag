"""Coverage Gap Plot 6: Anchor carry-drop temporal pattern (all models)."""
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
MARGIN_BOTTOM = 80
LINE_COLOR = "#c44e52"


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
    raise ValueError("Unable to parse coverage judgment")


def gather_rates(path: Path) -> Tuple[Dict[int, List[bool]], int]:
    step_map: Dict[int, List[bool]] = {}
    max_step = 0
    for entry in iter_jsonl(path):
        try:
            parsed = ensure_parsed(entry)
        except ValueError:
            continue
        for item in parsed.get("anchor_carry_drop", {}).get("per_step", []):
            step = item.get("step")
            if step is None:
                continue
            step = int(step)
            max_step = max(max_step, step)
            carry_drop = bool(item.get("carry_drop"))
            step_map.setdefault(step, []).append(carry_drop)
    return step_map, max_step


def percentage(values: Sequence[bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if v) / len(values)


def svg_line(step_map: Dict[int, List[bool]], max_step: int, output_path: Path) -> None:
    if max_step == 0:
        output_path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='600' height='400'><text x='50%' y='50%' text-anchor='middle'>No data</text></svg>", encoding="utf-8")
        return
    chart_width = SVG_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    chart_height = SVG_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    steps = list(range(1, max_step + 1))
    spacing = chart_width / max(len(steps) - 1, 1)

    svg_parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{SVG_WIDTH}' height='{SVG_HEIGHT}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{SVG_WIDTH/2}' y='{MARGIN_TOP/2}' text-anchor='middle' font-size='20' font-family='Arial'>Anchor carry-drop temporal pattern</text>",
        f"<text x='{SVG_WIDTH/2}' y='{SVG_HEIGHT-20}' text-anchor='middle' font-size='14' font-family='Arial'>Planner step</text>",
        f"<text x='{MARGIN_LEFT/2}' y='{SVG_HEIGHT/2}' text-anchor='middle' transform='rotate(-90 {MARGIN_LEFT/2},{SVG_HEIGHT/2})' font-size='14' font-family='Arial'>Carry-drop rate</text>",
        f"<line x1='{MARGIN_LEFT}' y1='{SVG_HEIGHT-MARGIN_BOTTOM}' x2='{SVG_WIDTH-MARGIN_RIGHT}' y2='{SVG_HEIGHT-MARGIN_BOTTOM}' stroke='black' />",
        f"<line x1='{MARGIN_LEFT}' y1='{MARGIN_TOP}' x2='{MARGIN_LEFT}' y2='{SVG_HEIGHT-MARGIN_BOTTOM}' stroke='black' />",
    ]
    for tick in range(0, 6):
        value = tick / 5
        y = SVG_HEIGHT - MARGIN_BOTTOM - value * chart_height
        svg_parts.append(f"<line x1='{MARGIN_LEFT-5}' y1='{y:.2f}' x2='{MARGIN_LEFT}' y2='{y:.2f}' stroke='black' />")
        svg_parts.append(f"<text x='{MARGIN_LEFT-8}' y='{y+4:.2f}' text-anchor='end' font-size='12' font-family='Arial'>{value:.1f}</text>")

    points = []
    for idx, step in enumerate(steps):
        rate = percentage(step_map.get(step, []))
        x = MARGIN_LEFT + idx * spacing
        y = SVG_HEIGHT - MARGIN_BOTTOM - rate * chart_height
        points.append((x, y, rate))
        svg_parts.append(f"<circle cx='{x:.2f}' cy='{y:.2f}' r='5' fill='{LINE_COLOR}' opacity='0.85' />")
        svg_parts.append(f"<text x='{x:.2f}' y='{SVG_HEIGHT-MARGIN_BOTTOM+25}' text-anchor='middle' font-size='12' font-family='Arial'>{step}</text>")
        svg_parts.append(f"<text x='{x:.2f}' y='{y-8:.2f}' text-anchor='middle' font-size='11' font-family='Arial'>{rate:.2f}</text>")
    if len(points) >= 2:
        path = " ".join(f"{x:.2f},{y:.2f}" for x, y, _ in points)
        svg_parts.append(f"<polyline points='{path}' fill='none' stroke='{LINE_COLOR}' stroke-width='2' />")

    svg_parts.append("</svg>")
    output_path.write_text("\n".join(svg_parts), encoding="utf-8")


def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Anchor carry-drop temporal plot")
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("src/rag_analysis/output/responses_bedrock_mistral.mistral-large-2402-v1:0_reverified_coverage_gap_judgments.jsonl"),
        help="Coverage gap JSONL",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("src/rag_analysis/rag_plots/coverage_anchor_carry_temporal.svg"),
        help="Output SVG",
    )
    return ap


def main() -> None:
    args = build_cli().parse_args()
    step_map, max_step = gather_rates(args.input)
    svg_line(step_map, max_step, args.output)


if __name__ == "__main__":
    main()
