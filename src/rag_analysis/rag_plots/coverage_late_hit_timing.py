"""Coverage Gap Plot 1: Late Hit Timing Distribution (violin-style SVG)."""
from __future__ import annotations

import argparse
import json
from math import exp
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

SVG_WIDTH = 900
SVG_HEIGHT = 500
MARGIN_LEFT = 100
MARGIN_RIGHT = 60
MARGIN_TOP = 70
MARGIN_BOTTOM = 70
PALETTE = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974", "#64b5cd"]


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


def gather_deltas(path: Path) -> Dict[int, List[float]]:
    by_hop: Dict[int, List[float]] = {}
    for entry in iter_jsonl(path):
        try:
            parsed = ensure_parsed(entry)
        except ValueError:
            continue
        for hop in parsed.get("late_hit_per_hop", {}).get("per_hop", []):
            hop_index = hop.get("hop_index")
            first_hit = hop.get("first_hit_step")
            if hop_index is None or first_hit is None:
                continue
            hop_index = int(hop_index)
            delta = float(first_hit) - float(hop_index)
            by_hop.setdefault(hop_index, []).append(delta)
    return by_hop


def kernel_density(values: Sequence[float], points: Sequence[float], bandwidth: float) -> List[float]:
    densities: List[float] = []
    if not values:
        return [0.0 for _ in points]
    n = len(values)
    constant = 1.0 / (bandwidth * (2.0 * 3.14159265359) ** 0.5)
    for x in points:
        total = 0.0
        for v in values:
            u = (x - v) / bandwidth
            total += exp(-0.5 * u * u)
        densities.append(constant * total / n)
    return densities


def build_violin_path(center_x: float, y_coords: Sequence[float], densities: Sequence[float], scale: float) -> str:
    if not y_coords:
        return ""
    left_points = []
    right_points = []
    for y, density in zip(y_coords, densities):
        offset = density * scale
        left_points.append((center_x - offset, y))
        right_points.append((center_x + offset, y))
    path_points = left_points + list(reversed(right_points))
    if not path_points:
        return ""
    parts = [f"M {path_points[0][0]:.2f} {path_points[0][1]:.2f}"]
    for x, y in path_points[1:]:
        parts.append(f"L {x:.2f} {y:.2f}")
    parts.append("Z")
    return " ".join(parts)


def create_svg(by_hop: Dict[int, List[float]], output_path: Path) -> None:
    if not by_hop:
        output_path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='600' height='400'><text x='50%' y='50%' text-anchor='middle'>No data</text></svg>", encoding="utf-8")
        return
    hop_indices = sorted(by_hop)
    all_values = [delta for values in by_hop.values() for delta in values]
    min_delta = min(all_values)
    max_delta = max(all_values)
    if min_delta == max_delta:
        min_delta -= 1
        max_delta += 1
    chart_width = SVG_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    chart_height = SVG_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    x_spacing = chart_width / max(len(hop_indices), 1)

    y_points = 60
    y_values = [min_delta + (max_delta - min_delta) * i / (y_points - 1) for i in range(y_points)]
    bandwidth = max(0.2, (max_delta - min_delta) / 20)
    max_density = 0.0
    densities_per_hop: Dict[int, List[float]] = {}
    for hop in hop_indices:
        densities = kernel_density(by_hop[hop], y_values, bandwidth)
        densities_per_hop[hop] = densities
        max_density = max(max_density, max(densities) if densities else 0.0)
    scale = (x_spacing * 0.45) / max_density if max_density > 0 else 0.0

    svg_lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{SVG_WIDTH}' height='{SVG_HEIGHT}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{SVG_WIDTH/2}' y='{MARGIN_TOP/2}' text-anchor='middle' font-size='20' font-family='Arial'>Late Hit Timing Distribution</text>",
        f"<text x='{MARGIN_LEFT/2}' y='{SVG_HEIGHT/2}' text-anchor='middle' font-size='14' font-family='Arial' transform='rotate(-90 {MARGIN_LEFT/2},{SVG_HEIGHT/2})'>(first_hit - hop_index)</text>",
    ]
    # Axis lines
    svg_lines.append(f"<line x1='{MARGIN_LEFT}' y1='{SVG_HEIGHT-MARGIN_BOTTOM}' x2='{SVG_WIDTH-MARGIN_RIGHT}' y2='{SVG_HEIGHT-MARGIN_BOTTOM}' stroke='black' />")
    svg_lines.append(f"<line x1='{MARGIN_LEFT}' y1='{MARGIN_TOP}' x2='{MARGIN_LEFT}' y2='{SVG_HEIGHT-MARGIN_BOTTOM}' stroke='black' />")
    # Y labels
    for tick in range(int(min_delta), int(max_delta) + 1):
        y = SVG_HEIGHT - MARGIN_BOTTOM - (tick - min_delta) / (max_delta - min_delta) * chart_height
        svg_lines.append(f"<line x1='{MARGIN_LEFT-5}' y1='{y:.2f}' x2='{MARGIN_LEFT}' y2='{y:.2f}' stroke='black' />")
        svg_lines.append(f"<text x='{MARGIN_LEFT-8}' y='{y+4:.2f}' text-anchor='end' font-size='12' font-family='Arial'>{tick}</text>")

    for idx, hop in enumerate(hop_indices):
        center_x = MARGIN_LEFT + x_spacing * idx + x_spacing / 2
        svg_lines.append(f"<text x='{center_x:.2f}' y='{SVG_HEIGHT-MARGIN_BOTTOM+30}' text-anchor='middle' font-size='12' font-family='Arial'>Hop {hop}</text>")
        path = build_violin_path(center_x, [SVG_HEIGHT - MARGIN_BOTTOM - (y - min_delta) / (max_delta - min_delta) * chart_height for y in y_values], densities_per_hop[hop], scale)
        color = PALETTE[idx % len(PALETTE)]
        if path:
            svg_lines.append(f"<path d='{path}' fill='{color}' opacity='0.6' stroke='{color}' stroke-width='1' />")

    svg_lines.append("</svg>")
    output_path.write_text("\n".join(svg_lines), encoding="utf-8")


def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Late hit timing violin-style plot")
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("src/rag_analysis/output/responses_bedrock_mistral.mistral-large-2402-v1:0_reverified_coverage_gap_judgments.jsonl"),
        help="Coverage gap judgments JSONL",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("src/rag_analysis/rag_plots/coverage_late_hit_timing.svg"),
        help="Output SVG path",
    )
    return ap


def main() -> None:
    args = build_cli().parse_args()
    by_hop = gather_deltas(args.input)
    create_svg(by_hop, args.output)


if __name__ == "__main__":
    main()
