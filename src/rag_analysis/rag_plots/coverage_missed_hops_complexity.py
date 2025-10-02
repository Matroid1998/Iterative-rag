"""Coverage Gap Plot 5: Missed hop patterns by question complexity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

SVG_WIDTH = 900
SVG_HEIGHT = 500
MARGIN_LEFT = 80
MARGIN_RIGHT = 40
MARGIN_TOP = 60
MARGIN_BOTTOM = 80
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
    raise ValueError("Unable to parse coverage judgment")


def compute_stack(path: Path) -> Tuple[List[int], Dict[int, Dict[int, float]]]:
    totals: Dict[int, int] = {}
    missed_counts: Dict[int, Dict[int, int]] = {}
    for entry in iter_jsonl(path):
        num_hops = entry.get("number_of_hops")
        if not isinstance(num_hops, int):
            continue
        try:
            parsed = ensure_parsed(entry)
        except ValueError:
            continue
        missed = parsed.get("retrieval_coverage_gap", {}).get("missed_hops", [])
        totals[num_hops] = totals.get(num_hops, 0) + 1
        hop_bucket = missed_counts.setdefault(num_hops, {})
        if isinstance(missed, list):
            for hop_idx in missed:
                try:
                    hop_idx = int(hop_idx)
                except (TypeError, ValueError):
                    continue
                hop_bucket[hop_idx] = hop_bucket.get(hop_idx, 0) + 1
    proportions: Dict[int, Dict[int, float]] = {}
    for num_hops, hop_map in missed_counts.items():
        total = totals.get(num_hops, 0)
        if total == 0:
            continue
        proportions[num_hops] = {hop_idx: count / total for hop_idx, count in hop_map.items()}
    complexities = sorted(totals)
    return complexities, proportions


def svg_stacked_bars(complexities: List[int], data: Dict[int, Dict[int, float]], output_path: Path) -> None:
    if not complexities:
        output_path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='600' height='400'><text x='50%' y='50%' text-anchor='middle'>No data</text></svg>", encoding="utf-8")
        return
    chart_width = SVG_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    chart_height = SVG_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    bar_spacing = chart_width / max(len(complexities), 1)
    bar_width = bar_spacing * 0.6

    svg_parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{SVG_WIDTH}' height='{SVG_HEIGHT}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{SVG_WIDTH/2}' y='{MARGIN_TOP/2}' text-anchor='middle' font-size='20' font-family='Arial'>Missed hop patterns by question complexity</text>",
        f"<text x='{SVG_WIDTH/2}' y='{SVG_HEIGHT-20}' text-anchor='middle' font-size='14' font-family='Arial'>Number of hops in question</text>",
        f"<text x='{MARGIN_LEFT/2}' y='{SVG_HEIGHT/2}' text-anchor='middle' transform='rotate(-90 {MARGIN_LEFT/2},{SVG_HEIGHT/2})' font-size='14' font-family='Arial'>Proportion of runs</text>",
        f"<line x1='{MARGIN_LEFT}' y1='{SVG_HEIGHT-MARGIN_BOTTOM}' x2='{SVG_WIDTH-MARGIN_RIGHT}' y2='{SVG_HEIGHT-MARGIN_BOTTOM}' stroke='black' />",
        f"<line x1='{MARGIN_LEFT}' y1='{MARGIN_TOP}' x2='{MARGIN_LEFT}' y2='{SVG_HEIGHT-MARGIN_BOTTOM}' stroke='black' />",
    ]
    for tick in range(0, 6):
        value = tick / 5
        y = SVG_HEIGHT - MARGIN_BOTTOM - value * chart_height
        svg_parts.append(f"<line x1='{MARGIN_LEFT-5}' y1='{y:.2f}' x2='{MARGIN_LEFT}' y2='{y:.2f}' stroke='black' />")
        svg_parts.append(f"<text x='{MARGIN_LEFT-8}' y='{y+4:.2f}' text-anchor='end' font-size='12' font-family='Arial'>{value:.1f}</text>")

    legend_items = set()
    for idx, complexity in enumerate(complexities):
        hop_map = data.get(complexity, {})
        total_height = 0.0
        x = MARGIN_LEFT + idx * bar_spacing + (bar_spacing - bar_width) / 2
        for hop_idx, proportion in sorted(hop_map.items())[::-1]:
            bar_height = proportion * chart_height
            y = SVG_HEIGHT - MARGIN_BOTTOM - total_height - bar_height
            color = PALETTE[(hop_idx - 1) % len(PALETTE)]
            svg_parts.append(
                f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_width:.2f}' height='{bar_height:.2f}' fill='{color}' opacity='0.85' />"
            )
            svg_parts.append(
                f"<text x='{x+bar_width/2:.2f}' y='{y + bar_height/2:.2f}' text-anchor='middle' font-size='11' font-family='Arial' fill='white'>{proportion:.2f}</text>"
            )
            total_height += bar_height
            legend_items.add(hop_idx)
        svg_parts.append(
            f"<text x='{x + bar_width/2:.2f}' y='{SVG_HEIGHT-MARGIN_BOTTOM+25}' text-anchor='middle' font-size='12' font-family='Arial'>{complexity}</text>"
        )

    if legend_items:
        legend_x = SVG_WIDTH - MARGIN_RIGHT - 150
        legend_y = MARGIN_TOP
        legend_height = 18 * len(legend_items) + 10
        svg_parts.append(f"<rect x='{legend_x-10}' y='{legend_y-20}' width='150' height='{legend_height}' fill='#f5f5f5' stroke='#cccccc' />")
        for i, hop_idx in enumerate(sorted(legend_items)):
            color = PALETTE[(hop_idx - 1) % len(PALETTE)]
            y_pos = legend_y + i * 18
            svg_parts.append(f"<rect x='{legend_x}' y='{y_pos}' width='12' height='12' fill='{color}' stroke='black' stroke-width='0.4' />")
            svg_parts.append(f"<text x='{legend_x + 18}' y='{y_pos + 10}' font-size='12' font-family='Arial'>Hop {hop_idx}</text>")

    svg_parts.append("</svg>")
    output_path.write_text("\n".join(svg_parts), encoding="utf-8")


def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Missed hop patterns plot")
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("src/rag_analysis/output/responses_bedrock_mistral.mistral-large-2402-v1:0_reverified_coverage_gap_judgments.jsonl"),
        help="Coverage gap JSONL",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("src/rag_analysis/rag_plots/coverage_missed_hops_complexity.svg"),
        help="Output SVG",
    )
    return ap


def main() -> None:
    args = build_cli().parse_args()
    complexities, data = compute_stack(args.input)
    svg_stacked_bars(complexities, data, args.output)


if __name__ == "__main__":
    main()
