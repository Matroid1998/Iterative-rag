"""Coverage Gap Plot 2: Model-level gap and late-hit rates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

SVG_WIDTH = 800
SVG_HEIGHT = 480
MARGIN_LEFT = 80
MARGIN_RIGHT = 40
MARGIN_TOP = 60
MARGIN_BOTTOM = 80
BAR_COLORS = ["#4c72b0", "#c44e52"]


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


def compute_rates(path: Path) -> Dict[str, Tuple[float, float]]:
    stats: Dict[str, List[Tuple[bool, bool]]] = {}
    for entry in iter_jsonl(path):
        model = entry.get("model") or "unknown"
        try:
            parsed = ensure_parsed(entry)
        except ValueError:
            continue
        has_gap = bool(parsed.get("retrieval_coverage_gap", {}).get("has_gap"))
        any_late_hit = bool(parsed.get("late_hit_per_hop", {}).get("any_late_hit"))
        stats.setdefault(model, []).append((has_gap, any_late_hit))
    rates: Dict[str, Tuple[float, float]] = {}
    for model, pairs in stats.items():
        if not pairs:
            continue
        total = len(pairs)
        gap_rate = sum(1 for g, _ in pairs if g) / total
        late_rate = sum(1 for _, l in pairs if l) / total
        rates[model] = (gap_rate, late_rate)
    return rates


def svg_grouped_bars(rates: Dict[str, Tuple[float, float]], output_path: Path) -> None:
    if not rates:
        output_path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='600' height='400'><text x='50%' y='50%' text-anchor='middle'>No data</text></svg>", encoding="utf-8")
        return
    models = sorted(rates)
    chart_width = SVG_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    chart_height = SVG_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    group_spacing = chart_width / max(len(models), 1)
    bar_width = group_spacing * 0.3

    svg_parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{SVG_WIDTH}' height='{SVG_HEIGHT}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{SVG_WIDTH/2}' y='{MARGIN_TOP/2}' text-anchor='middle' font-family='Arial' font-size='20'>Coverage gap & late-hit rates by model</text>",
        f"<text x='{MARGIN_LEFT/2}' y='{SVG_HEIGHT/2}' text-anchor='middle' transform='rotate(-90 {MARGIN_LEFT/2},{SVG_HEIGHT/2})' font-size='14' font-family='Arial'>Proportion of runs</text>",
        f"<line x1='{MARGIN_LEFT}' y1='{SVG_HEIGHT-MARGIN_BOTTOM}' x2='{SVG_WIDTH-MARGIN_RIGHT}' y2='{SVG_HEIGHT-MARGIN_BOTTOM}' stroke='black' />",
        f"<line x1='{MARGIN_LEFT}' y1='{MARGIN_TOP}' x2='{MARGIN_LEFT}' y2='{SVG_HEIGHT-MARGIN_BOTTOM}' stroke='black' />",
    ]
    for tick in range(0, 6):
        value = tick / 5
        y = SVG_HEIGHT - MARGIN_BOTTOM - value * chart_height
        svg_parts.append(f"<line x1='{MARGIN_LEFT-5}' y1='{y:.2f}' x2='{MARGIN_LEFT}' y2='{y:.2f}' stroke='black' />")
        svg_parts.append(f"<text x='{MARGIN_LEFT-8}' y='{y+4:.2f}' text-anchor='end' font-family='Arial' font-size='12'>{value:.1f}</text>")

    for idx, model in enumerate(models):
        gap_rate, late_rate = rates[model]
        group_x = MARGIN_LEFT + idx * group_spacing + group_spacing / 2
        for j, value in enumerate((gap_rate, late_rate)):
            x = group_x + (j - 0.5) * (bar_width + 10)
            bar_height = value * chart_height
            y = SVG_HEIGHT - MARGIN_BOTTOM - bar_height
            svg_parts.append(
                f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_width:.2f}' height='{bar_height:.2f}' fill='{BAR_COLORS[j]}' opacity='0.85'/>"
            )
            svg_parts.append(
                f"<text x='{x+bar_width/2:.2f}' y='{y-5:.2f}' text-anchor='middle' font-size='11' font-family='Arial'>{value:.2f}</text>"
            )
        svg_parts.append(
            f"<text x='{group_x:.2f}' y='{SVG_HEIGHT-MARGIN_BOTTOM+30}' text-anchor='middle' font-size='12' font-family='Arial'>{model}</text>"
        )
    svg_parts.append("</svg>")
    output_path.write_text("\n".join(svg_parts), encoding="utf-8")


def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Model-level coverage gap rates plot")
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("src/rag_analysis/output/responses_bedrock_mistral.mistral-large-2402-v1:0_reverified_coverage_gap_judgments.jsonl"),
        help="Coverage gap JSONL",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("src/rag_analysis/rag_plots/coverage_model_rates.svg"),
        help="Output SVG",
    )
    return ap


def main() -> None:
    args = build_cli().parse_args()
    rates = compute_rates(args.input)
    svg_grouped_bars(rates, args.output)


if __name__ == "__main__":
    main()
