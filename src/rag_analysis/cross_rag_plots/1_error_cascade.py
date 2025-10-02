"""
Plot 1: Error Cascade Analysis
Custom Sankey-style Diagram linking coverage gaps → query issues → hallucination issues for a selected system.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from cross_plot_utils import load_cross_data

STATE_ORDER = {
    "coverage": [(False, "No gap", "#55a868"), (True, "Gap", "#c44e52")],
    "query": [(False, "Query OK", "#4c72b0"), (True, "Query issue", "#dd8452")],
    "halluc": [(False, "Halluc OK", "#8172b2"), (True, "Hallucination", "#c44e52")],
}
STATE_COLOR = {(stage, state): color for stage, items in STATE_ORDER.items() for state, _label, color in items}


@dataclass
class Band:
    left_state: Tuple[str, bool]
    right_state: Tuple[str, bool]
    value: float


def compute_bands(data, total: int) -> List[Band]:
    bands: List[Band] = []
    # Stage transitions
    for (cov_state, q_state), val in data['cov_query'].items():
        if val <= 0:
            continue
        bands.append(Band(("coverage", cov_state), ("query", q_state), val / total))
    for (q_state, hall_state), val in data['query_hall'].items():
        if val <= 0:
            continue
        bands.append(Band(("query", q_state), ("halluc", hall_state), val / total))
    return bands


def build_positions(stage_counts: Dict[str, Dict[bool, float]]) -> Dict[Tuple[str, bool], Tuple[float, float, float]]:
    positions = {}
    for idx, stage in enumerate(["coverage", "query", "halluc"]):
        counts = stage_counts[stage]
        total = sum(counts.values())
        y = 0.0
        for state, label, _color in STATE_ORDER[stage]:
            height = 0.0
            if total > 0:
                height = counts.get(state, 0.0) / total
            positions[(stage, state)] = (idx, y, y + height)
            y += height
    return positions


def draw_alluvial(ax, positions, bands):
    node_width = 0.5
    for stage_idx, stage in enumerate(["coverage", "query", "halluc"]):
        for state, label, color in STATE_ORDER[stage]:
            x0 = stage_idx
            x1 = stage_idx + node_width
            y0, y1 = positions[(stage, state)][1:]
            height = y1 - y0
            if height <= 0:
                continue
            rect = plt.Rectangle((x0, y0), node_width, height, color=color, alpha=0.6)
            ax.add_patch(rect)
            ax.text(x0 + node_width / 2, y0 + height / 2, label, ha='center', va='center', fontsize=11, color='black')

    # Prepare offsets for flows
    offsets_left: Dict[Tuple[str, bool], float] = {key: positions[key][1] for key in positions}
    offsets_right: Dict[Tuple[str, bool], float] = {key: positions[key][1] for key in positions}

    for band in bands:
        (stage_l, state_l) = band.left_state
        (stage_r, state_r) = band.right_state
        width = band.value
        if width <= 0:
            continue
        left_x = positions[(stage_l, state_l)][0] + node_width
        right_x = positions[(stage_r, state_r)][0]

        y0_left = offsets_left[(stage_l, state_l)]
        y1_left = y0_left + width
        offsets_left[(stage_l, state_l)] = y1_left

        y0_right = offsets_right[(stage_r, state_r)]
        y1_right = y0_right + width
        offsets_right[(stage_r, state_r)] = y1_right

        xs = np.linspace(left_x, right_x, 30)
        top = np.linspace(y1_left, y1_right, 30)
        bottom = np.linspace(y0_left, y0_right, 30)
        path_x = np.concatenate([xs, xs[::-1]])
        path_y = np.concatenate([top, bottom[::-1]])

        color = STATE_COLOR.get((stage_l, state_l), '#999999')
        ax.fill(path_x, path_y, color=color, alpha=0.35, edgecolor='none')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Error cascade Sankey diagram")
    parser.add_argument("--system", type=str, default=None, help="System identifier (default: first available)")
    parser.add_argument("--output", type=Path, default=None, help="Optional override output path")
    return parser.parse_args()


def main():
    args = parse_args()
    run_df, _cov_df, _cov_step_df, _qual_step_df, _hall_df, _late_df = load_cross_data()
    plot_dir = Path(__file__).resolve().parent

    systems = sorted(run_df["system"].dropna().unique())
    if not systems:
        print("No systems found for cascade plot.")
        return

    system = args.system or systems[0]
    data = run_df[run_df["system"] == system].copy()
    if data.empty:
        print(f"No records for system {system}.")
        return

    data["has_gap"] = data.get("has_gap", False).fillna(False).astype(bool)
    data["any_query_issue"] = data.get("any_query_issue", False).fillna(False).astype(bool)
    data["hallucination_issue"] = data.get("hallucination_issue", False).fillna(False).astype(bool)

    total = len(data)
    if total == 0:
        print("No runs to visualise.")
        return

    stage_counts = {
        "coverage": data["has_gap"].value_counts().reindex([False, True], fill_value=0).to_dict(),
        "query": data["any_query_issue"].value_counts().reindex([False, True], fill_value=0).to_dict(),
        "halluc": data["hallucination_issue"].value_counts().reindex([False, True], fill_value=0).to_dict(),
    }

    cov_query = data.groupby(["has_gap", "any_query_issue"]).size().to_dict()
    query_hall = data.groupby(["any_query_issue", "hallucination_issue"]).size().to_dict()

    flow_data = {
        'cov_query': {(bool(k1), bool(k2)): v for (k1, k2), v in cov_query.items()},
        'query_hall': {(bool(k1), bool(k2)): v for (k1, k2), v in query_hall.items()},
    }

    bands = compute_bands(flow_data, total)
    positions = build_positions(stage_counts)

    fig, ax = plt.subplots(figsize=(12, 6))
    draw_alluvial(ax, positions, bands)
    ax.axis('off')
    ax.set_title(f"Error cascade for {system} (n={total})", fontsize=18, fontweight='bold')

    output_path = args.output or (plot_dir / f"error_cascade_{system}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
