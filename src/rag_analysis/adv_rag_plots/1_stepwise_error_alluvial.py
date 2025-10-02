"""
Plot 1: Step-by-step error evolution alluvial based on query quality flags.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adv_plot_utils import load_all_data

STEPS = [1, 2, 3]
FLAG_PRIORITY = [
    ("query_off_topic", "Off-topic", "#c44e52"),
    ("query_over_broad", "Over-broad", "#dd8452"),
    ("query_compound", "Compound", "#55a868"),
    ("query_vague", "Vague", "#4c72b0"),
]
DEFAULT = ("Clean", "#8172b2")
LABEL_COLORS = {label: color for _, label, color in FLAG_PRIORITY}
LABEL_COLORS[DEFAULT[0]] = DEFAULT[1]
STATE_COLORS = LABEL_COLORS


def label_for_step(row: Dict[str, bool]) -> Tuple[str, str]:
    for col, label, color in FLAG_PRIORITY:
        if row.get(col, False):
            return label, color
    return DEFAULT


def build_distributions(step_df):
    records = {}
    for step in STEPS:
        subset = step_df[step_df["step"] == step]
        labels = subset.apply(lambda r: label_for_step(r), axis=1)
        counts: Dict[str, int] = {}
        for label, _color in labels:
            counts[label] = counts.get(label, 0) + 1
        records[step] = counts
    return records


def build_bands(step_df) -> Tuple[List[Tuple[int, str, str, str]], List[Tuple[Tuple[int, str], Tuple[int, str], float]]]:
    nodes: List[Tuple[int, str, str, str]] = []  # (stage idx, state label, display, color)
    flows: List[Tuple[Tuple[int, str], Tuple[int, str], float]] = []

    snapshots = {}
    for step in STEPS:
        subset = step_df[step_df["step"] == step]
        snapshot = subset.apply(lambda r: label_for_step(r)[0], axis=1)
        snapshots[step] = snapshot
        counts = snapshot.value_counts()
        for label in counts.index:
            color = LABEL_COLORS.get(label, DEFAULT[1])
            nodes.append((step, label, label, color))

    for idx in range(len(STEPS) - 1):
        s1 = STEPS[idx]
        s2 = STEPS[idx + 1]
        if s1 not in snapshots or s2 not in snapshots:
            continue
        df = pd.DataFrame({"s1": snapshots[s1], "s2": snapshots[s2]})
        counts = df.groupby(["s1", "s2"]).size()
        total = counts.sum()
        for (l1, l2), value in counts.items():
            flows.append(((s1, l1), (s2, l2), value / total if total else 0))
    return nodes, flows


def draw_alluvial(nodes, flows, output_path: Path):
    stages = {step for step, *_ in nodes}
    stage_positions: Dict[Tuple[int, str], Tuple[float, float, float]] = {}
    for step in sorted(stages):
        labels = [n for n in nodes if n[0] == step]
        y = 0.0
        for _step, label, _display, color in labels:
            outgoing = sum(flow[2] for flow in flows if flow[0] == (step, label))
            incoming = sum(flow[2] for flow in flows if flow[1] == (step, label))
            height = max(outgoing, incoming)
            height = max(height, 0.001)
            stage_positions[(step, label)] = (step, y, y + height, color)
            y += height
    fig, ax = plt.subplots(figsize=(12, 6))
    node_width = 0.4
    for (step, label), (xpos, y0, y1, color) in stage_positions.items():
        rect = plt.Rectangle((step, y0), node_width, y1 - y0, color=color, alpha=0.6)
        ax.add_patch(rect)
        ax.text(step + node_width / 2, (y0 + y1) / 2, label, ha='center', va='center', fontsize=10)

    offsets = {key: pos[1] for key, pos in stage_positions.items()}
    for (left_stage, left_label), (right_stage, right_label), value in flows:
        if value <= 0:
            continue
        left = stage_positions.get((left_stage, left_label))
        right = stage_positions.get((right_stage, right_label))
        if not left or not right:
            continue
        y0_left = offsets[(left_stage, left_label)]
        y1_left = y0_left + value
        offsets[(left_stage, left_label)] = y1_left

        y0_right = offsets[(right_stage, right_label)]
        y1_right = y0_right + value
        offsets[(right_stage, right_label)] = y1_right

        xs = np.linspace(left_stage + node_width, right_stage, 30)
        top = np.linspace(y1_left, y1_right, 30)
        bottom = np.linspace(y0_left, y0_right, 30)
        color = STATE_COLORS.get(left_label, '#999999')
        ax.fill(np.concatenate([xs, xs[::-1]]), np.concatenate([top, bottom[::-1]]), color=color, alpha=0.3, edgecolor='none')

    ax.set_xlim(min(STEPS), max(STEPS) + 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Step-by-step query flag evolution (first 3 steps)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()


def main():
    _run_df, _coverage_df, _coverage_step_df, quality_step_df, _hall_df, _late_df = load_all_data()
    if quality_step_df.empty:
        print("Quality step data missing.")
        return
    subset = quality_step_df.copy()
    subset = subset[subset["step"].isin(STEPS)]
    if subset.empty:
        print("No steps within requested range.")
        return
    nodes, flows = build_bands(subset)
    plot_dir = Path(__file__).resolve().parent
    output_path = plot_dir / "stepwise_error_alluvial.png"
    draw_alluvial(nodes, flows, output_path)


if __name__ == "__main__":
    main()
