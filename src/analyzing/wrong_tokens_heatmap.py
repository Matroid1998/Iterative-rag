#!/usr/bin/env python3
"""Generate heatmaps showing wrong-answer counts by hop count and token usage."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Human-friendly model names.
MODEL_NAME_MAP: Dict[str, str] = {
    "responses_bedrock_mistral.mistral-large-2402-v1:0": "Mistral Large",
    "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0": "Claude 3.7 Sonnet",
    "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning": "Claude 3.7 Sonnet",
    "responses_bedrock_us.deepseek.r1-v1:0": "DeepSeek R1",
    "responses_bedrock_us.deepseek.r1-v1:0-reasoning": "DeepSeek R1",
    "responses_openai_gpt-4o_reverified": "GPT-4o (reverified)",
    "responses_openai_gpt-5": "GPT-5",
}

REASONING_MODEL_KEYS = {
    "responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning",
    "responses_bedrock_us.deepseek.r1-v1:0-reasoning",
    "responses_openai_gpt-5",
}


def determine_layout(n_items: int) -> tuple[int, int]:
    if n_items == 0:
        return 1, 1
    if n_items == 1:
        return 1, 1
    if n_items == 2:
        return 1, 2
    if n_items == 3:
        return 1, 3
    if n_items == 4:
        return 2, 2
    if n_items <= 6:
        return 2, 3
    return 3, 3


def flatten_axes(axes) -> List:
    if hasattr(axes, "flat"):
        return list(axes.flat)
    if isinstance(axes, (list, tuple)):
        flat: List = []
        for entry in axes:
            flat.extend(flatten_axes(entry))
        return flat
    return [axes]


def sortable_hop_key(hop: str) -> Tuple[int, float, str]:
    try:
        numeric = float(hop)
    except (TypeError, ValueError):
        return (1, float("inf"), hop)
    return (0, numeric, hop)


def collect_wrong_records(path: Path, token_field: str) -> List[Tuple[str, float]]:
    """Return (hop, tokens) pairs for wrong answers in the given JSONL file."""
    records: List[Tuple[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"Skipping {path.name}:{line_number} (invalid JSON: {error})")
                continue

            if entry.get("is_correct") is True:
                continue

            hops = entry.get("number_of_hops")
            token_value = entry.get(token_field)
            if hops is None or token_value is None:
                continue
            try:
                tokens = float(token_value)
            except (TypeError, ValueError):
                continue
            records.append((str(hops), tokens))
    return records


def compute_token_bins(values: Sequence[float]) -> np.ndarray:
    """Return bin edges using box-plot quantiles (min, Q1, median, Q3, max)."""
    if len(values) == 0:
        return np.array([0.0, 1.0])
    arr = np.asarray(values, dtype=float)
    quantiles = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    edges = np.quantile(arr, quantiles)
    edges = np.unique(edges)
    if edges.size < 2:
        single = edges[0]
        return np.array([single - 0.5, single + 0.5])
    return edges


def format_bin_labels(edges: np.ndarray, quantile_labels: Sequence[str]) -> List[str]:
    labels: List[str] = []
    for idx, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        descriptor = quantile_labels[idx] if idx < len(quantile_labels) else ""
        if np.isclose(start, end):
            labels.append(f"{start:.0f}")
        else:
            labels.append(f"{start:.0f}-{end:.0f}")
    return labels


def build_heatmap_matrix(records: List[Tuple[str, float]], edges: np.ndarray, quantile_labels: Sequence[str]) -> Tuple[List[str], List[str], np.ndarray]:
    if not records:
        return [], [], np.zeros((0, 0), dtype=int)

    hop_labels = sorted({hop for hop, _ in records}, key=sortable_hop_key)
    bin_labels = format_bin_labels(edges, quantile_labels)

    matrix = np.zeros((len(hop_labels), len(bin_labels)), dtype=int)

    for hop, tokens in records:
        row = hop_labels.index(hop)
        col = int(np.searchsorted(edges, tokens, side="right") - 1)
        col = max(0, min(col, matrix.shape[1] - 1))
        matrix[row, col] += 1

    return hop_labels, bin_labels, matrix


def prepare_heatmap_data(paths: Sequence[Path], token_field: str) -> List[Tuple[str, List[str], List[str], np.ndarray]]:
    data = []
    all_values: List[float] = []
    per_model_records: List[Tuple[str, List[Tuple[str, float]]]] = []

    for path in paths:
        records = collect_wrong_records(path, token_field)
        per_model_records.append((path.stem, records))
        all_values.extend(token for _, token in records)

    edges = compute_token_bins(all_values)
    bin_count = max(0, len(edges) - 1)
    quantile_names: list[str] = []
    quantile_labels = []
    for idx in range(bin_count):
        if idx < len(quantile_names):
            quantile_labels.append(quantile_names[idx])
        else:
            quantile_labels.append(f"bin {idx + 1}")
    if edges.size:
        print(f"{token_field} quantile edges (min, Q1, median, Q3, max where distinct): {edges.tolist()}")

    for model_key, records in per_model_records:
        hop_labels, bin_labels, matrix = build_heatmap_matrix(records, edges, quantile_labels)
        display_name = MODEL_NAME_MAP.get(model_key, model_key)
        data.append((display_name, hop_labels, bin_labels, matrix))
    return data


def plot_heatmaps(data: List[Tuple[str, List[str], List[str], np.ndarray]], output_path: Path, title: str) -> None:
    if not data:
        raise RuntimeError("No data available to plot heatmaps.")

    n_rows, n_cols = determine_layout(len(data))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 5.5 * n_rows), sharex=False, sharey=False)
    axes_list = flatten_axes(axes)

    vmax = max((matrix.max() for _, _, _, matrix in data), default=1)
    if vmax <= 0:
        vmax = 1

    for ax, (display_name, hop_labels, bin_labels, matrix) in zip(axes_list, data):
        if matrix.size == 0:
            ax.text(0.5, 0.5, "No wrong answers", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            im = ax.imshow(matrix, aspect="auto", cmap="Reds", vmin=0, vmax=vmax)
            ax.set_xticks(range(len(bin_labels)))
            ax.set_xticklabels(bin_labels, rotation=45, ha="right")
            ax.set_yticks(range(len(hop_labels)))
            ax.set_yticklabels(hop_labels)
        ax.set_title(display_name, fontsize=10)
        ax.set_xlabel("Token bins")
        ax.set_ylabel("Number of hops")

    for ax in axes_list[len(data):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)

    # Add a single colorbar aligned with all axes that contain data.
    valid_axes = [ax for ax, item in zip(axes_list, data) if item[3].size > 0]
    if valid_axes:
        im = None
        for axis in axes_list:
            if axis.images:
                im = axis.images[0]
        if im is not None:
            cbar = fig.colorbar(im, ax=valid_axes, fraction=0.03, pad=0.12)
            cbar.set_label("Wrong answers")

    fig.subplots_adjust(top=0.9, hspace=0.35, wspace=0.25)
    fig.savefig(output_path, dpi=300)
    print(f"Saved figure to {output_path}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    responses_dir = repo_root / "src" / "responses"
    plots_dir = repo_root / "src" / "plots"

    jsonl_files = sorted(responses_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise RuntimeError(f"No JSONL files found in {responses_dir}")

    reasoning_paths = [path for path in jsonl_files if path.stem in REASONING_MODEL_KEYS]
    reasoning_data = prepare_heatmap_data(reasoning_paths, "reasoning_tokens")
    if reasoning_data:
        plot_heatmaps(reasoning_data, plots_dir / "wrong_reasoning_tokens_heatmap.png", "Wrong answers vs reasoning tokens and hops")

    output_data = prepare_heatmap_data(jsonl_files, "output_tokens")
    if output_data:
        plot_heatmaps(output_data, plots_dir / "wrong_output_tokens_heatmap.png", "Wrong answers vs output tokens and hops")


if __name__ == "__main__":
    main()
