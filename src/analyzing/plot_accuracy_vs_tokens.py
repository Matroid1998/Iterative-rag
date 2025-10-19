#!/usr/bin/env python3
"""Plot scatter plot of model accuracy vs average output tokens in Iterative RAG setup."""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from config import (
    get_responses_dir,
    PLOTS_DIR,
    ITERATIVE_MODEL_ENTRIES,
    MODEL_COLOR_MAP,
    DEFAULT_MODEL_COLOR,
)


def get_model_accuracy(csv_path: Path) -> Dict[str, float]:
    """Read accuracy values from reverify_accuracies.csv file."""
    accuracy_map = {}
    
    # Mapping between JSONL filenames and display names
    filename_to_display = {}
    for jsonl_filename, display_name in ITERATIVE_MODEL_ENTRIES:
        filename_to_display[jsonl_filename] = display_name
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                folder = row.get('folder', '')
                file_name = row.get('file_name', '')
                
                # Only use Iterative-RAG folder entries
                if folder != 'Iterative-RAG':
                    continue
                
                # Get the display name for this file
                display_name = filename_to_display.get(file_name)
                if display_name is None:
                    continue
                
                # Get accuracy (stored as decimal in CSV, convert to percentage)
                accuracy_decimal = float(row.get('accuracy', 0))
                accuracy_percent = accuracy_decimal * 100
                
                accuracy_map[display_name] = accuracy_percent
                
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
    
    return accuracy_map


def calculate_average_output_tokens(path: Path) -> float:
    """Calculate average output tokens for a model from JSONL file."""
    total_tokens = 0
    count = 0
    
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"Skipping {path.name}:{line_number} (invalid JSON: {error})")
                continue

            # Get output tokens from root level
            raw_value = record.get("output_tokens")
            
            try:
                output_tokens = float(raw_value)
                total_tokens += output_tokens
                count += 1
            except (TypeError, ValueError):
                continue
    
    return total_tokens / count if count > 0 else 0.0


def main() -> None:
    responses_dir = get_responses_dir()
    csv_path = responses_dir.parent / "results" / "reverify_accuracies.csv"
    
    if not csv_path.exists():
        raise RuntimeError(f"CSV file not found: {csv_path}")
    
    # Get accuracy values from reverify_accuracies.csv
    accuracy_map = get_model_accuracy(csv_path)
    
    if not accuracy_map:
        raise RuntimeError("No accuracy data found in CSV file")
    
    # Collect data points
    data_points: List[Tuple[str, float, float]] = []
    
    for filename, display_name in ITERATIVE_MODEL_ENTRIES:
        path = responses_dir / filename
        
        if not path.exists():
            print(f"Warning: File not found: {path}")
            continue
        
        # Get accuracy for this model
        accuracy = accuracy_map.get(display_name)
        if accuracy is None:
            print(f"Warning: No accuracy data for {display_name}")
            continue
        
        # Calculate average output tokens
        avg_output_tokens = calculate_average_output_tokens(path)
        
        if avg_output_tokens > 0:
            data_points.append((display_name, accuracy, avg_output_tokens))
            print(f"{display_name}: Accuracy = {accuracy:.2f}%, Avg Output Tokens = {avg_output_tokens:.2f}")
    
    if not data_points:
        raise RuntimeError("No data points collected")
    
    # Create both versions of the plot
    plot_accuracy_vs_tokens(data_points, use_log_scale=True)
    plot_accuracy_vs_tokens(data_points, use_log_scale=False)


def plot_accuracy_vs_tokens(data_points: List[Tuple[str, float, float]], use_log_scale: bool = True) -> None:
    """Create scatter plot of accuracy vs average output tokens."""
    
    # Separate data
    labels = [label for label, _, _ in data_points]
    accuracies = [acc for _, acc, _ in data_points]
    avg_tokens = [tokens for _, _, tokens in data_points]
    colors = [MODEL_COLOR_MAP.get(label, DEFAULT_MODEL_COLOR) for label in labels]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create scatter plot
    scatter = ax.scatter(accuracies, avg_tokens, s=200, c=colors, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Add labels for each point
    for i, label in enumerate(labels):
        ax.annotate(
            label,
            (accuracies[i], avg_tokens[i]),
            xytext=(8, 8),
            textcoords='offset points',
            fontsize=9,
            ha='left',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8)
        )
    
    # Set labels and title
    ax.set_xlabel("Accuracy in Iterative RAG (%)", fontsize=12, fontweight='bold')
    
    if use_log_scale:
        ax.set_ylabel("Average Output Tokens (log scale)", fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        title_suffix = " (Log Scale)"
        filename_suffix = ""
    else:
        ax.set_ylabel("Average Output Tokens", fontsize=12, fontweight='bold')
        title_suffix = " (Linear Scale)"
        filename_suffix = "_linear"
    
    ax.set_title(f"Model Accuracy vs Average Output Tokens{title_suffix}", fontsize=14, fontweight='bold')
    
    # Add grid
    if use_log_scale:
        ax.grid(True, linestyle='--', alpha=0.4, which='both')
    else:
        ax.grid(True, linestyle='--', alpha=0.4)
    
    # Set x-axis range with some padding
    x_min, x_max = min(accuracies), max(accuracies)
    x_range = x_max - x_min
    ax.set_xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)
    
    # Add correlation info
    if len(accuracies) > 1:
        # Calculate correlation coefficient
        if use_log_scale:
            correlation = np.corrcoef(accuracies, np.log10(avg_tokens))[0, 1]
            corr_label = 'Pearson r (with log tokens)'
        else:
            correlation = np.corrcoef(accuracies, avg_tokens)[0, 1]
            corr_label = 'Pearson r'
        
        # Add text box with correlation
        textstr = f'{corr_label}: {correlation:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    fig.tight_layout()
    
    # Save figure
    output_path = PLOTS_DIR / f"accuracy_vs_output_tokens{filename_suffix}.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure to {output_path}")
    
    # Print summary statistics (only once for log scale version)
    if use_log_scale:
        print(f"\nSummary Statistics:")
        print(f"  Accuracy range: {min(accuracies):.2f}% - {max(accuracies):.2f}%")
        print(f"  Avg tokens range: {min(avg_tokens):.2f} - {max(avg_tokens):.2f}")
        print(f"  Number of models: {len(data_points)}")


if __name__ == "__main__":
    main()
