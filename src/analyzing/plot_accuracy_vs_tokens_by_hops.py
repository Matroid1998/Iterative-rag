#!/usr/bin/env python3
"""
Generate scatter plots showing model accuracy vs average output tokens,
separated by question hop count (single-hop vs multi-hop).

Single-hop: 1 hop questions only
Multi-hop: 2, 3, and 4 hop questions
"""

import json
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# Import configuration
from config import (
    get_responses_dir,
    PLOTS_DIR,
    ITERATIVE_MODEL_ENTRIES,
    MODEL_COLOR_MAP,
    DEFAULT_MODEL_COLOR,
)

# Paths
RESPONSES_DIR = get_responses_dir()

def get_model_accuracy_by_hops(csv_path: str, hop_filter):
    """
    Calculate accuracy for each model from reverify_accuracies.csv,
    filtered by hop count.
    
    Args:
        csv_path: Path to reverify_accuracies.csv
        hop_filter: 'single' for 1-hop, 'multi' for 2-4 hops
    
    Returns:
        dict: {display_name: accuracy_percentage}
    """
    # Create mapping from JSONL filename to display name
    filename_to_display = {}
    for filename, display_name in ITERATIVE_MODEL_ENTRIES:
        filename_to_display[filename] = display_name
    
    # Read all responses to count correct answers by hop
    model_hop_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for filename, display_name in ITERATIVE_MODEL_ENTRIES:
        jsonl_path = RESPONSES_DIR / filename
        
        if not jsonl_path.exists():
            print(f"Warning: JSONL file not found: {jsonl_path}")
            continue
        
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    hops = data.get('number_of_hops', data.get('raw', {}).get('number_of_hops'))
                    
                    if hops is None:
                        continue
                    
                    # Apply hop filter
                    if hop_filter == 'single' and hops != 1:
                        continue
                    elif hop_filter == 'multi' and hops == 1:
                        continue
                    
                    # Count this question
                    model_hop_stats[display_name]['total'] += 1
                    if data.get('is_correct', False):
                        model_hop_stats[display_name]['correct'] += 1
                        
                except json.JSONDecodeError:
                    continue
    
    # Calculate accuracies
    accuracies = {}
    for display_name, stats in model_hop_stats.items():
        if stats['total'] > 0:
            accuracy = (stats['correct'] / stats['total']) * 100
            accuracies[display_name] = accuracy
            print(f"{display_name} ({hop_filter}-hop): {stats['correct']}/{stats['total']} = {accuracy:.2f}%")
    
    return accuracies

def calculate_average_output_tokens_by_hops(display_name: str, hop_filter):
    """
    Calculate average output tokens for a model from JSONL files,
    filtered by hop count.
    
    Args:
        display_name: Model display name
        hop_filter: 'single' for 1-hop, 'multi' for 2-4 hops
    
    Returns:
        float: Average output tokens (or None if not available)
    """
    # Find the entry in ITERATIVE_MODEL_ENTRIES
    filename = None
    for fname, dname in ITERATIVE_MODEL_ENTRIES:
        if dname == display_name:
            filename = fname
            break
    
    if not filename:
        return None
    
    jsonl_path = RESPONSES_DIR / filename
    
    if not jsonl_path.exists():
        return None
    
    total_tokens = 0
    count = 0
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                hops = data.get('number_of_hops', data.get('raw', {}).get('number_of_hops'))
                
                if hops is None:
                    continue
                
                # Apply hop filter
                if hop_filter == 'single' and hops != 1:
                    continue
                elif hop_filter == 'multi' and hops == 1:
                    continue
                
                output_tokens = data.get('output_tokens')
                if output_tokens is not None and isinstance(output_tokens, (int, float)) and output_tokens > 0:
                    total_tokens += output_tokens
                    count += 1
                    
            except json.JSONDecodeError:
                continue
    
    if count > 0:
        return total_tokens / count
    return None

def plot_accuracy_vs_tokens_side_by_side(data_points_single, data_points_multi, output_filename):
    """
    Create side-by-side scatter plots of accuracy vs average output tokens.
    
    Args:
        data_points_single: list of tuples (model_name, accuracy, avg_tokens) for single-hop
        data_points_multi: list of tuples (model_name, accuracy, avg_tokens) for multi-hop
        output_filename: Output filename (without path)
    """
    if not data_points_single or not data_points_multi:
        print(f"Missing data points!")
        return
    
    # Calculate uniform axis ranges
    all_accuracies = [x[1] for x in data_points_single] + [x[1] for x in data_points_multi]
    all_tokens = [x[2] for x in data_points_single] + [x[2] for x in data_points_multi]
    
    acc_min, acc_max = min(all_accuracies), max(all_accuracies)
    token_min, token_max = min(all_tokens), max(all_tokens)
    
    # Add 5% padding to ranges
    acc_padding = (acc_max - acc_min) * 0.05
    token_padding = (token_max - token_min) * 0.05
    
    acc_range = (acc_min - acc_padding, acc_max + acc_padding)
    token_range = (token_min - token_padding, token_max + token_padding)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot single-hop (left)
    models_single = [x[0] for x in data_points_single]
    accuracies_single = [x[1] for x in data_points_single]
    avg_tokens_single = [x[2] for x in data_points_single]
    colors_single = [MODEL_COLOR_MAP.get(model, DEFAULT_MODEL_COLOR) for model in models_single]
    
    ax1.scatter(accuracies_single, avg_tokens_single, s=200, c=colors_single, 
               alpha=0.7, edgecolors='black', linewidth=1.5)
    
    for model, acc, tokens in zip(models_single, accuracies_single, avg_tokens_single):
        ax1.annotate(model, (acc, tokens), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))
    
    # Calculate correlation for single-hop
    if len(accuracies_single) > 1:
        correlation_single = np.corrcoef(accuracies_single, avg_tokens_single)[0, 1]
        ax1.text(0.02, 0.98, f'Pearson r = {correlation_single:.3f}',
                transform=ax1.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.set_xlabel('Accuracy (%) in Iterative RAG', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Output Tokens', fontsize=12, fontweight='bold')
    ax1.set_title('Single-Hop Questions (1 hop)', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xlim(acc_range)
    ax1.set_ylim(token_range)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Plot multi-hop (right)
    models_multi = [x[0] for x in data_points_multi]
    accuracies_multi = [x[1] for x in data_points_multi]
    avg_tokens_multi = [x[2] for x in data_points_multi]
    colors_multi = [MODEL_COLOR_MAP.get(model, DEFAULT_MODEL_COLOR) for model in models_multi]
    
    ax2.scatter(accuracies_multi, avg_tokens_multi, s=200, c=colors_multi,
               alpha=0.7, edgecolors='black', linewidth=1.5)
    
    for model, acc, tokens in zip(models_multi, accuracies_multi, avg_tokens_multi):
        ax2.annotate(model, (acc, tokens), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))
    
    # Calculate correlation for multi-hop
    if len(accuracies_multi) > 1:
        correlation_multi = np.corrcoef(accuracies_multi, avg_tokens_multi)[0, 1]
        ax2.text(0.02, 0.98, f'Pearson r = {correlation_multi:.3f}',
                transform=ax2.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax2.set_xlabel('Accuracy (%) in Iterative RAG', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Output Tokens', fontsize=12, fontweight='bold')
    ax2.set_title('Multi-Hop Questions (2-4 hops)', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xlim(acc_range)
    ax2.set_ylim(token_range)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    # Main title
    fig.suptitle('Model Accuracy vs Average Output Tokens by Question Complexity', 
                fontsize=15, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    output_path = PLOTS_DIR / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved combined figure to {output_path}")
    plt.close()
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Single-Hop - Accuracy: {min(accuracies_single):.2f}% to {max(accuracies_single):.2f}%, Tokens: {min(avg_tokens_single):.2f} to {max(avg_tokens_single):.2f}")
    print(f"  Multi-Hop - Accuracy: {min(accuracies_multi):.2f}% to {max(accuracies_multi):.2f}%, Tokens: {min(avg_tokens_multi):.2f} to {max(avg_tokens_multi):.2f}")
    print(f"  Uniform axis ranges - Accuracy: {acc_range[0]:.2f}% to {acc_range[1]:.2f}%, Tokens: {token_range[0]:.2f} to {token_range[1]:.2f}")

def main():
    """Main execution function."""
    csv_path = Path(__file__).parent.parent / 'results' / 'reverify_accuracies.csv'
    
    if not csv_path.exists():
        print(f"Error: reverify_accuracies.csv not found at {csv_path}")
        return
    
    # Process single-hop questions (1 hop)
    print("=" * 70)
    print("PROCESSING SINGLE-HOP QUESTIONS (1 hop)")
    print("=" * 70)
    
    accuracies_single = get_model_accuracy_by_hops(csv_path, 'single')
    
    data_points_single = []
    for display_name, accuracy in accuracies_single.items():
        avg_tokens = calculate_average_output_tokens_by_hops(display_name, 'single')
        if avg_tokens is not None:
            data_points_single.append((display_name, accuracy, avg_tokens))
            print(f"{display_name}: Accuracy = {accuracy:.2f}%, Avg Output Tokens = {avg_tokens:.2f}")
    
    # Process multi-hop questions (2-4 hops)
    print("\n" + "=" * 70)
    print("PROCESSING MULTI-HOP QUESTIONS (2-4 hops)")
    print("=" * 70)
    
    accuracies_multi = get_model_accuracy_by_hops(csv_path, 'multi')
    
    data_points_multi = []
    for display_name, accuracy in accuracies_multi.items():
        avg_tokens = calculate_average_output_tokens_by_hops(display_name, 'multi')
        if avg_tokens is not None:
            data_points_multi.append((display_name, accuracy, avg_tokens))
            print(f"{display_name}: Accuracy = {accuracy:.2f}%, Avg Output Tokens = {avg_tokens:.2f}")
    
    # Create side-by-side plot with uniform axes
    print("\n" + "=" * 70)
    print("CREATING SIDE-BY-SIDE PLOT WITH UNIFORM AXES")
    print("=" * 70)
    
    plot_accuracy_vs_tokens_side_by_side(data_points_single, data_points_multi,
                                         'accuracy_vs_output_tokens_by_hops.png')
    
    print("\n" + "=" * 70)
    print("COMPLETED: Generated side-by-side comparison plot")
    print("=" * 70)

if __name__ == '__main__':
    main()
