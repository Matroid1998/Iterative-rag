#!/usr/bin/env python3
"""
Plot specificity score bins by retrieval step for each model.

This shows how query specificity evolves across retrieval steps
and whether models maintain specificity throughout the iterative process.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def get_base_path() -> Path:
    """Get the base path for the project."""
    return Path(__file__).resolve().parents[4]


def get_quality_model_entries() -> List[Tuple[Path, str]]:
    """Get list of (quality_file_path, display_name) tuples."""
    base = get_base_path()
    quality_dir = base  / "data" / "results" / "failure_modes"
    
    model_names = {
        "bedrock_mistral.mistral-large-2402-v1:0": "Mistral Large 2402",
        "bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning": "Claude 3.7 Sonnet Thinking",
        "bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0": "Claude 3.7 Sonnet",
        "bedrock_us.deepseek.r1-v1:0-reasoning": "DeepSeek R1",
        "bedrock_us.meta.llama3-3-70b-instruct-v1:0": "Llama 3.3 70B Instruct",
        "openai_gpt-4o": "GPT-4o",
        "openai_gpt-5": "GPT-5",
        "openrouter_anthropic__claude-sonnet-4.5": "Claude Sonnet 4.5",
        "openrouter_google__gemini-2.5-pro": "Gemini 2.5 Pro",
        "openrouter_x-ai__grok-4-fast": "Grok 4 Fast",
        "openrouter_z-ai__glm-4.6": "GLM 4.6",
    }
    
    entries = []
    for quality_file in sorted(quality_dir.glob("*quality_judement.jsonl")):
        stem = quality_file.stem
        
        if stem.endswith("_quality_judement"):
            stem = stem[:-len("_quality_judement")]
        
        if stem.startswith("2_"):
            stem = stem[2:]
        
        raw_name = stem
        if stem.endswith("_reverified"):
            raw_name = stem[:-len("_reverified")]
        
        model_key = raw_name
        if model_key.startswith("responses_"):
            model_key = model_key[len("responses_"):]
        
        display_name = model_names.get(model_key, model_key)
        entries.append((quality_file, display_name))
    
    return entries


def analyze_specificity_by_step(quality_file: Path) -> Dict[int, Dict[str, int]]:
    """
    Analyze specificity scores by retrieval step.
    
    Returns:
        Dict[step_num -> Dict[bin_label -> count]]
    """
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
    
    # Track counts by step
    step_stats = defaultdict(lambda: {label: 0 for label in bin_labels})
    
    with open(quality_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
            except:
                continue
            
            parsed = data.get('parsed_judgment', {})
            per_step = parsed.get('per_step', [])
            
            for step_data in per_step:
                step_num = step_data.get('step', 0)
                qc = step_data.get('query_quality', {})
                spec_score = qc.get('specificity_score')
                
                if spec_score is None:
                    continue
                
                # Find which bin
                for i in range(len(bins) - 1):
                    if i == len(bins) - 2:
                        if bins[i] <= spec_score <= bins[i+1]:
                            bin_label = bin_labels[i]
                            break
                    else:
                        if bins[i] <= spec_score < bins[i+1]:
                            bin_label = bin_labels[i]
                            break
                else:
                    continue
                
                step_stats[step_num][bin_label] += 1
    
    return dict(step_stats)


def plot_specificity_by_step(model_stats: Dict[str, Dict[int, Dict[str, int]]], output_path: Path):
    """Plot specificity score bins by step for each model."""
    
    # Sort models alphabetically
    sorted_models = sorted(model_stats.keys())
    
    # Find max step across all models
    max_step = max(max(step_data.keys()) for step_data in model_stats.values() if step_data)
    
    # Create subplots
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    bin_labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
    # Use more distinct colors that are easier to read
    colors = ['#d62728', '#ff7f0e', '#ffdd57', '#9467bd', '#2ca02c']
    
    for idx, model in enumerate(sorted_models):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        step_data = model_stats[model]
        
        # Get all steps for this model
        model_steps = sorted(step_data.keys())
        if not model_steps:
            ax.axis('off')
            continue
        
        # Use all steps from 1 to max_step for this model
        steps = list(range(1, max(model_steps) + 1))
        
        # Prepare data for stacked bar chart
        step_labels = [f'S{s}' for s in steps]
        x = np.arange(len(steps))
        
        # Calculate percentages for each bin at each step
        bin_percentages = {label: [] for label in bin_labels}
        
        for step in steps:
            if step in step_data:
                total = sum(step_data[step].values())
                if total > 0:
                    for bin_label in bin_labels:
                        pct = (step_data[step][bin_label] / total) * 100
                        bin_percentages[bin_label].append(pct)
                else:
                    for bin_label in bin_labels:
                        bin_percentages[bin_label].append(0)
            else:
                # Step doesn't exist for this model, add zeros
                for bin_label in bin_labels:
                    bin_percentages[bin_label].append(0)
        
        # Create stacked bar chart
        bottom = np.zeros(len(steps))
        for i, bin_label in enumerate(bin_labels):
            ax.bar(x, bin_percentages[bin_label], bottom=bottom,
                  label=bin_label, color=colors[i], alpha=0.9, edgecolor='black', linewidth=0.8)
            bottom += bin_percentages[bin_label]
        
        # Customize subplot
        ax.set_title(model, fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel('Retrieval Step', fontsize=9)
        ax.set_ylabel('Percentage (%)', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(step_labels, fontsize=8)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add legend only on first subplot with better visibility
        if idx == 0:
            ax.legend(fontsize=8, loc='upper left', title='Specificity Score', 
                     framealpha=0.95, edgecolor='black')
    
    # Hide unused subplots
    for idx in range(len(sorted_models), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Specificity Score Distribution by Retrieval Step - Per Model\n' +
                'Higher scores (green) = more specific queries',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved specificity by step plot: {output_path}")


def main():
    """Main execution function."""
    base = get_base_path()
    output_dir = base / "data" / "plots" / "general"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Analyzing specificity scores by retrieval step...")
    
    model_stats = {}
    
    for quality_path, display_name in get_quality_model_entries():
        step_stats = analyze_specificity_by_step(quality_path)
        model_stats[display_name] = step_stats
        
        # Print summary
        steps = sorted(step_stats.keys())
        if steps:
            print(f"  {display_name:30s}: Steps {min(steps)}-{max(steps)}")
        else:
            print(f"  {display_name:30s}: No data")
    
    print(f"\nTotal models analyzed: {len(model_stats)}")
    
    # Generate plot
    print("\nGenerating plot...")
    plot_specificity_by_step(
        model_stats,
        output_dir / "specificity_by_step_per_model.png"
    )
    
    print("\n✅ Specificity by step analysis completed!")


if __name__ == "__main__":
    main()
