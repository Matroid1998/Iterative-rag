#!/usr/bin/env python3
"""
Plot the percentage of fusion, over-broad queries, and vague queries per model.

This script analyzes query characteristics across different models to identify
patterns in how models formulate their retrieval queries.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

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
        "openrouter_anthropic_claude_sonnet_4_5_reasoning": "Claude Sonnet 4.5",
        "openrouter_google__gemini-2.5-pro": "Gemini 2.5 Pro",
        "openrouter_x-ai__grok-4-fast": "Grok 4 Fast",
        "openrouter_z-ai__glm-4.6": "GLM 4.6",
    }
    
    files = list(quality_dir.glob("*quality_judgement.jsonl")) + list(quality_dir.glob("*quality_judement.jsonl"))
    
    entries = []
    for quality_file in sorted(files):
        stem = quality_file.stem
        
        if stem.endswith("_quality_judement"):
            stem = stem[:-len("_quality_judement")]
        elif stem.endswith("_quality_judgement"):
            stem = stem[:-len("_quality_judgement")]
        
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


def analyze_query_characteristics(quality_file: Path) -> Dict[str, float]:
    """
    Analyze query characteristics from a quality judgment file.
    
    Returns:
        Dict with percentages for fusion, over_broad, and vague queries.
    """
    stats = {
        'fusion_count': 0,
        'over_broad_count': 0,
        'vague_count': 0,
        'off_topic_count': 0,
        'total_steps': 0,
    }
    
    with open(quality_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            parsed = data.get('parsed_judgment', {})
            per_step = parsed.get('per_step', [])
            
            for step_data in per_step:
                stats['total_steps'] += 1
                
                # Check fusion/skip
                fusion_or_skip = step_data.get('fusion', False)
                if fusion_or_skip:
                    stats['fusion_count'] += 1
                
                # Check query quality flags
                qc = step_data.get('query_quality', {})
                if qc.get('over_broad', False):
                    stats['over_broad_count'] += 1
                if qc.get('vague', False):
                    stats['vague_count'] += 1
                if qc.get('off_topic', False):
                    stats['off_topic_count'] += 1
    
    # Calculate percentages
    total = stats['total_steps']
    if total > 0:
        return {
            'fusion': (stats['fusion_count'] / total) * 100,
            'over_broad': (stats['over_broad_count'] / total) * 100,
            'vague': (stats['vague_count'] / total) * 100,
            'off_topic': (stats['off_topic_count'] / total) * 100,
            'total_steps': total,
        }
    else:
        return {
            'fusion': 0,
            'over_broad': 0,
            'vague': 0,
            'off_topic': 0,
            'total_steps': 0,
        }


def plot_query_characteristics(model_stats: Dict[str, Dict[str, float]], output_path: Path):
    """Plot query characteristics (fusion, over-broad, vague) per model."""
    
    # Sort models by fusion percentage (descending)
    sorted_models = sorted(model_stats.items(), key=lambda x: x[1]['fusion'], reverse=True)
    model_names = [name for name, _ in sorted_models]
    
    # Extract percentages
    fusion_pcts = [stats['fusion'] for _, stats in sorted_models]
    over_broad_pcts = [stats['over_broad'] for _, stats in sorted_models]
    vague_pcts = [stats['vague'] for _, stats in sorted_models]
    off_topic_pcts = [stats['off_topic'] for _, stats in sorted_models]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(model_names))
    width = 0.15
    
    # Create bars
    bars1 = ax.bar(x - 2*width, fusion_pcts, width, label='Fusion', 
                   color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x - width, over_broad_pcts, width, label='Over-Broad Query', 
                   color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.2)
    bars3 = ax.bar(x, vague_pcts, width, label='Vague Query', 
                   color='#f39c12', alpha=0.85, edgecolor='black', linewidth=1.2)
    bars5 = ax.bar(x + width, off_topic_pcts, width, label='Off-Topic Query', 
                   color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0.5:  # Only show if bar is tall enough
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontweight='bold', fontsize=8, rotation=90)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    add_labels(bars5)
    
    # Customize plot
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('Percentage of Steps (%)', fontsize=13, fontweight='bold')
    ax.set_title('Query Characteristics by Model\nFusion, Over-Broad, Vague, and Off-Topic Queries',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(max(fusion_pcts), max(over_broad_pcts), max(vague_pcts), max(off_topic_pcts)) * 1.15)
    
    # Add statistics in text box - REMOVED per user request
    # avg_fusion = np.mean(fusion_pcts)
    # ...
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved query characteristics plot: {output_path}")


def main():
    """Main execution function."""
    base = get_base_path()
    output_dir = base / "data" / "plots" / "general"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Analyzing query characteristics across models...")
    
    model_stats = {}
    for quality_path, display_name in get_quality_model_entries():
        stats = analyze_query_characteristics(quality_path)
        model_stats[display_name] = stats
        print(f"  {display_name:30s}: {stats['fusion']:5.1f}% fusion, "
              f"{stats['over_broad']:5.1f}% over-broad, {stats['vague']:5.1f}% vague, "
              f"{stats['off_topic']:5.1f}% off-topic "
              f"({stats['total_steps']} steps)")
    
    print(f"\nTotal models analyzed: {len(model_stats)}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    all_fusion = [stats['fusion'] for stats in model_stats.values()]
    all_over_broad = [stats['over_broad'] for stats in model_stats.values()]
    all_vague = [stats['vague'] for stats in model_stats.values()]
    all_off_topic = [stats['off_topic'] for stats in model_stats.values()]
    
    print(f"\nFusion/Skip:")
    print(f"  Average: {np.mean(all_fusion):.1f}%")
    print(f"  Range: {np.min(all_fusion):.1f}% - {np.max(all_fusion):.1f}%")
    
    print(f"\nOver-Broad Queries:")
    print(f"  Average: {np.mean(all_over_broad):.1f}%")
    print(f"  Range: {np.min(all_over_broad):.1f}% - {np.max(all_over_broad):.1f}%")
    
    print(f"\nVague Queries:")
    print(f"  Average: {np.mean(all_vague):.1f}%")
    print(f"  Range: {np.min(all_vague):.1f}% - {np.max(all_vague):.1f}%")
    

    
    print(f"\nOff-Topic Queries:")
    print(f"  Average: {np.mean(all_off_topic):.1f}%")
    print(f"  Range: {np.min(all_off_topic):.1f}% - {np.max(all_off_topic):.1f}%")
    
    # Generate plot
    print("\nGenerating plot...")
    plot_query_characteristics(
        model_stats,
        output_dir / "model_query_characteristics.png"
    )
    
    print("\n✅ Query characteristics analysis completed!")


if __name__ == "__main__":
    main()
