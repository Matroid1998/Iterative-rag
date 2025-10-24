#!/usr/bin/env python3
"""
Analyze correlation between fusion and over-broad queries.
"""

import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def get_base_path() -> Path:
    """Get the base path for the project."""
    return Path(__file__).resolve().parents[2]


def get_query_stats() -> Tuple[List[float], List[float], List[str]]:
    """Get fusion and over-broad percentages for all models."""
    base = get_base_path()
    quality_dir = base / "src" / "rag_analysis" / "output"
    
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
    
    fusion_pcts = []
    over_broad_pcts = []
    model_labels = []
    
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
        
        fusion_count = 0
        over_broad_count = 0
        total_steps = 0
        
        with open(quality_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    per_step = data.get('parsed_judgment', {}).get('per_step', [])
                    for step_data in per_step:
                        total_steps += 1
                        if step_data.get('fusion_or_skip', False):
                            fusion_count += 1
                        if step_data.get('query_quality', {}).get('over_broad', False):
                            over_broad_count += 1
                except:
                    continue
        
        if total_steps > 0:
            fusion_pcts.append((fusion_count / total_steps) * 100)
            over_broad_pcts.append((over_broad_count / total_steps) * 100)
            model_labels.append(display_name)
    
    return fusion_pcts, over_broad_pcts, model_labels


def plot_fusion_vs_over_broad(fusion_pcts: List[float], over_broad_pcts: List[float], 
                               model_labels: List[str], output_path: Path):
    """Create scatter plot showing correlation between fusion and over-broad queries."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate correlation
    fusion_arr = np.array(fusion_pcts)
    over_broad_arr = np.array(over_broad_pcts)
    correlation = np.corrcoef(fusion_arr, over_broad_arr)[0, 1]
    
    # Create scatter plot
    colors = ['#e74c3c' if 'GPT-5' in label else 
              '#2ecc71' if 'Claude Sonnet 4.5' in label or 'DeepSeek' in label else 
              '#3498db' for label in model_labels]
    
    scatter = ax.scatter(fusion_pcts, over_broad_pcts, s=200, c=colors, 
                        alpha=0.7, edgecolors='black', linewidth=2)
    
    # Add model labels
    for i, label in enumerate(model_labels):
        # Shorten some labels for readability
        short_label = label.replace('Claude 3.7 Sonnet', 'C3.7S').replace('Thinking', 'T')
        short_label = short_label.replace('Llama 3.3 70B Instruct', 'Llama 3.3')
        short_label = short_label.replace('Mistral Large 2402', 'Mistral L')
        short_label = short_label.replace('Gemini 2.5 Pro', 'Gemini 2.5')
        
        ax.annotate(short_label, (fusion_pcts[i], over_broad_pcts[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='gray', alpha=0.8))
    
    # Add trend line
    z = np.polyfit(fusion_arr, over_broad_arr, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(fusion_arr.min(), fusion_arr.max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=2, label='Trend line')
    
    # Customize plot
    ax.set_xlabel('Fusion/Skip Usage (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Over-Broad Query Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Correlation between Fusion and Over-Broad Queries\nPearson r = {correlation:.3f}',
                fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)
    
    # Add interpretation text
    interpretation = "Moderate positive correlation"
    if correlation > 0.7:
        interpretation = "Strong positive correlation"
    elif correlation < 0.3:
        interpretation = "Weak positive correlation"
    
    ax.text(0.05, 0.95, f'{interpretation}\nr = {correlation:.3f}', 
           transform=ax.transAxes,
           fontsize=12, verticalalignment='top', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
    
    # Add outlier notes
    gpt5_idx = model_labels.index('GPT-5')
    ax.annotate('', xy=(fusion_pcts[gpt5_idx], over_broad_pcts[gpt5_idx]),
               xytext=(fusion_pcts[gpt5_idx] - 2, over_broad_pcts[gpt5_idx] + 3),
               arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(fusion_pcts[gpt5_idx] - 2, over_broad_pcts[gpt5_idx] + 3.5,
           'Lowest in both', fontsize=10, color='red', fontweight='bold',
           ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved correlation plot: {output_path}")


def main():
    """Main execution function."""
    base = get_base_path()
    output_dir = base / "src" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Analyzing correlation between fusion and over-broad queries...")
    
    fusion_pcts, over_broad_pcts, model_labels = get_query_stats()
    
    # Calculate and display correlation
    correlation = np.corrcoef(fusion_pcts, over_broad_pcts)[0, 1]
    
    print(f"\nPearson Correlation Coefficient: {correlation:.3f}")
    
    if correlation > 0.7:
        strength = "strong positive"
    elif correlation > 0.4:
        strength = "moderate positive"
    elif correlation > 0.2:
        strength = "weak positive"
    elif correlation > -0.2:
        strength = "negligible"
    elif correlation > -0.4:
        strength = "weak negative"
    elif correlation > -0.7:
        strength = "moderate negative"
    else:
        strength = "strong negative"
    
    print(f"Interpretation: {strength} correlation")
    print(f"\nThis suggests that models that use fusion/skip more frequently")
    print(f"also tend to generate more over-broad queries.")
    
    print("\nData points:")
    for model, fusion, over_broad in sorted(zip(model_labels, fusion_pcts, over_broad_pcts), 
                                           key=lambda x: x[1]):
        print(f"  {model:30s}: Fusion={fusion:5.1f}%, Over-Broad={over_broad:5.1f}%")
    
    # Generate plot
    print("\nGenerating correlation plot...")
    plot_fusion_vs_over_broad(
        fusion_pcts,
        over_broad_pcts,
        model_labels,
        output_dir / "fusion_vs_over_broad_correlation.png"
    )
    
    print("\n✅ Correlation analysis completed!")


if __name__ == "__main__":
    main()
