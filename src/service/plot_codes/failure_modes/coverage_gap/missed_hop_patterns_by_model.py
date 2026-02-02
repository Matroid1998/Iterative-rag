#!/usr/bin/env python3
"""
Plot: Missed Hop Patterns by Model
Shows missed hop patterns for each model in a multi-panel plot.
Insight: Which models have better hop coverage? Which hops do they miss?
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Model name mapping
MODEL_NAME_MAP = {
    'responses_openai_gpt-4o_reverified': 'GPT-4o',
    'responses_openai_gpt-5_reverified': 'GPT-5',
    'responses_openrouter_google__gemini-2.5-pro_reverified': 'Gemini 2.5 Pro',
    'responses_openrouter_x-ai__grok-4-fast_reverified': 'Grok 4 Fast',
    'responses_openrouter_z-ai__glm-4.6_reverified': 'GLM 4.6',
    'responses_bedrock_us.deepseek.r1-v1:0-reasoning_reverified': 'DeepSeek R1',
    'responses_bedrock_mistral.mistral-large-2402-v1:0_reverified': 'Mistral Large',
    'responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0_reverified': 'Claude 3.7 Sonnet',
    'responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning_reverified': 'Claude 3.7 Sonnet + Reasoning',
    'responses_openrouter_anthropic__claude-sonnet-4.5_reverified': 'Claude Sonnet 4.5',
    '2_responses_bedrock_us.meta.llama3-3-70b-instruct-v1:0_reverified': 'Llama 3.3 70B',
    'responses_bedrock_us.meta.llama3-3-70b-instruct-v1:0_reverified': 'Llama 3.3 70B',
}

def load_missed_hop_patterns_by_model(output_dir):
    """Load missed hop patterns organized by model and number of hops."""
    # Structure: {model_name: {num_hops: {missed_hop_index: count, 'total': count}}}
    model_patterns = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for file_path in sorted(glob.glob(str(output_dir / '*coverage_gap_judgments.jsonl'))):
        filename = Path(file_path).stem
        base_filename = filename.replace('_coverage_gap_judgments', '')
        model_name = MODEL_NAME_MAP.get(base_filename, base_filename)
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    
                    parsed = data.get('parsed_judgment', {})
                    coverage = parsed.get('retrieval_coverage_gap', {})
                    
                    # Determine number of hops from late_hit_per_hop data
                    late_hit = parsed.get('late_hit_per_hop', {})
                    per_hop = late_hit.get('per_hop', [])
                    
                    if per_hop:
                        num_hops = max(h.get('hop_index', 0) for h in per_hop)
                    else:
                        num_hops = 1  # Default assumption
                    
                    model_patterns[model_name][num_hops]['total'] += 1
                    
                    # Check for missed hops
                    if coverage.get('has_gap'):
                        missed_hops = coverage.get('missed_hops', [])
                        for hop in missed_hops:
                            model_patterns[model_name][num_hops][hop] += 1
                
                except json.JSONDecodeError:
                    continue
    
    return model_patterns


def create_multi_model_plot(model_patterns, output_path):
    """Create multi-panel plot showing missed hop patterns for each model."""
    models = sorted(model_patterns.keys())
    n_models = len(models)
    
    # Create subplot grid (3 rows x 4 cols for 11 models)
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    # Get global max hop index for consistent coloring
    all_missed_hops = set()
    all_num_hops = set()
    for patterns in model_patterns.values():
        all_num_hops.update(patterns.keys())
        for hop_data in patterns.values():
            all_missed_hops.update(k for k in hop_data.keys() if k != 'total')
    
    max_hop = max(all_missed_hops) if all_missed_hops else 4
    num_hops_list = sorted(all_num_hops)
    
    # Color palette
    colors = plt.cm.Set3(np.linspace(0, 1, max_hop))
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        patterns = model_patterns[model]
        
        # Build data for this model
        x = np.arange(len(num_hops_list))
        width = 0.6
        
        # Build data matrix for stacking
        data_matrix = []
        for hop_idx in range(1, max_hop + 1):
            row = []
            for num_hops in num_hops_list:
                total = patterns[num_hops]['total']
                missed_count = patterns[num_hops].get(hop_idx, 0)
                percentage = 100 * missed_count / total if total > 0 else 0
                row.append(percentage)
            data_matrix.append(row)
        
        bottom = np.zeros(len(num_hops_list))
        
        # Stack bars
        for hop_idx, percentages in enumerate(data_matrix, 1):
            bars = ax.bar(x, percentages, width, 
                         bottom=bottom, 
                         color=colors[hop_idx - 1],
                         edgecolor='black', 
                         linewidth=0.8, 
                         alpha=0.85,
                         label=f'Hop {hop_idx}')
            
            # Add value labels for significant values
            for i, (bar, pct) in enumerate(zip(bars, percentages)):
                if pct > 3:  # Only label if > 3%
                    height = bar.get_height()
                    y_pos = bottom[i] + height / 2
                    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                           f'{pct:.0f}%',
                           ha='center', va='center', 
                           fontsize=7, fontweight='bold')
            
            bottom += percentages
        
        # Add total percentage labels on top
        for i, x_pos in enumerate(x):
            total_pct = bottom[i]
            total_count = patterns[num_hops_list[i]]['total']
            if total_pct > 0:
                ax.text(x_pos, total_pct + 1, 
                       f'{total_pct:.1f}%',
                       ha='center', va='bottom', 
                       fontsize=8, fontweight='bold',
                       color='darkred')
            # Add sample size below x-axis
            ax.text(x_pos, -2, f'n={total_count}',
                   ha='center', va='top', fontsize=7, color='gray')
        
        # Customize subplot
        ax.set_title(model, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Question Hops', fontsize=10)
        ax.set_ylabel('% Missed', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{n}' for n in num_hops_list])
        ax.set_ylim(0, max(bottom) * 1.15 if max(bottom) > 0 else 10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Only show legend on first subplot
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8, ncol=2)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    # Overall title
    fig.suptitle('Missed Hop Patterns by Model\n(Which hops get missed in multi-hop questions?)',
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved multi-model plot to {output_path}")
    plt.close()
    
    # Print statistics
    print("\n" + "="*80)
    print("MISSED HOP PATTERNS BY MODEL")
    print("="*80)
    
    for model in models:
        print(f"\n{model}:")
        patterns = model_patterns[model]
        
        for num_hops in sorted(patterns.keys()):
            hop_data = patterns[num_hops]
            total = hop_data['total']
            
            # Calculate total questions with gaps
            total_with_gaps = sum(hop_data.get(h, 0) for h in range(1, max_hop + 1))
            gap_rate = 100 * total_with_gaps / total if total > 0 else 0
            
            print(f"  {num_hops}-hop (n={total}): {gap_rate:.1f}% gap rate", end='')
            
            # Show which hops were missed
            missed_details = []
            for hop_idx in range(1, max_hop + 1):
                missed_count = hop_data.get(hop_idx, 0)
                if missed_count > 0:
                    pct = 100 * missed_count / total
                    missed_details.append(f"hop{hop_idx}={pct:.1f}%")
            
            if missed_details:
                print(f" ({', '.join(missed_details)})")
            else:
                print()


def main():
    # Setup paths
    base_dir = Path(__file__).resolve().parents[5]
    output_dir = base_dir  / "data" / "results" / "failure_modes"
    plot_dir = base_dir  / "data" / "plots" / "failure_modes" / "coverage_gap"
    plot_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading missed hop patterns by model...")
    model_patterns = load_missed_hop_patterns_by_model(output_dir)
    
    if not model_patterns:
        print("No missed hop data found!")
        return
    
    # Create plot
    output_path = plot_dir / "missed_hop_patterns_by_model.png"
    create_multi_model_plot(model_patterns, output_path)


if __name__ == "__main__":
    main()
