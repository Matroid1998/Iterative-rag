"""
Plot 4: Steps per Run Distribution
Shows distribution of len(per_step) across runs per model
Overlays average (first_hit_step - hop_index) to show retrieval efficiency
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
from advanced_utils import (
    load_all_judgments, 
    create_merged_dataset,
    normalize_model_name,
    calculate_avg_retrieval_delay
)


def calculate_steps_and_efficiency(merged_data):
    """Calculate step distribution and retrieval efficiency per model."""
    
    model_data = defaultdict(lambda: {
        'steps': [],
        'retrieval_delays': [],
    })
    
    for rec in merged_data:
        model = normalize_model_name(rec['model'])
        quality = rec.get('quality', {})
        coverage = rec.get('coverage', {})
        
        # Number of steps
        per_step = quality.get('per_step', [])
        num_steps = len(per_step)
        if num_steps > 0:
            model_data[model]['steps'].append(num_steps)
        
        # Retrieval efficiency (delay)
        delay = calculate_avg_retrieval_delay(coverage)
        if delay > 0:
            model_data[model]['retrieval_delays'].append(delay)
    
    return model_data


def create_steps_efficiency_plot(model_data):
    """Create combined histogram and efficiency overlay."""
    
    if not model_data:
        print("No model data to plot")
        return None
    
    models = sorted(model_data.keys())
    num_models = len(models)
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_models, 1, figsize=(14, 4 * num_models))
    
    if num_models == 1:
        axes = [axes]
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        data = model_data[model]
        
        steps = data['steps']
        delays = data['retrieval_delays']
        
        if not steps:
            ax.text(0.5, 0.5, f'No data for {model}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Calculate statistics
        avg_steps = np.mean(steps)
        median_steps = np.median(steps)
        avg_delay = np.mean(delays) if delays else 0
        
        # Histogram of steps
        bins = np.arange(0.5, max(steps) + 1.5, 1)
        counts, _, patches = ax.hist(steps, bins=bins, alpha=0.7, color='#3498db', 
                                     edgecolor='black', linewidth=1.2, label='Step Count Distribution')
        
        # Color bars by efficiency
        max_count = max(counts)
        for i, patch in enumerate(patches):
            step_val = i + 1
            # Color gradient: fewer steps = greener
            if step_val <= 2:
                patch.set_facecolor('#2ecc71')
            elif step_val <= 3:
                patch.set_facecolor('#f39c12')
            else:
                patch.set_facecolor('#e74c3c')
        
        # Add average line
        ax.axvline(avg_steps, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {avg_steps:.2f} steps', alpha=0.8)
        ax.axvline(median_steps, color='purple', linestyle='-.', linewidth=2, 
                  label=f'Median: {median_steps:.1f} steps', alpha=0.8)
        
        # Add retrieval delay annotation
        if delays:
            # Create secondary y-axis for delay
            ax2 = ax.twinx()
            
            # Plot delay as scatter at each step count
            step_to_delays = defaultdict(list)
            for step, delay in zip(steps, delays):
                step_to_delays[step].append(delay)
            
            step_vals = sorted(step_to_delays.keys())
            avg_delays_by_step = [np.mean(step_to_delays[s]) for s in step_vals]
            
            ax2.plot(step_vals, avg_delays_by_step, 'o-', color='#e67e22', 
                    linewidth=2, markersize=8, label=f'Avg Retrieval Delay', alpha=0.8)
            
            ax2.set_ylabel('Avg Retrieval Delay\n(first_hit_step - hop_index)', 
                          fontsize=11, fontweight='bold', color='#e67e22')
            ax2.tick_params(axis='y', labelcolor='#e67e22')
            ax2.set_ylim(0, max(avg_delays_by_step) * 1.2)
            ax2.legend(loc='upper right', frameon=True, fontsize=9)
        
        # Formatting
        ax.set_xlabel('Number of Steps per Run', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency (Number of Runs)', fontsize=11, fontweight='bold')
        ax.set_title(f'{model}: Step Distribution & Retrieval Efficiency', 
                    fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.legend(loc='upper left', frameon=True, fontsize=9)
        
        # Add statistics text box
        textstr = f'N = {len(steps)} runs\n'
        textstr += f'Avg Steps: {avg_steps:.2f}\n'
        textstr += f'Median Steps: {median_steps:.1f}\n'
        if delays:
            textstr += f'Avg Delay: {avg_delay:.2f} steps'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    return fig


def print_efficiency_analysis(model_data):
    """Print detailed efficiency analysis."""
    print("\n" + "="*60)
    print("STEPS PER RUN & RETRIEVAL EFFICIENCY ANALYSIS")
    print("="*60)
    
    models = sorted(model_data.keys())
    
    print(f"\n{'Model':<25} {'Avg Steps':<12} {'Median':<10} {'Avg Delay':<12} {'N':<8}")
    print("-" * 70)
    
    for model in models:
        data = model_data[model]
        steps = data['steps']
        delays = data['retrieval_delays']
        
        if not steps:
            continue
        
        avg_steps = np.mean(steps)
        median_steps = np.median(steps)
        avg_delay = np.mean(delays) if delays else 0
        
        print(f"{model:<25} {avg_steps:>6.2f}       {median_steps:>4.1f}      "
              f"{avg_delay:>6.2f}       {len(steps):<8}")
    
    # Find most efficient model
    print("\nEfficiency Rankings:")
    
    efficiency_scores = {}
    for model in models:
        data = model_data[model]
        steps = data['steps']
        delays = data['retrieval_delays']
        
        if not steps:
            continue
        
        avg_steps = np.mean(steps)
        avg_delay = np.mean(delays) if delays else 0
        
        # Lower is better for both
        efficiency_score = avg_steps + (avg_delay * 0.5)  # Weight delay less
        efficiency_scores[model] = efficiency_score
    
    ranked = sorted(efficiency_scores.items(), key=lambda x: x[1])
    
    for rank, (model, score) in enumerate(ranked, 1):
        print(f"{rank}. {model}: {score:.2f} (lower is better)")
    
    # Distribution insights
    print("\nStep Distribution Insights:")
    for model in models:
        data = model_data[model]
        steps = data['steps']
        
        if not steps:
            continue
        
        step_counts = {s: steps.count(s) for s in set(steps)}
        mode_steps = max(step_counts, key=step_counts.get)
        mode_pct = (step_counts[mode_steps] / len(steps)) * 100
        
        print(f"• {model}: Most common = {mode_steps} steps ({mode_pct:.1f}% of runs)")


def main():
    output_dir = Path(__file__).resolve().parents[1] / 'output'
    
    print("Loading all judgments...")
    coverage, quality, hallucination = load_all_judgments(output_dir)
    
    print(f"Loaded: {len(coverage)} coverage, {len(quality)} quality, {len(hallucination)} hallucination")
    
    print("Merging datasets...")
    merged_data = create_merged_dataset(coverage, quality, hallucination)
    print(f"Merged: {len(merged_data)} records")
    
    print("Calculating steps and efficiency...")
    model_data = calculate_steps_and_efficiency(merged_data)
    
    print_efficiency_analysis(model_data)
    
    print("\nCreating steps distribution plot...")
    fig = create_steps_efficiency_plot(model_data)
    
    if fig:
        output_path = Path(__file__).parent / '4_steps_per_run.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    else:
        print("✗ Failed to create plot")


if __name__ == '__main__':
    main()
