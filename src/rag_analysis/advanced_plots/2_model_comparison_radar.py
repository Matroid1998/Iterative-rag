"""
Plot 2: Model Comparison Radar Chart
Shows: accuracy, avg_steps, specificity, on_topic, sufficiency, coverage, (low) miscalibration
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_system_plots.cross_system_utils import (
    load_all_judgments, 
    create_merged_dataset,
    normalize_model_name,
    extract_model_from_filename
)


def load_accuracy_from_csv(csv_file):
    """Load accuracy data from reverify_accuracies.csv file."""
    import csv
    accuracy_map = {}
    
    if not csv_file.exists():
        print(f"Warning: {csv_file} not found")
        return accuracy_map
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = row.get('folder', '')
            if folder != 'Iterative-RAG':
                continue
            
            file_name = row.get('file_name', '')
            accuracy = float(row.get('accuracy', 0)) * 100  # Convert to percentage
            
            # Extract model name from file name
            model_name = file_name.replace('responses_', '').replace('_reverified.jsonl', '')
            model_name = model_name.replace('bedrock_', '').replace('openai_', '')
            model_name = normalize_model_name(model_name)
            
            accuracy_map[model_name] = accuracy
    
    return accuracy_map


def calculate_model_metrics(merged_data, accuracy_map):
    """Calculate all radar chart metrics per model."""
    
    model_stats = defaultdict(lambda: {
        'total': 0,
        'steps': [],
        'specificity': [],
        'on_topic': 0,
        'sufficient': 0,
        'good_coverage': 0,
        'calibrated': 0,
    })
    
    for rec in merged_data:
        model = normalize_model_name(rec['model'])
        quality = rec.get('quality', {})
        coverage = rec.get('coverage', {})
        hallucination = rec.get('hallucination', {})
        
        stats = model_stats[model]
        stats['total'] += 1
        
        # Steps
        per_step = quality.get('per_step', [])
        if per_step:
            stats['steps'].append(len(per_step))
            
            # Specificity - average across all steps
            step_specificities = []
            for step in per_step:
                spec = step.get('query_quality', {}).get('specificity_score', 0)
                if spec > 0:
                    step_specificities.append(spec)
            if step_specificities:
                stats['specificity'].append(np.mean(step_specificities))
        
        # On-topic (not off-topic in any step)
        is_on_topic = True
        for step in per_step:
            if step.get('query_quality', {}).get('off_topic', False):
                is_on_topic = False
                break
        if is_on_topic:
            stats['on_topic'] += 1
        
        # Sufficiency - nested in composition_and_faithfulness
        comp_faith = hallucination.get('composition_and_faithfulness', {})
        sufficiency = comp_faith.get('sufficiency_score_est', 0)
        if sufficiency >= 0.6:
            stats['sufficient'] += 1
        
        # Coverage (no gaps)
        has_gap = coverage.get('any_coverage_gap', False)
        if not has_gap:
            stats['good_coverage'] += 1
        
        # Calibration (not miscalibrated) - nested in confidence_miscalibration
        conf_misc = hallucination.get('confidence_miscalibration', {})
        is_miscalibrated = conf_misc.get('is_miscalibrated', False)
        if not is_miscalibrated:
            stats['calibrated'] += 1
    
    # Calculate averages and percentages
    results = {}
    for model, stats in model_stats.items():
        if stats['total'] == 0:
            continue
        
        results[model] = {
            'accuracy': accuracy_map.get(model, 0.0),  # Already a percentage from CSV
            'avg_steps': np.mean(stats['steps']) if stats['steps'] else 0,
            'specificity': np.mean(stats['specificity']) * 100 if stats['specificity'] else 0,
            'on_topic_rate': (stats['on_topic'] / stats['total']) * 100,
            'sufficiency_rate': (stats['sufficient'] / stats['total']) * 100,
            'coverage_rate': (stats['good_coverage'] / stats['total']) * 100,
            'calibration_rate': (stats['calibrated'] / stats['total']) * 100,
        }
    
    return results


def create_radar_chart(model_metrics):
    """Create radar chart comparing models."""
    
    if not model_metrics:
        print("No model metrics to plot")
        return None
    
    # Define categories (metrics)
    categories = [
        'Accuracy',
        'Specificity',
        'On-Topic',
        'Sufficiency',
        'Coverage',
        'Calibration',
        'Avg Steps'
    ]
    
    # Prepare data
    models = list(model_metrics.keys())
    num_vars = len(categories)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Colors for models
    colors = [
        '#e74c3c', '#3498db', '#2ecc71', '#f39c12', 
        '#9b59b6', '#1abc9c', '#e67e22', '#34495e'
    ]
    
    # Find max steps for normalization
    max_steps = max(metrics['avg_steps'] for metrics in model_metrics.values())
    
    # Plot each model
    for idx, model in enumerate(models):
        metrics = model_metrics[model]
        
        # Extract values
        # Normalize steps to 0-100 scale for radar chart
        values = [
            metrics['accuracy'],
            metrics['specificity'],
            metrics['on_topic_rate'],
            metrics['sufficiency_rate'],
            metrics['coverage_rate'],
            metrics['calibration_rate'],
            (metrics['avg_steps'] / max_steps) * 100,  # Normalize to 0-100
        ]
        
        # Complete the circle
        values += values[:1]
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])
    
    # Fix axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    
    # Set y-axis limits
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], size=9)
    ax.set_rlabel_position(0)
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Title and legend
    plt.title(
        'Model Comparison: Multi-Dimensional Performance Profile',
        size=14,
        fontweight='bold',
        pad=30
    )
    
    ax.legend(
        loc='upper right',
        bbox_to_anchor=(1.3, 1.1),
        frameon=True,
        fontsize=10
    )
    
    plt.tight_layout()
    return fig


def main():
    output_dir = Path('/media/torontoai/Iterative-rag/src/rag_analysis/output')
    csv_file = Path('/media/torontoai/Iterative-rag/src/results/reverify_accuracies.csv')
    
    print("Loading accuracy from CSV...")
    accuracy_map = load_accuracy_from_csv(csv_file)
    print(f"Loaded accuracy for {len(accuracy_map)} models")
    for model, acc in accuracy_map.items():
        print(f"  {model}: {acc:.2f}%")
    
    print("\nLoading all judgments...")
    coverage, quality, hallucination = load_all_judgments(output_dir)
    
    print(f"Loaded: {len(coverage)} coverage, {len(quality)} quality, {len(hallucination)} hallucination")
    
    print("Merging datasets...")
    merged_data = create_merged_dataset(coverage, quality, hallucination)
    print(f"Merged: {len(merged_data)} records")
    
    print("Calculating model metrics...")
    model_metrics = calculate_model_metrics(merged_data, accuracy_map)
    
    print("\nModel Metrics Summary:")
    for model, metrics in model_metrics.items():
        print(f"\n{model}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.2f}")
    
    print("\nCreating radar chart...")
    fig = create_radar_chart(model_metrics)
    
    if fig:
        output_path = Path(__file__).parent / '2_model_comparison_radar.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    else:
        print("✗ Failed to create plot")


if __name__ == '__main__':
    main()
