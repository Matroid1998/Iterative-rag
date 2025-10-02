"""
Plot 1: Query Degradation Over Steps
Multi-line chart with X=step number, Y=score, separate lines for specificity_score and on_topic_score, faceted by model.
Insight: Do queries get worse as RAG iterates?
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def load_query_scores_by_step(output_dir):
    """Load specificity and on-topic scores by step for each model."""
    # Structure: {model: {step: {'specificity': [], 'on_topic': []}}}
    model_step_scores = defaultdict(lambda: defaultdict(lambda: {'specificity': [], 'on_topic': []}))
    
    for file_path in glob.glob(str(output_dir / '*quality_judement.jsonl')):
        # Extract model name from filename
        filename = Path(file_path).name
        model_name = filename.replace('responses_', '').replace('_reverified_quality_judement.jsonl', '')
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    parsed = data.get('parsed_judgment', {})
                    
                    for step_data in parsed.get('per_step', []):
                        step = step_data.get('step')
                        if step is None or step > 5:  # Limit to first 5 steps
                            continue
                        
                        quality = step_data.get('query_quality', {})
                        
                        spec_score = quality.get('specificity_score')
                        if spec_score is not None:
                            model_step_scores[model_name][step]['specificity'].append(float(spec_score))
                        
                        on_topic = quality.get('on_topic_score')
                        if on_topic is not None:
                            model_step_scores[model_name][step]['on_topic'].append(float(on_topic))
                
                except json.JSONDecodeError:
                    continue
    
    return model_step_scores


def create_faceted_line_chart(model_step_scores, output_path):
    """Create faceted multi-line chart showing query score degradation."""
    models = sorted(model_step_scores.keys())
    n_models = len(models)
    
    # Create subplots - 2 columns
    n_cols = 2
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        scores_by_step = model_step_scores[model]
        
        steps = sorted(scores_by_step.keys())
        
        # Calculate means for each step
        spec_means = []
        spec_stds = []
        on_topic_means = []
        on_topic_stds = []
        
        for step in steps:
            spec_scores = scores_by_step[step]['specificity']
            on_topic_scores = scores_by_step[step]['on_topic']
            
            if spec_scores:
                spec_means.append(np.mean(spec_scores))
                spec_stds.append(np.std(spec_scores))
            else:
                spec_means.append(None)
                spec_stds.append(None)
            
            if on_topic_scores:
                on_topic_means.append(np.mean(on_topic_scores))
                on_topic_stds.append(np.std(on_topic_scores))
            else:
                on_topic_means.append(None)
                on_topic_stds.append(None)
        
        # Plot lines with error bands
        if any(s is not None for s in spec_means):
            ax.plot(steps, spec_means, marker='o', linewidth=2.5, markersize=8,
                   label='Specificity Score', color='#4c72b0', alpha=0.9)
            # Add confidence band
            spec_array = np.array([s if s is not None else 0 for s in spec_means])
            std_array = np.array([s if s is not None else 0 for s in spec_stds])
            ax.fill_between(steps, spec_array - std_array, spec_array + std_array,
                           alpha=0.2, color='#4c72b0')
        
        if any(s is not None for s in on_topic_means):
            ax.plot(steps, on_topic_means, marker='s', linewidth=2.5, markersize=8,
                   label='On-Topic Score', color='#55a868', alpha=0.9)
            # Add confidence band
            on_topic_array = np.array([s if s is not None else 0 for s in on_topic_means])
            std_array = np.array([s if s is not None else 0 for s in on_topic_stds])
            ax.fill_between(steps, on_topic_array - std_array, on_topic_array + std_array,
                           alpha=0.2, color='#55a868')
        
        # Add trend lines
        if len(steps) > 2 and any(s is not None for s in spec_means):
            clean_steps = [s for s, v in zip(steps, spec_means) if v is not None]
            clean_spec = [v for v in spec_means if v is not None]
            if len(clean_steps) > 1:
                z = np.polyfit(clean_steps, clean_spec, 1)
                p = np.poly1d(z)
                ax.plot(steps, p(steps), "--", color='#4c72b0', linewidth=1.5, 
                       alpha=0.5, label=f'Spec Trend: {z[0]:.3f}x')
        
        # Customize subplot
        short_name = model.replace('bedrock_', '').replace('openai_', '').replace('us.anthropic.', '')
        ax.set_title(short_name, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Step Number', fontsize=10)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_ylim(0.55, 1.05)  # Focus on 0.6-1.0 range where values actually fall
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left', fontsize=8)
        ax.set_xticks(steps)
    
    # Hide empty subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Query Quality Degradation Over Steps\n(Do queries get worse as RAG iterates?)',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved query degradation chart to {output_path}")
    plt.close()
    
    # Print statistics
    print("\n" + "="*80)
    print("QUERY DEGRADATION ANALYSIS")
    print("="*80)
    
    for model in models:
        scores_by_step = model_step_scores[model]
        steps = sorted(scores_by_step.keys())
        
        print(f"\n{model}:")
        print(f"  {'Step':<6} {'Spec Mean':>12} {'On-Topic Mean':>15} {'N':>8}")
        print("  " + "-"*45)
        
        for step in steps:
            spec_scores = scores_by_step[step]['specificity']
            on_topic_scores = scores_by_step[step]['on_topic']
            
            spec_mean = np.mean(spec_scores) if spec_scores else 0
            on_topic_mean = np.mean(on_topic_scores) if on_topic_scores else 0
            n = len(spec_scores) if spec_scores else len(on_topic_scores)
            
            print(f"  {step:<6} {spec_mean:>12.3f} {on_topic_mean:>15.3f} {n:>8}")
        
        # Calculate trend
        if len(steps) > 2:
            spec_means_list = [np.mean(scores_by_step[s]['specificity']) 
                              for s in steps if scores_by_step[s]['specificity']]
            if len(spec_means_list) > 1:
                z = np.polyfit(range(len(spec_means_list)), spec_means_list, 1)
                trend = "DEGRADING" if z[0] < -0.01 else "STABLE" if abs(z[0]) < 0.01 else "IMPROVING"
                print(f"  Specificity Trend: {trend} (slope: {z[0]:.4f})")


def main():
    # Setup paths
    base_dir = Path(__file__).resolve().parents[2]
    output_dir = base_dir / "rag_analysis" / "output"
    plot_dir = base_dir / "rag_analysis" / "quality_rag_plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading query scores by step...")
    model_step_scores = load_query_scores_by_step(output_dir)
    
    if not model_step_scores:
        print("No query score data found!")
        return
    
    # Create plot
    output_path = plot_dir / "query_degradation_over_steps.png"
    create_faceted_line_chart(model_step_scores, output_path)


if __name__ == "__main__":
    main()
