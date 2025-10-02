"""
Plot 7: Scores Distribution and Trends
Violin or box plots of specificity_score and on_topic_score per model.
Also includes trend lines (avg by step).
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def load_score_distributions(output_dir):
    """Load specificity and on-topic score distributions."""
    # Structure: {model: {'specificity': [], 'on_topic': [], 'by_step': {step: {'spec': [], 'topic': []}}}}
    model_scores = defaultdict(lambda: {'specificity': [], 'on_topic': [], 'by_step': defaultdict(lambda: {'spec': [], 'topic': []})})
    
    for file_path in glob.glob(str(output_dir / '*quality_judement.jsonl')):
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
                        quality = step_data.get('query_quality', {})
                        
                        spec_score = quality.get('specificity_score')
                        on_topic_score = quality.get('on_topic_score')
                        
                        if spec_score is not None:
                            model_scores[model_name]['specificity'].append(float(spec_score))
                            if step is not None and step <= 5:
                                model_scores[model_name]['by_step'][step]['spec'].append(float(spec_score))
                        
                        if on_topic_score is not None:
                            model_scores[model_name]['on_topic'].append(float(on_topic_score))
                            if step is not None and step <= 5:
                                model_scores[model_name]['by_step'][step]['topic'].append(float(on_topic_score))
                
                except json.JSONDecodeError:
                    continue
    
    return model_scores


def create_distribution_and_trend_plots(model_scores, output_path):
    """Create combined distribution and trend plots."""
    models = sorted(model_scores.keys())
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # === TOP LEFT: Violin plot for Specificity Score ===
    ax1 = fig.add_subplot(gs[0, 0])
    
    spec_data = [model_scores[m]['specificity'] for m in models]
    positions = np.arange(1, len(models) + 1)
    
    parts = ax1.violinplot(spec_data, positions=positions, widths=0.7,
                           showmeans=True, showmedians=True, showextrema=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor('#4c72b0')
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
    
    ax1.set_ylabel('Specificity Score', fontsize=11, fontweight='bold')
    ax1.set_title('Specificity Score Distribution by Model', fontsize=12, fontweight='bold')
    ax1.set_xticks(positions)
    ax1.set_xticklabels([m.replace('bedrock_', '').replace('openai_', '').replace('us.anthropic.', '')
                         for m in models], rotation=45, ha='right', fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # === TOP RIGHT: Violin plot for On-Topic Score ===
    ax2 = fig.add_subplot(gs[0, 1])
    
    topic_data = [model_scores[m]['on_topic'] for m in models]
    
    parts = ax2.violinplot(topic_data, positions=positions, widths=0.7,
                           showmeans=True, showmedians=True, showextrema=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor('#55a868')
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
    
    ax2.set_ylabel('On-Topic Score', fontsize=11, fontweight='bold')
    ax2.set_title('On-Topic Score Distribution by Model', fontsize=12, fontweight='bold')
    ax2.set_xticks(positions)
    ax2.set_xticklabels([m.replace('bedrock_', '').replace('openai_', '').replace('us.anthropic.', '')
                         for m in models], rotation=45, ha='right', fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # === BOTTOM LEFT: Specificity trend by step ===
    ax3 = fig.add_subplot(gs[1, 0])
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    for i, model in enumerate(models):
        by_step = model_scores[model]['by_step']
        steps = sorted(by_step.keys())
        
        spec_means = []
        for step in steps:
            scores = by_step[step]['spec']
            if scores:
                spec_means.append(np.mean(scores))
            else:
                spec_means.append(None)
        
        if any(s is not None for s in spec_means):
            short_name = model.replace('bedrock_', '').replace('openai_', '').replace('us.anthropic.', '')
            ax3.plot(steps, spec_means, marker='o', linewidth=2, markersize=6,
                    label=short_name, color=colors[i], alpha=0.8)
    
    ax3.set_xlabel('Step Number', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Avg Specificity Score', fontsize=11, fontweight='bold')
    ax3.set_title('Specificity Score Trend by Step', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)
    
    # === BOTTOM RIGHT: On-Topic trend by step ===
    ax4 = fig.add_subplot(gs[1, 1])
    
    for i, model in enumerate(models):
        by_step = model_scores[model]['by_step']
        steps = sorted(by_step.keys())
        
        topic_means = []
        for step in steps:
            scores = by_step[step]['topic']
            if scores:
                topic_means.append(np.mean(scores))
            else:
                topic_means.append(None)
        
        if any(s is not None for s in topic_means):
            short_name = model.replace('bedrock_', '').replace('openai_', '').replace('us.anthropic.', '')
            ax4.plot(steps, topic_means, marker='s', linewidth=2, markersize=6,
                    label=short_name, color=colors[i], alpha=0.8)
    
    ax4.set_xlabel('Step Number', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Avg On-Topic Score', fontsize=11, fontweight='bold')
    ax4.set_title('On-Topic Score Trend by Step', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.05)
    
    plt.suptitle('Query Quality Scores: Distribution and Trends\n(How do score distributions and trends vary by model?)',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved scores distribution and trends to {output_path}")
    plt.close()
    
    # Print statistics
    print("\n" + "="*80)
    print("SCORE DISTRIBUTIONS")
    print("="*80)
    print(f"{'Model':<50} {'Spec Mean':>12} {'Spec Std':>12} {'Topic Mean':>12} {'Topic Std':>12}")
    print("-"*100)
    
    for model in models:
        spec_scores = model_scores[model]['specificity']
        topic_scores = model_scores[model]['on_topic']
        
        spec_mean = np.mean(spec_scores) if spec_scores else 0
        spec_std = np.std(spec_scores) if spec_scores else 0
        topic_mean = np.mean(topic_scores) if topic_scores else 0
        topic_std = np.std(topic_scores) if topic_scores else 0
        
        print(f"{model:<50} {spec_mean:>12.3f} {spec_std:>12.3f} {topic_mean:>12.3f} {topic_std:>12.3f}")


def main():
    # Setup paths
    base_dir = Path(__file__).resolve().parents[2]
    output_dir = base_dir / "rag_analysis" / "output"
    plot_dir = base_dir / "rag_analysis" / "quality_rag_plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading score distributions...")
    model_scores = load_score_distributions(output_dir)
    
    if not model_scores:
        print("No score distribution data found!")
        return
    
    # Create plot
    output_path = plot_dir / "scores_distribution_trends.png"
    create_distribution_and_trend_plots(model_scores, output_path)


if __name__ == "__main__":
    main()
