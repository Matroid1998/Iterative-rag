"""
Plot 1: Late Hit Timing Distribution - Aggregate
Aggregate heatmaps showing when each hop is first retrieved across all models.
Creates two plots: one for correct answers, one for incorrect answers.
Each plot has 4 subplots (1-hop, 2-hop, 3-hop, 4-hop questions).
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_system_plots.cross_system_utils import extract_model_from_filename


def load_question_hop_counts(qa_file_path):
    """Load question to hop count mapping from chemrxiv_qa.json."""
    with open(qa_file_path, 'r') as f:
        qa_data = json.load(f)
    
    question_hops = {}
    for item in qa_data:
        question = item.get('q', '')
        path = item.get('path', [])
        if question and path:
            question_hops[question] = len(path)
    
    return question_hops


def load_late_hit_data_aggregate(output_dir, question_hops, correctness_filter):
    """Load late hit timing data aggregated across all models.
    
    Args:
        output_dir: Directory containing judgment files
        question_hops: Dict mapping questions to hop counts
        correctness_filter: 'correct' or 'incorrect'
    
    Returns:
        Structure: {num_hops: {(hop_index, first_hit_step): count}}
    """
    aggregate_data = defaultdict(lambda: defaultdict(int))
    
    # Load correctness information from hallucination judgments
    correctness_map = {}  # {(model, question): is_correct}
    for file_path in glob.glob(str(output_dir / '*hallucination_judgment.jsonl')):
        model_name = extract_model_from_filename(Path(file_path).name)
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    question = data.get('question', '')
                    parsed = data.get('parsed_judgment', {})
                    composition = parsed.get('composition_and_faithfulness', {})
                    # Answer is correct if sufficiency_score is 1.0 and no composition failure
                    is_correct = (composition.get('sufficiency_score_est', 0) == 1.0 and 
                                 not composition.get('composition_failure', True))
                    correctness_map[(model_name, question)] = is_correct
                except json.JSONDecodeError:
                    continue
    
    # Load coverage gap data
    for file_path in glob.glob(str(output_dir / '*coverage_gap_judgments.jsonl')):
        model_name = extract_model_from_filename(Path(file_path).name)
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    question = data.get('question', '')
                    
                    # Get the actual number of hops from chemrxiv_qa.json
                    num_hops = question_hops.get(question)
                    if num_hops is None:
                        continue
                    
                    # Filter by correctness
                    is_correct = correctness_map.get((model_name, question), False)
                    if correctness_filter == 'correct' and not is_correct:
                        continue
                    if correctness_filter == 'incorrect' and is_correct:
                        continue
                    
                    parsed = data.get('parsed_judgment', {})
                    late_hit = parsed.get('late_hit_per_hop', {})
                    per_hop = late_hit.get('per_hop', [])
                    
                    if not per_hop:
                        continue
                    
                    # Record each hop's first hit step
                    for hop_data in per_hop:
                        hop_index = hop_data.get('hop_index')
                        first_hit_step = hop_data.get('first_hit_step')
                        
                        # Skip if first_hit_step is None (means hop was never retrieved)
                        if hop_index is not None and first_hit_step is not None:
                            aggregate_data[num_hops][(hop_index, first_hit_step)] += 1
                
                except json.JSONDecodeError:
                    continue
    
    return aggregate_data


def create_aggregate_heatmap_plot(hops_data, output_path, correctness_label):
    """Create 4-subplot heatmap plot aggregated across all models.
    
    Args:
        hops_data: Data structure with hop timing
        output_path: Path to save the plot
        correctness_label: "Correct" or "Incorrect"
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    # Process each hop count (1-4)
    for idx, num_hops in enumerate([1, 2, 3, 4]):
        ax = axes[idx]
        hop_step_data = hops_data.get(num_hops, {})
        
        if not hop_step_data:
            # No data for this hop count
            ax.text(0.5, 0.5, f'No data for {num_hops}-hop questions',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title(f'{num_hops}-Hop Questions', fontsize=12, fontweight='bold')
            ax.axis('off')
            continue
        
        # Determine matrix dimensions
        max_hop = max(hop for hop, step in hop_step_data.keys())
        max_step = max(step for hop, step in hop_step_data.keys())
        
        # Create matrix: rows = steps, cols = hops
        matrix = np.zeros((max_step, max_hop))
        
        for (hop_index, first_hit_step), count in hop_step_data.items():
            # hop_index is 1-indexed, first_hit_step is 1-indexed
            if hop_index <= max_hop and first_hit_step <= max_step:
                matrix[first_hit_step - 1, hop_index - 1] = count
        
        # Create heatmap
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        
        # Set ticks
        ax.set_xticks(np.arange(max_hop))
        ax.set_yticks(np.arange(max_step))
        ax.set_xticklabels([f'Hop {i+1}' for i in range(max_hop)], fontsize=10)
        ax.set_yticklabels([f'Step {i+1}' for i in range(max_step)], fontsize=9)
        
        # Add text annotations
        for i in range(max_step):
            for j in range(max_hop):
                value = matrix[i, j]
                if value > 0:
                    text_color = 'white' if value > matrix.max() / 2 else 'black'
                    ax.text(j, i, f'{int(value)}',
                           ha='center', va='center', color=text_color,
                           fontsize=8, fontweight='bold')
        
        # Labels and title
        ax.set_xlabel('Hop Index', fontsize=11, fontweight='bold')
        ax.set_ylabel('First Hit Step', fontsize=11, fontweight='bold')
        ax.set_title(f'{num_hops}-Hop Questions (n={sum(hop_step_data.values())} observations)',
                    fontsize=12, fontweight='bold', pad=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Number of Cases', fontsize=10, fontweight='bold')
    
    # Overall title with correctness label
    title = f'Hop Retrieval Timing: All Models Aggregated ({correctness_label} Answers)'
    title += '\n(At which step is each hop first retrieved?)'
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved aggregate heatmap plot to {output_path}")
    plt.close()


def main():
    # Setup paths
    base_dir = Path(__file__).resolve().parents[2]
    output_dir = base_dir / "rag_analysis" / "output"
    plot_dir = base_dir / "rag_analysis" / "cov_rag_plots"
    qa_file_path = base_dir.parent / "data" / "corpus" / "chemrxiv_qa.json"
    plot_dir.mkdir(exist_ok=True)
    
    # Load question hop counts from chemrxiv_qa.json
    print("Loading question hop counts from chemrxiv_qa.json...")
    question_hops = load_question_hop_counts(qa_file_path)
    print(f"Loaded hop counts for {len(question_hops)} questions")
    
    # Generate aggregate plots for correct and incorrect answers
    for correctness_type in ['correct', 'incorrect']:
        label = correctness_type.capitalize()
        
        print(f"\n{'='*60}")
        print(f"Processing {label} Answers (All Models Aggregated)")
        print('='*60)
        
        # Load data
        print(f"Loading late hit timing data for {label.lower()} answers...")
        aggregate_data = load_late_hit_data_aggregate(output_dir, question_hops, correctness_type)
        
        if not aggregate_data:
            print(f"No late hit data found for {label.lower()} answers!")
            continue
        
        # Create aggregate plot
        output_path = plot_dir / f"late_hit_timing_distribution_All_Models_{correctness_type}.png"
        create_aggregate_heatmap_plot(aggregate_data, output_path, label)
        
        # Print summary
        print(f"\nAll Models Aggregated - {label} statistics:")
        total_observations = 0
        for num_hops in [1, 2, 3, 4]:
            hop_data = aggregate_data.get(num_hops, {})
            if hop_data:
                total = sum(hop_data.values())
                total_observations += total
                print(f"  {num_hops}-hop questions: {total} observations")
        print(f"  Total: {total_observations} observations")
    
    print("\nDone! Generated aggregate late hit timing distribution plots.")


if __name__ == "__main__":
    main()
