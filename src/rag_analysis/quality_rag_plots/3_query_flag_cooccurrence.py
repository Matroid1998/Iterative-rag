"""
Plot 3: Query Flag Co-occurrence Matrix
Heatmap showing how often query flags (vague, over_broad, compound, off_topic) appear together.
Insight: Are certain query problems correlated?
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def load_query_flags(output_dir):
    """Load all query flag combinations."""
    flag_combinations = []
    
    for file_path in glob.glob(str(output_dir / '*quality_judement.jsonl')):
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    parsed = data.get('parsed_judgment', {})
                    
                    for step_data in parsed.get('per_step', []):
                        quality = step_data.get('query_quality', {})
                        
                        flags = {
                            'vague': quality.get('vague', False),
                            'over_broad': quality.get('over_broad', False),
                            'compound': quality.get('compound', False),
                            'off_topic': quality.get('off_topic', False),
                        }
                        
                        flag_combinations.append(flags)
                
                except json.JSONDecodeError:
                    continue
    
    return flag_combinations


def create_cooccurrence_matrix(flag_combinations):
    """Create co-occurrence matrix for query flags."""
    flags = ['vague', 'over_broad', 'compound', 'off_topic']
    n = len(flags)
    
    # Initialize matrix
    matrix = np.zeros((n, n))
    
    # Count co-occurrences
    for combo in flag_combinations:
        for i, flag1 in enumerate(flags):
            for j, flag2 in enumerate(flags):
                if combo.get(flag1) and combo.get(flag2):
                    matrix[i, j] += 1
    
    # Calculate percentages (of total steps)
    total_steps = len(flag_combinations)
    matrix_pct = (matrix / total_steps) * 100
    
    return matrix_pct, flags


def create_heatmap(matrix, flags, output_path):
    """Create heatmap of flag co-occurrences."""
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # Create heatmap
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=max(10, matrix.max()))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Co-occurrence Rate (%)', fontsize=12, fontweight='bold')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(flags)))
    ax.set_yticks(np.arange(len(flags)))
    ax.set_xticklabels(flags, fontsize=11)
    ax.set_yticklabels(flags, fontsize=11)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(flags)):
        for j in range(len(flags)):
            value = matrix[i, j]
            # Bold diagonal (individual occurrence rates)
            weight = 'bold' if i == j else 'normal'
            color = 'white' if value > matrix.max() * 0.5 else 'black'
            text = ax.text(j, i, f'{value:.1f}%',
                          ha="center", va="center", color=color,
                          fontsize=11, fontweight=weight)
    
    # Add title
    ax.set_title('Query Flag Co-occurrence Matrix\n(How often do query problems appear together?)',
                fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.set_xticks(np.arange(len(flags))-.5, minor=True)
    ax.set_yticks(np.arange(len(flags))-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved co-occurrence heatmap to {output_path}")
    plt.close()
    
    # Print correlation analysis
    print("\n" + "="*80)
    print("QUERY FLAG CO-OCCURRENCE ANALYSIS")
    print("="*80)
    
    print("\nIndividual flag occurrence rates (diagonal):")
    for i, flag in enumerate(flags):
        print(f"  {flag}: {matrix[i, i]:.1f}%")
    
    print("\nStrongest co-occurrences (off-diagonal):")
    correlations = []
    for i in range(len(flags)):
        for j in range(i + 1, len(flags)):
            if matrix[i, j] > 0.1:
                correlations.append((flags[i], flags[j], matrix[i, j]))
    
    correlations.sort(key=lambda x: x[2], reverse=True)
    for flag1, flag2, rate in correlations[:10]:
        print(f"  {flag1} + {flag2}: {rate:.1f}%")
    
    # Calculate conditional probabilities
    print("\nConditional probabilities:")
    for i in range(len(flags)):
        if matrix[i, i] > 0:
            print(f"\n  Given {flags[i]} is True:")
            for j in range(len(flags)):
                if i != j and matrix[i, j] > 0:
                    conditional = (matrix[i, j] / matrix[i, i]) * 100
                    print(f"    P({flags[j]}|{flags[i]}) = {conditional:.1f}%")


def main():
    # Setup paths
    base_dir = Path(__file__).resolve().parents[2]
    output_dir = base_dir / "rag_analysis" / "output"
    plot_dir = base_dir / "rag_analysis" / "quality_rag_plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading query flags...")
    flag_combinations = load_query_flags(output_dir)
    
    if not flag_combinations:
        print("No query flag data found!")
        return
    
    print(f"Loaded {len(flag_combinations)} query steps")
    
    # Create co-occurrence matrix
    matrix, flags = create_cooccurrence_matrix(flag_combinations)
    
    # Create plot
    output_path = plot_dir / "query_flag_cooccurrence.png"
    create_heatmap(matrix, flags, output_path)


if __name__ == "__main__":
    main()
