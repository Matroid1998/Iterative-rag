"""
Plot 1: Error Cascade Analysis (Sankey Diagram)

Shows the flow from Coverage Gap → Query Quality Issues → Hallucination.

Insight: Understand if coverage gaps lead to poor queries, which then lead to hallucinations.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.sankey import Sankey
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_system_plots.cross_system_utils import (
    load_all_judgments, create_merged_dataset, normalize_model_name,
    has_coverage_gap, has_poor_query_quality, has_composition_failure
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'rag_analysis' / 'output'
PLOT_DIR = Path(__file__).resolve().parent


def create_sankey_data(merged_data, model_filter=None):
    """Create flow data for Sankey diagram."""
    if model_filter:
        data = [r for r in merged_data if normalize_model_name(r['model']) == model_filter]
    else:
        data = merged_data
    
    # Count flows
    total = len(data)
    
    # Stage 1: Coverage
    has_gap = [r for r in data if has_coverage_gap(r.get('coverage', {}))]
    no_gap = [r for r in data if not has_coverage_gap(r.get('coverage', {}))]
    
    # Stage 2: Query Quality (from gap and no-gap)
    gap_poor_query = [r for r in has_gap if has_poor_query_quality(r.get('quality', {}))]
    gap_good_query = [r for r in has_gap if not has_poor_query_quality(r.get('quality', {}))]
    
    no_gap_poor_query = [r for r in no_gap if has_poor_query_quality(r.get('quality', {}))]
    no_gap_good_query = [r for r in no_gap if not has_poor_query_quality(r.get('quality', {}))]
    
    # Stage 3: Hallucination (from each query quality state)
    gap_poor_hall = [r for r in gap_poor_query if has_composition_failure(r.get('hallucination', {}))]
    gap_poor_ok = [r for r in gap_poor_query if not has_composition_failure(r.get('hallucination', {}))]
    
    gap_good_hall = [r for r in gap_good_query if has_composition_failure(r.get('hallucination', {}))]
    gap_good_ok = [r for r in gap_good_query if not has_composition_failure(r.get('hallucination', {}))]
    
    no_gap_poor_hall = [r for r in no_gap_poor_query if has_composition_failure(r.get('hallucination', {}))]
    no_gap_poor_ok = [r for r in no_gap_poor_query if not has_composition_failure(r.get('hallucination', {}))]
    
    no_gap_good_hall = [r for r in no_gap_good_query if has_composition_failure(r.get('hallucination', {}))]
    no_gap_good_ok = [r for r in no_gap_good_query if not has_composition_failure(r.get('hallucination', {}))]
    
    return {
        'total': total,
        'has_gap': len(has_gap),
        'no_gap': len(no_gap),
        'gap_poor_query': len(gap_poor_query),
        'gap_good_query': len(gap_good_query),
        'no_gap_poor_query': len(no_gap_poor_query),
        'no_gap_good_query': len(no_gap_good_query),
        'gap_poor_hall': len(gap_poor_hall),
        'gap_poor_ok': len(gap_poor_ok),
        'gap_good_hall': len(gap_good_hall),
        'gap_good_ok': len(gap_good_ok),
        'no_gap_poor_hall': len(no_gap_poor_hall),
        'no_gap_poor_ok': len(no_gap_poor_ok),
        'no_gap_good_hall': len(no_gap_good_hall),
        'no_gap_good_ok': len(no_gap_good_ok),
    }


def plot_custom_sankey(data, model_name):
    """Create a custom Sankey-style flow diagram."""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    total = data['total']
    
    # Define positions for 3 stages
    stage_x = [0.1, 0.5, 0.9]
    
    # Stage 1: Coverage (left)
    gap_height = data['has_gap'] / total
    no_gap_height = data['no_gap'] / total
    
    # Stage 2: Query Quality (middle)
    gap_poor_height = data['gap_poor_query'] / total
    gap_good_height = data['gap_good_query'] / total
    no_gap_poor_height = data['no_gap_poor_query'] / total
    no_gap_good_height = data['no_gap_good_query'] / total
    
    # Stage 3: Hallucination (right)
    all_hall = (data['gap_poor_hall'] + data['gap_good_hall'] + 
                data['no_gap_poor_hall'] + data['no_gap_good_hall'])
    all_ok = (data['gap_poor_ok'] + data['gap_good_ok'] + 
              data['no_gap_poor_ok'] + data['no_gap_good_ok'])
    
    hall_height = all_hall / total
    ok_height = all_ok / total
    
    # Colors
    gap_color = '#e74c3c'
    no_gap_color = '#2ecc71'
    poor_query_color = '#f39c12'
    good_query_color = '#3498db'
    hall_color = '#c0392b'
    ok_color = '#27ae60'
    
    # Draw Stage 1: Coverage
    y_offset = 0.1
    gap_y = y_offset
    ax.add_patch(mpatches.Rectangle((stage_x[0]-0.05, gap_y), 0.1, gap_height, 
                                     facecolor=gap_color, alpha=0.7, edgecolor='black'))
    ax.text(stage_x[0], gap_y + gap_height/2, f'Coverage Gap\n{data["has_gap"]} ({100*gap_height:.1f}%)',
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    no_gap_y = gap_y + gap_height + 0.02
    ax.add_patch(mpatches.Rectangle((stage_x[0]-0.05, no_gap_y), 0.1, no_gap_height,
                                     facecolor=no_gap_color, alpha=0.7, edgecolor='black'))
    ax.text(stage_x[0], no_gap_y + no_gap_height/2, f'No Gap\n{data["no_gap"]} ({100*no_gap_height:.1f}%)',
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Draw Stage 2: Query Quality
    query_y_start = y_offset
    
    # Gap → Poor Query
    gap_poor_y = query_y_start
    if data['gap_poor_query'] > 0:
        ax.add_patch(mpatches.Rectangle((stage_x[1]-0.05, gap_poor_y), 0.1, gap_poor_height,
                                         facecolor=poor_query_color, alpha=0.7, edgecolor='black'))
        ax.text(stage_x[1], gap_poor_y + gap_poor_height/2, 
                f'Poor Query\n{data["gap_poor_query"]}',
                ha='center', va='center', fontsize=8, fontweight='bold')
        # Draw flow
        draw_flow(ax, stage_x[0]+0.05, gap_y + gap_height/2, 
                 stage_x[1]-0.05, gap_poor_y + gap_poor_height/2, gap_poor_height, gap_color, 0.2)
    
    # Gap → Good Query
    gap_good_y = gap_poor_y + gap_poor_height + 0.01
    if data['gap_good_query'] > 0:
        ax.add_patch(mpatches.Rectangle((stage_x[1]-0.05, gap_good_y), 0.1, gap_good_height,
                                         facecolor=good_query_color, alpha=0.7, edgecolor='black'))
        ax.text(stage_x[1], gap_good_y + gap_good_height/2,
                f'Good Query\n{data["gap_good_query"]}',
                ha='center', va='center', fontsize=8, fontweight='bold')
        draw_flow(ax, stage_x[0]+0.05, gap_y + gap_height/2,
                 stage_x[1]-0.05, gap_good_y + gap_good_height/2, gap_good_height, gap_color, 0.2)
    
    # No Gap → Poor Query
    no_gap_poor_y = gap_good_y + gap_good_height + 0.01
    if data['no_gap_poor_query'] > 0:
        ax.add_patch(mpatches.Rectangle((stage_x[1]-0.05, no_gap_poor_y), 0.1, no_gap_poor_height,
                                         facecolor=poor_query_color, alpha=0.7, edgecolor='black'))
        ax.text(stage_x[1], no_gap_poor_y + no_gap_poor_height/2,
                f'Poor Query\n{data["no_gap_poor_query"]}',
                ha='center', va='center', fontsize=8, fontweight='bold')
        draw_flow(ax, stage_x[0]+0.05, no_gap_y + no_gap_height/2,
                 stage_x[1]-0.05, no_gap_poor_y + no_gap_poor_height/2, no_gap_poor_height, no_gap_color, 0.2)
    
    # No Gap → Good Query
    no_gap_good_y = no_gap_poor_y + no_gap_poor_height + 0.01
    if data['no_gap_good_query'] > 0:
        ax.add_patch(mpatches.Rectangle((stage_x[1]-0.05, no_gap_good_y), 0.1, no_gap_good_height,
                                         facecolor=good_query_color, alpha=0.7, edgecolor='black'))
        ax.text(stage_x[1], no_gap_good_y + no_gap_good_height/2,
                f'Good Query\n{data["no_gap_good_query"]}',
                ha='center', va='center', fontsize=8, fontweight='bold')
        draw_flow(ax, stage_x[0]+0.05, no_gap_y + no_gap_height/2,
                 stage_x[1]-0.05, no_gap_good_y + no_gap_good_height/2, no_gap_good_height, no_gap_color, 0.2)
    
    # Draw Stage 3: Hallucination outcome
    # Aggregate all flows to final outcomes
    hall_y = y_offset
    ax.add_patch(mpatches.Rectangle((stage_x[2]-0.05, hall_y), 0.1, hall_height,
                                     facecolor=hall_color, alpha=0.7, edgecolor='black'))
    ax.text(stage_x[2], hall_y + hall_height/2,
            f'Composition\nFailure\n{all_hall} ({100*hall_height:.1f}%)',
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    ok_y = hall_y + hall_height + 0.02
    ax.add_patch(mpatches.Rectangle((stage_x[2]-0.05, ok_y), 0.1, ok_height,
                                     facecolor=ok_color, alpha=0.7, edgecolor='black'))
    ax.text(stage_x[2], ok_y + ok_height/2,
            f'OK\n{all_ok} ({100*ok_height:.1f}%)',
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Stage labels
    ax.text(stage_x[0], 0.05, 'Coverage', ha='center', va='center', 
            fontsize=14, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax.text(stage_x[1], 0.05, 'Query Quality', ha='center', va='center',
            fontsize=14, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax.text(stage_x[2], 0.05, 'Outcome', ha='center', va='center',
            fontsize=14, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    title = f'Error Cascade Analysis: Coverage → Query → Hallucination\n{model_name} (n={total})'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    return fig


def draw_flow(ax, x1, y1, x2, y2, height, color, alpha):
    """Draw a flow band between two points."""
    # Simple polygon for flow
    y_offset = height / 2
    verts = [
        (x1, y1 - y_offset),
        (x1, y1 + y_offset),
        (x2, y2 + y_offset),
        (x2, y2 - y_offset),
        (x1, y1 - y_offset),
    ]
    poly = mpatches.Polygon(verts, facecolor=color, alpha=alpha, edgecolor='none')
    ax.add_patch(poly)


def main():
    """Generate error cascade Sankey diagram."""
    cov_records, qual_records, hall_records = load_all_judgments(OUTPUT_DIR)
    merged = create_merged_dataset(cov_records, qual_records, hall_records)
    
    # Filter to records with all three judgments
    complete = [r for r in merged if 'quality' in r and 'hallucination' in r]
    
    print(f"\nTotal merged records: {len(merged)}")
    print(f"Records with all 3 judgments: {len(complete)}")
    
    # Get unique models
    models = list(set(normalize_model_name(r['model']) for r in complete))
    models.sort()
    
    # Create plot for each model (or first model if many)
    for model in models[:1]:  # Start with first model
        print(f"\nAnalyzing {model}...")
        data = create_sankey_data(complete, model)
        
        # Print statistics
        print(f"\nCoverage Stage:")
        print(f"  Has Gap: {data['has_gap']} ({100*data['has_gap']/data['total']:.1f}%)")
        print(f"  No Gap: {data['no_gap']} ({100*data['no_gap']/data['total']:.1f}%)")
        
        print(f"\nQuery Quality Stage:")
        print(f"  Gap → Poor Query: {data['gap_poor_query']}")
        print(f"  Gap → Good Query: {data['gap_good_query']}")
        print(f"  No Gap → Poor Query: {data['no_gap_poor_query']}")
        print(f"  No Gap → Good Query: {data['no_gap_good_query']}")
        
        print(f"\nOutcome Stage:")
        total_hall = (data['gap_poor_hall'] + data['gap_good_hall'] + 
                     data['no_gap_poor_hall'] + data['no_gap_good_hall'])
        total_ok = (data['gap_poor_ok'] + data['gap_good_ok'] + 
                   data['no_gap_poor_ok'] + data['no_gap_good_ok'])
        print(f"  Composition Failure: {total_hall} ({100*total_hall/data['total']:.1f}%)")
        print(f"  OK: {total_ok} ({100*total_ok/data['total']:.1f}%)")
        
        # Key cascade statistics
        if data['has_gap'] > 0:
            gap_to_poor = 100 * data['gap_poor_query'] / data['has_gap']
            print(f"\nCascade: Gap → Poor Query: {gap_to_poor:.1f}%")
        
        if data['gap_poor_query'] > 0:
            poor_to_hall = 100 * data['gap_poor_hall'] / data['gap_poor_query']
            print(f"Cascade: Gap+Poor Query → Hallucination: {poor_to_hall:.1f}%")
        
        fig = plot_custom_sankey(data, model)
        
        plt.tight_layout()
        output_path = PLOT_DIR / f'1_error_cascade_{model.replace(" ", "_")}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved: {output_path}")
        plt.close()


if __name__ == '__main__':
    main()
