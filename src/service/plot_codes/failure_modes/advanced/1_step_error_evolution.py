"""
Plot 1: Step-by-Step Error Evolution (Alluvial Diagram)
Shows how query quality evolves from step 1 → 2 → 3
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict
from advanced_utils import (
    load_all_judgments, 
    create_merged_dataset,
    get_query_flags,
    get_quality_category
)


def create_alluvial_plot(merged_data, model_name='GPT-5'):
    """Create alluvial diagram showing query quality trajectory."""
    
    # Filter for specific model
    model_data = [rec for rec in merged_data if model_name.lower() in rec['model'].lower()]
    
    if not model_data:
        print(f"No data for {model_name}")
        return
    
    # Track transitions: step1_cat -> step2_cat -> step3_cat
    transitions = defaultdict(int)
    
    for rec in model_data:
        quality = rec.get('quality', {})
        per_step = quality.get('per_step', [])
        
        if len(per_step) < 1:
            continue
        
        # Get categories for first 3 steps
        categories = []
        for i in range(min(3, len(per_step))):
            step = per_step[i]
            flags = get_query_flags(step)
            cat = get_quality_category(flags)
            categories.append(cat)
        
        # Pad if needed
        while len(categories) < 3:
            categories.append('done')
        
        # Record transition
        transition_key = tuple(categories)
        transitions[transition_key] += 1
    
    # Aggregate flows
    step1_counts = defaultdict(int)
    step2_counts = defaultdict(int)
    step3_counts = defaultdict(int)
    
    step1_to_2 = defaultdict(lambda: defaultdict(int))
    step2_to_3 = defaultdict(lambda: defaultdict(int))
    
    for (s1, s2, s3), count in transitions.items():
        step1_counts[s1] += count
        step2_counts[s2] += count
        step3_counts[s3] += count
        step1_to_2[s1][s2] += count
        step2_to_3[s2][s3] += count
    
    # Define category order and colors
    categories = ['clean', 'anchored', 'compound', 'poor', 'off_topic', 'done']
    colors = {
        'clean': '#2ecc71',
        'anchored': '#3498db',
        'compound': '#f39c12',
        'poor': '#e67e22',
        'off_topic': '#e74c3c',
        'done': '#95a5a6'
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define positions
    step_x = [0, 1, 2]
    total_height = 100
    
    # Draw nodes for each step
    def draw_column(x_pos, counts, label):
        total = sum(counts.values())
        if total == 0:
            return {}
        
        y_positions = {}
        current_y = 0
        
        for cat in categories:
            count = counts.get(cat, 0)
            if count == 0:
                continue
            
            height = (count / total) * total_height
            y_center = current_y + height / 2
            
            # Draw rectangle
            rect = mpatches.Rectangle(
                (x_pos - 0.08, current_y),
                0.16,
                height,
                facecolor=colors[cat],
                edgecolor='black',
                linewidth=1.5,
                alpha=0.7
            )
            ax.add_patch(rect)
            
            # Add label
            if height > 3:  # Only label if big enough
                ax.text(
                    x_pos, y_center, 
                    f"{cat}\n{count}",
                    ha='center', va='center',
                    fontsize=9,
                    fontweight='bold'
                )
            
            y_positions[cat] = (current_y, current_y + height)
            current_y += height
        
        # Add step label
        ax.text(x_pos, -8, label, ha='center', va='top', fontsize=12, fontweight='bold')
        
        return y_positions
    
    # Draw flows between nodes
    def draw_flows(x1, y_pos1, x2, y_pos2, flows):
        for cat1, targets in flows.items():
            if cat1 not in y_pos1:
                continue
            
            for cat2, count in targets.items():
                if cat2 not in y_pos2 or count == 0:
                    continue
                
                # Calculate positions
                y1_start, y1_end = y_pos1[cat1]
                y2_start, y2_end = y_pos2[cat2]
                
                # Proportion of source category going to target
                total_from_cat1 = sum(flows[cat1].values())
                prop = count / total_from_cat1
                
                # Source segment
                seg_height = (y1_end - y1_start) * prop
                y1_mid = y1_start + seg_height / 2
                y1_start_seg = y1_start
                y1_end_seg = y1_start + seg_height
                
                # Update for next iteration
                flows[cat1][cat2] = (y1_start_seg, y1_end_seg)
                y_pos1[cat1] = (y1_end_seg, y1_end)
                
                # Target segment
                total_to_cat2 = sum(flow.get(cat2, 0) for flow in flows.values())
                if cat2 not in hasattr(draw_flows, 'target_y_tracker'):
                    if not hasattr(draw_flows, 'target_y_tracker'):
                        draw_flows.target_y_tracker = {}
                    draw_flows.target_y_tracker[cat2] = y2_start
                
                y2_seg_height = (y2_end - y2_start) * (count / (step2_counts[cat2] if x2 == 1 else step3_counts[cat2]))
                y2_start_seg = draw_flows.target_y_tracker[cat2]
                y2_end_seg = y2_start_seg + y2_seg_height
                draw_flows.target_y_tracker[cat2] = y2_end_seg
                
                # Draw bezier curve
                vertices = [
                    (x1 + 0.08, (y1_start_seg + y1_end_seg) / 2),
                    (x1 + 0.4, (y1_start_seg + y1_end_seg) / 2),
                    (x2 - 0.4, (y2_start_seg + y2_end_seg) / 2),
                    (x2 - 0.08, (y2_start_seg + y2_end_seg) / 2),
                ]
                
                codes = [
                    mpatches.Path.MOVETO,
                    mpatches.Path.CURVE4,
                    mpatches.Path.CURVE4,
                    mpatches.Path.CURVE4,
                ]
                
                # Draw thick and thin edges for ribbon effect
                for y_offset in [seg_height/2, -seg_height/2]:
                    verts_offset = [
                        (v[0], v[1] + y_offset) for v in vertices
                    ]
                    path = mpatches.Path(verts_offset, codes)
                    patch = mpatches.PathPatch(
                        path,
                        facecolor='none',
                        edgecolor=colors[cat1],
                        linewidth=max(0.5, seg_height / 4),
                        alpha=0.3
                    )
                    ax.add_patch(patch)
    
    # Draw columns
    y_pos_1 = draw_column(step_x[0], step1_counts, "Step 1")
    y_pos_2 = draw_column(step_x[1], step2_counts, "Step 2")
    y_pos_3 = draw_column(step_x[2], step3_counts, "Step 3")
    
    # Draw flows (simplified version)
    # Due to complexity, we'll draw simple connecting lines
    for cat1, targets in step1_to_2.items():
        if cat1 not in y_pos_1:
            continue
        y1_start, y1_end = y_pos_1[cat1]
        y1_mid = (y1_start + y1_end) / 2
        
        for cat2, count in targets.items():
            if cat2 not in y_pos_2 or count == 0:
                continue
            y2_start, y2_end = y_pos_2[cat2]
            y2_mid = (y2_start + y2_end) / 2
            
            # Draw line
            ax.plot(
                [step_x[0] + 0.08, step_x[1] - 0.08],
                [y1_mid, y2_mid],
                color=colors[cat1],
                alpha=0.2,
                linewidth=max(0.5, count / 10)
            )
    
    for cat2, targets in step2_to_3.items():
        if cat2 not in y_pos_2:
            continue
        y2_start, y2_end = y_pos_2[cat2]
        y2_mid = (y2_start + y2_end) / 2
        
        for cat3, count in targets.items():
            if cat3 not in y_pos_3 or count == 0:
                continue
            y3_start, y3_end = y_pos_3[cat3]
            y3_mid = (y3_start + y3_end) / 2
            
            # Draw line
            ax.plot(
                [step_x[1] + 0.08, step_x[2] - 0.08],
                [y2_mid, y3_mid],
                color=colors[cat2],
                alpha=0.2,
                linewidth=max(0.5, count / 10)
            )
    
    # Formatting
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(-10, total_height + 5)
    ax.set_aspect('auto')
    ax.axis('off')
    
    plt.title(
        f'Query Quality Evolution: Step-by-Step Trajectory ({model_name})',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors[cat], edgecolor='black', label=cat.replace('_', ' ').title())
        for cat in categories
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        frameon=True,
        fontsize=10
    )
    
    plt.tight_layout()
    return fig


def main():
    output_dir = Path(__file__).resolve().parents[5] / "data" / "results" / "failure_modes"
    
    print("Loading all judgments...")
    coverage, quality, hallucination = load_all_judgments(output_dir)
    
    print(f"Loaded: {len(coverage)} coverage, {len(quality)} quality, {len(hallucination)} hallucination")
    
    print("Merging datasets...")
    merged_data = create_merged_dataset(coverage, quality, hallucination)
    print(f"Merged: {len(merged_data)} records")
    
    print("Creating alluvial plot...")
    fig = create_alluvial_plot(merged_data)
    
    if fig:
        output_path = Path(__file__).resolve().parents[5] / "data" / "plots" / "failure_modes" / "advanced" / "1_step_error_evolution.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    else:
        print("✗ Failed to create plot")


if __name__ == '__main__':
    main()
