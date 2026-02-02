#!/usr/bin/env python3
"""
Create a scatter plot of cost vs accuracy for all models.
Cost is calculated based on total input/output tokens and model-specific pricing.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def get_base_path() -> Path:
    """Get the base path for the project."""
    return Path(__file__).resolve().parents[2]


def load_model_data():
    """
    Load data from all CSV files and calculate costs.
    
    Returns:
        DataFrame with columns: Model, Accuracy, Cost, Input_Tokens, Output_Tokens
    """
    base = get_base_path()
    results_dir = base / "src" / "results" / "new_results_csv"
    
    # Pricing per 1M tokens (input, output)
    pricing = {
        "Mistral Large": (4.00, 12.00),
        "Claude 3.7 + Reasoning": (3.00, 15.00),
        "Claude 3.7 Sonnet": (3.00, 15.00),
        "DeepSeek R1": (1.35, 5.40),
        "Llama 3.3 70B": (0.72, 0.72),
        "GPT-4o": (2.50, 10.00),
        "GPT-5": (1.25, 10.00),
        "Claude Sonnet 4.5": (3.00, 15.00),
        "Gemini 2.5 Pro": (1.25, 10.00),
        "Grok 4 Fast": (0.20, 0.50),
        "GLM 4.6": (0.40, 1.75),
    }
    
    # Map file names to display names
    file_to_model = {
        "bedrock_mistral.mistral-large-2402-v1_0": "Mistral Large",
        "claude-3-7-sonnet-20250219-v1_0-reasoning": "Claude 3.7 + Reasoning",
        "claude-3-7-sonnet-20250219-v1_0.csv": "Claude 3.7 Sonnet",
        "deepseek.r1-v1_0-reasoning": "DeepSeek R1",
        "llama3-3-70b-instruct": "Llama 3.3 70B",
        "gpt-4o": "GPT-4o",
        "gpt-5": "GPT-5", # Catch all gpt-5
        "claude-sonnet-4.5": "Claude Sonnet 4.5",
        "gemini-2.5-pro": "Gemini 2.5 Pro",
        "grok-4-fast": "Grok 4 Fast",
        "glm-4.6": "GLM 4.6",
    }
    
    data = []
    
    for csv_file in results_dir.glob("*.csv"):
        # Explicit skip for GPT-5.1
        if "gpt-5.1" in csv_file.name:
            continue
            
        # Find matching model name
        model_name = None
        for pattern, name in file_to_model.items():
            if pattern in csv_file.name:
                model_name = name
                break
        
        if not model_name:
            print(f"Warning: Could not match {csv_file.name} to a model")
            continue
        
        # Load CSV
        df = pd.read_csv(csv_file)
        
        if len(df) == 0:
            print(f"Warning: Empty file {csv_file.name}")
            continue
        
        # Extract data (first row)
        row = df.iloc[0]
        accuracy = row['Accuracy (%)']
        total_input_tokens = row['Total Input Tokens']
        total_output_tokens = row['Total Output Tokens']
        
        # Get pricing
        if model_name not in pricing:
            print(f"Warning: No pricing for {model_name}")
            continue
        
        input_price, output_price = pricing[model_name]
        
        # Calculate cost (pricing is per 1M tokens)
        cost = (total_input_tokens / 1_000_000 * input_price + 
                total_output_tokens / 1_000_000 * output_price)
        
        data.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Cost': cost,
            'Input_Tokens': total_input_tokens,
            'Output_Tokens': total_output_tokens,
        })
        
        print(f"Loaded {model_name}: Accuracy={accuracy:.2f}%, Cost=${cost:.2f}")
    
    return pd.DataFrame(data)


def create_scatter_plot(df: pd.DataFrame, output_dir: Path):
    """Create and save the cost vs accuracy scatter plot."""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create color map based on model families
    # Define sorted order to match anchor_carry_by_step.png colors
    # Based on sorting of raw keys (files starting with 2_, bedrock_, openai_, openrouter_)
    sorted_models = [
        "Llama 3.3 70B",            # 2_bedrock...
        "Mistral Large",            # bedrock_mistral...
        "Claude 3.7 Sonnet",        # bedrock_us.anthropic...v1:0
        "Claude 3.7 + Reasoning",   # bedrock_us.anthropic...v1:0-reasoning (prefix comes first)
        "DeepSeek R1",              # bedrock_us.deepseek...
        "GPT-4o",                   # openai_gpt-4o
        "GPT-5",                    # openai_gpt-5 (Takes GPT-5.1's old spot/color)
        "Claude Sonnet 4.5",        # openrouter_anthropic...
        "Gemini 2.5 Pro",           # openrouter_google...
        "Grok 4 Fast",              # openrouter_x-ai...
        "GLM 4.6",                  # openrouter_z-ai...
    ]
    
    # Generate tab10 colors
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Create color map
    color_map = {}
    for i, model in enumerate(sorted_models):
        color_map[model] = colors[i % 10]
    
    # Plot each model
    for idx, row in df.iterrows():
        model = row['Model']
        color = color_map.get(model, '#808080')
        
        # Fallback if model not in sorted list (shouldn't happen with current data)
        if model not in color_map and model not in sorted_models:
             # Find index if it was in the list? Or just append?
             pass

        ax.scatter(row['Cost'], row['Accuracy'], 
                  s=200, c=[color], alpha=0.7, 
                  edgecolors='black', linewidth=1.5,
                  label=row['Model'])
        
        # Add model name labels with smart positioning
        # Simple text, larger font, no highlight/bbox
        # Move label to the right of the marker (s=200 -> radius ~7 points)
        # Exception: Claude 3.7 + Reasoning label to the left
        
        if row['Model'] == 'Claude 3.7 + Reasoning':
            offset_x = -12
            offset_y = 0
            ha = 'right'
        else:
            offset_x = 12
            offset_y = 0
            ha = 'left'
        
        ax.annotate(row['Model'], 
                   (row['Cost'], row['Accuracy']),
                   xytext=(offset_x, offset_y),
                   textcoords='offset points',
                   fontsize=12,              # Increased font size
                   fontweight='bold',
                   color='black',            # Neutral color
                   ha=ha,
                   va='center')
    
    # Styling
    ax.set_xlabel('Total Cost ($)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Model Cost vs Accuracy', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=min(df['Accuracy']) - 2, top=max(df['Accuracy']) + 2)
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / "cost_vs_accuracy_scatter.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")
    
    plt.close()
    
    # Create a summary table
    df_sorted = df.sort_values('Accuracy', ascending=False)
    
    print("\n" + "=" * 80)
    print("MODEL RANKINGS BY ACCURACY")
    print("=" * 80)
    print(f"{'Rank':<6} {'Model':<30} {'Accuracy':<12} {'Cost':<10} {'$/1% Acc':<10}")
    print("-" * 80)
    
    for i, (idx, row) in enumerate(df_sorted.iterrows(), 1):
        cost_per_acc = row['Cost'] / row['Accuracy']
        print(f"{i:<6} {row['Model']:<30} {row['Accuracy']:>6.2f}%     ${row['Cost']:>7.2f}   ${cost_per_acc:>7.4f}")
    
    print("\n" + "=" * 80)
    print("MODEL RANKINGS BY COST")
    print("=" * 80)
    
    df_sorted = df.sort_values('Cost')
    print(f"{'Rank':<6} {'Model':<30} {'Cost':<12} {'Accuracy':<12}")
    print("-" * 80)
    
    for i, (idx, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"{i:<6} {row['Model']:<30} ${row['Cost']:>7.2f}     {row['Accuracy']:>6.2f}%")
    
    print("\n" + "=" * 80)
    print("BEST VALUE MODELS (Highest Accuracy per Dollar)")
    print("=" * 80)
    
    df['Acc_per_Dollar'] = df['Accuracy'] / df['Cost']
    df_sorted = df.sort_values('Acc_per_Dollar', ascending=False)
    
    print(f"{'Rank':<6} {'Model':<30} {'Acc/$ Ratio':<15} {'Accuracy':<12} {'Cost':<10}")
    print("-" * 80)
    
    for i, (idx, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"{i:<6} {row['Model']:<30} {row['Acc_per_Dollar']:>8.4f}       {row['Accuracy']:>6.2f}%     ${row['Cost']:>7.2f}")


def main():
    """Main execution."""
    base = get_base_path()
    output_dir = base / "data" / "plots" / "general"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("COST VS ACCURACY ANALYSIS")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading model data and calculating costs...")
    df = load_model_data()
    
    if len(df) == 0:
        print("Error: No data loaded!")
        return
    
    print(f"\n✅ Loaded data for {len(df)} models")
    
    # Create plot
    print("\nCreating scatter plot...")
    create_scatter_plot(df, output_dir)
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
