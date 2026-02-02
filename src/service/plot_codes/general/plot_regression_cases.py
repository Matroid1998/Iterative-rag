#!/usr/bin/env python3
"""
Generate plot showing Parametric Suppression Rate (PSR).

PSR measures the proportion of questions answered correctly using only 
parametric knowledge (No Context) that became incorrect when using 
Iterative RAG with retrieved context.

Formula: PSR = |Q_correct^NoCtx ∩ Q_incorrect^IterRAG| / |Q_correct^NoCtx|
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Set

import matplotlib.pyplot as plt
import numpy as np

from config import ITERATIVE_MODEL_ENTRIES, get_model_color


# Define reasoning models
REASONING_MODELS = {
    "Claude 3.7 Sonnet Thinking",
    "DeepSeek R1",
    "GPT-o1",
    "GPT-o3",
    "Claude Sonnet 4.5",
    "GPT-5",
    "Gemini 2.5 Pro",
    "Grok 4 Fast",
    "GLM 4.6"
}


def sort_models_non_reasoning_first(model_names: List[str]) -> List[str]:
    """Sort models so non-reasoning models appear first, then reasoning models."""
    non_reasoning = []
    reasoning = []
    
    for model in model_names:
        if model in REASONING_MODELS:
            reasoning.append(model)
        else:
            non_reasoning.append(model)
    
    return non_reasoning + reasoning


def iter_records(path: Path) -> Iterable[dict]:
    """Iterate over JSONL records."""
    if not path.exists():
        return
    
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError:
                continue


def extract_question(record: dict) -> str | None:
    """Extract question text from a record."""
    # Try direct question field
    question = record.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()
    
    # Try nested in raw_response
    raw_response = record.get("raw_response")
    if isinstance(raw_response, dict):
        q = raw_response.get("question")
        if isinstance(q, str) and q.strip():
            return q.strip()
    
    # Try nested in raw
    raw = record.get("raw")
    if isinstance(raw, dict):
        q = raw.get("question")
        if isinstance(q, str) and q.strip():
            return q.strip()
    
    return None


def load_correct_questions(jsonl_path: Path) -> Set[str]:
    """Load all questions that were answered correctly from a JSONL file."""
    correct_questions = set()
    
    for record in iter_records(jsonl_path):
        question = extract_question(record)
        if not question:
            continue
        
        is_correct = bool(record.get("is_correct", False))
        if is_correct:
            correct_questions.add(question)
    
    return correct_questions


def calculate_parametric_suppression(
    no_context_dir: Path,
    iterative_dir: Path,
    model_entries: List[tuple[str, str]]
) -> Dict[str, float]:
    """
    Calculate Parametric Suppression Rate (PSR) for each model.
    
    PSR = |Q_correct^NoCtx ∩ Q_incorrect^IterRAG| / |Q_correct^NoCtx|
    
    Measures the proportion of no-context correct answers that became 
    incorrect with Iterative RAG.
    """
    psr_rates = {}
    
    for filename, display_name in model_entries:
        # Try to find matching files in both directories
        no_context_path = no_context_dir / filename
        iterative_path = iterative_dir / filename
        
        # Try various filename variations for no_context
        if not no_context_path.exists():
            # Try without _reverified suffix
            filename_alt = filename.replace("_reverified", "")
            no_context_path = no_context_dir / filename_alt
        
        if not no_context_path.exists():
            # Try with -reasoning suffix instead of _reverified
            filename_alt = filename.replace("_reverified.jsonl", "-reasoning.jsonl")
            no_context_path = no_context_dir / filename_alt
        
        if not no_context_path.exists():
            # Try with -reasoning_reverified suffix
            filename_alt = filename.replace("_reverified.jsonl", "-reasoning_reverified.jsonl")
            no_context_path = no_context_dir / filename_alt
        
        if not no_context_path.exists():
            # Try removing _reverified and checking for base name
            base_name = filename.replace("_reverified.jsonl", ".jsonl")
            no_context_path = no_context_dir / base_name
        
        if not no_context_path.exists() or not iterative_path.exists():
            print(f"Warning: Missing files for {display_name}")
            print(f"  No Context: {no_context_path.exists()} - {no_context_path.name if no_context_path else 'N/A'}")
            print(f"  Iterative: {iterative_path.exists()} - {iterative_path.name if iterative_path else 'N/A'}")
            continue
        
        # Load correct questions from both setups
        no_context_correct = load_correct_questions(no_context_path)
        iterative_correct = load_correct_questions(iterative_path)
        
        # Calculate Parametric Suppression Rate
        # PSR = |Q_correct^NoCtx ∩ Q_incorrect^IterRAG| / |Q_correct^NoCtx|
        suppressed_questions = no_context_correct - iterative_correct
        
        if len(no_context_correct) > 0:
            psr = (len(suppressed_questions) / len(no_context_correct)) * 100
        else:
            psr = 0.0
        
        psr_rates[display_name] = psr
        
        print(f"{display_name}:")
        print(f"  No Context correct: {len(no_context_correct)}")
        print(f"  Iterative correct: {len(iterative_correct)}")
        print(f"  Suppressed questions: {len(suppressed_questions)}")
        print(f"  PSR: {psr:.2f}%")
    
    return psr_rates


def plot_parametric_suppression(
    psr_rates: Dict[str, float],
    output_path: Path
) -> None:
    """Create bar plot showing Parametric Suppression Rate per model."""
    if not psr_rates:
        print("No data to plot!")
        return
    
    # Sort models
    model_names = list(psr_rates.keys())
    model_names = sort_models_non_reasoning_first(model_names)
    
    # Get PSR rates in sorted order
    rates = [psr_rates[model] for model in model_names]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(model_names))
    
    bars = ax.bar(x, rates, color='gray', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, rates)):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Parametric Suppression Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_title('Parametric Suppression Rate (PSR): Proportion of No-Context Correct Answers Suppressed by Iterative RAG', 
                fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylim(0, max(rates) * 1.15 if rates else 10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add horizontal average line
    if rates:
        avg_psr = np.mean(rates)
        ax.axhline(y=avg_psr, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                   label=f'Average: {avg_psr:.1f}%')
        ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved plot to: {output_path}")


def main():
    """Main execution function."""
    project_root = Path(__file__).resolve().parents[2]
    no_context_dir = project_root / "src" / "response-jsonl-without-context"
    iterative_dir = project_root / "src" / "responses_reverified"
    output_path = project_root / "data" / "plots" / "general" / "parametric_suppression_rate.png"
    
    print("Calculating Parametric Suppression Rate (PSR)...")
    print("=" * 60)
    
    psr_rates = calculate_parametric_suppression(
        no_context_dir,
        iterative_dir,
        ITERATIVE_MODEL_ENTRIES
    )
    
    print("\n" + "=" * 60)
    print(f"Total models analyzed: {len(psr_rates)}")
    if psr_rates:
        avg_psr = np.mean(list(psr_rates.values()))
        print(f"Average PSR: {avg_psr:.2f}%")
    
    print("\nGenerating plot...")
    plot_parametric_suppression(psr_rates, output_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
