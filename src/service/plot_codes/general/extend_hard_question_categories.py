#!/usr/bin/env python3
"""
Extend hard_question_categories.json to include categories 1, 2, 5, 6, and 7
in addition to the existing 9, 10, and 11.

Categories represent the number of models that answered incorrectly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

from config import ITERATIVE_MODEL_ENTRIES


def load_all_model_responses(responses_dir: Path) -> Dict[str, Dict[str, bool]]:
    """
    Load all model responses.
    
    Returns:
        Dict[model_name][question] = is_correct
    """
    all_model_data = {}
    
    for filename, display_name in ITERATIVE_MODEL_ENTRIES:
        file_path = responses_dir / filename
        if not file_path.exists():
            print(f"Warning: {filename} not found")
            continue
        
        model_questions = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    
                    # Extract question
                    question = record.get("question")
                    if not question:
                        raw = record.get("raw") or record.get("raw_response")
                        if isinstance(raw, dict):
                            question = raw.get("question")
                    
                    if not question or not isinstance(question, str):
                        continue
                    
                    question = question.strip()
                    is_correct = bool(record.get("is_correct", False))
                    
                    model_questions[question] = is_correct
                
                except json.JSONDecodeError:
                    continue
        
        if model_questions:
            all_model_data[display_name] = model_questions
    
    return all_model_data


def categorize_questions_by_difficulty(
    all_model_data: Dict[str, Dict[str, bool]],
    categories: List[int]
) -> Dict[int, List[Dict]]:
    """
    Categorize questions by number of models that answered incorrectly.
    
    Args:
        all_model_data: Dict[model_name][question] = is_correct
        categories: List of category numbers to include (e.g., [1, 2, 5, 6, 7, 9, 10, 11])
    
    Returns:
        Dict[category_number] = List of question dictionaries
    """
    # Find common questions
    question_sets = [set(data.keys()) for data in all_model_data.values()]
    common_questions = set.intersection(*question_sets)
    
    print(f"Total models: {len(all_model_data)}")
    print(f"Common questions: {len(common_questions)}")
    
    # Categorize questions by how many models got them wrong
    question_categories = defaultdict(list)
    
    for question in sorted(common_questions):
        models_wrong = []
        models_correct = []
        
        for model_name, model_data in all_model_data.items():
            if question in model_data:
                if model_data[question]:
                    models_correct.append(model_name)
                else:
                    models_wrong.append(model_name)
        
        wrong_count = len(models_wrong)
        
        # Only include if in requested categories
        if wrong_count in categories:
            question_categories[wrong_count].append({
                "question": question,
                "models_wrong": sorted(models_wrong),
                "models_correct": sorted(models_correct)
            })
    
    return dict(question_categories)


def generate_extended_categories_file(
    responses_dir: Path,
    output_path: Path,
    categories: List[int]
) -> None:
    """Generate extended hard_question_categories.json file."""
    
    print("Loading model responses...")
    all_model_data = load_all_model_responses(responses_dir)
    
    print(f"\nCategorizing questions for categories: {categories}")
    categorized_questions = categorize_questions_by_difficulty(all_model_data, categories)
    
    # Print statistics
    print("\n" + "=" * 70)
    print("QUESTION DISTRIBUTION BY DIFFICULTY")
    print("=" * 70)
    for category in sorted(categorized_questions.keys()):
        count = len(categorized_questions[category])
        print(f"  {category:2d} models wrong: {count:4d} questions")
    print("=" * 70)
    
    # Convert to string keys for JSON
    output_data = {str(k): v for k, v in categorized_questions.items()}
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved extended categories to: {output_path}")
    print(f"  Total categories: {len(output_data)}")
    print(f"  Total questions: {sum(len(v) for v in output_data.values())}")


def main() -> None:
    """Main execution function."""
    base = Path(__file__).resolve().parents[3]
    responses_dir = base / "responses_reverified"
    output_dir = base / "results" / "unanswered_questions"
    output_path = output_dir / "hard_question_categories.json"
    
    # Define categories to include
    # Original: [9, 10, 11] (hard questions)
    # Extended: [1, 2, 5, 6, 7, 9, 10, 11]
    categories = [1, 2, 5, 6, 7, 9, 10, 11]
    
    print("=" * 70)
    print("EXTENDING HARD QUESTION CATEGORIES")
    print("=" * 70)
    print(f"Original categories: 9, 10, 11")
    print(f"Adding categories: 1, 2, 5, 6, 7")
    print(f"Final categories: {categories}")
    print("=" * 70)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate extended file
    generate_extended_categories_file(responses_dir, output_path, categories)
    
    print("\n" + "=" * 70)
    print("COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
