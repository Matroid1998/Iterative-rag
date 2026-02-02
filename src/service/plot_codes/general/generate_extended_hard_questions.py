#!/usr/bin/env python3
"""
Generate extended hard question categories file including categories 0-11.
Categories represent how many models got the question wrong:
- 0: All models correct (easiest questions)
- 1-2: Very easy (only 1-2 models wrong)
- 5-7: Medium difficulty
- 9-11: Hard questions (9+ models wrong)
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

from config import ITERATIVE_MODEL_ENTRIES


def load_all_model_responses(responses_dir: Path) -> Dict[str, Dict[str, Dict]]:
    """
    Load all model responses.
    
    Returns:
        Dict[model_name][question] = {
            'is_correct': bool,
            'output_tokens': int,
            ...
        }
    """
    all_data = {}
    
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
                    
                    # Extract metrics
                    is_correct = bool(record.get("is_correct", False))
                    output_tokens = record.get("output_tokens")
                    
                    model_questions[question] = {
                        'is_correct': is_correct,
                        'output_tokens': int(output_tokens) if isinstance(output_tokens, (int, float)) and output_tokens > 0 else 0
                    }
                
                except json.JSONDecodeError:
                    continue
        
        if model_questions:
            all_data[display_name] = model_questions
            print(f"Loaded {display_name}: {len(model_questions)} questions")
    
    return all_data


def categorize_questions_by_difficulty(all_data: Dict[str, Dict[str, Dict]]) -> Dict[int, List[Dict]]:
    """
    Categorize questions by how many models got them wrong.
    
    Returns:
        Dict[wrong_count] = [
            {
                'question': str,
                'models_wrong': List[str],
                'models_correct': List[str]
            },
            ...
        ]
    """
    # Find common questions
    question_sets = [set(data.keys()) for data in all_data.values()]
    common_questions = set.intersection(*question_sets)
    
    print(f"\nTotal models: {len(all_data)}")
    print(f"Common questions: {len(common_questions)}")
    
    # Categorize by wrong count
    categories = defaultdict(list)
    
    for question in common_questions:
        wrong_models = []
        correct_models = []
        
        for model_name, model_data in all_data.items():
            if question in model_data:
                if model_data[question]['is_correct']:
                    correct_models.append(model_name)
                else:
                    wrong_models.append(model_name)
        
        wrong_count = len(wrong_models)
        
        categories[wrong_count].append({
            'question': question,
            'models_wrong': sorted(wrong_models),
            'models_correct': sorted(correct_models)
        })
    
    return dict(categories)


def main():
    """Main execution function."""
    base = Path(__file__).resolve().parents[3]
    responses_dir = base / "responses_reverified"
    output_dir = base / "results" / "unanswered_questions"
    output_file = output_dir / "hard_question_categories.json"
    
    print("=" * 70)
    print("GENERATING EXTENDED HARD QUESTION CATEGORIES")
    print("=" * 70)
    
    # Load all model responses
    print("\nLoading model responses...")
    all_data = load_all_model_responses(responses_dir)
    
    if not all_data:
        print("Error: No model data loaded!")
        return
    
    # Categorize questions
    print("\nCategorizing questions by difficulty...")
    all_categories = categorize_questions_by_difficulty(all_data)
    
    # Categories to extract (all from 0 to 11)
    categories_to_extract = [0, 1, 2, 5, 6, 7, 9, 10, 11]
    
    # Filter and prepare output
    output_data = {}
    
    print("\n" + "=" * 70)
    print("CATEGORY DISTRIBUTION")
    print("=" * 70)
    
    for category in sorted(categories_to_extract):
        if category in all_categories:
            output_data[str(category)] = all_categories[category]
            count = len(all_categories[category])
            
            if category == 0:
                desc = "All models correct (easiest)"
            elif category <= 2:
                desc = "Very easy"
            elif category <= 7:
                desc = "Medium difficulty"
            else:
                desc = "Hard questions"
            
            print(f"{category:>2} models wrong: {count:>4} questions - {desc}")
        else:
            print(f"{category:>2} models wrong: {0:>4} questions - (none found)")
    
    # Save to file
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print(f"✅ Saved to: {output_file}")
    print(f"   Total categories: {len(output_data)}")
    print(f"   Total questions: {sum(len(v) for v in output_data.values())}")
    print("=" * 70)


if __name__ == "__main__":
    main()
