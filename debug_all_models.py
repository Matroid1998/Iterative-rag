import json
from pathlib import Path
from typing import List, Tuple

def get_base_path() -> Path:
    return Path("/home/mehdi/Projects/Iterative-rag")

def get_quality_model_entries() -> List[Tuple[Path, Path, str]]:
    """Get list of (quality_file_path, reverified_file_path, display_name) tuples."""
    base = get_base_path()
    quality_dir = base / "src" / "rag_analysis" / "output"
    reverified_dir = base / "src" / "responses_reverified"
    
    # Model display names mapping
    model_names = {
        "bedrock_mistral.mistral-large-2402-v1:0": "Mistral Large 2402",
        "bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning": "Claude 3.7 Sonnet Thinking",
        "bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0": "Claude 3.7 Sonnet",
        "bedrock_us.deepseek.r1-v1:0-reasoning": "DeepSeek R1",
        "bedrock_us.meta.llama3-3-70b-instruct-v1:0": "Llama 3.3 70B Instruct",
        "openai_gpt-4o": "GPT-4o",
        "openai_gpt-5": "GPT-5",
        "openrouter_anthropic__claude-sonnet-4.5": "Claude Sonnet 4.5",
        "openrouter_google__gemini-2.5-pro": "Gemini 2.5 Pro",
        "openrouter_x-ai__grok-4-fast": "Grok 4 Fast",
        "openrouter_z-ai__glm-4.6": "GLM 4.6",
    }
    
    entries = []
    # Use the updated discovery logic from the main script if possible, or just glob both
    files = list(quality_dir.glob("*quality_judement.jsonl")) + list(quality_dir.glob("*quality_judgement.jsonl"))
    
    for quality_file in sorted(files):
        stem = quality_file.stem
        
        if stem.endswith("_quality_judement"):
            stem = stem[:-len("_quality_judement")]
        elif stem.endswith("_quality_judgement"):
            stem = stem[:-len("_quality_judgement")]
        
        if stem.startswith("2_"):
            stem = stem[2:]
        
        raw_name = stem
        if stem.endswith("_reverified"):
            raw_name = stem[:-len("_reverified")]
        
        reverified_file = reverified_dir / f"{stem}.jsonl"
        
        if not reverified_file.exists():
            print(f"Missing reverified file for: {stem}")
            continue
        
        model_key = raw_name
        if model_key.startswith("responses_"):
            model_key = model_key[len("responses_"):]
        
        display_name = model_names.get(model_key, model_key)
        entries.append((quality_file, reverified_file, display_name))
    
    return entries

def check_all_models():
    entries = get_quality_model_entries()
    print(f"Found {len(entries)} model entries")
    
    total_steps_all = 0
    total_questions_all = 0
    
    for quality_file, reverified_file, display_name in entries:
        print(f"\nChecking {display_name}...")
        
        # Load is_correct map
        is_correct_map = {}
        with open(reverified_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    question = data.get('raw_response', {}).get('question', '') or data.get('question', '')
                    is_correct_map[question] = True
                except:
                    pass
        
        matched_count = 0
        steps_count = 0
        quality_questions = 0
        
        with open(quality_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    question = data.get('question', '')
                    quality_questions += 1
                    
                    if question in is_correct_map:
                        matched_count += 1
                        parsed = data.get('parsed_judgment', {})
                        per_step = parsed.get('per_step', [])
                        steps_count += len(per_step)
                except:
                    pass
        
        print(f"  Quality questions: {quality_questions}")
        print(f"  Matched questions: {matched_count}")
        print(f"  Total steps: {steps_count}")
        
        total_steps_all += steps_count
        total_questions_all += matched_count

    print(f"\nTotal matched questions: {total_questions_all}")
    print(f"Total steps found: {total_steps_all}")

if __name__ == "__main__":
    check_all_models()
