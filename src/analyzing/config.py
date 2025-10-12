"""Centralized configuration for analysis scripts."""
from pathlib import Path
from typing import Dict, List

# Base paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"

# Response directories (in order of preference)
RESPONSE_DIRS = [
    SRC_DIR / "responses_reverified",
    SRC_DIR / "responses",
    SRC_DIR / "response-jsonl-with-context",
    SRC_DIR / "response-jsonl-without-context",
]

# Output directory
PLOTS_DIR = SRC_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Model name normalization patterns
MODEL_NAME_PATTERNS = {
    "mistral-large": "Mistral Large",
    "mistral-small": "Mistral Small",
    "claude-3-7-sonnet-20250219-v1:0-reasoning": "Claude 3.7 Sonnet (reasoning)",
    "claude-3-7-sonnet-20250219-v1:0": "Claude 3.7 Sonnet",
    "claude-3-5-sonnet": "Claude 3.5 Sonnet",
    "deepseek-r1-v1:0-reasoning": "DeepSeek R1 (reasoning)",
    "deepseek-r1-v1:0": "DeepSeek R1",
    "deepseek-r1-distill": "DeepSeek R1 Distill",
    "deepseek-chat": "DeepSeek Chat",
    "gpt-4o-mini": "GPT-4o Mini",
    "gpt-4o": "GPT-4o",
    "gpt-5": "GPT-5",
    "o1-mini": "o1 Mini",
    "o3-mini": "o3 Mini",
    "llama3-3-70b": "Llama 3.3 70B",
    "llama-3": "Llama 3",
    "qwq-32b": "QwQ 32B",
    "gemma-3-27b": "Gemma 3 27B",
    "gemini": "Gemini",
}

# Reasoning model identifiers (partial matches)
REASONING_INDICATORS = [
    "reasoning",
    "o1-",
    "o3-",
    "qwq",
    "deepseek-r1",
]


def get_responses_dir() -> Path:
    """Get the first available responses directory."""
    for dir_path in RESPONSE_DIRS:
        if dir_path.exists():
            return dir_path
    raise FileNotFoundError(f"No response directories found. Checked: {RESPONSE_DIRS}")


def normalize_model_key(file_stem: str) -> str:
    """Normalize model identifier from filename."""
    # Remove common prefixes and suffixes
    key = file_stem.replace("responses_", "")
    key = key.replace("_reverified", "")
    
    # Remove provider prefixes
    for prefix in ["bedrock_", "bedrock_us.", "bedrock_mistral.", "openai_", "openrouter_", 
                   "anthropic.", "meta.", "google__", "qwen__", "deepseek__", "us."]:
        key = key.replace(prefix, "")
    
    return key


def get_display_name(file_stem: str) -> str:
    """Get human-readable display name for a model."""
    normalized = normalize_model_key(file_stem)
    
    # Check for exact matches first
    for pattern, display_name in MODEL_NAME_PATTERNS.items():
        if pattern in normalized:
            return display_name
    
    # Fallback: capitalize and clean up
    return normalized.replace("_", " ").replace("-", " ").title()


def is_reasoning_model(file_stem: str) -> bool:
    """Check if a model file is a reasoning model."""
    normalized = normalize_model_key(file_stem).lower()
    return any(indicator in normalized for indicator in REASONING_INDICATORS)


def discover_jsonl_files(directory: Path = None) -> List[Path]:
    """Discover all JSONL files in the responses directory."""
    if directory is None:
        directory = get_responses_dir()
    
    return sorted(directory.glob("*.jsonl"))


def discover_reasoning_jsonl_files(directory: Path = None) -> List[Path]:
    """Discover only reasoning model JSONL files."""
    all_files = discover_jsonl_files(directory)
    return [f for f in all_files if is_reasoning_model(f.stem)]


def discover_non_reasoning_jsonl_files(directory: Path = None) -> List[Path]:
    """Discover only non-reasoning model JSONL files."""
    all_files = discover_jsonl_files(directory)
    return [f for f in all_files if not is_reasoning_model(f.stem)]
