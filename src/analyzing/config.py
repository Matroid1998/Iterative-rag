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

# Model name normalization patterns (order matters - check more specific patterns first)
MODEL_NAME_PATTERNS = {
    "mistral-large-2402": "Mistral Large 2402",
    "mistral-large": "Mistral Large",
    "mistral-small": "Mistral Small",
    "claude-3-7-sonnet-20250219-v1-0-reasoning": "Claude 3.7 Sonnet Thinking",
    "claude-3-7-sonnet-20250219-v1:0-reasoning": "Claude 3.7 Sonnet Thinking",
    "claude-3-7-sonnet-20250219-v1-0": "Claude 3.7 Sonnet",
    "claude-3-7-sonnet-20250219-v1:0": "Claude 3.7 Sonnet",
    "claude-3-5-sonnet": "Claude 3.5 Sonnet",
    "claude_sonnet_4_5": "Claude Sonnet 4.5",
    "claude-sonnet-4.5": "Claude Sonnet 4.5",
    "r1-v1-0-reasoning": "DeepSeek R1",
    "r1-v1:0-reasoning": "DeepSeek R1",
    "r1-v1-0": "DeepSeek R1",
    "r1-v1:0": "DeepSeek R1",
    "deepseek-r1-distill": "DeepSeek R1 Distill",
    "deepseek-chat": "DeepSeek Chat",
    "gpt-4o-mini": "GPT-4o Mini",
    "gpt-4o": "GPT-4o",
    "gpt-5": "GPT-5",
    "o1-mini": "o1 Mini",
    "o3-mini": "o3 Mini",
    "llama3-3-70b-instruct": "Llama 3.3 70B Instruct",
    "llama3-3-70b": "Llama 3.3 70B",
    "llama-3": "Llama 3",
    "qwq-32b": "QwQ 32B",
    "gemma-3-27b": "Gemma 3 27B",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini": "Gemini",
    "grok-4-fast": "Grok 4 Fast",
    "glm-4.6": "GLM 4.6",
}

# Reasoning model identifiers (partial matches)
REASONING_INDICATORS = [
    "reasoning",
    "o1-",
    "o3-",
    "qwq",
    "deepseek-r1",
    "gpt-5",
    "claude-sonnet-4.5",
    "gemini-2.5-pro",
    "grok-4-fast",
    "glm-4.6",
]

MODEL_COLOR_MAP = {
    "Mistral Large 2402": "#d62728",
    "Claude 3.7 Sonnet Thinking": "#2ca02c",
    "Claude 3.7 Sonnet": "#1f77b4",
    "DeepSeek R1": "#ff7f0e",
    "Llama 3.3 70B Instruct": "#9467bd",
    "GPT-4o": "#8c564b",
    "GPT-5": "#e377c2",
    "Claude Sonnet 4.5": "#7f7f7f",
    "Gemini 2.5 Pro": "#bcbd22",
    "Grok 4 Fast": "#17becf",
    "GLM 4.6": "#aec7e8",
}

DEFAULT_MODEL_COLOR = "#7f7f7f"

ITERATIVE_MODEL_ENTRIES = [
    ("responses_bedrock_mistral.mistral-large-2402-v1:0_reverified.jsonl", "Mistral Large 2402"),
    ("responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning_reverified.jsonl", "Claude 3.7 Sonnet Thinking"),
    ("responses_bedrock_us.anthropic.claude-3-7-sonnet-20250219-v1:0_reverified.jsonl", "Claude 3.7 Sonnet"),
    ("responses_bedrock_us.deepseek.r1-v1:0-reasoning_reverified.jsonl", "DeepSeek R1"),
    ("responses_bedrock_us.meta.llama3-3-70b-instruct-v1:0_reverified.jsonl", "Llama 3.3 70B Instruct"),
    ("responses_openai_gpt-4o_reverified.jsonl", "GPT-4o"),
    ("responses_openai_gpt-5_reverified.jsonl", "GPT-5"),
    ("responses_openrouter_anthropic_claude_sonnet_4_5_reasoning.jsonl", "Claude Sonnet 4.5"),
    ("responses_openrouter_google__gemini-2.5-pro_reverified.jsonl", "Gemini 2.5 Pro"),
    ("responses_openrouter_x-ai__grok-4-fast_reverified.jsonl", "Grok 4 Fast"),
    # ("responses_openrouter_z-ai__glm-4.6_reverified.jsonl", "GLM 4.6"),  # Excluded from knowledge gap analysis
]

ITERATIVE_MODEL_DISPLAY_NAMES = [display for _, display in ITERATIVE_MODEL_ENTRIES]


def get_iterative_model_entries(existing_only: bool = True) -> List[tuple[Path, str]]:
    """Return (path, display_name) pairs for reverified response files."""
    entries: List[tuple[Path, str]] = []
    base = SRC_DIR / "responses_reverified"
    for filename, display_name in ITERATIVE_MODEL_ENTRIES:
        path = base / filename
        if existing_only and not path.exists():
            continue
        entries.append((path, display_name))
    return entries


def get_iterative_display_names(existing_only: bool = True) -> List[str]:
    """Return ordered list of display names for available models."""
    return [display for _, display in get_iterative_model_entries(existing_only=existing_only)]


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
    key = key.replace("__", "_")
    # Don't replace : yet - we need it for pattern matching
    
    # Remove provider prefixes (order matters - more specific first)
    for prefix in ["bedrock_us.", "bedrock_mistral.", "bedrock_", "openai_", "openrouter_", 
                   "anthropic.", "anthropic_", "us.anthropic.", 
                   "us.deepseek.", "deepseek.", "meta.", 
                   "google__", "qwen__", "deepseek__", "x-ai__", "z-ai__", "us."]:
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


def get_model_color(display_name: str) -> str:
    """Return consistent color for a model display name."""
    return MODEL_COLOR_MAP.get(display_name, DEFAULT_MODEL_COLOR)


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
