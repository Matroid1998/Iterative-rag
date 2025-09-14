"""
Minimal app configuration to control the LLM provider and planning knobs.

Rules:
- Only read API keys from environment.
- All other options (provider, model, passages_top_k, etc.) are defined here explicitly.
"""

from typing import Optional
import os

from repo.planning.planner_iface import JSONPlanner
from service.planner_llm import (
    LLMClient,
    OllamaLLM,
    HFInferenceLLM,
    make_json_planner,
)
from service.structured_llm_adapter import StructuredLLMClient

# --------------------- Explicit settings (edit in code) -----------------------
# Choose one: "openai" | "ollama" | "hf" | "none"
LLM_PROVIDER: str = "openai"

# OpenAI model name
OPENAI_MODEL: str = "gpt-4o"

# Ollama local model + endpoint
OLLAMA_MODEL: str = "llama3.1:8b-instruct"
OLLAMA_ENDPOINT: str = "http://localhost:11434/api/chat"

# Hugging Face Inference model + base URL
HF_MODEL: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
HF_BASE_URL: str = "https://api-inference.huggingface.co/models"

# Planning knobs
PASSAGES_TOP_K: int = 20
PLANNER_DEFAULT_K: int = 8
PLANNER_MAX_ACTIONS: int = 6
ALLOW_KG: bool = False


# ---------------------------- API keys (env only) -----------------------------

def _env(name: str) -> Optional[str]:
    v = os.getenv(name)
    return v if (v and v.strip()) else None


def build_llm_from_settings() -> Optional[LLMClient]:
    provider = (LLM_PROVIDER or "none").lower()
    if provider == "openai":
        api_key = _env("OPENAI_API_KEY")
        if not api_key:
            return None
        # Use StructuredLLM-backed client for OpenAI
        return StructuredLLMClient(provider="openai", model=OPENAI_MODEL)
    if provider == "ollama":
        return OllamaLLM(model=OLLAMA_MODEL, endpoint=OLLAMA_ENDPOINT)
    if provider == "hf":
        api_key = _env("HF_API_KEY")
        if not api_key:
            return None
        return HFInferenceLLM(model=HF_MODEL, api_key=api_key, base_url=HF_BASE_URL)
    return None


def build_json_planner_from_settings() -> Optional[JSONPlanner]:
    llm = build_llm_from_settings()
    if llm is None:
        return None
    return make_json_planner(
        llm,
        allow_kg=ALLOW_KG,
        default_k=PLANNER_DEFAULT_K,
        max_actions=PLANNER_MAX_ACTIONS,
        passages_top_k=PASSAGES_TOP_K,
    )
