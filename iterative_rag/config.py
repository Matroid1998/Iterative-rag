"""
Central configuration for the iterative_rag package.

This module is intentionally dependency-light: importing it pulls in nothing
beyond the standard library, so path constants can be read from anywhere
(figures, endpoints, diagnostics) without dragging in torch / chromadb / openai.
The heavy LLM/planner factories defer their imports until called.

Rules:
- API keys are read from the environment only.
- Filesystem paths default to a layout rooted at the repository, but every
  path can be overridden with an ``IRAG_*`` environment variable.
- Provider / model / planning knobs are defined explicitly here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing heavy modules at module load time
    from iterative_rag.system.planner import JSONPlanner
    from iterative_rag.system.llm import LLMClient


# ----------------------------------------------------------------------------
# Filesystem layout
# ----------------------------------------------------------------------------

def _path_env(name: str, default: Path) -> Path:
    v = os.getenv(name)
    return Path(v).expanduser().resolve() if v and v.strip() else default


# iterative_rag/config.py -> repo root is two levels up.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = _path_env("IRAG_DATA_ROOT", REPO_ROOT / "data")
DOCS_DIR: Path = DATA_DIR / "docs"
CORPUS_DIR: Path = DOCS_DIR / "chemrxiv_graph_v2_texts"
QA_DATASET: Path = DOCS_DIR / "chemrxiv_qa.json"
GRAPH_JSON: Path = DOCS_DIR / "chemrxiv_graph_v2.json"

RESPONSES_DIR: Path = _path_env("IRAG_RESPONSES_DIR", REPO_ROOT / "responses")
# Sub-collections of response JSONL (original directory names preserved).
ITERATIVE_RESPONSES_DIR: Path = RESPONSES_DIR / "responses_reverified"
WITH_CONTEXT_RESPONSES_DIR: Path = RESPONSES_DIR / "response-jsonl-with-context"
WITHOUT_CONTEXT_RESPONSES_DIR: Path = RESPONSES_DIR / "response-jsonl-without-context"
COT_RESPONSES_DIR: Path = RESPONSES_DIR / "responses_cot"

DIAGNOSTICS_DIR: Path = _path_env("IRAG_DIAGNOSTICS_DIR", REPO_ROOT / "diagnostics_output")
RESULTS_DIR: Path = _path_env("IRAG_RESULTS_DIR", REPO_ROOT / "results")
FIGURES_DIR: Path = _path_env("IRAG_FIGURES_DIR", REPO_ROOT / "paper_figures")

# Vector store
CHROMA_DIR: Path = _path_env("IRAG_CHROMA_DIR", REPO_ROOT / "chroma_store")
DEFAULT_COLLECTION: str = os.getenv("IRAG_COLLECTION", "chemrxiv_graph")
DEFAULT_EMBED_MODEL: str = os.getenv("IRAG_EMBED_MODEL", "BASF-AI/ChEmbed")
DEFAULT_DEVICE: str = os.getenv("IRAG_DEVICE", "cpu")


# ----------------------------------------------------------------------------
# LLM provider + planning knobs
# ----------------------------------------------------------------------------
# Choose one: "openai" | "ollama" | "hf" | "none"
LLM_PROVIDER: str = os.getenv("IRAG_LLM_PROVIDER", "openai")

OPENAI_MODEL: str = os.getenv("IRAG_OPENAI_MODEL", "gpt-4o")

OLLAMA_MODEL: str = "llama3.1:8b-instruct"
OLLAMA_ENDPOINT: str = "http://localhost:11434/api/chat"

HF_MODEL: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
HF_BASE_URL: str = "https://api-inference.huggingface.co/models"

# Default LLM used by the diagnostics (LLM-as-judge) endpoint.
JUDGE_PROVIDER: str = os.getenv("IRAG_JUDGE_PROVIDER", "openai")
JUDGE_MODEL: str = os.getenv("IRAG_JUDGE_MODEL", "gpt-5-mini")

# Planning knobs
PASSAGES_TOP_K: int = 20
PLANNER_DEFAULT_K: int = 8
PLANNER_MAX_ACTIONS: int = 7
ALLOW_KG: bool = False


# ----------------------------------------------------------------------------
# API keys (environment only) + factories (lazy imports)
# ----------------------------------------------------------------------------

def _env(name: str) -> Optional[str]:
    v = os.getenv(name)
    return v if (v and v.strip()) else None


def build_llm_from_settings() -> Optional["LLMClient"]:
    """Build the configured LLM client, or None if unavailable (no key / 'none')."""
    provider = (LLM_PROVIDER or "none").lower()
    if provider == "openai":
        if not _env("OPENAI_API_KEY"):
            return None
        from iterative_rag.system.structured_llm import StructuredLLMClient
        return StructuredLLMClient(provider="openai", model=OPENAI_MODEL)
    if provider == "ollama":
        from iterative_rag.system.llm import OllamaLLM
        return OllamaLLM(model=OLLAMA_MODEL, endpoint=OLLAMA_ENDPOINT)
    if provider == "hf":
        api_key = _env("HF_API_KEY")
        if not api_key:
            return None
        from iterative_rag.system.llm import HFInferenceLLM
        return HFInferenceLLM(model=HF_MODEL, api_key=api_key, base_url=HF_BASE_URL)
    return None


def build_json_planner_from_settings() -> Optional["JSONPlanner"]:
    """Build the configured JSON planner, or None if no LLM is available."""
    llm = build_llm_from_settings()
    if llm is None:
        return None
    from iterative_rag.system.llm import make_json_planner
    return make_json_planner(
        llm,
        allow_kg=ALLOW_KG,
        default_k=PLANNER_DEFAULT_K,
        max_actions=PLANNER_MAX_ACTIONS,
        passages_top_k=PASSAGES_TOP_K,
    )
