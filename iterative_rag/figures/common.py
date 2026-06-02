"""Shared data loading + styling for the paper figures.

Centralizes everything the figure modules need:
- the canonical set of paper models (display names, colors, order),
- robust mapping of response/judgment files to a model (filenames vary across regimes),
- loaders for the three regimes (No-Context / Gold-Context / Iterative) and the
  LLM-judge diagnostic JSONL,
- small matplotlib styling + save helpers.

All figures write to ``config.FIGURES_DIR`` (``paper_figures/`` by default).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from iterative_rag import config

# --------------------------------------------------------------------------- #
# Canonical paper models (the 11 evaluated in the paper), in display order.
# --------------------------------------------------------------------------- #
PAPER_MODELS: List[str] = [
    "Mistral Large 2402",
    "Claude 3.7 Sonnet Thinking",
    "Claude 3.7 Sonnet",
    "DeepSeek R1",
    "Llama 3.3 70B Instruct",
    "GPT-4o",
    "GPT-5",
    "Claude Sonnet 4.5",
    "Gemini 2.5 Pro",
    "Grok 4 Fast",
    "GLM 4.6",
]

MODEL_COLOR_MAP: Dict[str, str] = {
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
DEFAULT_COLOR = "#7f7f7f"

REASONING_MODELS = {
    "Claude 3.7 Sonnet Thinking", "DeepSeek R1", "GPT-5",
    "Claude Sonnet 4.5", "Gemini 2.5 Pro", "Grok 4 Fast", "GLM 4.6",
}

# Order matters: check more specific patterns first.
_MODEL_PATTERNS = [
    ("mistral-large-2402", "Mistral Large 2402"),
    ("mistral.mistral-large", "Mistral Large 2402"),
    ("claude-3-7-sonnet-20250219-v1:0-reasoning", "Claude 3.7 Sonnet Thinking"),
    ("claude-3-7-sonnet-20250219-v1_0-reasoning", "Claude 3.7 Sonnet Thinking"),
    ("claude-3-7-sonnet", "Claude 3.7 Sonnet"),
    ("claude_sonnet_4_5", "Claude Sonnet 4.5"),
    ("claude-sonnet-4.5", "Claude Sonnet 4.5"),
    ("claude-sonnet-4-5", "Claude Sonnet 4.5"),
    ("deepseek.r1", "DeepSeek R1"),
    ("deepseek-r1", "DeepSeek R1"),
    ("r1-v1:0", "DeepSeek R1"),
    ("llama3-3-70b", "Llama 3.3 70B Instruct"),
    ("gpt-4o", "GPT-4o"),
    ("gpt-5", "GPT-5"),
    ("gemini-2.5-pro", "Gemini 2.5 Pro"),
    ("grok-4-fast", "Grok 4 Fast"),
    ("glm-4.6", "GLM 4.6"),
]


def display_name_for(stem: str) -> Optional[str]:
    """Map a response/judgment filename stem to a canonical paper-model name (or None)."""
    s = stem.lower()
    # gpt-4o must not match gpt-4o-mini
    if "gpt-4o-mini" in s or "gpt-4o-2024" in s and "mini" in s:
        return None
    for pat, name in _MODEL_PATTERNS:
        if pat in s:
            # avoid mini/distill variants leaking into paper models
            if name == "GPT-4o" and "mini" in s:
                return None
            if name == "GPT-5" and ("gpt-5.1" in s or "gpt-5-mini" in s):
                return None
            if name == "DeepSeek R1" and "distill" in s:
                return None
            return name
    return None


def color_for(display: str) -> str:
    return MODEL_COLOR_MAP.get(display, DEFAULT_COLOR)


# --------------------------------------------------------------------------- #
# JSONL loading
# --------------------------------------------------------------------------- #

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def question_of(rec: Dict[str, Any]) -> str:
    raw = rec.get("raw") or {}
    if isinstance(raw, dict) and raw.get("question"):
        return raw["question"]
    rr = rec.get("raw_response") or {}
    if isinstance(rr, dict) and rr.get("question"):
        return rr["question"]
    return rec.get("question") or ""


_REGIME_DIRS = {
    "iterative": config.ITERATIVE_RESPONSES_DIR,
    "no_context": config.WITHOUT_CONTEXT_RESPONSES_DIR,
    "gold": config.WITH_CONTEXT_RESPONSES_DIR,
}


def discover_regime_files(regime: str) -> Dict[str, Path]:
    """Best file per paper model for a regime (prefers '_reverified' files)."""
    d = _REGIME_DIRS[regime]
    chosen: Dict[str, Path] = {}
    if not Path(d).is_dir():
        return chosen
    for p in sorted(Path(d).glob("*.jsonl")):
        name = display_name_for(p.stem)
        if not name or name not in PAPER_MODELS:
            continue
        # prefer reverified version if both exist
        if name not in chosen or ("reverified" in p.stem and "reverified" not in chosen[name].stem):
            chosen[name] = p
    return chosen


def load_regime(regime: str) -> Dict[str, List[Dict[str, Any]]]:
    """{display_name -> list of records} for a regime, paper models only."""
    return {name: load_jsonl(p) for name, p in discover_regime_files(regime).items()}


def correct_by_question(regime: str) -> Dict[str, Dict[str, bool]]:
    """{display_name -> {question -> is_correct}} for a regime."""
    out: Dict[str, Dict[str, bool]] = {}
    for name, recs in load_regime(regime).items():
        out[name] = {question_of(r): bool(r.get("is_correct")) for r in recs if question_of(r)}
    return out


def accuracy_by_model(regime: str) -> Dict[str, float]:
    """{display_name -> accuracy %} for a regime."""
    out: Dict[str, float] = {}
    for name, recs in load_regime(regime).items():
        vals = [bool(r.get("is_correct")) for r in recs]
        if vals:
            out[name] = 100.0 * sum(vals) / len(vals)
    return out


# --------------------------------------------------------------------------- #
# Iterative dataframe + diagnostics
# --------------------------------------------------------------------------- #

def _qa_hops_map() -> Dict[str, int]:
    try:
        data = json.loads(config.QA_DATASET.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[str, int] = {}
    for r in data:
        q = r.get("q") or r.get("question")
        if q:
            out[q] = len(r.get("path") or [])
    return out


def iterative_dataframe():
    """A pandas DataFrame, one row per (model, question) iterative-RAG run."""
    import pandas as pd
    hops_map = _qa_hops_map()
    rows = []
    for name, recs in load_regime("iterative").items():
        for r in recs:
            q = question_of(r)
            rr = r.get("raw_response") or {}
            hops = r.get("number_of_hops") or hops_map.get(q) or 0
            rows.append({
                "model": name,
                "question": q,
                "is_correct": bool(r.get("is_correct")),
                "hops": int(hops or 0),
                "steps": int((rr.get("steps") if isinstance(rr, dict) else None) or 0),
                "output_tokens": int(r.get("output_tokens") or 0),
                "reasoning_tokens": int(r.get("reasoning_tokens") or 0),
                "input_tokens": int(r.get("input_tokens") or 0),
                "latency": float(r.get("latency") or 0.0),
            })
    return pd.DataFrame(rows)


_JUDGE_SUFFIX = {
    "coverage": "_coverage_gap_judgments",
    "hallucination": "_hallucination_judgment",
    "quality": "_quality_judgement",
}


def load_judgments(kind: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """{display_name -> {question -> parsed_judgment}} from diagnostics_output/."""
    suffix = _JUDGE_SUFFIX[kind]
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    d = config.DIAGNOSTICS_DIR
    if not Path(d).is_dir():
        return out
    for p in sorted(Path(d).glob(f"*{suffix}.jsonl")):
        name = display_name_for(p.stem)
        if not name or name not in PAPER_MODELS:
            continue
        per_q: Dict[str, Dict[str, Any]] = {}
        for r in load_jsonl(p):
            q = r.get("question")
            pj = r.get("parsed_judgment")
            if q and isinstance(pj, dict):
                per_q[q] = pj
        if per_q:
            out.setdefault(name, {}).update(per_q)
    return out


def gold_context_tokens_map() -> Dict[str, int]:
    """{question -> gold-context length in whitespace tokens} from the oracle path texts."""
    try:
        data = json.loads(config.QA_DATASET.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[str, int] = {}
    for r in data:
        q = r.get("q") or r.get("question")
        if not q:
            continue
        n = 0
        for hop in r.get("path") or []:
            n += len((hop.get("text") or "").split())
        out[q] = n
    return out


def difficulty_map(df=None) -> Dict[str, str]:
    """{question -> 'easy'|'medium'|'hard'} by how many models answer it wrong (iterative).

    easy: <=2 models wrong; hard: >=9 models wrong; medium otherwise (paper stratification).
    """
    if df is None:
        df = iterative_dataframe()
    out: Dict[str, str] = {}
    for q, grp in df.groupby("question"):
        n_wrong = int((~grp["is_correct"]).sum())
        out[q] = "easy" if n_wrong <= 2 else ("hard" if n_wrong >= 9 else "medium")
    return out


# --------------------------------------------------------------------------- #
# Styling / saving
# --------------------------------------------------------------------------- #

def use_style() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def models_present(*regimes: str) -> List[str]:
    """Paper models (in order) that have data in all given regimes."""
    sets = [set(discover_regime_files(r)) for r in regimes]
    common = set.intersection(*sets) if sets else set()
    return [m for m in PAPER_MODELS if m in common]


def save(fig, out_dir: Path, name: str) -> Path:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path
