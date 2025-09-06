from typing import Protocol, List, Dict, Any, Optional, Tuple, Sequence
import json
import re

from protocols.actions import (
    Action,
    RetrieveText,
    RetrieveKG,
    ProposeAnswer,
    Stop,
    parse_actions_json,
    validate_actions,
)


# ----------------------------- Public Interfaces ------------------------------

class Planner(Protocol):
    """Planner returns a list of JSON actions for the orchestrator to execute."""
    def plan(self, question: str, state: Dict[str, Any]) -> List[Action]:
        ...


class LLMClient(Protocol):
    """
    Minimal LLM interface to keep the planner decoupled from any vendor SDK.
    Implement .complete(system, user) -> str, returning the *raw* LLM text.
    """
    def complete(self, system: str, user: str) -> str:
        ...


# ------------------------------ Implementations -------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You are a retrieval planner for multi-hop QA over unstructured text.\n"
    "You MUST output ONLY a JSON array of actions. No prose.\n"
    "Allowed actions:\n"
    '  {"action":"retrieve_text","query":"...","k":8,"where":null,"where_document":null}\n'
    '  {"action":"propose_answer","needs_citations":true}\n'
    "Rules:\n"
    "- Prefer minimal steps. 1–3 actions is typical.\n"
    "- Use retrieve_text for each sub-question you need evidence for.\n"
    "- Finish with propose_answer when you believe you have enough evidence.\n"
    "- Do NOT include comments or extra fields.\n"
)

class JSONPlanner:
    """
    LLM-driven planner that expects the model to return a JSON list of actions.
    Safely parses & validates, with fallbacks if the LLM outputs extra text.
    """
    def __init__(
        self,
        llm: LLMClient,
        allow_kg: bool = False,
        default_k: int = 8,
        max_actions: int = 6,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.allow_kg = bool(allow_kg)
        self.default_k = int(default_k)
        self.max_actions = int(max_actions)
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    def plan(self, question: str, state: Dict[str, Any]) -> List[Action]:
        # Build a compact user prompt with optional context hints
        entities = state.get("entities") or []
        subgoals = state.get("open_subgoals") or []
        filters = state.get("filters") or None

        user_prompt = (
            f"Question: {question}\n"
            f"Known entities: {entities}\n"
            f"Open subgoals: {subgoals}\n"
            f"Default k: {self.default_k}\n"
            f"Filters (optional): {filters}\n"
        )

        raw = self.llm.complete(self.system_prompt, user_prompt)

        # Extract a JSON array from the raw LLM text (be robust to extra prose)
        json_text, ok = _extract_json_array(raw)
        if not ok:
            # Fallback: a simple plan (dense-only)
            return _fallback_plan(question, self.default_k)

        try:
            actions = parse_actions_json(json_text)
        except Exception:
            return _fallback_plan(question, self.default_k)

        # Optionally strip KG actions if KG is not enabled yet
        if not self.allow_kg:
            actions = [a for a in actions if not isinstance(a, RetrieveKG)]

        # Enforce caps and defaults
        actions = actions[: self.max_actions]
        actions = _fill_defaults(actions, default_k=self.default_k, filters=filters)

        # Validate & fallback if needed
        errs = validate_actions(actions)
        if errs:
            # If invalid, return a safe, minimal plan
            return _fallback_plan(question, self.default_k)

        # Ensure the plan is actionable: must contain at least one action
        if not actions:
            return _fallback_plan(question, self.default_k)

        return actions


class RuleBasedPlanner:
    """
    A tiny rule-based planner: retrieves once (or twice if there are subgoals),
    then proposes an answer. Useful for smoke tests and offline runs.
    """
    def __init__(self, default_k: int = 8) -> None:
        self.default_k = int(default_k)

    def plan(self, question: str, state: Dict[str, Any]) -> List[Action]:
        filters = state.get("filters") or None
        subgoals = state.get("open_subgoals") or []
        actions: List[Action] = []

        # First retrieval on the full question
        actions.append(RetrieveText(query=question, k=self.default_k, where=filters, where_document=None))

        # If there are explicit subgoals, retrieve on the first one too
        if isinstance(subgoals, list) and len(subgoals) > 0 and isinstance(subgoals[0], str):
            actions.append(RetrieveText(query=subgoals[0], k=max(4, self.default_k // 2), where=filters, where_document=None))

        actions.append(ProposeAnswer(needs_citations=True))
        return actions


# ------------------------------- Utilities ------------------------------------

def _extract_json_array(text: str) -> Tuple[str, bool]:
    """
    Extract the *first* top-level JSON array from an arbitrary string.
    Returns (json_text, ok). ok=False if nothing array-like is found.
    """
    # Fast path: already looks like a JSON array
    s = text.strip()
    if s.startswith("[") and s.endswith("]"):
        return s, True

    # Fallback: regex to find the first [...] block (non-greedy, DOTALL)
    m = re.search(r"\[\s*{.*?}\s*\]", s, flags=re.DOTALL)
    if not m:
        return "", False
    return m.group(0), True


def _fill_defaults(actions: Sequence[Action], default_k: int, filters: Optional[Dict[str, Any]]) -> List[Action]:
    """Ensure reasonable defaults (e.g., k, filters) on retrieve_text actions."""
    filled: List[Action] = []
    for a in actions:
        if isinstance(a, RetrieveText):
            k = a.k if a.k and a.k > 0 else default_k
            where = a.where if a.where is not None else filters
            filled.append(RetrieveText(query=a.query, k=k, where=where, where_document=a.where_document))
        else:
            filled.append(a)
    return filled


def _fallback_plan(question: str, default_k: int) -> List[Action]:
    """A safe plan if the LLM output is unusable."""
    return [
        RetrieveText(query=question, k=default_k, where=None, where_document=None),
        ProposeAnswer(needs_citations=True),
    ]
