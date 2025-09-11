# protocols/actions.py
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Union, Literal
import json

# ---- Action dataclasses ------------------------------------------------------

@dataclass(frozen=True)
class RetrieveText:
    action: Literal["retrieve_text"] = "retrieve_text"
    query: str = ""
    k: int = 8
    # Match your TextRetriever signature for easy pass-through:
    where: Optional[Dict[str, Any]] = None
    where_document: Optional[Dict[str, Any]] = None
    # New: model-suggested partial answer hypothesis for this step
    partial_answer: Optional[str] = None

@dataclass(frozen=True)
class RetrieveKG:
    action: Literal["retrieve_kg"] = "retrieve_kg"
    seed_entities: List[str] = field(default_factory=list)
    relation: Optional[str] = None
    max_hops: int = 2

@dataclass(frozen=True)
class ProposeAnswer:
    action: Literal["propose_answer"] = "propose_answer"
    needs_citations: bool = True
    # Optional: when the planner LLM decides to answer directly, carry it here
    answer: Optional[str] = None

@dataclass(frozen=True)
class Stop:
    action: Literal["stop"] = "stop"

Action = Union[RetrieveText, RetrieveKG, ProposeAnswer, Stop]

# ---- (De)serialization helpers ----------------------------------------------

def action_to_dict(a: Action) -> Dict[str, Any]:
    return asdict(a)

def action_from_dict(d: Dict[str, Any]) -> Action:
    kind = d.get("action")
    if kind == "retrieve_text":
        return RetrieveText(
            query=d.get("query", ""),
            k=int(d.get("k", 8)),
            where=d.get("where"),
            where_document=d.get("where_document"),
            partial_answer=(d.get("partial_answer") if isinstance(d.get("partial_answer"), str) else None),
        )
    if kind == "retrieve_kg":
        return RetrieveKG(
            seed_entities=d.get("seed_entities", []) or [],
            relation=d.get("relation"),
            max_hops=int(d.get("max_hops", 2)),
        )
    if kind == "propose_answer":
        return ProposeAnswer(
            needs_citations=bool(d.get("needs_citations", True)),
            answer=d.get("answer") if isinstance(d.get("answer"), str) else None,
        )
    if kind == "stop":
        return Stop()
    raise ValueError(f"Unknown action: {kind}")

def actions_to_json(actions: List[Action]) -> str:
    return json.dumps([action_to_dict(a) for a in actions], ensure_ascii=False)

def parse_actions_json(s: str) -> List[Action]:
    data = json.loads(s)
    if not isinstance(data, list):
        raise ValueError("Planner must return a JSON list of actions.")
    return [action_from_dict(x) for x in data]

# ---- Optional light validation ----------------------------------------------

def validate_actions(actions: List[Action]) -> List[str]:
    """
    Return a list of human-readable warnings/errors (empty list if OK).
    Useful to guard the orchestrator against bad planner output.
    """
    errors: List[str] = []
    for i, a in enumerate(actions):
        if isinstance(a, RetrieveText):
            if not a.query:
                errors.append(f"[{i}] retrieve_text.query is empty")
            if a.k <= 0:
                errors.append(f"[{i}] retrieve_text.k must be > 0")
        if isinstance(a, RetrieveKG):
            if not a.seed_entities:
                errors.append(f"[{i}] retrieve_kg.seed_entities is empty")
            if a.max_hops <= 0:
                errors.append(f"[{i}] retrieve_kg.max_hops must be > 0")
        if isinstance(a, ProposeAnswer):
            # Enforce that planner-proposed answers include the answer text
            if a.answer is None or str(a.answer).strip() == "":
                errors.append(f"[{i}] propose_answer.answer is required and must be non-empty")
    return errors
