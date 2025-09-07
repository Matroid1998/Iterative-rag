"""
# service/rag/orchestrator.py
Iterative RAG orchestrator (text-only, training-free).

Loop:
  1) Ask the planner for exactly one JSON action (retrieve_text / propose_answer / stop)
  2) Execute tools (TextRetriever) for retrieve_text
  3) Update state with new evidence and hints (previous_queries, last_passages)
  4) Stop when the planner proposes an answer (or budget is exhausted)

This file stays agnostic to concrete LLM vendors and index backends:
- Planner must implement .plan(question, state) -> List[Action]
- Text retriever must implement .retrieve(query, k, where, where_document) -> List[hit dicts]
- Composer can be a callable (question, evidence) -> {"answer": str, "citations": ...}
"""

from typing import Any, Callable, Dict, List, Optional
import time
import hashlib

from protocols.actions import (
    Action,
    RetrieveText,
    RetrieveKG,
    ProposeAnswer,
    Stop,
    action_to_dict,
)

# Type alias for evidence hits (as returned by TextRetriever / index wrapper)
Hit = Dict[str, Any]

ComposeFunc = Callable[[str, List[Hit]], Dict[str, Any]]  # returns {"answer": str, ...}


class Orchestrator:
    """Iterative controller for unstructured-text RAG."""

    def __init__(
        self,
        *,
        planner: Any,
        text_retriever: Any,
        compose_answer: Optional[ComposeFunc] = None,
        max_steps: int = 6,
        dedupe: bool = True,
        keep_top_evidence: Optional[int] = None,  # optionally cap evidence list length
    ) -> None:
        """
        planner: implements .plan(question, state)->List[Action]
        text_retriever: implements .retrieve(query, k, where=None, where_document=None)->List[Hit]
        compose_answer: callable(question, evidence)->dict with at least {"answer": str}
        max_steps: how many planner iterations to allow
        dedupe: drop duplicate evidence by id
        keep_top_evidence: if set, keep only top-N most recent/strong evidence items
        """
        self.planner = planner
        self.text_retriever = text_retriever
        self.compose_answer = compose_answer or _default_composer
        self.max_steps = int(max_steps)
        self.dedupe = bool(dedupe)
        self.keep_top_evidence = keep_top_evidence if keep_top_evidence is None else int(keep_top_evidence)

    # -------------------------------------------------------------------------

    def run(
        self,
        question: str,
        *,
        initial_state: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
        k_default: int = 8,
        trace: bool = True,
    ) -> Dict[str, Any]:
        """Execute the iterative loop for a single question."""
        t0 = time.time()
        state: Dict[str, Any] = {
            "evidence": [],
            "entities": [],
            "open_subgoals": [],
            "filters": filters or None,
            "previous_queries": [],
            "last_passages": [],
        }
        if initial_state:
            # Shallow merge into our defaults
            for k, v in initial_state.items():
                state[k] = v

        actions_trace: List[Dict[str, Any]] = []

        for step in range(self.max_steps):
            actions = self.planner.plan(question, state)

            if not isinstance(actions, list) or not actions:
                # Planner returned nothing usable — default to a single retrieval
                actions = [RetrieveText(query=question, k=k_default)]

            # Execute only the first action, then loop
            action = actions[0]

            if isinstance(action, RetrieveText):
                hits = self.text_retriever.retrieve(
                    query=action.query,
                    k=action.k,
                    where=action.where,
                    where_document=action.where_document,
                )
                before = len(state["evidence"])
                self._add_evidence(state, hits)
                after = len(state["evidence"])

                # Track planner hints
                prev_qs = state.get("previous_queries") or []
                prev_qs.append(action.query)
                state["previous_queries"] = prev_qs
                state["last_passages"] = list(hits or [])

                if trace:
                    actions_trace.append({
                        "action": action_to_dict(action),
                        "added": after - before,
                        "k": action.k,
                    })

                # Optional evidence capping
                if self.keep_top_evidence is not None and len(state["evidence"]) > self.keep_top_evidence:
                    state["evidence"] = state["evidence"][-self.keep_top_evidence :]
                continue

            if isinstance(action, RetrieveKG):
                # KG not enabled yet; ignore safely (or record no-op in trace)
                if trace:
                    actions_trace.append({
                        "action": action_to_dict(action),
                        "note": "retrieve_kg ignored (KG disabled)",
                    })
                continue

            if isinstance(action, ProposeAnswer):
                composed = self.compose_answer(question, state["evidence"])
                # Prefer planner-provided answer if present
                answer = getattr(action, "answer", None) or composed.get("answer")
                return {
                    "question": question,
                    "answer": answer,
                    "evidence": state["evidence"],
                    "citations": composed.get("citations"),
                    "actions_trace": actions_trace if trace else None,
                    "steps": step + 1,
                    "stop_reason": "proposed_answer",
                    "elapsed_sec": round(time.time() - t0, 3),
                }

            if isinstance(action, Stop):
                return {
                    "question": question,
                    "answer": None,
                    "evidence": state["evidence"],
                    "actions_trace": actions_trace if trace else None,
                    "steps": step + 1,
                    "stop_reason": "stop_action",
                    "elapsed_sec": round(time.time() - t0, 3),
                }

            # Unknown action — ignore but keep trace
            if trace:
                actions_trace.append({
                    "action": {"action": getattr(action, "action", "unknown")},
                    "error": "unknown_action_type",
                })

        # Budget exhausted
        return {
            "question": question,
            "answer": None,
            "evidence": state["evidence"],
            "actions_trace": actions_trace if trace else None,
            "steps": self.max_steps,
            "stop_reason": "budget_exhausted",
            "elapsed_sec": round(time.time() - t0, 3),
        }

    # -------------------------------------------------------------------------

    def _add_evidence(self, state: Dict[str, Any], hits: List[Hit]) -> None:
        """Append new evidence hits to state (with optional de-duplication)."""
        if not hits:
            return
        if not self.dedupe:
            state["evidence"].extend(hits)
            return

        # Build a set of existing IDs for fast membership tests
        seen_ids = set()
        for ev in state["evidence"]:
            eid = str(ev.get("id") or _hash_fallback(ev))
            seen_ids.add(eid)

        for h in hits:
            hid = str(h.get("id") or _hash_fallback(h))
            if hid in seen_ids:
                continue
            seen_ids.add(hid)
            state["evidence"].append(h)


# ------------------------------- helpers --------------------------------------

def _hash_fallback(hit: Hit) -> str:
    """Stable-ish hash for hits without an explicit 'id'."""
    text = str(hit.get("text", ""))
    meta = str(hit.get("metadata", ""))
    h = hashlib.sha1((text + "||" + meta).encode("utf-8")).hexdigest()
    return h


def _default_composer(question: str, evidence: List[Hit]) -> Dict[str, Any]:
    """
    Minimal, deterministic composer:
    - returns the top-1 passage as the 'answer' draft (or empty)
    - builds naive 'citations' from metadata
    Replace this with an LLM-based composer in your service layer.
    """
    if not evidence:
        return {"answer": "", "citations": []}

    # Choose best by whichever score field is available
    def best_key(h: Hit) -> float:
        if "score" in h and h["score"] is not None:
            return float(h["score"])
        if "score_dense" in h and h["score_dense"] is not None:
            return float(h["score_dense"])
        return 0.0

    best = sorted(evidence, key=best_key, reverse=True)[0]
    citations = [{
        "id": best.get("id"),
        "title": (best.get("metadata") or {}).get("title"),
        "score": best.get("score"),
    }]
    # For a safe default, just surface the top passage text
    return {"answer": best.get("text", ""), "citations": citations}

