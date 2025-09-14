from typing import Protocol, List, Dict, Any, Optional, Tuple, Sequence
import os
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

DEFAULT_SYSTEM_PROMPT = "\n".join([
    "You are an iterative retrieval planner for multi-hop QA over unstructured text in chemistry.",
    "Return ONLY a JSON array containing EXACTLY ONE action. No prose. No comments.",
    "",
    "Allowed actions (schemas):",
    '{"action":"retrieve_text","query":"...","k":8,"where":null,"where_document":null,"partial_answer":"..."}',
    '{"action":"propose_answer","needs_citations":true,"answer":"..."}',
    "",
    "Inputs provided each call:",
    "",
    "original_question: full user question",
    "previous_queries: list of your prior retrieve_text queries for this question",
    "partial_answers: list of your prior partial answers (most recent last)",
    "passages: latest query's passages in full, plus top-2 from each earlier query",
    "planner_step: 1-based index of this planning call.",
    "planner_max_actions: maximum number of planning calls allowed.",
    "final_call: boolean; true when this is your last allowed LLM call",
    "",
    "Policy:",
    "",
    "Decompose: Identify a set of sub-questions needed to answer original_question.",
    "One-at-a-time: Each call must handle exactly ONE sub-question and return one action.",
    "Answer to the point. Your answers should be to the point. No need to explain a lot for propose_answer. You will observer some examples of how you should answwer later.",
    "If passages is empty: output one retrieve_text targeting the most critical sub-question.",
    "If passages do NOT resolve all sub-questions: output one retrieve_text for the NEXT unresolved sub-question only.",
    'Only when ALL sub-questions are supported by the retrieved passages: output one propose_answer with the final answer string in "answer".',
    "Final-call rule: If final_call is true, this is the last call to the LLM. You MUST generate an answer now based on the provided passages (the evidence) and take the propose_answer action. Do NOT output retrieve_text.",
    "When returning retrieve_text and passages is non-empty (i.e., not the first call), also include a concise 'partial_answer' summarizing your best current hypothesis based on the provided passages. On the very first call (no passages), omit 'partial_answer'.",
    "Query formation:",
    "- Include exact entities and properties from original_question (e.g., 'pKa of acetic acid').",
    "- Avoid vague keywords; target the missing fact precisely.",
    "Use previous_queries and partial_answers to refine and avoid repetition. Add specificity (entities, dates, synonyms, abbreviations) as you progress.",
    "Never guess. If evidence is insufficient, retrieve_text again rather than proposing an answer.",
    "Do NOT include any keys other than those shown. The JSON array must contain exactly one object.",
    "In answer field keep put the answer only. No need for explanation.",
    "",
    "Chemistry two-hop example (simulated):",
    "original_question: Which compound mentioned is aromatic, and what is the pKa of acetic acid?",
    "Call 1 output:",
    "[",
    '{"action":"retrieve_text","query":"Which compound is aromatic?","k":8,"where":null,"where_document":null}',
    "]",
    "Call 2 output:",
    "[",
    '{"action":"retrieve_text","query":"pKa of acetic acid","k":8,"where":null,"where_document":null,"partial_answer":"Aromatic compound identified; now need acetic acid pKa."}',
    "]",
    "Call 3 output:",
    "[",
    '{"action":"propose_answer","needs_citations":true,"answer":"Benzene is aromatic; acetic acid pKa approx 4.76."}',
    "]",
    "Here are some examples of how to broke the question down into atomic sub-questions and the final answer.",
    "Original question:What is the name of the molecular fragment that is responsible for delocalizing unpaired electrons in a radical formed on an aromatic hydrocarbon—one that also appears in compounds like an antiemetic featuring a pyridine ring and known for its role as a colorless, flammable industrial solvent precursor?",
    "Sub-questions: 1) Which molecular fragment is responsible for delocalizing its unpaired electrons across the benzene ring as seen in the described radical?",
    "2)Which aromatic hydrocarbon—a colorless, flammable liquid with a sweet odor that serves as an industrial solvent and precursor to various chemicals—is found in drugs such as Netupitant?",
    "3) Which antiemetic compound, a monocarboxylic acid amide used in combination with palonosetron for the prevention of chemotherapy-induced nausea and vomiting, contains a pyridine ring in its structure?",
    "Final answer: triazinyl",
    "Original question: What member of the Janus kinase family is inhibited by an orally available inhibitor with a pyrrolopyrimidine structure marketed under the brand name Xeljanz, which is used for treating a chronic autoimmune disorder characterized by warm, swollen, and painful joints and potential involvement of the skin, eyes, lungs, and heart?",
    "Sub-questions: 1) Which member of the Janus kinase family, a tyrosine kinase encoded by a gene described as a tyrosine-protein kinase, is inhibited by tofacitinib?",
    "2) Which orally available Janus kinase inhibitor, characterized by a pyrrolopyrimidine structure and marketed under the brand name Xeljanz, is approved for treating moderate to severe rheumatoid arthritis?",
    "3) Which chronic autoimmune disorder, known for causing warm, swollen, and painful joints and potentially affecting other parts of the body such as the skin, eyes, lungs, and heart, is approved to be treated with baricitinib, one of the first JAK inhibitors introduced in the USA and Europe?",
    "Final answer: JAK3",
    "The following example can't be broken down into sub questions.",
    "Original question:Which conductive polymer composite, known for its excellent material properties especially when incorporated with transition metal oxides, is used in fuel cells and photoelectrochemical water splitting?",
    "Sub-questions: 1) Which conductive polymer composite, known for its excellent material properties especially when incorporated with transition metal oxides, is used in fuel cells and photoelectrochemical water splitting?",
    "Fianal answer: polyaniline",
    "As you can see the answers are to the point and short. Now it's your turn.",
])

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
        passages_top_k: int = 5,
    ) -> None:
        self.llm = llm
        self.allow_kg = bool(allow_kg)
        self.default_k = int(default_k)
        self.max_actions = int(max_actions)
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.passages_top_k = int(passages_top_k)
        # Debug printing toggle via env: PLANNER_DEBUG=1|true
        self._debug = 1
        self._dbg_calls = 0

    def plan(self, question: str, state: Dict[str, Any]) -> List[Action]:
        # Build iterative user prompt expected by the system prompt
        filters = state.get("filters") or None
        prev_queries = state.get("previous_queries") or []
        evidence = state.get("evidence") or []

        def _score_key(h: Dict[str, Any]) -> float:
            if h.get("score") is not None:
                return float(h["score"])
            if h.get("score_dense") is not None:
                return float(h["score_dense"])
            return 0.0

        # Build passage set: include ALL from the latest query, plus top-2 from each earlier query
        last_hits = list(state.get("last_passages") or [])
        # Determine latest step if annotated; else fall back to id exclusion
        latest_step = None
        for h in last_hits:
            if isinstance(h, dict) and ("source_step" in h):
                latest_step = h.get("source_step")
                break

        if latest_step is not None:
            others = [h for h in (evidence or []) if h.get("source_step") != latest_step]
        else:
            last_ids = set()
            for h in last_hits:
                if isinstance(h, dict) and (h.get("id") is not None):
                    last_ids.add(str(h.get("id")))
            others = [h for h in (evidence or []) if str(h.get("id")) not in last_ids]

        # Group older hits by their origin query (preferred) or step
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for h in others:
            key = str(h.get("source_query") or h.get("source_step") or "ungrouped")
            groups.setdefault(key, []).append(h)
        # Pick top-2 per older group
        older_top: List[Dict[str, Any]] = []
        for key, items in groups.items():
            items_sorted = sorted(items, key=_score_key, reverse=True)
            older_top.extend(items_sorted[:2])

        chosen = list(last_hits) + older_top

        # Serialize passages compactly: include id/title if present and text
        out_lines: List[str] = []
        for i, h in enumerate(chosen, 1):
            pid = str(h.get("id") or i)
            meta = h.get("metadata") or {}
            title = meta.get("title")
            header = f"[{i}] id={pid}" + (f" | title={title}" if title else "")
            text = str(h.get("text", "")).strip()
            # Clip very long texts to keep prompt size manageable
            if len(text) > 1200:
                text = text[:1200] + " ..."
            out_lines.append(header)
            out_lines.append(text)
        passages_block = "\n".join(out_lines) if out_lines else "[]"

        # Note: On the very first call, chosen==[] (no passages). The system prompt
        # instructs the model to return a retrieve_text WITHOUT 'partial_answer' on
        # that first turn. Subsequent turns include partial_answers and passages.
        # Include planning budget annotations for the model
        planner_step = state.get("planner_step") or None
        planner_max_actions = state.get("planner_max_actions") or self.max_actions
        is_final = bool(state.get("is_final_call") or False)

        user_prompt = (
            f"original_question: {question}\n"
            f"previous_queries: {prev_queries}\n"
            f"partial_answers: {state.get('partial_answers') or []}\n"
            f"passages:\n{passages_block}\n"
            f"planner_step: {planner_step}\n"
            f"planner_max_actions: {planner_max_actions}\n"
            f"final_call: {is_final}\n"
        )

        # Optional debug prints
        if self._debug:
            self._dbg_calls += 1
            if self._dbg_calls == 1:
                print("[Planner] System prompt (once):\n" + self.system_prompt)
                print("-" * 60)
            print(f"[Planner] User prompt (call {self._dbg_calls}):\n" + user_prompt)
            print("-" * 60)

        raw = self.llm.complete(self.system_prompt, user_prompt)

        # Extract a JSON array from the raw LLM text (be robust to extra prose)
        json_text, ok = _extract_json_array(raw)
        if not ok:
            # Fallback: return exactly one action based on whether we have evidence
            if bool(state.get("is_final_call")):
                return [ProposeAnswer(needs_citations=True)]
            if state.get("evidence"):
                return [ProposeAnswer(needs_citations=True)]
            return [RetrieveText(query=question, k=self.default_k, where=filters, where_document=None)]

        try:
            actions = parse_actions_json(json_text)
        except Exception:
            if bool(state.get("is_final_call")):
                return [ProposeAnswer(needs_citations=True)]
            if state.get("evidence"):
                return [ProposeAnswer(needs_citations=True)]
            return [RetrieveText(query=question, k=self.default_k, where=filters, where_document=None)]

        # Optionally strip KG actions if KG is not enabled yet
        if not self.allow_kg:
            actions = [a for a in actions if not isinstance(a, RetrieveKG)]

        # Enforce EXACTLY ONE action from the LLM output by truncation
        if len(actions) > 1:
            actions = actions[:1]
        # Fill defaults on the single action
        actions = _fill_defaults(actions, default_k=self.default_k, filters=filters)

        # Validate & fallback if needed
        errs = validate_actions(actions)
        if errs:
            # If invalid, fall back to a single actionable step
            if bool(state.get("is_final_call")):
                return [ProposeAnswer(needs_citations=True)]
            if state.get("evidence"):
                return [ProposeAnswer(needs_citations=True)]
            return [RetrieveText(query=question, k=self.default_k, where=filters, where_document=None)]

        # Ensure the plan is actionable: must contain at least one action
        if not actions:
            if bool(state.get("is_final_call")):
                return [ProposeAnswer(needs_citations=True)]
            if state.get("evidence"):
                return [ProposeAnswer(needs_citations=True)]
            return [RetrieveText(query=question, k=self.default_k, where=filters, where_document=None)]

        # Enforce propose_answer on final call even if the LLM suggested retrieval
        if bool(state.get("is_final_call")) and not isinstance(actions[0], ProposeAnswer):
            return [ProposeAnswer(needs_citations=True)]

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
