"""
LLM-backed planner wiring for iterative RAG.

This module offers small adapters that implement the LLMClient Protocol
defined in repo/planning/planner_iface.py, so you can plug different LLM
providers without changing the planner/orchestrator code.

Adapters:
  - OpenAIChatLLM      (uses 'openai' python package, chat.completions)
  - OllamaLLM          (local HTTP server at http://localhost:11434)
  - HFInferenceLLM     (Hugging Face Inference API via HTTP)

Also includes:
  - make_json_planner(...) : returns a JSONPlanner with a solid system prompt

All adapters must implement: complete(system: str, user: str) -> str
and return the *raw* model output (the planner will extract JSON).
"""

from typing import Any, Dict, List, Optional, Callable
import json

from repo.planning.planner_iface import LLMClient, JSONPlanner


# -------------------------- Prompt helper -------------------------------------

def default_planner_system_prompt(allow_kg: bool = False) -> str:
    """
    Iterative one-action planner prompt (chemistry domain), aligned to orchestrator loop.
    Note: allow_kg is ignored in this prompt version (text-only retrieval).
    """
    return "\n".join([
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
        "planner_step: 1-based index of this planning call",
        "planner_max_actions: maximum number of planning calls allowed",
        "final_call: boolean; true when this is your last allowed LLM call",
        "",
        "Policy:",
        "",
        "Decompose: Identify the minimal set of atomic sub-questions needed to answer original_question.",
        "One-at-a-time: Each call must handle exactly ONE sub-question and return one action.",
        "If passages is empty: output one retrieve_text targeting the most critical sub-question.",
        "If passages do NOT resolve all sub-questions: output one retrieve_text for the NEXT unresolved sub-question only.",
        "Only when ALL sub-questions are supported by the retrieved passages: output one propose_answer with the final answer string in \"answer\".",
        "Final-call rule: If final_call is true, this is your last call to the LLM. You MUST generate an answer now based on the provided passages (the evidence) and return exactly one propose_answer. Do NOT output retrieve_text.",
        "When returning retrieve_text and passages is non-empty (i.e., not the first call), also include a concise 'partial_answer' summarizing your best current hypothesis based on the provided passages. On the very first call (no passages), omit 'partial_answer'.",
        "Query formation:",
        "- Do NOT combine multiple sub-questions in a single query (avoid conjunctions like 'and').",
        "- Include exact entities and properties from original_question (e.g., 'pKa of acetic acid').",
        "- Avoid vague keywords; target the missing fact precisely.",
        "Use previous_queries and partial_answers to refine and avoid repetition. Add specificity (entities, dates, synonyms, abbreviations) as you progress.",
        "Never guess. If evidence is insufficient, retrieve_text again rather than proposing an answer.",
        "Do NOT include any keys other than those shown. The JSON array must contain exactly one object.",
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
        "",
        "Anti-pattern (do NOT do this):",
        "[",
        '{"action":"retrieve_text","query":"aromatic compounds and pKa of acetic acid","k":8,"where":null,"where_document":null}',
        "]",
        "Reason: Combines two sub-questions; always target one missing fact per retrieve_text.",
    ])
# -------------------------- Factory -------------------------------------------

def make_json_planner(
    llm: LLMClient,
    *,
    allow_kg: bool = False,
    default_k: int = 8,
    max_actions: int = 6,
    system_prompt: Optional[str] = None,
    passages_top_k: int = 5,
) -> JSONPlanner:
    """
    Convenience factory to build a JSONPlanner with a good default system prompt.
    """
    sys_prompt = system_prompt or default_planner_system_prompt(allow_kg=allow_kg)
    return JSONPlanner(
        llm=llm,
        allow_kg=allow_kg,
        default_k=default_k,
        max_actions=max_actions,
        system_prompt=sys_prompt,
        passages_top_k=passages_top_k,
    )


# --------------------------- Adapters -----------------------------------------

class OpenAIChatLLM(LLMClient):
    """
    OpenAI Chat Completions adapter (requires 'openai' package >= 1.0).
    Example:
        llm = OpenAIChatLLM(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        text = llm.complete(system="...", user="...")
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 400,
        timeout: float = 60.0,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.organization = organization
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        self.extra_headers = extra_headers or {}

    def complete(self, system: str, user: str) -> str:
        try:
            # OpenAI SDK (>=1.0 style)
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError(
                "OpenAIChatLLM requires the 'openai' package (>=1.0). "
                "Install via `pip install openai`."
            ) from e

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            organization=self.organization,
            timeout=self.timeout,
            default_headers=self.extra_headers,
        )
        # Chat Completions API
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = (resp.choices[0].message.content or "").strip()
        return content


class OllamaLLM(LLMClient):
    """
    Ollama chat adapter (local server).
    Requires 'requests': pip install requests

    Example:
        llm = OllamaLLM(model="llama3.1:8b-instruct")
        text = llm.complete(system="...", user="...")
    """

    def __init__(
        self,
        *,
        model: str = "llama3.1:8b-instruct",
        endpoint: str = "http://localhost:11434/api/chat",
        temperature: float = 0.0,
        num_predict: int = 400,
        timeout: float = 120.0,
        headers: Optional[Dict[str, str]] = None,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.temperature = float(temperature)
        self.num_predict = int(num_predict)
        self.timeout = float(timeout)
        self.headers = headers or {}
        self.extra_options = extra_options or {}

    def complete(self, system: str, user: str) -> str:
        try:
            import requests
        except Exception as e:
            raise RuntimeError("OllamaLLM requires the 'requests' package.") from e

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
                **self.extra_options,
            },
            "stream": False,
        }
        r = requests.post(self.endpoint, json=payload, timeout=self.timeout, headers=self.headers)
        r.raise_for_status()
        data = r.json()
        # Ollama returns a list of messages in 'message' for chat endpoint
        msg = data.get("message", {})
        content = (msg.get("content") or "").strip()
        return content


class HFInferenceLLM(LLMClient):
    """
    Hugging Face Inference API adapter (text-generation endpoints).
    Requires 'requests'.

    Notes:
      - This is a *generic* text-generation call; different models may need
        slightly different inputs. We prepend a simple system/user template.
      - Set 'stream' to False; planner expects one-shot text.
    Example:
        llm = HFInferenceLLM(model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key=HF_TOKEN)
        text = llm.complete(system="...", user="...")
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        max_new_tokens: int = 400,
        temperature: float = 0.0,
        top_p: float = 1.0,
        base_url: str = "https://api-inference.huggingface.co/models",
        headers: Optional[Dict[str, str]] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.timeout = float(timeout)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        if self.api_key:
            self.headers.setdefault("Authorization", f"Bearer {self.api_key}")
        self.extra_payload = extra_payload or {}

    def complete(self, system: str, user: str) -> str:
        try:
            import requests
        except Exception as e:
            raise RuntimeError("HFInferenceLLM requires the 'requests' package.") from e

        url = f"{self.base_url}/{self.model}"
        # Simple prompt template; many instruct models handle this well.
        prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "return_full_text": False,
            },
            "options": {"use_cache": True, "wait_for_model": True},
        }
        # Allow user overrides
        if self.extra_payload:
            # shallow-merge parameters if present
            params = payload.get("parameters", {})
            params.update(self.extra_payload.get("parameters", {}))
            payload["parameters"] = params
            for k, v in self.extra_payload.items():
                if k != "parameters":
                    payload[k] = v

        r = requests.post(url, headers=self.headers, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        # HF Inference returns a list of dicts with 'generated_text'
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return (data[0]["generated_text"] or "").strip()
        # Some endpoints return a dict with 'generated_text' or errors
        if isinstance(data, dict) and "generated_text" in data:
            return (data["generated_text"] or "").strip()
        # Fallback to raw JSON string
        return json.dumps(data)


# --------------------------- Composer (LLM) -----------------------------------

def default_composer_system_prompt() -> str:
    """
    System prompt for evidence-grounded, chemistry-focused answer generation.

    Rules:
    - Use ONLY the provided passages as sources of truth.
    - If passages are insufficient or conflict, answer with empty string.
    - Be concise, precise, and avoid speculation.
    - Optionally reference passage ids inline like [id: ...] if helpful.
    """
    return "\n".join([
        "You are a chemistry QA assistant.",
        "Answer ONLY using the passages provided by the retrieval system.",
        "If the passages are insufficient or conflicting, say so explicitly.",
        "Be concise, precise, and avoid speculation.",
    ])


def make_llm_composer(
    llm: LLMClient,
    *,
    system_prompt: Optional[str] = None,
    max_passages: int = 6,
    max_chars: int = 18000,
    cite_top_k: int = 3,
) -> Callable[[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Build a composer callable that asks an LLM to write the final answer based on retrieved passages.

    - llm: any LLMClient (OpenAIChatLLM, OllamaLLM, HFInferenceLLM, ...)
    - system_prompt: optional override; otherwise uses default_composer_system_prompt()
    - max_passages: include at most this many passages in the prompt
    - max_chars: total character budget for concatenated passages (approximate)
    - cite_top_k: number of citations to return (best-effort, top by score)

    Returns a composer(question, evidence)->{"answer": str, "citations": [...]}
    """
    sys_prompt = system_prompt or default_composer_system_prompt()

    def _choose_passages(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Prefer items with an overall score, then dense score
        def key(h: Dict[str, Any]) -> float:
            if h.get("score") is not None:
                return float(h["score"])  # fused score, higher is better
            if h.get("score_dense") is not None:
                return float(h["score_dense"])  # similarity
            # lexical-only or unknown; push lower
            return 0.0

        items = sorted(list(evidence or []), key=key, reverse=True)
        out: List[Dict[str, Any]] = []
        used = 0
        for h in items:
            txt = str(h.get("text", "")).strip()
            if not txt:
                continue
            if used + len(txt) > max_chars and out:
                break
            out.append(h)
            used += len(txt)
            if len(out) >= max_passages:
                break
        return out

    def _pick_citations(evidence: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        items = list(evidence or [])
        def key(h: Dict[str, Any]) -> float:
            if h.get("score") is not None:
                return float(h["score"])  # fused
            if h.get("score_dense") is not None:
                return float(h["score_dense"])  # similarity
            return 0.0
        items.sort(key=key, reverse=True)
        cites: List[Dict[str, Any]] = []
        for h in items[: max(0, int(k))]:
            meta = h.get("metadata") or {}
            cites.append({
                "id": h.get("id"),
                "title": meta.get("title"),
                "score": h.get("score"),
            })
        return cites

    def composer(question: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        chosen = _choose_passages(evidence)
        if not chosen:
            # Nothing to ground on
            return {"answer": "I don't have enough information to answer from the provided passages.", "citations": []}

        # Build a compact user message with labeled passages
        lines: List[str] = []
        lines.append(f"Question: {question}")
        lines.append("")
        lines.append("Passages:")
        for i, h in enumerate(chosen, 1):
            pid = str(h.get("id") or i)
            title = (h.get("metadata") or {}).get("title")
            header = f"[{i}] id={pid}" + (f" | title={title}" if title else "")
            text = str(h.get("text", "")).strip()
            lines.append(header)
            lines.append(text)
            lines.append("")
        lines.append("Instructions: Answer strictly using the passages above. If insufficient, say so. Be concise.")
        user_msg = "\n".join(lines)

        answer = llm.complete(system=sys_prompt, user=user_msg).strip()
        citations = _pick_citations(chosen, cite_top_k)
        return {"answer": answer, "citations": citations}

    return composer
