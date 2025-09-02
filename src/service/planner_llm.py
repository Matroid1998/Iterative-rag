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

from typing import Any, Dict, List, Optional
import json

from repo.planning.planner_iface import LLMClient, JSONPlanner


# -------------------------- Prompt helper -------------------------------------

def default_planner_system_prompt(allow_kg: bool = False) -> str:
    """
    System prompt that instructs the model to return ONLY a JSON array of actions.
    If allow_kg=False, we omit the retrieve_kg schema from the instructions.
    """
    base = [
        "You are a retrieval planner for multi-hop QA over unstructured text.",
        "Return ONLY a JSON array of actions. No prose. No comments.",
        "Allowed actions:",
        '  {"action":"retrieve_text","query":"...","k":8,"where":null,"where_document":null}',
        '  {"action":"propose_answer","needs_citations":true}',
        "Rules:",
        "- Keep plans short (1–3 actions typical).",
        "- Use retrieve_text for each sub-question you need evidence for.",
        "- Finish with propose_answer once you have enough evidence.",
        "- Do NOT include any other keys or text.",
    ]
    if allow_kg:
        base.insert(4, '  {"action":"retrieve_kg","seed_entities":["..."],"relation":null,"max_hops":2}')
        base.insert(6, "- Prefer retrieve_kg only when relations/entities are explicit.")
    return "\n".join(base)


# -------------------------- Factory -------------------------------------------

def make_json_planner(
    llm: LLMClient,
    *,
    allow_kg: bool = False,
    default_k: int = 8,
    max_actions: int = 6,
    system_prompt: Optional[str] = None,
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
            ],
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
