"""
LLMClient adapter that reuses the StructuredLLM clients from src/benchmark/evaluator.py
so the planner/composer run through the same provider/model configuration.

Implements the minimal LLMClient interface: complete(system, user) -> str

Instrumentation only: adds per-call usage ledger without changing behavior.
"""

from typing import Optional, Dict, Any, List, Tuple

from repo.planning.planner_iface import LLMClient
try:  # prefer package-qualified import
    from src.benchmark.evaluator import (
        StructuredLLM,
        Provider,
        Answer,
        OPENAI_REASONING_MODELS,
    )
except Exception:  # fallback to namespace import when PYTHONPATH=src
    from benchmark.evaluator import (
        StructuredLLM,
        Provider,
        Answer,
        OPENAI_REASONING_MODELS,
    )


class StructuredLLMClient(LLMClient):
    """
    Wraps StructuredLLM to produce raw chat completions for planner/composer.

    Notes:
    - Uses the underlying provider SDK initialized by StructuredLLM
    - Returns raw string content (no JSON enforcement)
    - Currently supports OpenAI and Ollama; extend as needed
    """

    def __init__(
        self,
        *,
        provider: str = "openai",
        model: str = "gpt-4o",
        temperature: float = 0.2,
        max_tokens: int = 600,
        extra_options: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ) -> None:
        prov = provider if isinstance(provider, Provider) else Provider(provider)
        # Initialize StructuredLLM to set up the underlying client with env keys
        # We use Answer schema as a placeholder; we won't call the structured path.
        self._core = StructuredLLM(
            provider=prov,
            model_id=model,
            output_format=Answer,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        self._provider = prov
        self._model = model
        self._reasoning_model = (
            self._provider == Provider.OPENAI and self._model in OPENAI_REASONING_MODELS
        )
        self._temperature = None if self._reasoning_model else float(temperature)
        self._max_tokens = max_tokens
        self._extra = extra_options or {}
        self._debug = bool(debug)
        # Simple cumulative usage counters across calls
        self._usage: Dict[str, int] = {
            "input": 0,
            "output": 0,
            "reasoning": 0,
        }
        # Per-call ledger for detailed tracking
        self._calls: List[Dict[str, Any]] = []

    # ------------------------ helpers ------------------------
    @staticmethod
    def _label_role(system_prompt: str) -> str:
        s = (system_prompt or "").lower()
        if (
            "return only a json array" in s
            or "allowed actions (schemas)" in s
            or "iterative retrieval planner" in s
            or "planner_step" in s
        ):
            return "planner"
        if "you are a chemistry qa assistant" in s:
            return "composer"
        return "unknown"

    @staticmethod
    def _extract_json_array(text: str) -> Tuple[str, bool]:
        if not text:
            return "", False
        s = text.strip()
        if s.startswith("[") and s.endswith("]"):
            return s, True
        import re
        m = re.search(r"\[\s*{.*?}\s*\]", s, flags=re.DOTALL)
        if not m:
            return "", False
        return m.group(0), True

    @classmethod
    def _extract_partial_and_proposed(cls, raw: str) -> Tuple[Optional[str], Optional[str]]:
        if not raw:
            return None, None
        try:
            import json as _json
            arr_txt, ok = cls._extract_json_array(raw)
            if not ok:
                return None, None
            data = _json.loads(arr_txt)
            if isinstance(data, list) and data:
                obj = data[0]
                if isinstance(obj, dict):
                    pa = obj.get("partial_answer")
                    if isinstance(pa, str) and pa.strip():
                        partial = pa.strip()
                    else:
                        partial = None
                    proposed = None
                    if obj.get("action") == "propose_answer":
                        ans = obj.get("answer")
                        if isinstance(ans, str) and ans.strip():
                            proposed = ans.strip()
                    return partial, proposed
        except Exception:
            pass
        return None, None

    @staticmethod
    def _extract_bedrock_text_and_reason(content: Any) -> Tuple[str, Optional[str]]:
        out_text = ""
        reason_text = None
        try:
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        t = item.get("type")
                        if t in {"reasoning_content", "thinking", "reasoning"}:
                            rc = item.get("reasoning_content")
                            if isinstance(rc, dict) and rc.get("text"):
                                reason_text = (rc.get("text") or "").strip()
                            elif item.get("text"):
                                reason_text = (item.get("text") or "").strip()
                # prefer first explicit text node for final output
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        out_text = (item.get("text") or "").strip()
                        break
                if not out_text:
                    # concatenate any text-like nodes as fallback
                    texts: List[str] = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") in {"text", "output_text"} and item.get("text"):
                            texts.append(item.get("text") or "")
                    out_text = ("".join(texts)).strip()
            elif isinstance(content, str):
                out_text = content.strip()
        except Exception:
            pass
        return out_text, reason_text

    def get_and_reset_usage(self) -> Dict[str, Any]:
        out = {
            "input": int(self._usage.get("input", 0)),
            "output": int(self._usage.get("output", 0)),
            "reasoning": int(self._usage.get("reasoning", 0)),
            "calls": list(self._calls),
        }
        self._usage = {"input": 0, "output": 0, "reasoning": 0}
        self._calls = []
        return out

    def complete(self, system: str, user: str) -> str:
        # Build minimal messages per provider expectations
        if self._provider == Provider.OPENAI or self._provider == Provider.NVIDIA:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            if self._debug:
                print(f"[LLM:{self._provider.value}:{self._model}] >>> system:\n{system}\n")
                print(f"[LLM:{self._provider.value}:{self._model}] >>> user:\n{user}\n")
            import time as _t
            _t0 = _t.time()
            reasoning_kwargs: Dict[str, Any] = {}
            if self._model == "gpt-5":
                reasoning_kwargs["reasoning_effort"] = "medium"
            request_kwargs: Dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "max_completion_tokens": self._max_tokens,
                **reasoning_kwargs,
            }
            if self._temperature is not None:
                request_kwargs["temperature"] = self._temperature
            resp = self._core.client.chat.completions.create(**request_kwargs)
            choice = resp.choices[0]
            out = (choice.message.content or "").strip()
            elapsed_ms = int((_t.time() - _t0) * 1000)
            try:
                self._usage["input"] += int(getattr(resp.usage, "prompt_tokens", 0) or 0)
                self._usage["output"] += int(getattr(resp.usage, "completion_tokens", 0) or 0)
                details = getattr(resp.usage, "completion_tokens_details", None)
                r_tok = 0
                if details is not None:
                    r_tok = int(getattr(details, "reasoning_tokens", 0) or 0)
                    self._usage["reasoning"] += r_tok
                # Per-call ledger
                role = self._label_role(system)
                partial, proposed = self._extract_partial_and_proposed(out)
                self._calls.append({
                    "provider": self._provider.value,
                    "model": self._model,
                    "role": role,
                    "input_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
                    "output_tokens": int(getattr(resp.usage, "completion_tokens", 0) or 0),
                    "reasoning_tokens": r_tok,
                    "reasoning_text": None,
                    "partial_answer": partial,
                    "proposed_answer": proposed,
                    "elapsed_ms": elapsed_ms,
                })
            except Exception:
                pass
            if self._debug:
                print(f"[LLM:{self._provider.value}:{self._model}] <<< output:\n{out}\n")
            return out

        if self._provider == Provider.OLLAMA:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            opts = {"temperature": self._temperature, "num_predict": self._max_tokens}
            opts.update(self._extra.get("options", {}))
            if self._debug:
                print(f"[LLM:{self._provider.value}:{self._model}] >>> system:\n{system}\n")
                print(f"[LLM:{self._provider.value}:{self._model}] >>> user:\n{user}\n")
            import time as _t
            _t0 = _t.time()
            resp = self._core.client.chat(
                model=self._model,
                messages=messages,
                options=opts,
            )
            out = (getattr(resp, "message", {}).get("content") or "").strip()
            try:
                # Ollama chat response usage fields
                in_tok = int(getattr(resp, "prompt_eval_count", 0) or 0)
                out_tok = int(getattr(resp, "eval_count", 0) or 0)
                self._usage["input"] += in_tok
                self._usage["output"] += out_tok
                role = self._label_role(system)
                partial, proposed = self._extract_partial_and_proposed(out)
                self._calls.append({
                    "provider": self._provider.value,
                    "model": self._model,
                    "role": role,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "reasoning_tokens": 0,
                    "reasoning_text": None,
                    "partial_answer": partial,
                    "proposed_answer": proposed,
                    "elapsed_ms": int((_t.time() - _t0) * 1000),
                })
            except Exception:
                pass
            if self._debug:
                print(f"[LLM:{self._provider.value}:{self._model}] <<< output:\n{out}\n")
            return out

        if self._provider == Provider.OPENROUTER:
            # Use OpenRouter HTTP API via requests
            import os
            import requests
            payload = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
            }
            if self._debug:
                print(f"[LLM:{self._provider.value}:{self._model}] >>> system:\n{system}\n")
                print(f"[LLM:{self._provider.value}:{self._model}] >>> user:\n{user}\n")
            import time as _t
            _t0 = _t.time()
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY','')}",
                },
                json=payload,
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            out = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            try:
                usage = data.get("usage") or {}
                in_tok = int(usage.get("prompt_tokens", 0) or 0)
                out_tok = int(usage.get("completion_tokens", 0) or 0)
                self._usage["input"] += in_tok
                self._usage["output"] += out_tok
                role = self._label_role(system)
                partial, proposed = self._extract_partial_and_proposed(out)
                self._calls.append({
                    "provider": self._provider.value,
                    "model": self._model,
                    "role": role,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "reasoning_tokens": 0,
                    "reasoning_text": None,
                    "partial_answer": partial,
                    "proposed_answer": proposed,
                    "elapsed_ms": int((_t.time() - _t0) * 1000),
                })
            except Exception:
                pass
            if self._debug:
                print(f"[LLM:{self._provider.value}:{self._model}] <<< output:\n{out}\n")
            return out

        if self._provider == Provider.BEDROCK:
            # Use a plain Converse client (no structured output) so planner/composer get raw text like GPT‑4o/DeepSeek.
            try:
                from langchain_aws.chat_models.bedrock_converse import ChatBedrockConverse
                import time as _t
                _t0 = _t.time()
                # Send system + user as a single user turn to match earlier working behavior
                messages = [{"role": "user", "content": [{"text": system + "\n\n" + user}]}]
                llm = ChatBedrockConverse(
                    client=self._core.client,
                    model_id=getattr(self._core, "model_id", self._model),
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )
                if self._debug:
                    print(f"[LLM:{self._provider.value}:{self._model}] >>> system+user as text\n{system}\n\n{user}\n")
                resp = llm.invoke(messages)
                # Extract plain text and reasoning text
                content = getattr(resp, "content", None)
                out, reason_txt = self._extract_bedrock_text_and_reason(content)
                # Track usage if available
                in_tok = out_tok = 0
                try:
                    um = getattr(resp, "usage_metadata", None)
                    if isinstance(um, dict):
                        in_tok = int(um.get("input_tokens", 0) or 0)
                        out_tok = int(um.get("output_tokens", 0) or 0)
                        self._usage["input"] += in_tok
                        self._usage["output"] += out_tok
                except Exception:
                    pass
                # Estimate reasoning tokens if we have reason text
                r_tok = 0
                if isinstance(reason_txt, str) and reason_txt.strip():
                    r_tok = len(reason_txt.strip().split())
                    self._usage["reasoning"] += r_tok
                # Per-call ledger
                role = self._label_role(system)
                partial, proposed = self._extract_partial_and_proposed(out)
                self._calls.append({
                    "provider": self._provider.value,
                    "model": self._model,
                    "role": role,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "reasoning_tokens": r_tok,
                    "reasoning_text": reason_txt,
                    "partial_answer": partial,
                    "proposed_answer": proposed,
                    "elapsed_ms": int((_t.time() - _t0) * 1000),
                })
                if self._debug:
                    print(f"[LLM:{self._provider.value}:{self._model}] <<< output:\n{out}\n")
                return out
            except Exception:
                return ""

        # Unsupported provider
        raise ValueError(f"Unsupported provider for StructuredLLMClient: {self._provider}")
