"""
LLMClient adapter that reuses the StructuredLLM clients from src/benchmark/evaluator.py
so the planner/composer run through the same provider/model configuration.

Implements the minimal LLMClient interface: complete(system, user) -> str
"""

from typing import Optional, Dict, Any

from repo.planning.planner_iface import LLMClient
try:  # prefer package-qualified import
    from src.benchmark.evaluator import StructuredLLM, Provider, Answer
except Exception:  # fallback to namespace import when PYTHONPATH=src
    from benchmark.evaluator import StructuredLLM, Provider, Answer


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
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._extra = extra_options or {}
        self._debug = bool(debug)
        # Simple cumulative usage counters across calls
        self._usage: Dict[str, int] = {
            "input": 0,
            "output": 0,
            "reasoning": 0,
        }

    def get_and_reset_usage(self) -> Dict[str, int]:
        out = dict(self._usage)
        self._usage = {"input": 0, "output": 0, "reasoning": 0}
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
            resp = self._core.client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_completion_tokens=self._max_tokens,
            )
            choice = resp.choices[0]
            out = (choice.message.content or "").strip()
            try:
                self._usage["input"] += int(getattr(resp.usage, "prompt_tokens", 0) or 0)
                self._usage["output"] += int(getattr(resp.usage, "completion_tokens", 0) or 0)
                details = getattr(resp.usage, "completion_tokens_details", None)
                if details is not None:
                    self._usage["reasoning"] += int(getattr(details, "reasoning_tokens", 0) or 0)
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
            resp = self._core.client.chat(
                model=self._model,
                messages=messages,
                options=opts,
            )
            out = (getattr(resp, "message", {}).get("content") or "").strip()
            try:
                # Ollama chat response usage fields
                self._usage["input"] += int(getattr(resp, "prompt_eval_count", 0) or 0)
                self._usage["output"] += int(getattr(resp, "eval_count", 0) or 0)
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
                self._usage["input"] += int(usage.get("prompt_tokens", 0) or 0)
                self._usage["output"] += int(usage.get("completion_tokens", 0) or 0)
            except Exception:
                pass
            if self._debug:
                print(f"[LLM:{self._provider.value}:{self._model}] <<< output:\n{out}\n")
            return out

        if self._provider == Provider.BEDROCK:
            # Use a plain Converse client (no structured output) so planner/composer get raw text like GPT‑4o/DeepSeek.
            try:
                from langchain_aws.chat_models.bedrock_converse import ChatBedrockConverse
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
                # Extract plain text content
                content = getattr(resp, "content", None)
                out = ""
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            out = (item.get("text") or "").strip()
                            break
                elif isinstance(content, str):
                    out = content.strip()
                # Track usage if available
                try:
                    um = getattr(resp, "usage_metadata", None)
                    if isinstance(um, dict):
                        self._usage["input"] += int(um.get("input_tokens", 0) or 0)
                        self._usage["output"] += int(um.get("output_tokens", 0) or 0)
                except Exception:
                    pass
                if self._debug:
                    print(f"[LLM:{self._provider.value}:{self._model}] <<< output:\n{out}\n")
                return out
            except Exception:
                return ""

        # Unsupported provider
        raise ValueError(f"Unsupported provider for StructuredLLMClient: {self._provider}")
