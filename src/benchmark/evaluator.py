import time
import json
import random
import tqdm
import math
import os
import re

import ollama
import boto3
from botocore.config import Config
from openai import OpenAI, NotGiven

from enum import Enum
from typing import Dict, Any, Union, List, Tuple, Callable, Optional
from langchain_aws.chat_models.bedrock_converse import ChatBedrockConverse
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import concurrent.futures
from threading import Lock
from datetime import datetime
import requests

load_dotenv()


def read_json(file_path: str) -> dict:
    """Read the JSON file from the given file path."""
    with open(file_path, "r") as file:
        return json.load(file)


class Answer(BaseModel):
    answer: str = Field(..., description="A short answer to the question")


class AreSimilar(BaseModel):
    are_the_same: bool = Field(..., description="Whether the two answers are the same.")


class LegalVerificationScore(BaseModel):
    score: int = Field(..., description="Legal accuracy score from 1 to 10", ge=1, le=10)


JSON_ENFORCE = (
    "Please return only the raw JSON string that strictly conforms to the following JSON schema, with no additional text: {json_schema}"
    "Example: {example}"
)


class Provider(str, Enum):
    OPENAI = "openai"
    BEDROCK = "bedrock"
    NVIDIA = "nvidia"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


OPENAI_REASONING_MODELS = {
    "o1",
    "o1-preview",
    "o1-mini",
    "o3-mini",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5.1-2025-11-13",
    "gpt-5.1",
}


class ModelRegistry:
    """Registry to manage provider-model relationships and benchmarking state."""

    PROVIDER_MODELS = {
        Provider.OPENAI: [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-5",
            "gpt-5-mini",
            # "o1",
            "o1-mini",
            "o3-mini",
            "gpt-5.1-2025-11-13",
            "gpt-5.1"
        ],
        Provider.BEDROCK: [
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "us.meta.llama3-3-70b-instruct-v1:0",
            "mistral.mistral-large-2402-v1:0",
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning",
            "us.deepseek.r1-v1:0-reasoning",
        ],
        Provider.OLLAMA: [
            # "gemma3:27b",
            # "deepseek-r1:32b-reasoning",
        ],
        Provider.OPENROUTER: [
            "anthropic/claude-sonnet-4.5",
            "google/gemini-2.5-pro",
            "z-ai/glm-4.6",
            "x-ai/grok-4-fast",
            "deepseek/deepseek-chat-v3.1",
            "google/gemini-3-pro-preview"
        ],
        Provider.NVIDIA: [
            # "deepseek-ai/deepseek-r1",
        ],
    }

    def __init__(self):
        self.completed_benchmarks = set()

    def get_models_for_provider(self, provider: Provider) -> List[str]:
        """Get all models supported by a specific provider."""
        return self.PROVIDER_MODELS.get(provider, [])

    def is_valid_model(self, provider: Provider, model: str) -> bool:
        """Check if a model is valid for a given provider."""
        return model in self.PROVIDER_MODELS.get(provider, [])

    def get_all_provider_model_combinations(self) -> List[Tuple[Provider, str]]:
        """Get all valid provider-model combinations."""
        combinations = []
        for provider in Provider:
            for model in self.get_models_for_provider(provider):
                combinations.append((provider, model))
        return combinations

    def mark_as_completed(self, provider: Provider, model: str):
        """Mark a provider-model combination as completed."""
        self.completed_benchmarks.add((provider, model))


class StructuredLLM:
    def __init__(
        self,
        provider: Union[Provider, str],
        model_id: str,
        output_format: BaseModel,
        temperature: float = 0.2,
        max_completion_tokens: int = 8192,
    ):
        self.provider = (
            provider if isinstance(provider, Provider) else Provider(provider)
        )
        self.model_id = model_id
        self.output_format = output_format
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.is_reasoning = False
        self.thinking_params = None

        model_registry = ModelRegistry()
        if not model_registry.is_valid_model(self.provider, self.model_id):
            raise ValueError(
                f"Model '{self.model_id}' is not supported by provider '{self.provider}'"
            )

        if "reasoning" in self.model_id:
            self.model_id = self.model_id.replace("-reasoning", "")
            self.is_reasoning = True
            if "claude-3-7" in self.model_id:
                self.temperature = 1.0
                self.thinking_params = {
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": self.max_completion_tokens - 256,
                    }
                }
            else:
                self.temperature = 0.6

        if self.provider in [Provider.OPENAI, Provider.NVIDIA, Provider.OPENROUTER]:
            self.api_key = self._get_api_key()
        self.client = self._initialize_client()
        if self.provider == Provider.BEDROCK:
            self.bedrock_llm = self._get_bedrock_llm()

        if self.provider == Provider.OPENAI and self.model_id in OPENAI_REASONING_MODELS:
            self.temperature = NotGiven()

    def _get_api_key(self) -> str:
        """Get API key from environment variables based on the provider."""
        key_mapping = {
            Provider.OPENAI: "OPENAI_API_KEY",
            Provider.NVIDIA: "NVIDIA_API_KEY",
            Provider.OPENROUTER: "OPENROUTER_API_KEY",
        }

        if self.provider in key_mapping:
            env_var = key_mapping[self.provider]
            api_key = os.environ.get(env_var)
            if not api_key:
                raise ValueError(f"{env_var} environment variable is not set")
            return api_key

    def _initialize_client(self):
        """Initialize the appropriate client based on provider."""
        if self.provider == Provider.OPENAI:
            return OpenAI(api_key=self.api_key)
        elif self.provider == Provider.NVIDIA:
            return OpenAI(
                base_url="https://integrate.api.nvidia.com/v1", api_key=self.api_key
            )
        elif self.provider == Provider.BEDROCK:
            session = boto3.session.Session()
            configured_region = session.region_name
            return boto3.client(
                "bedrock-runtime",
                region_name=configured_region,
                config=Config(
                    connect_timeout=300,
                    read_timeout=1000,
                    retries={"max_attempts": 3},
                ),
            )
        elif self.provider == Provider.OLLAMA:
            return ollama.Client(host="http://localhost:11434")
        elif self.provider == Provider.OPENROUTER:
            return None
        else:
            raise ValueError(f"Invalid provider: {self.provider}")

    def _parse_json_from_text(self, text_to_parse: str) -> BaseModel:
        """Extract and parse JSON from text string."""
        try:
            parsed_json = JsonOutputParser().invoke(text_to_parse)
            parsed_output = self.output_format.model_validate(parsed_json)
        except Exception as e:
            regex_match = re.search(r"(\{.*\})", text_to_parse, re.DOTALL)
            if regex_match:
                cleaned_text = regex_match.group(1)
                try:
                    parsed_json = JsonOutputParser().invoke(cleaned_text)
                    parsed_output = self.output_format.model_validate(parsed_json)
                except Exception as e:
                    print(f"Error parsing JSON: {e}")
                    parsed_output = self._generate_empty_output()
            else:
                parsed_output = self._generate_empty_output()

        return parsed_output

    def _extract_from_content(
        self, content: Union[str, List[AIMessage]]
    ) -> Tuple[str, str]:
        """Extract raw_response and reasoning from model response content."""
        raw_response, reason = None, None
        try:
            if isinstance(content, str):
                raw_response = content
            elif isinstance(content, list):
                # We only have reasoning content if `content` is a list
                reason = next(
                    (
                        item["reasoning_content"]["text"]
                        for item in content
                        if item.get("type") == "reasoning_content"
                    ),
                    None,
                )
                # We don't use tool call for these models, just raw json output
                if self.model_id in [
                    "us.deepseek.r1-v1:0",
                    "mistral.mistral-large-2402-v1:0",
                ]:
                    raw_response = next(
                        (
                            item["text"]
                            for item in content
                            if item.get("type") == "text"
                        ),
                        None,
                    )
                else:
                    raw_response = next(
                        (
                            item["input"]
                            for item in content
                            if item.get("type") == "tool_use"
                        ),
                        None,
                    )
        except Exception as e:
            raw_response = None

        return raw_response, reason

    @staticmethod
    def _parse_raw_reasoning_output(raw_output: str) -> Tuple[str, str]:
        """Parse the raw reasoning output from the AI model."""
        pattern = r"<think>\s*(.*?)\s*</think>\s*(.*)"
        match = re.search(pattern, raw_output, re.DOTALL)
        if match:
            reasoning_tokens = match.group(1).strip()
            final_output = match.group(2).strip()
        else:
            reasoning_tokens = None
            final_output = raw_output

        return reasoning_tokens, final_output

    def _get_bedrock_llm(self):
        """Get the Bedrock LLM model based on the model name and temperature."""
        llm = ChatBedrockConverse(
            client=self.client,
            model_id=self.model_id,
            max_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            additional_model_request_fields=self.thinking_params,
        )
        llm = (
            llm.with_structured_output(self.output_format, include_raw=True)
            if self.model_id
            not in ["us.deepseek.r1-v1:0", "mistral.mistral-large-2402-v1:0"]
            else llm
        )
        return llm

    def _call_bedrock(self, messages: list[dict]) -> Dict[str, Any]:
        """Call the Bedrock LLM model with the given message."""
        try:
            reason = None
            response = self.bedrock_llm.invoke(messages)
            if self.model_id in [
                "us.deepseek.r1-v1:0",
                "mistral.mistral-large-2402-v1:0",
            ]:
                content = response.content
                raw_response, reason = self._extract_from_content(content)
                parsed_output = self._parse_json_from_text(raw_response)
            else:
                parsed_output = response["parsed"]
                response = response["raw"]
                raw_response, reason = self._extract_from_content(response.content)

            usage_metadata = response.usage_metadata
            latency = response.response_metadata["metrics"]["latencyMs"][0]

            output = {
                "raw_response": raw_response,
                "parsed_output": parsed_output,
                "date": datetime.now(),
                "latency": latency,
                "input_tokens": usage_metadata["input_tokens"],
                "output_tokens": usage_metadata["output_tokens"],
                "reasoning": reason,
            }
            return output
        except Exception as e:
            print(f"Error calling Bedrock LLM: {e}")
            return {
                "raw_response": None,
                "parsed_output": None,
                "date": datetime.now(),
                "latency": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning": None,
                "error": str(e),
            }

    def _call_ollama(self, messages: str) -> Dict[str, Any]:
        """Call the Ollama API with the given messages."""
        reason = None
        response = self.client.chat(
            model=self.model_id,
            messages=messages,
            format=self.output_format.model_json_schema()
            if not self.is_reasoning
            else None,
            options={
                "temperature": self.temperature,
                "num_predicr": self.max_completion_tokens,
            },
        )
        raw_response = response.message.content
        if self.is_reasoning:
            reason, raw_response = self._parse_raw_reasoning_output(raw_response)
        parsed_output = self._parse_json_from_text(raw_response)
        latency = int(
            (response.prompt_eval_duration / 1e6) + (response.eval_duration / 1e6)
        )

        output = {
            "raw_response": raw_response,
            "parsed_output": parsed_output,
            "date": datetime.now(),
            "latency": latency,
            "input_tokens": response.prompt_eval_count,
            "output_tokens": response.eval_count,
            "reasoning": reason,
        }
        return output

    def _call_openai(self, messages: str) -> Dict[str, Any]:
        """Call the OpenAI API with the given messages."""
        try:
            now = time.time()
            reasoning_args = {}
            if self.model_id == "gpt-5":
                reasoning_args["reasoning_effort"] = "medium"
            if self.model_id == "o1-mini":
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_completion_tokens=self.max_completion_tokens,
                    **reasoning_args,
                )
                raw_response = response.choices[0].message.content
                parsed_output = self._parse_json_from_text(raw_response)
            else:
                response = self.client.beta.chat.completions.parse(
                    model=self.model_id,
                    messages=messages,
                    response_format=self.output_format,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_completion_tokens,
                    **reasoning_args,
                )
                raw_response = response.choices[0].message.content
                parsed_output = response.choices[0].message.parsed
            elapsed_ms = (time.time() - now) * 1000
            output = {
                "raw_response": raw_response,
                "parsed_output": parsed_output,
                "date": datetime.now(),
                "latency": elapsed_ms,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "reasoning_tokens": response.usage.completion_tokens_details.reasoning_tokens,
            }
            return output
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return {
                "raw_response": None,
                "parsed_output": None,
                "date": datetime.now(),
                "latency": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "error": str(e),
            }

    def _call_openrouter(self, messages: str) -> Dict[str, Any]:
        """Call the OpenRouter API with the given messages."""
        reason = None
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": self.output_format.__name__,
                "strict": True,
                "schema": self.output_format.model_json_schema(),
            },
        }

        reasoning_settings = {
            "enabled": True,
            "exclude": False,
            "effort": "medium",
        }

        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": response_format,
            "max_tokens": self.max_completion_tokens,
            "reasoning": reasoning_settings,
            "include_reasoning": True,
            "usage": {"include": True},
        }

        if self.model_id == "qwen/qwq-32b":
            payload["top_k"] = 40
            payload["top_p"] = 0.95

        if self.is_reasoning:
            payload.pop("response_format")

        max_retries = 3
        for retry in range(max_retries):
            try:
                now = time.time()
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                response_json = response.json()
                elapsed_ms = (time.time() - now) * 1000

                raw_message = response_json["choices"][0]["message"]
                reasoning_segments: List[str] = []
                raw_response = raw_message.get("content")
                if isinstance(raw_response, str):
                    raw_response = raw_response.strip()
                elif isinstance(raw_response, list):
                    assembled = []
                    for chunk in raw_response:
                        text_value = ""
                        if isinstance(chunk, dict):
                            chunk_type = (chunk.get("type") or "").lower()
                            candidate = chunk.get("text")
                            if isinstance(candidate, str):
                                text_value = candidate
                            else:
                                candidate = chunk.get("content")
                                if isinstance(candidate, str):
                                    text_value = candidate
                                elif isinstance(candidate, list):
                                    text_value = "".join(
                                        part if isinstance(part, str) else ""
                                        for part in candidate
                                    )
                            if chunk_type and "reason" in chunk_type:
                                if isinstance(text_value, str) and text_value.strip():
                                    reasoning_segments.append(text_value.strip())
                                text_value = ""
                        elif isinstance(chunk, str):
                            text_value = chunk
                        if text_value:
                            assembled.append(text_value)
                    raw_response = "".join(assembled).strip()
                parsed_output = self._parse_json_from_text(raw_response)

                reasoning_payload = raw_message.get("reasoning")
                if isinstance(reasoning_payload, str):
                    if reasoning_payload.strip():
                        reasoning_segments.append(reasoning_payload.strip())
                elif isinstance(reasoning_payload, list):
                    for item in reasoning_payload:
                        if isinstance(item, str) and item.strip():
                            reasoning_segments.append(item.strip())
                        elif isinstance(item, dict):
                            text_val = (
                                item.get("text")
                                or item.get("content")
                                or item.get("reasoning")
                            )
                            if isinstance(text_val, str) and text_val.strip():
                                reasoning_segments.append(text_val.strip())
                elif isinstance(reasoning_payload, dict):
                    text_val = (
                        reasoning_payload.get("text")
                        or reasoning_payload.get("content")
                        or reasoning_payload.get("reasoning")
                    )
                    if isinstance(text_val, str) and text_val.strip():
                        reasoning_segments.append(text_val.strip())

                reasoning_details = raw_message.get("reasoning_details")
                if isinstance(reasoning_details, list):
                    for detail in reasoning_details:
                        if isinstance(detail, dict):
                            text_val = detail.get("text")
                            if isinstance(text_val, str) and text_val.strip():
                                reasoning_segments.append(text_val.strip())

                reason = (
                    "\n\n".join(segment for segment in reasoning_segments if segment)
                    or None
                )

                usage = response_json.get("usage") or {}
                prompt_tokens = usage.get("prompt_tokens") or 0
                completion_tokens = usage.get("completion_tokens") or 0
                reasoning_tokens = 0
                if isinstance(usage.get("completion_tokens_details"), dict):
                    reasoning_tokens = (
                        usage["completion_tokens_details"].get("reasoning_tokens") or 0
                    )
                elif isinstance(usage.get("reasoning"), dict):
                    reasoning_tokens = usage["reasoning"].get("tokens") or 0
                elif usage.get("reasoning_tokens") is not None:
                    reasoning_tokens = usage.get("reasoning_tokens") or 0

                return {
                    "raw_response": raw_response,
                    "parsed_output": parsed_output,
                    "date": datetime.now(),
                    "latency": elapsed_ms,
                    "input_tokens": int(prompt_tokens),
                    "output_tokens": int(completion_tokens),
                    "reasoning": reason,
                    "reasoning_tokens": int(reasoning_tokens or 0),
                }
            except Exception as e:
                if retry < max_retries - 1:
                    time.sleep(2 * (retry + 1))  # exponential backoff
                else:
                    return {
                        "raw_response": None,
                        "parsed_output": self._generate_empty_output(),
                        "date": datetime.now(),
                        "latency": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "reasoning": None,
                        "reasoning_tokens": 0,
                        "error": str(e),
                    }

    def _call_nvidia(self, messages: str) -> Dict[str, Any]:
        now = time.time()
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=0.6,
            top_p=0.7,
            max_tokens=self.max_completion_tokens,
        )
        elapsed_ms = (time.time() - now) * 1000
        response_text = response.choices[0].message.content

        reason, raw_response = self._parse_raw_reasoning_output(response_text)

        parsed_output = self._parse_json_from_text(raw_response)

        output = {
            "raw_response": raw_response,
            "parsed_output": parsed_output,
            "date": datetime.now(),
            "latency": elapsed_ms,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "reasoning": reason,
        }
        return output

    def _generate_empty_output(self):
        """Create an empty instance of the output format."""
        field_types = self.output_format.__annotations__
        fields = {}
        for field_name, field_type in field_types.items():
            if field_type == str:
                fields[field_name] = ""
            elif field_type == bool:
                fields[field_name] = False
            elif field_type == int:
                fields[field_name] = 0
            elif field_type == float:
                fields[field_name] = 0.0
            else:
                fields[field_name] = None
        return self.output_format(**fields)

    def __call__(self, prompt: str) -> Dict[str, Any]:
        """Evaluate the given messages using the appropriate LLM model."""
        # TODO: Add support for dynamic example, instead of hardcoding
        json_schema = JSON_ENFORCE.format(
            json_schema=self.output_format.model_json_schema(),
            example='{"answer": "Aspirin"}',
        )
        prompt = f"{prompt}\n{json_schema}"

        if self.provider == Provider.BEDROCK:
            messages = [{"role": "user", "content": [{"text": prompt}]}]
            return self._call_bedrock(messages)
        elif self.provider == Provider.OPENAI:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            return self._call_openai(messages)
        elif self.provider == Provider.NVIDIA:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            return self._call_nvidia(messages)
        elif self.provider == Provider.OLLAMA:
            messages = [{"role": "user", "content": prompt}]
            return self._call_ollama(messages)
        elif self.provider == Provider.OPENROUTER:
            messages = [{"role": "user", "content": prompt}]
            return self._call_openrouter(messages)


class Evaluate:
    def __init__(
        self,
        qa_llm: StructuredLLM,
        records: list[dict],
        use_context: bool,
        responses_save_path: str = None,
        verifier_provider: Provider = Provider.OPENAI,
        verifier_model: str = "gpt-5-mini",
        num_workers: int = 2,
        bedrock_cooldown: float = 0.5,
        domain: str = "chemistry",
    ):
        self.qa_llm = qa_llm
        # Use different verifier output format for legal domain
        verifier_output_format = LegalVerificationScore if domain == "legal" else AreSimilar
        self.verifier_llm = StructuredLLM(
            provider=verifier_provider,
            model_id=verifier_model,
            output_format=verifier_output_format,
        )
        self.records = records
        self.responses_save_path = responses_save_path
        self.num_workers = num_workers
        self.bedrock_cooldown = bedrock_cooldown
        self.use_context = use_context
        self.domain = domain

        self.file_lock = Lock()

        records_per_worker = len(records) / num_workers
        self.batch_size = max(1, math.ceil(records_per_worker))

        self.qa_llm_params = {
            "provider": qa_llm.provider,
            "model_id": qa_llm.model_id,
            "output_format": qa_llm.output_format,
            "temperature": 0.0,
            "max_completion_tokens": qa_llm.max_completion_tokens,
        }

        if qa_llm.is_reasoning:
            self.qa_llm_params["model_id"] = f"{qa_llm.model_id}-reasoning"

        self.verifier_llm_params = {
            "provider": verifier_provider,
            "model_id": verifier_model,
            "output_format": verifier_output_format,
        }

        # Iterative RAG setup (lazy init per worker for thread safety)
        # Resolve persist path relative to the 'src' root regardless of CWD
        base_src = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        rag_path = os.path.join(base_src, "chroma_store")
        if not os.path.exists(rag_path):
            alt_root = os.path.abspath(os.path.join(base_src, "..", "chroma_store"))
            if os.path.exists(alt_root):
                rag_path = alt_root
        self.rag_persist_path = rag_path
        
        # Domain-specific configuration
        if self.domain == "legal":
            self.rag_collection_name = "KoblexQwen"
            self.embedding_model = "Qwen/Qwen3-Embedding-0.6B"
        else:  # chemistry (default)
            self.rag_collection_name = "chemrxiv_graph"
            self.embedding_model = "BASF-AI/ChEmbed"

        # Create a single shared RAG service to avoid multiple model loads
        # This prevents the meta tensor error from parallel worker model initialization
        self._shared_rag_service = None
        self._shared_llm_client = None
        self._rag_lock = Lock()

    def _get_shared_rag_service(self):
        """Get or create the shared RAG service instance (thread-safe)"""
        with self._rag_lock:
            if self._shared_rag_service is None:
                from protocols.embedding_config import EmbedderConfig
                from service.rag_text_service import RagTextService
                from service.structured_llm_adapter import StructuredLLMClient
                from service.planner_llm import make_json_planner, make_llm_composer

                llm_client = StructuredLLMClient(
                    provider=self.qa_llm_params["provider"],
                    model=self.qa_llm_params["model_id"],
                    temperature=float(self.qa_llm_params.get("temperature", 0.0) or 0.0),
                    max_tokens=int(self.qa_llm_params.get("max_completion_tokens", 600) or 600),
                    debug=False,
                )
                planner = make_json_planner(
                    llm_client,
                    allow_kg=False,
                    default_k=8,
                    max_actions=6,
                    passages_top_k=5,
                )
                composer = make_llm_composer(llm_client)

                embed_cfg = EmbedderConfig(
                    model_name=self.embedding_model,
                    device="cpu"
                )
                
                self._shared_rag_service = RagTextService(
                    persist_path=self.rag_persist_path,
                    collection_name=self.rag_collection_name,
                    embedder_cfg=embed_cfg,
                    planner=planner,
                    composer=composer,
                    max_steps=6,
                )
                self._shared_llm_client = llm_client
            return self._shared_rag_service, self._shared_llm_client

    def _save_result_to_jsonl(self, result: dict):
        """Save a single result to the JSONL file incrementally"""
        if not self.responses_save_path:
            return
        with self.file_lock:
            with open(self.responses_save_path, "a") as f:
                f.write(json.dumps(result, default=str) + "\n")

    def _verify_entity(
        self, expected: str, candidate: str, worker_verifier_llm=None, question: str = "", context: str = ""
    ) -> tuple:
        """
        Verify if the candidate answer matches the expected answer.
        
        For chemistry: Returns (is_correct: bool, None)
        For legal: Returns (is_correct: bool, verification_score: int)
        """
        verifier_llm = worker_verifier_llm or self.verifier_llm

        if candidate.strip().lower() == expected.strip().lower():
            # Exact match - return true with max score for legal
            if self.domain == "legal":
                return True, 10
            return True, None
        
        # Legal domain uses scoring-based verification
        if self.domain == "legal":
            LEGAL_VERIFY_PROMPT = (
                "<Task Description>\n"
                "You will be given a complex legal question along with the\n"
                "relevant legal provisions that can be used to resolve it.\n\n"
                "<Evaluation Criteria>\n"
                "Evaluate the legal accuracy of the response on a scale 1 ∼ 10.\n\n"
                "<Evaluation Steps>\n"
                "Check whether the prediction properly answers the question.\n"
                "Check whether the prediction contradicts or omits any legal\n"
                "provisions in the context.\n"
                "Heavily penalize when the legal conclusion differs in detail\n"
                "from the expected output.\n"
                "Heavily penalize if the prediction contradicts or omits any\n"
                "specific elements from the context.\n"
                "Heavily penalize responses that include statements like 'The\n"
                "given context does not include the answer, but generally'.\n\n"
                "Output Format\n"
                "Return ONLY a single integer from 1 to 10. No text, labels, or punctuation.\n\n"
                "Now it's your turn to answer:\n\n"
                "Query\n"
                f"Question:\n{question}\n\n"
                f"Context:\n{context}\n\n"
                f"Expected output:\n{expected}\n\n"
                f"Prediction:\n{candidate}"
            )
            response = verifier_llm(LEGAL_VERIFY_PROMPT)
            parsed = response.get("parsed_output") if isinstance(response, dict) else None
            score = int(getattr(parsed, "score", 1))
            is_correct = score >= 8
            return is_correct, score
        
        # Chemistry domain uses boolean verification
        else:
            VERIFY_PROMPT = (
                "Task\n"
                "Decide whether Expected and Candidate name the SAME chemical entity.\n"
                "\n"
                "What counts as the SAME\n"
                "- Aliases, common vs IUPAC names, and formulas refer to the same thing (e.g., lithium chloride = LiCl; acetic acid = ethanoic acid).\n"
                "- Minor packaging/context words do not change identity: material, compound, sample, reagent, powder, nanopowder, precursor, solution.\n"
                "- The Candidate may be a long sentence or paragraph with explanations; as long as it explicitly names the same entity as the answer, count it as the same.\n"
                "\n"
                "What is NOT the same\n"
                "- Different polymorph/crystal structure/phase (wurtzite ZnO vs rocksalt ZnO).\n"
                "- Different charge state or ion vs neutral; cation vs anion (Li vs Li+; chloride ion vs HCl).\n"
                "- Different oxidation state or stoichiometry (FeCl2 vs FeCl3).\n"
                "- Different hydration/solvation (CuSO4 vs CuSO4*5H2O).\n"
                "- Different stereochemistry or isotopic labeling (L- vs D-; 13C-labeled vs unlabeled).\n"
                "- Salt vs parent acid/base (acetate vs acetic acid).\n"
                "- Class/family vs specific member (alkali metal chloride vs lithium chloride) unless the specific Expected entity is explicitly named.\n"
                "- Candidate only mentions Expected to negate or contrast it (uses words like 'not', 'instead of', 'different from', 'vs') while naming a different main entity.\n"
                "\n"
                "Decision rule\n"
                "- If Candidate explicitly names the same entity as Expected (even inside a longer explanation), answer: true.\n"
                "- Otherwise, answer: false.\n"
                "\n"
                "Output\n"
                "Answer with exactly: true or false\n"
                "\n"
                "Examples\n"
                "Expected: wurtzite ZnO\n"
                "Candidate: The ZnO polymorph used as the precursor in the synthesis of rsZnO according to high-pressure nanopowder synthesis methods is wurtzite ZnO (wZnO).\n"
                "Answer: true\n"
                "\n"
                "Expected: wurtzite ZnO\n"
                "Candidate: The product was rocksalt ZnO (rs-ZnO), not wurtzite ZnO.\n"
                "Answer: false\n"
                "\n"
                "Now it is your turn to answer:\n"
                f"Expected: {expected}\n"
                f"Candidate: {candidate}\n"
                "Answer with true or false."
            )
            response = verifier_llm(VERIFY_PROMPT)
            parsed = response.get("parsed_output") if isinstance(response, dict) else None
            is_correct = bool(getattr(parsed, "are_the_same", False))
            return is_correct, None

    def _create_worker_llms(self):
        """Create new LLM instances for workers to avoid thread safety issues"""
        qa_llm = StructuredLLM(**self.qa_llm_params)
        verifier_llm = StructuredLLM(**self.verifier_llm_params)
        return qa_llm, verifier_llm

    def _create_worker_rag_service(self):
        """Create a RagTextService instance per worker to avoid shared-state issues.
        Returns (rag_service, llm_client) so we can read token usage.
        """
        try:
            from protocols.embedding_config import EmbedderConfig
            from service.rag_text_service import RagTextService
            from service.structured_llm_adapter import StructuredLLMClient
            from service.planner_llm import make_json_planner, make_llm_composer
            from repo.embeddings.embeddeing_models import HFEmbedder
            from repo.index.chroma_index import ChromaTextIndex
            from repo.retrievers.multi_retriever import MultiCollectionRetriever
            import chromadb
        except Exception as e0:
            # If running as a script from src/benchmark, add parent 'src' to sys.path and retry
            import sys as _sys, os as _os
            _parent = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
            if _parent not in _sys.path:
                _sys.path.insert(0, _parent)
            try:
                from protocols.embedding_config import EmbedderConfig
                from service.rag_text_service import RagTextService
                from service.structured_llm_adapter import StructuredLLMClient
                from service.planner_llm import make_json_planner, make_llm_composer
                from repo.embeddings.embeddeing_models import HFEmbedder
                from repo.index.chroma_index import ChromaTextIndex
                from repo.retrievers.multi_retriever import MultiCollectionRetriever
                import chromadb
            except Exception as e:
                raise RuntimeError(
                    f"RagTextService import failed. Ensure the iterative RAG modules are available. Cause: {e.__class__.__name__}: {e} (initial: {e0.__class__.__name__}: {e0})"
                ) from e

        # Align planner/composer LLM with the benchmarked provider/model
        llm_client = StructuredLLMClient(
            provider=self.qa_llm_params["provider"],
            model=self.qa_llm_params["model_id"],
            temperature=float(self.qa_llm_params.get("temperature", 0.0) or 0.0),
            max_tokens=int(self.qa_llm_params.get("max_completion_tokens", 600) or 600),
            debug=False,
        )
        planner = make_json_planner(
            llm_client,
            allow_kg=False,
            default_k=8,
            max_actions=6,
            passages_top_k=5,
        )
        composer = make_llm_composer(llm_client)

        # Configure embedding model based on domain
        # EmbedderConfig is a frozen dataclass, so we must pass all params at creation
        embed_cfg = EmbedderConfig(
            model_name=self.embedding_model,
            device="cpu"
        )
        
        svc = RagTextService(
            persist_path=self.rag_persist_path,
            collection_name=self.rag_collection_name,
            embedder_cfg=embed_cfg,
            planner=planner,
            composer=composer,
            max_steps=6,
        )
        # Do NOT build a merged retriever - use only the domain-specific collection
        # This avoids embedding dimension mismatches between collections
        # (chemistry uses 1024-dim BASF-AI/ChEmbed, legal uses 768-dim Qwen)
        return svc, llm_client

    def _process_record_without_context(self, record, worker_llms=None):
        """Process a single record without context (thread-safe)"""
        qa_llm, verifier_llm = worker_llms or (self.qa_llm, self.verifier_llm)

        question = record["question"]
        expected = record["expected"]

        NO_CONTEXT_PROMPT = (
            "Provide a single entity name as an answer to the following question:\n"
            f"Question: {question}"
        )

        response = qa_llm(NO_CONTEXT_PROMPT)

        if response["parsed_output"] is not None:
            candidate = response["parsed_output"].answer
            is_correct = (
                self._verify_entity(expected, candidate, verifier_llm)
                if expected
                else False
            )
        else:
            candidate = None
            is_correct = False

        result = {
            "candidate": candidate,
            "is_correct": is_correct,
            "raw_response": response.get("raw_response", None),
            "context_used": False,
            "input_tokens": response.get("input_tokens", 0),
            "output_tokens": response.get("output_tokens", 0),
            "reasoning_tokens": response.get("reasoning_tokens", 0),
            "latency": round(response.get("latency", 0), 2),
            "date": response.get("date", datetime.now()).strftime("%Y-%m-%d %H:%M"),
            "reasoning": response.get("reasoning", None),
            "raw": record,
            "error": response.get("error", None),
        }

        self._save_result_to_jsonl(result)

        return result

    def _process_record_with_context(self, record, worker_llms=None):
        """Process a single record with context (thread-safe)"""
        qa_llm, verifier_llm = worker_llms or (self.qa_llm, self.verifier_llm)

        question = record["question"]
        context = record["context"]
        expected = record["expected"]

        WITH_CONTEXT_PROMPT = (
            "Provide a single entity name as an answer to the following question based on the context:\n"
            f"Question: {question}\n\n"
            f"Context: {context}"
        )

        response = qa_llm(WITH_CONTEXT_PROMPT)

        if response["parsed_output"] is not None:
            candidate = response["parsed_output"].answer
            is_correct = (
                self._verify_entity(expected, candidate, verifier_llm)
                if expected
                else False
            )
        else:
            candidate = None
            is_correct = False

        result = {
            "candidate": candidate,
            "is_correct": is_correct,
            "raw_response": response.get("raw_response", None),
            "context_used": True,
            "input_tokens": response.get("input_tokens", 0),
            "output_tokens": response.get("output_tokens", 0),
            "reasoning_tokens": response.get("reasoning_tokens", 0),
            "latency": round(response.get("latency", 0), 2),
            "date": response.get("date", datetime.now()).strftime("%Y-%m-%d %H:%M"),
            "reasoning": response.get("reasoning", None),
            "raw": record,
            "error": response.get("error", None),
        }

        self._save_result_to_jsonl(result)

        return result

    def _process_batch(self, batch, process_fn, progress_callback=None):
        """Process a batch of records with the given processing function"""
        # Create worker-specific LLM instances to avoid thread safety issues
        worker_llms = self._create_worker_llms()
        results = []

        for record in batch:
            result = process_fn(record, worker_llms)
            results.append(result)

            if progress_callback:
                progress_callback(1)

            if worker_llms[0].provider == Provider.BEDROCK:
                time.sleep(self.bedrock_cooldown)

        return results

    # --------------------- Iterative RAG processing ---------------------------

    def _process_record_iterative_rag(self, record, worker_ctx=None):
        """Process a single record using the iterative RAG pipeline (thread-safe)."""
        # worker_ctx: tuple(verifier_llm, rag_service, llm_client)
        if worker_ctx is None:
            _, verifier_llm = self._create_worker_llms()
            rag_service, llm_client = self._create_worker_rag_service()
        else:
            verifier_llm, rag_service, llm_client = worker_ctx

        question = record.get("question") or record.get("q")
        expected = record.get("expected") or record.get("a")
        number_of_hops = record.get("number_of_hops", 0)

        t0 = time.time()
        try:
            print(f"\n[Evaluate] Question: {question}")
            rag_result = rag_service.answer(question=question, k_default=8, trace=True)
            candidate = (rag_result or {}).get("answer") or ""
            try:
                steps = rag_result.get("steps")
                actions = rag_result.get("actions_trace") or []
                if steps is not None:
                    print(f"[Evaluate] Steps: {steps} | Actions: {len(actions)}")
                    if actions:
                        print(f"[Evaluate] First action: {actions[0]}")
            except Exception:
                pass
        except Exception as e:
            rag_result = {"error": str(e)}
            candidate = ""

        # Extract context for legal verification
        context_text = ""
        if self.domain == "legal" and isinstance(rag_result, dict):
            evidence = rag_result.get("evidence", [])
            if evidence:
                context_parts = []
                for ev in evidence[:3]:  # Use top 3 evidence pieces
                    if isinstance(ev, dict) and "text" in ev:
                        context_parts.append(ev["text"])
                context_text = "\n\n".join(context_parts)

        verification_score = None
        if candidate is not None and expected:
            is_correct, verification_score = self._verify_entity(
                expected or "", 
                candidate or "", 
                verifier_llm,
                question=question,
                context=context_text
            )
        else:
            is_correct = False
            verification_score = None if self.domain != "legal" else 1

        elapsed_ms = (time.time() - t0) * 1000.0
        # Gather token usage across planner/composer calls in this answer
        try:
            usage = llm_client.get_and_reset_usage()
        except Exception:
            usage = {"input": 0, "output": 0, "reasoning": 0, "calls": []}
        
        # Aggregate reasoning text from all calls
        reasoning_combined = None
        try:
            calls = usage.get("calls", [])
            reasoning_parts = []
            for i, call in enumerate(calls):
                rt = call.get("reasoning_text")
                if rt and isinstance(rt, str) and rt.strip():
                    reasoning_parts.append(f"[Step {i+1} - {call.get('role', 'unknown')}]\n{rt.strip()}")
            if reasoning_parts:
                reasoning_combined = "\n\n".join(reasoning_parts)
        except Exception:
            pass
        
        print(f"[Evaluate] Candidate: {candidate}")
        if expected:
            print(f"[Evaluate] Expected: {expected}")
        if self.domain == "legal" and verification_score is not None:
            print(f"[Evaluate] Verification Score: {verification_score}/10 | Correct: {is_correct} | Elapsed: {round(elapsed_ms,2)} ms")
        else:
            print(f"[Evaluate] Correct: {is_correct} | Elapsed: {round(elapsed_ms,2)} ms")
        
        # Calculate total reasoning tokens from all calls
        total_reasoning_tokens = int(usage.get("reasoning", 0))
        
        result = {
            "candidate": candidate,
            "is_correct": is_correct,
            "raw_response": rag_result,  # full RAG result for debugging
            "context_used": True,  # iterative RAG uses retrieved context
            "number_of_hops": int(number_of_hops or 0),
            "input_tokens": int(usage.get("input", 0)),
            "output_tokens": int(usage.get("output", 0)),
            "reasoning_tokens": total_reasoning_tokens,
            "llm_calls": list(usage.get("calls", [])),
            "latency": round(elapsed_ms, 2),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "reasoning": reasoning_combined,
            "raw": {"question": question, "expected": expected, "number_of_hops": int(number_of_hops or 0)},
            "error": rag_result.get("error", None) if isinstance(rag_result, dict) else None,
        }
        
        # Add verification_score for legal domain
        if self.domain == "legal":
            result["verification_score"] = verification_score

        self._save_result_to_jsonl(result)
        return result

    def _process_batch_rag(self, batch, progress_callback=None):
        """Process a batch of records using the shared RAG service and worker-specific verifier."""
        verifier_llm = StructuredLLM(**self.verifier_llm_params)
        # Use shared RAG service and its llm_client for proper token tracking
        rag_service, llm_client = self._get_shared_rag_service()
        
        worker_ctx = (verifier_llm, rag_service, llm_client)
        results = []
        for record in batch:
            res = self._process_record_iterative_rag(record, worker_ctx)
            results.append(res)
            if progress_callback:
                progress_callback(1)
        return results

    def _parallel_process_rag(self):
        """Process all records in parallel using iterative RAG."""
        results = []
        total_records = len(self.records)

        batches = [
            self.records[i : i + self.batch_size]
            for i in range(0, len(self.records), self.batch_size)
        ]

        with tqdm.tqdm(total=total_records, desc="Processing records (RAG)") as pbar:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_workers
            ) as executor:
                futures = []

                def update_progress(n):
                    pbar.update(n)

                for batch in batches:
                    futures.append(
                        executor.submit(
                            self._process_batch_rag,
                            batch=batch,
                            progress_callback=update_progress,
                        )
                    )

                for future in concurrent.futures.as_completed(futures):
                    batch_results = future.result()
                    results.extend(batch_results)

        assert len(results) == len(self.records), "Some records were not processed"

        return results

    def _parallel_process(self, process_fn: Callable):
        """Process all records in parallel using the provided function"""
        results = []
        total_records = len(self.records)

        batches = [
            self.records[i : i + self.batch_size]
            for i in range(0, len(self.records), self.batch_size)
        ]

        with tqdm.tqdm(total=total_records, desc="Processing records") as pbar:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_workers
            ) as executor:
                futures = []

                def update_progress(n):
                    pbar.update(n)

                for batch in batches:
                    futures.append(
                        executor.submit(
                            self._process_batch,
                            batch=batch,
                            process_fn=process_fn,
                            progress_callback=update_progress,
                        )
                    )

                for future in concurrent.futures.as_completed(futures):
                    batch_results = future.result()
                    results.extend(batch_results)

        assert len(results) == len(self.records), "Some records were not processed"

        return results

    def _eval_without_context(self):
        """Evaluate records without context using parallel processing"""
        self._parallel_process(self._process_record_without_context)

    def _eval_with_context(self):
        """Evaluate records with context using parallel processing"""
        self._parallel_process(self._process_record_with_context)

    def evaluate(self):
        """
        Evaluate the records based on the use_context flag.
        Results are saved to the JSONL file as they are generated.
        """
        # Replace legacy modes with iterative RAG evaluation
        self._parallel_process_rag()


class BenchmarkRunner:
    """Class to run benchmarks across multiple models and providers."""

    def __init__(
        self,
        records: list[dict],
        responses_dir: str = "responses",
        results_file: str = "results.csv",
        domain: str = "chemistry",
    ):
        self.records = records
        self.responses_dir = responses_dir
        self.results_file = results_file
        self.domain = domain
        self.model_registry = ModelRegistry()
        os.makedirs(responses_dir, exist_ok=True)

        if not os.path.exists(results_file):
            with open(results_file, "w") as f:
                if domain == "legal":
                    f.write(
                        "Model,Run,Accuracy (%),Avg Duration (s),Avg Input Tokens,Avg Output Tokens,Total Input Tokens,Total Output Tokens,Total Samples,Avg Verification Score\n"
                    )
                else:
                    f.write(
                        "Model,Run,Accuracy (%),Avg Duration (s),Avg Input Tokens,Avg Output Tokens,Total Input Tokens,Total Output Tokens,Total Samples\n"
                    )

    def _get_processed_questions(self, responses_path: str):
        """Get questions that have already been processed for a model"""
        questions_without_context = set()
        questions_with_context = set()

        if not os.path.exists(responses_path):
            return questions_without_context, questions_with_context

        try:
            with open(responses_path, "r") as f:
                for line in f:
                    try:
                        response = json.loads(line.strip())
                        if "raw" in response and "question" in response["raw"]:
                            question = response["raw"]["question"]
                            if response.get("context_used", False):
                                questions_with_context.add(question)
                            else:
                                questions_without_context.add(question)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading response file {responses_path}: {e}")

        return questions_without_context, questions_with_context

    def _filter_records_for_evaluation(self, responses_path: str):
        """Filter records that need to be evaluated"""
        questions_without_context, questions_with_context = (
            self._get_processed_questions(responses_path)
        )

        records_for_no_context = []
        records_for_context = []

        for record in self.records:
            question = record["question"]
            if question not in questions_without_context:
                records_for_no_context.append(record)
            if question not in questions_with_context and "context" in record:
                records_for_context.append(record)

        return records_for_no_context, records_for_context

    def _calculate_metrics_from_responses(self, responses_path, provider, model):
        """Calculate metrics from a response file and update the results CSV"""
        if not os.path.exists(responses_path):
            return None

        with_context_results = []
        without_context_results = []

        try:
            with open(responses_path, "r") as f:
                for line in f:
                    try:
                        response = json.loads(line.strip())
                        if response.get("context_used", False):
                            with_context_results.append(response)
                        else:
                            without_context_results.append(response)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading response file {responses_path}: {e}")
            return None

        model_name = f"{provider.value}-{model}"
        results = []

        if without_context_results:
            total = len(without_context_results)
            correct = sum(
                1 for r in without_context_results if r.get("is_correct", False)
            )
            accuracy = (correct / total) * 100 if total > 0 else 0
            avg_latency = (
                sum(r.get("latency", 0) for r in without_context_results) / total
                if total > 0
                else 0
            )
            avg_in_tokens = (
                sum(r.get("input_tokens", 0) for r in without_context_results) / total
                if total > 0
                else 0
            )
            avg_out_tokens = (
                sum(r.get("output_tokens", 0) for r in without_context_results) / total
                if total > 0
                else 0
            )
            total_in_tokens = sum(
                r.get("input_tokens", 0) for r in without_context_results
            )
            total_out_tokens = sum(
                r.get("output_tokens", 0) for r in without_context_results
            )

            results.append(
                {
                    "Model": model_name,
                    "Run": "Without Context",
                    "Accuracy (%)": round(accuracy, 2),
                    "Avg Duration (s)": round(avg_latency, 2),
                    "Avg Input Tokens": round(avg_in_tokens, 2),
                    "Avg Output Tokens": round(avg_out_tokens, 2),
                    "Total Input Tokens": total_in_tokens,
                    "Total Output Tokens": total_out_tokens,
                    "Total Samples": total,
                }
            )

        if with_context_results:
            total = len(with_context_results)
            correct = sum(1 for r in with_context_results if r.get("is_correct", False))
            accuracy = (correct / total) * 100 if total > 0 else 0
            avg_latency = (
                sum(r.get("latency", 0) for r in with_context_results) / total
                if total > 0
                else 0
            )
            avg_in_tokens = (
                sum(r.get("input_tokens", 0) for r in with_context_results) / total
                if total > 0
                else 0
            )
            avg_out_tokens = (
                sum(r.get("output_tokens", 0) for r in with_context_results) / total
                if total > 0
                else 0
            )
            total_in_tokens = sum(
                r.get("input_tokens", 0) for r in with_context_results
            )
            total_out_tokens = sum(
                r.get("output_tokens", 0) for r in with_context_results
            )
            
            # Calculate average verification score for legal domain
            avg_verification_score = None
            if self.domain == "legal":
                verification_scores = [
                    r.get("verification_score", 0) 
                    for r in with_context_results 
                    if r.get("verification_score") is not None
                ]
                if verification_scores:
                    avg_verification_score = sum(verification_scores) / len(verification_scores)

            result_dict = {
                "Model": model_name,
                "Run": "With Context",
                "Accuracy (%)": round(accuracy, 2),
                "Avg Duration (s)": round(avg_latency, 2),
                "Avg Input Tokens": round(avg_in_tokens, 2),
                "Avg Output Tokens": round(avg_out_tokens, 2),
                "Total Input Tokens": total_in_tokens,
                "Total Output Tokens": total_out_tokens,
                "Total Samples": total,
            }
            
            # Add verification score column for legal domain
            if self.domain == "legal" and avg_verification_score is not None:
                result_dict["Avg Verification Score"] = round(avg_verification_score, 2)
            
            results.append(result_dict)

        if results and self.results_file:
            self._update_results_csv(pd.DataFrame(results), model_name)
            return
        else:
            return pd.DataFrame(results)

    def _update_results_csv(self, new_df, model_name):
        """Update the results CSV file with new results, replacing existing entries for the same model"""
        if os.path.exists(self.results_file) and os.path.getsize(self.results_file) > 0:
            try:
                existing_df = pd.read_csv(self.results_file)
                existing_df = existing_df[existing_df["Model"] != model_name]
                if existing_df.empty:
                    combined_df = new_df
                else:
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)

                combined_df.to_csv(self.results_file, index=False)
            except Exception as e:
                print(f"Error updating results CSV: {e}")
                new_df.to_csv(self.results_file, index=False)
        else:
            new_df.to_csv(self.results_file, index=False)

    def run_benchmark_for_model(self, provider: Provider, model: str):
        """Run benchmark for a specific provider and model"""
        model_filename = model.replace("/", "__")
        responses_path = (
            f"{self.responses_dir}/responses_{provider.value}_{model_filename}.jsonl"
        )

        try:
            # For iterative RAG we treat every question as 'with context'
            _, questions_with_context = self._get_processed_questions(responses_path)
            # self.records are already normalized to {question, expected}
            records_with_context = [
                r for r in self.records if r.get("question") not in questions_with_context
            ]

            if not records_with_context:
                print(
                    f"All records already processed for {provider.value} model: {model}, skipping..."
                )
                self._calculate_metrics_from_responses(responses_path, provider, model)
                return

            structured_llm = StructuredLLM(
                provider=provider, model_id=model, output_format=Answer
            )

            if records_with_context:
                print(f"Evaluating {len(records_with_context)} records via Iterative RAG")
                # Allow override of num_workers via environment variable
                try:
                    num_workers = int(os.getenv("EVAL_WORKERS", "2"))
                except Exception:
                    num_workers = 2
                evaluator = Evaluate(
                    qa_llm=structured_llm,
                    records=records_with_context,
                    responses_save_path=responses_path,
                    use_context=True,
                    domain=self.domain,
                    num_workers=num_workers,
                )
                evaluator.evaluate()

            self._calculate_metrics_from_responses(responses_path, provider, model)

        except Exception as e:
            print(f"Error evaluating {provider.value} model {model}: {e}")

    def run_all_benchmarks(self):
        """Run benchmarks for all provider-model combinations."""
        for (
            provider,
            model,
        ) in self.model_registry.get_all_provider_model_combinations():
            print(f"\nEvaluating {provider.value} model: {model}")
            self.run_benchmark_for_model(provider, model)


if __name__ == "__main__":
    # Anchor default paths to the src/ tree so running from src/ works consistently
    _SRC_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Domain selection: chemistry (default) or legal
    DOMAIN = os.getenv("EVAL_DOMAIN", "chemistry").lower()
    
    # Domain-specific configuration
    if DOMAIN == "legal":
        RECORDS_PATH = os.path.join(_SRC_BASE, "docs", "koblex_transformed_enhanced.json")
        RESPONSES_DIR = os.path.join(_SRC_BASE, "responses_legal")
    else:  # chemistry (default)
        RECORDS_PATH = os.path.join(_SRC_BASE, "docs", "chemrxiv_qa.json")
        RESPONSES_DIR = os.path.join(_SRC_BASE, "responses")
    
    # Save results per provider/model to separate files (fallback to 'any' when unset)
    _prov_slug = (os.getenv("EVAL_PROVIDER") or "any").replace("/", "__").replace(":", "_")
    _model_slug = (os.getenv("EVAL_MODEL") or "any").replace("/", "__").replace(":", "_")
    RESULT_PATH = os.path.join(_SRC_BASE, f"results_{DOMAIN}_{_prov_slug}_{_model_slug}.csv")
    
    os.makedirs(RESPONSES_DIR, exist_ok=True)

    def _normalize_records(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in recs:
            q = r.get("question") or r.get("q")
            a = r.get("expected") or r.get("a")
            path = r.get("path")
            num_hops = len(path) if isinstance(path, list) else 0
            if not q:
                continue
            out.append({"question": q, "expected": a, "number_of_hops": num_hops})
        return out

    records_raw = read_json(RECORDS_PATH)
    records = _normalize_records(records_raw if isinstance(records_raw, list) else [])
    # Randomly select N questions for testing; remember the selection
    try:
        limit = int(os.getenv("EVAL_LIMIT", "5"))
    except Exception:
        limit = 5
    if limit and limit > 0 and len(records) > limit:
        # If provider/model are specified, persist a deterministic selection per combo
        env_provider = os.getenv("EVAL_PROVIDER") or "any"
        env_model = os.getenv("EVAL_MODEL") or "any"
        model_filename = (env_model or "any").replace("/", "__")
        selection_file = os.path.join(
            RESPONSES_DIR,
            f"selection_{env_provider}_{model_filename}_limit{limit}.json",
        )

        selected_questions: List[str] = []
        if os.path.exists(selection_file):
            try:
                with open(selection_file, "r") as sf:
                    selected_questions = json.load(sf) or []
            except Exception:
                selected_questions = []

        if not selected_questions:
            # Select a fresh random sample (no seeding), then persist for repeatability across runs
            sampled = random.sample(records, k=limit)
            selected_questions = [r.get("question") for r in sampled if r.get("question")]
            try:
                with open(selection_file, "w") as sf:
                    json.dump(selected_questions, sf, ensure_ascii=False, indent=2)
            except Exception:
                pass

        # Filter normalized records to match the saved selection order
        sel_set = set(selected_questions)
        # Preserve selection file order
        order_index = {q: i for i, q in enumerate(selected_questions)}
        records = [r for r in records if r.get("question") in sel_set]
        records.sort(key=lambda r: order_index.get(r.get("question"), 1_000_000))

    benchmark_runner = BenchmarkRunner(
        records=records, responses_dir=RESPONSES_DIR, results_file=RESULT_PATH, domain=DOMAIN
    )
    # Optional scope control for quick tests
    env_provider = os.getenv("EVAL_PROVIDER")
    env_model = os.getenv("EVAL_MODEL")
    
    print(f"\n{'='*80}")
    print(f"EVALUATION CONFIGURATION")
    print(f"{'='*80}")
    print(f"Domain: {DOMAIN.upper()}")
    print(f"Dataset: {RECORDS_PATH}")
    print(f"Records: {len(records)}")
    print(f"Responses Dir: {RESPONSES_DIR}")
    print(f"Results File: {RESULT_PATH}")
    print(f"{'='*80}\n")

    now = datetime.now()
    if env_provider:
        try:
            prov = Provider(env_provider)
        except Exception:
            print(f"Invalid EVAL_PROVIDER: {env_provider}. Falling back to all benchmarks.")
            prov = None
        if prov and env_model:
            print(f"Running benchmark for provider={prov.value}, model={env_model}")
            benchmark_runner.run_benchmark_for_model(prov, env_model)
        elif prov:
            print(f"Running benchmarks for provider={prov.value} (all configured models)")
            for m in benchmark_runner.model_registry.get_models_for_provider(prov):
                benchmark_runner.run_benchmark_for_model(prov, m)
        else:
            benchmark_runner.run_all_benchmarks()
    else:
        benchmark_runner.run_all_benchmarks()
    elapsed = datetime.now() - now
    elapsed_formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed.total_seconds()))
    print(f"Completed benchmarking in {elapsed_formatted}")
