"""
cot_evaluator.py

Evaluates multi-hop chemistry QA using a direct chain-of-thought LLM approach
(no iterative RAG). The LLM receives the question and pre-fetched paragraphs
(from path[].text in the dataset) and must reason step-by-step before
providing a final answer.

Output files per model run
--------------------------
1. responses_cot/responses_<provider>_<model>.json
   List of dicts, each containing:
   {"CoT": "...", "answer": "...", "is_correct": true/false}

2. results_cot_<provider>_<model>.csv
   Columns: Model, Run, Accuracy (%), Avg Input Tokens, Avg Output Tokens,
             Total Input Tokens, Total Output Tokens, Total Reasoning Tokens,
             Avg Reasoning Tokens, Total Samples
"""

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
import requests

load_dotenv()


# ---------------------------------------------------------------------------
# Pydantic output schemas
# ---------------------------------------------------------------------------

class ChemistryCoT(BaseModel):
    CoT: str = Field(..., description="Step-by-step chain-of-thought reasoning")
    answer: str = Field(..., description="Final concise answer")


class AreSimilar(BaseModel):
    are_the_same: bool = Field(
        ..., description="Whether the two answers name the same chemical entity."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_json(file_path: str) -> dict:
    """Read the JSON file from the given file path."""
    with open(file_path, "r") as file:
        return json.load(file)


JSON_ENFORCE = (
    "Please return only the raw JSON string that strictly conforms to the following "
    "JSON schema, with no additional text: {json_schema}"
    "Example: {example}"
)

# ---------------------------------------------------------------------------
# Provider / Model registry  (identical to evaluator.py)
# ---------------------------------------------------------------------------

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
    """Registry to manage provider-model relationships."""

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
            "gpt-5.1",
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
            "anthropic/claude-3.7-sonnet",
            "anthropic/claude-3.7-sonnet:thinking-reasoning",
            "anthropic/claude-sonnet-4.5",
            "google/gemini-2.5-pro",
            "z-ai/glm-4.6",
            "x-ai/grok-4-fast",
            "deepseek/deepseek-chat-v3.1",
            "google/gemini-3-pro-preview",
        ],
        Provider.NVIDIA: [
            # "deepseek-ai/deepseek-r1",
        ],
    }

    def __init__(self):
        self.completed_benchmarks = set()

    def get_models_for_provider(self, provider: Provider) -> List[str]:
        return self.PROVIDER_MODELS.get(provider, [])

    def is_valid_model(self, provider: Provider, model: str) -> bool:
        return model in self.PROVIDER_MODELS.get(provider, [])

    def get_all_provider_model_combinations(self) -> List[Tuple[Provider, str]]:
        combinations = []
        for provider in Provider:
            for model in self.get_models_for_provider(provider):
                combinations.append((provider, model))
        return combinations

    def mark_as_completed(self, provider: Provider, model: str):
        self.completed_benchmarks.add((provider, model))


# ---------------------------------------------------------------------------
# StructuredLLM  (identical to evaluator.py)
# ---------------------------------------------------------------------------

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
        try:
            parsed_json = JsonOutputParser().invoke(text_to_parse)
            parsed_output = self.output_format.model_validate(parsed_json)
        except Exception:
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
        raw_response, reason = None, None
        try:
            if isinstance(content, str):
                raw_response = content
            elif isinstance(content, list):
                reason = next(
                    (
                        item["reasoning_content"]["text"]
                        for item in content
                        if item.get("type") == "reasoning_content"
                    ),
                    None,
                )
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
        except Exception:
            raw_response = None
        return raw_response, reason

    @staticmethod
    def _parse_raw_reasoning_output(raw_output: str) -> Tuple[str, str]:
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

            return {
                "raw_response": raw_response,
                "parsed_output": parsed_output,
                "date": datetime.now(),
                "latency": latency,
                "input_tokens": usage_metadata["input_tokens"],
                "output_tokens": usage_metadata["output_tokens"],
                "reasoning_tokens": 0,
                "reasoning": reason,
            }
        except Exception as e:
            print(f"Error calling Bedrock LLM: {e}")
            return {
                "raw_response": None,
                "parsed_output": None,
                "date": datetime.now(),
                "latency": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "reasoning": None,
                "error": str(e),
            }

    def _call_ollama(self, messages: str) -> Dict[str, Any]:
        reason = None
        response = self.client.chat(
            model=self.model_id,
            messages=messages,
            format=self.output_format.model_json_schema()
            if not self.is_reasoning
            else None,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_completion_tokens,
            },
        )
        raw_response = response.message.content
        if self.is_reasoning:
            reason, raw_response = self._parse_raw_reasoning_output(raw_response)
        parsed_output = self._parse_json_from_text(raw_response)
        latency = int(
            (response.prompt_eval_duration / 1e6) + (response.eval_duration / 1e6)
        )
        return {
            "raw_response": raw_response,
            "parsed_output": parsed_output,
            "date": datetime.now(),
            "latency": latency,
            "input_tokens": response.prompt_eval_count,
            "output_tokens": response.eval_count,
            "reasoning_tokens": 0,
            "reasoning": reason,
        }

    def _call_openai(self, messages: str) -> Dict[str, Any]:
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
            return {
                "raw_response": raw_response,
                "parsed_output": parsed_output,
                "date": datetime.now(),
                "latency": elapsed_ms,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "reasoning_tokens": response.usage.completion_tokens_details.reasoning_tokens,
            }
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
            "effort": os.getenv("EVAL_REASONING_EFFORT", "medium"),  # low | medium | high
        }
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": response_format,
            "max_tokens": self.max_completion_tokens,
            "usage": {"include": True},
        }
        if self.is_reasoning:
            # Reasoning model: enable extended thinking, drop structured output format
            payload["reasoning"] = reasoning_settings
            payload["include_reasoning"] = True
            payload.pop("response_format")

        if self.model_id == "qwen/qwq-32b":
            payload["top_k"] = 40
            payload["top_p"] = 0.95

        max_retries = 3
        for retry in range(max_retries):
            try:
                now = time.time()
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
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
                    time.sleep(2 * (retry + 1))
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
        return {
            "raw_response": raw_response,
            "parsed_output": parsed_output,
            "date": datetime.now(),
            "latency": elapsed_ms,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "reasoning_tokens": 0,
            "reasoning": reason,
        }

    def _generate_empty_output(self):
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
        """Call the LLM with a plain text prompt and return a result dict."""
        # For providers that need explicit JSON schema enforcement appended
        json_schema = JSON_ENFORCE.format(
            json_schema=self.output_format.model_json_schema(),
            example='{"CoT": "step-by-step reasoning...", "answer": "entity name"}',
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


# ---------------------------------------------------------------------------
# Direct evaluator
# ---------------------------------------------------------------------------

# System prompt for multi-hop chemistry QA
DIRECT_QA_SYSTEM_PROMPT = """\
You are an expert for multi-hop Question Answering over unstructured text in chemistry.

Return ONLY a valid JSON object. No prose outside the JSON. No markdown formatting outside the JSON block. No comments.

Required schema:
{"CoT":"...","answer":"..."}

Inputs provided:
question: The full multi-hop user question.
paragraphs: The text passages containing the potential evidence.

Policy:
You must document your step-by-step reasoning in the "CoT" field before providing the final "answer". Follow these strict steps within "CoT":

Decompose: Explicitly break down the multi-hop "question" into an ordered list of atomic, single-hop sub-questions and explain why you are breaking it down in such a way.

Extract & Answer: Go through each sub-question one by one. Search the provided "paragraphs" for the evidence needed. 
Strict Grounding: Never use outside knowledge.
Synthesize: Based on the extracted evidence, determine the final answer. 
Final Output: Place your final, concise conclusion in the "answer" field.

Your 'CoT' must be highly detailed and explicitly include the following sections:
[Hypothesis Formulation]: State detailed initial thoughts on how to approach the sub-questions.
[Verification]: Double-check if the extracted quotes fully satisfy the sub-questions without making assumptions.

Chemistry Multi-Hop CoT Example:
question: What is the pKa of the acid used to synthesize aspirin?
paragraphs: [Passage 1] Aspirin is synthesized by the esterification of salicylic acid with acetic anhydride. [Passage 2] Salicylic acid is a lipophilic monohydroxybenzoic acid with a pKa of 2.97.

Output:
{
"CoT": "1. Decompose: The multi-hop question requires two steps: a) What acid is used to synthesize aspirin? b) What is the pKa of that specific acid? 2. Extract & Answer: For sub-question (a), Passage 1 states that aspirin is synthesized using salicylic acid. For sub-question (b), Passage 2 states that salicylic acid has a pKa of 2.97. 3. Synthesize: Both sub-questions are fully supported by the text. The target acid is salicylic acid, and its pKa is 2.97.",
"answer": "The acid used to synthesize aspirin is salicylic acid, and its pKa is 2.97."
}

Anti-pattern (do NOT do this):
{
"CoT": "Aspirin is made from salicylic acid and its pKa is 2.97.",
"answer": "2.97"
}
Reason: The CoT fails to explicitly break down the multi-hop question into sub-questions and does not show the step-by-step extraction of evidence from the passages.\
"""

# Verification prompt (same as evaluator.py VERIFY_PROMPT)
VERIFY_PROMPT_TEMPLATE = (
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
    "Expected: {expected}\n"
    "Candidate: {candidate}\n"
    "Answer with true or false."
)


def _build_paragraphs_text(path: list) -> str:
    """
    Given the list of path hops in a QA record, concatenate each hop's
    'text' field as numbered passages.
    """
    parts = []
    for i, hop in enumerate(path, start=1):
        text = hop.get("text", "").strip()
        if text:
            parts.append(f"[Passage {i}] {text}")
    return "\n\n".join(parts)


def _build_direct_prompt(question: str, paragraphs: str) -> str:
    return (
        f"{DIRECT_QA_SYSTEM_PROMPT}\n\n"
        f"question: {question}\n"
        f"paragraphs: {paragraphs}"
    )


class DirectEvaluate:
    """
    Evaluates multi-hop QA records by calling an LLM directly with the
    pre-fetched paragraphs—no iterative RAG system involved.
    """

    def __init__(
        self,
        qa_llm: StructuredLLM,
        records: list,
        responses_save_path: str = None,
        verifier_provider: Provider = Provider.OPENAI,
        verifier_model: str = "gpt-4o-mini",
        num_workers: int = 2,
        bedrock_cooldown: float = 0.5,
    ):
        self.qa_llm = qa_llm
        self.records = records
        self.responses_save_path = responses_save_path
        self.num_workers = num_workers
        self.bedrock_cooldown = bedrock_cooldown

        self.verifier_llm = StructuredLLM(
            provider=verifier_provider,
            model_id=verifier_model,
            output_format=AreSimilar,
        )

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
            "output_format": AreSimilar,
        }

    def _save_result(self, result: dict):
        """Append a single result record to the JSONL responses file."""
        if not self.responses_save_path:
            return
        with self.file_lock:
            with open(self.responses_save_path, "a") as f:
                f.write(json.dumps(result, default=str) + "\n")

    def _verify(self, expected: str, candidate: str, verifier_llm) -> bool:
        """Return True if the candidate matches the expected answer."""
        if candidate.strip().lower() == expected.strip().lower():
            return True
        prompt = VERIFY_PROMPT_TEMPLATE.format(
            expected=expected, candidate=candidate
        )
        response = verifier_llm(prompt)
        parsed = response.get("parsed_output") if isinstance(response, dict) else None
        return bool(getattr(parsed, "are_the_same", False))

    def _create_worker_llms(self):
        qa_llm = StructuredLLM(**self.qa_llm_params)
        verifier_llm = StructuredLLM(**self.verifier_llm_params)
        return qa_llm, verifier_llm

    def _process_record(self, record: dict, worker_llms=None) -> dict:
        """Process a single record: call LLM, verify, return result."""
        qa_llm, verifier_llm = worker_llms or (self.qa_llm, self.verifier_llm)

        question = record["question"]
        expected = record["expected"]
        paragraphs = record["paragraphs"]

        prompt = _build_direct_prompt(question, paragraphs)
        response = qa_llm(prompt)

        parsed = response.get("parsed_output")
        cot = getattr(parsed, "CoT", "") if parsed else ""
        answer = getattr(parsed, "answer", "") if parsed else ""

        is_correct = False
        if answer and expected:
            is_correct = self._verify(expected, answer, verifier_llm)

        result = {
            "CoT": cot,
            "answer": answer,
            "is_correct": is_correct,
            # Additional metadata (not in the 3-key spec but useful for debugging)
            "question": question,
            "expected": expected,
            "input_tokens": response.get("input_tokens", 0),
            "output_tokens": response.get("output_tokens", 0),
            "reasoning_tokens": response.get("reasoning_tokens", 0),
            "latency": round(response.get("latency", 0), 2),
            "date": response.get("date", datetime.now()).strftime("%Y-%m-%d %H:%M"),
            "error": response.get("error", None),
        }

        self._save_result(result)
        return result

    def _process_batch(self, batch, progress_callback=None):
        worker_llms = self._create_worker_llms()
        results = []
        for record in batch:
            result = self._process_record(record, worker_llms)
            results.append(result)
            if progress_callback:
                progress_callback(1)
            if worker_llms[0].provider == Provider.BEDROCK:
                time.sleep(self.bedrock_cooldown)
        return results

    def evaluate(self) -> List[dict]:
        """Run evaluation across all records using thread pool."""
        results = []
        total_records = len(self.records)
        batches = [
            self.records[i : i + self.batch_size]
            for i in range(0, len(self.records), self.batch_size)
        ]

        with tqdm.tqdm(total=total_records, desc="CoT LLM evaluation") as pbar:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_workers
            ) as executor:
                futures = [
                    executor.submit(
                        self._process_batch,
                        batch=batch,
                        progress_callback=lambda n: pbar.update(n),
                    )
                    for batch in batches
                ]
                for future in concurrent.futures.as_completed(futures):
                    results.extend(future.result())

        return results


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

class CotBenchmarkRunner:
    """
    Runs benchmarks for all configured provider-model combinations using the
    direct chain-of-thought (CoT) LLM approach (no RAG).
    """

    CSV_COLUMNS = [
        "Model",
        "Run",
        "Accuracy (%)",
        "Avg Input Tokens",
        "Avg Output Tokens",
        "Total Input Tokens",
        "Total Output Tokens",
        "Total Reasoning Tokens",
        "Avg Reasoning Tokens",
        "Total Samples",
    ]

    def __init__(
        self,
        records: list,
        responses_dir: str = "responses_cot",
        results_file: str = "results_cot.csv",
    ):
        self.records = records
        self.responses_dir = responses_dir
        self.results_file = results_file
        self.model_registry = ModelRegistry()
        os.makedirs(responses_dir, exist_ok=True)

        if not os.path.exists(results_file):
            pd.DataFrame(columns=self.CSV_COLUMNS).to_csv(results_file, index=False)


    # ------------------------------------------------------------------
    # Resume support: detect already-processed questions
    # ------------------------------------------------------------------

    def _get_processed_questions(self, responses_path: str) -> set:
        processed = set()
        if not os.path.exists(responses_path):
            return processed
        try:
            with open(responses_path, "r") as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        q = rec.get("question")
                        if q:
                            processed.add(q)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Warning: could not read {responses_path}: {e}")
        return processed

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    def _compute_and_save_metrics(self, responses_path: str, provider: Provider, model: str):
        if not os.path.exists(responses_path):
            return

        results = []
        try:
            with open(responses_path, "r") as f:
                for line in f:
                    try:
                        results.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading {responses_path}: {e}")
            return

        if not results:
            return

        total = len(results)
        correct = sum(1 for r in results if r.get("is_correct", False))
        accuracy = (correct / total) * 100 if total > 0 else 0.0

        avg_in = sum(r.get("input_tokens", 0) for r in results) / total
        avg_out = sum(r.get("output_tokens", 0) for r in results) / total
        total_in = sum(r.get("input_tokens", 0) for r in results)
        total_out = sum(r.get("output_tokens", 0) for r in results)
        total_reasoning = sum(r.get("reasoning_tokens", 0) for r in results)
        avg_reasoning = total_reasoning / total if total > 0 else 0.0

        model_name = f"{provider.value}-{model}"
        row = {
            "Model": model_name,
            "Run": "Direct",
            "Accuracy (%)": round(accuracy, 2),
            "Avg Input Tokens": round(avg_in, 2),
            "Avg Output Tokens": round(avg_out, 2),
            "Total Input Tokens": total_in,
            "Total Output Tokens": total_out,
            "Total Reasoning Tokens": total_reasoning,
            "Avg Reasoning Tokens": round(avg_reasoning, 2),
            "Total Samples": total,
        }

        self._update_results_csv(pd.DataFrame([row]), model_name)
        print(
            f"  → {model_name}: accuracy={accuracy:.1f}%, samples={total}, "
            f"in={total_in}, out={total_out}, reasoning={total_reasoning}"
        )


    def _update_results_csv(self, new_df: pd.DataFrame, model_name: str):
        if os.path.exists(self.results_file) and os.path.getsize(self.results_file) > 0:
            try:
                existing = pd.read_csv(self.results_file)
                existing = existing[existing["Model"] != model_name]
                combined = pd.concat([existing, new_df], ignore_index=True)
                combined.to_csv(self.results_file, index=False)
                return
            except Exception as e:
                print(f"Warning: could not update CSV, overwriting. ({e})")
        new_df.to_csv(self.results_file, index=False)

    # ------------------------------------------------------------------
    # Per-model benchmark
    # ------------------------------------------------------------------

    def run_benchmark_for_model(self, provider: Provider, model: str):
        model_filename = model.replace("/", "__").replace(":", "_")
        responses_path = os.path.join(
            self.responses_dir, f"responses_cot_{provider.value}_{model_filename}.json"
        )

        try:
            processed_qs = self._get_processed_questions(responses_path)
            records_to_run = [
                r for r in self.records if r["question"] not in processed_qs
            ]

            if not records_to_run:
                print(
                    f"All records already processed for {provider.value}/{model}, skipping."
                )
                self._compute_and_save_metrics(responses_path, provider, model)
                return

            print(
                f"\nEvaluating {len(records_to_run)} records via CoT LLM "
                f"[{provider.value}/{model}]"
            )

            try:
                num_workers = int(os.getenv("EVAL_WORKERS", "2"))
            except Exception:
                num_workers = 2

            qa_llm = StructuredLLM(
                provider=provider,
                model_id=model,
                output_format=ChemistryCoT,
            )

            evaluator = DirectEvaluate(
                qa_llm=qa_llm,
                records=records_to_run,
                responses_save_path=responses_path,
                num_workers=num_workers,
            )
            evaluator.evaluate()
            self._compute_and_save_metrics(responses_path, provider, model)

        except Exception as e:
            print(f"Error evaluating {provider.value}/{model}: {e}")

    def run_all_benchmarks(self):
        for provider, model in self.model_registry.get_all_provider_model_combinations():
            print(f"\nEvaluating {provider.value} model: {model}")
            self.run_benchmark_for_model(provider, model)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _normalize_records(raw: list) -> list:
    """
    Convert raw chemrxiv_qa.json entries into the internal record format.

    Each internal record:
      - question   : str
      - expected   : str
      - paragraphs : str  (all path[].text concatenated as numbered passages)
      - num_hops   : int
    """
    out = []
    for item in raw:
        question = item.get("q") or item.get("question")
        expected = item.get("a") or item.get("expected")
        path = item.get("path", [])
        if not question:
            continue
        paragraphs = _build_paragraphs_text(path) if isinstance(path, list) else ""
        out.append(
            {
                "question": question,
                "expected": expected or "",
                "paragraphs": paragraphs,
                "num_hops": len(path) if isinstance(path, list) else 0,
            }
        )
    return out


if __name__ == "__main__":
    _SRC_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    RECORDS_PATH = os.path.join(_SRC_BASE, "docs", "chemrxiv_qa.json")
    RESPONSES_DIR = os.path.join(_SRC_BASE, "responses_cot")
    RESULT_PATH = os.path.join(_SRC_BASE, "results_cot.csv")

    _prov_slug = (os.getenv("EVAL_PROVIDER") or "any").replace("/", "__").replace(":", "_")
    _model_slug = (os.getenv("EVAL_MODEL") or "any").replace("/", "__").replace(":", "_")
    RESULT_PATH = os.path.join(
        _SRC_BASE, f"results_cot_{_prov_slug}_{_model_slug}.csv"
    )

    os.makedirs(RESPONSES_DIR, exist_ok=True)

    raw_records = read_json(RECORDS_PATH)
    records = _normalize_records(raw_records if isinstance(raw_records, list) else [])

    # Optional record limit — always picks a fresh random sample each run
    try:
        limit = int(os.getenv("EVAL_LIMIT", "0"))
    except Exception:
        limit = 0

    if limit and limit > 0 and len(records) > limit:
        records = random.sample(records, k=limit)

    print(f"\n{'='*80}")
    print("COT LLM EVALUATION CONFIGURATION")
    print(f"{'='*80}")
    print(f"Dataset : {RECORDS_PATH}")
    print(f"Records : {len(records)}")
    print(f"Responses Dir: {RESPONSES_DIR}")
    print(f"Results File : {RESULT_PATH}")
    print(f"{'='*80}\n")

    runner = CotBenchmarkRunner(
        records=records,
        responses_dir=RESPONSES_DIR,
        results_file=RESULT_PATH,
    )

    env_provider = os.getenv("EVAL_PROVIDER")
    env_model = os.getenv("EVAL_MODEL")

    now = datetime.now()
    if env_provider:
        try:
            prov = Provider(env_provider)
        except Exception:
            print(f"Invalid EVAL_PROVIDER: {env_provider}. Running all benchmarks.")
            prov = None
        if prov and env_model:
            runner.run_benchmark_for_model(prov, env_model)
        elif prov:
            for m in runner.model_registry.get_models_for_provider(prov):
                runner.run_benchmark_for_model(prov, m)
        else:
            runner.run_all_benchmarks()
    else:
        runner.run_all_benchmarks()

    elapsed = datetime.now() - now
    elapsed_fmt = time.strftime("%H:%M:%S", time.gmtime(elapsed.total_seconds()))
    print(f"\nCompleted in {elapsed_fmt}")
