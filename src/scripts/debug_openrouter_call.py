# """
# Ad-hoc diagnostic script to inspect the raw payload returned by OpenRouter.

# Run with: `python3 src/scripts/debug_openrouter_call.py`
# Requires OPENROUTER_API_KEY to be set in the environment.
# """

# from __future__ import annotations

# import argparse
# import json
# import os
# import sys
# from typing import Any, Dict

# import requests


# API_URL = "https://openrouter.ai/api/v1/chat/completions"
# DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"
# DEFAULT_MAX_COMPLETION_TOKENS = 2048
# MIN_REASONING_TOKENS = 1024

# def _build_headers(api_key: str) -> Dict[str, str]:
#     headers = {
#         "Authorization": f"Bearer {api_key}",
#         "Content-Type": "application/json",
#     }
#     referer = os.getenv("OPENROUTER_SITE_URL")
#     title = os.getenv("OPENROUTER_TITLE")
#     if referer:
#         headers["HTTP-Referer"] = referer
#     if title:
#         headers["X-Title"] = title
#     return headers


# def _build_reasoning_settings(model: str, max_completion_tokens: int) -> Dict[str, Any]:
#     return {
#         "enabled": True,
#         "exclude": False,
#         "effort": "medium",
#     }


# def _parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Debug OpenRouter response schema.")
#     parser.add_argument(
#         "--model",
#         default=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL),
#         help=f"Model to query (default: {DEFAULT_MODEL!r})",
#     )
#     parser.add_argument(
#         "--max-tokens",
#         type=int,
#         default=DEFAULT_MAX_COMPLETION_TOKENS,
#         help=f"Max completion tokens (default: {DEFAULT_MAX_COMPLETION_TOKENS})",
#     )
#     parser.add_argument(
#         "--temperature",
#         type=float,
#         default=0.2,
#         help="Sampling temperature (default: 0.2)",
#     )
#     return parser.parse_args()


# def main() -> int:
#     args = _parse_args()
#     api_key = os.getenv("OPENROUTER_API_KEY")
#     if not api_key:
#         print("OPENROUTER_API_KEY environment variable is not set", file=sys.stderr)
#         return 1

#     messages = [
#         {"role": "system", "content": "You are a concise assistant."},
#         {
#             "role": "user",
#             "content": "Explain what a coordination complex is in one sentence.",
#         },
#     ]

#     payload: Dict[str, Any] = {
#         "model": args.model,
#         "messages": messages,
#         "temperature": args.temperature,
#         "max_tokens": args.max_tokens,
#         "reasoning": _build_reasoning_settings(args.model, args.max_tokens),
#         "usage": {
#             "include": True,
#         },
#         "include_reasoning": True,
#     }

#     try:
#         response = requests.post(
#             API_URL,
#             headers=_build_headers(api_key),
#             json=payload,
#             timeout=60,
#         )
#         response.raise_for_status()
#     except requests.HTTPError as http_exc:
#         print(
#             f"OpenRouter returned an error: {http_exc.response.status_code} {http_exc.response.text}",
#             file=sys.stderr,
#         )
#         return 2
#     except Exception as exc:  # pragma: no cover - debugging path
#         print(f"Failed to call OpenRouter: {exc}", file=sys.stderr)
#         return 3

#     try:
#         content = response.json()
#     except ValueError:
#         print("Response was not valid JSON:", file=sys.stderr)
#         print(response.text)
#         return 4

#     print(json.dumps(content, indent=2, sort_keys=True))
#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())
import requests
import json

response = requests.get(
  url="https://openrouter.ai/api/v1/key",
  headers={
    "Authorization": f"Bearer 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"
  }
)

print(json.dumps(response.json(), indent=2))
