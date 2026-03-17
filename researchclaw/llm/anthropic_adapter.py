"""Anthropic Messages API adapter for ResearchClaw."""

import json
import logging
import urllib.error
from typing import Any

try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

logger = logging.getLogger(__name__)

_JSON_MODE_INSTRUCTION = (
    "You MUST respond with valid JSON only. "
    "Do not include any text outside the JSON object."
)


class AnthropicAdapter:
    """Adapter to call Anthropic Messages API and return OpenAI-compatible response."""

    def __init__(self, base_url: str, api_key: str, timeout_sec: int = 300):
        if not HAS_HTTPX:
            raise ImportError(
                "httpx is required for Anthropic adapter. Install: pip install httpx"
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_sec = timeout_sec

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        """Call Anthropic Messages API and return OpenAI-compatible response.

        Raises urllib.error.HTTPError on API errors so the upstream retry
        logic in LLMClient._call_with_retry works unchanged.
        """
        # Extract system message if present
        system_msg = None
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_messages.append(msg)

        # Prepend JSON instruction when json_mode is requested
        if json_mode:
            system_msg = (
                f"{_JSON_MODE_INSTRUCTION}\n\n{system_msg}"
                if system_msg
                else _JSON_MODE_INSTRUCTION
            )

        # Build Anthropic request
        body: dict[str, Any] = {
            "model": model,
            "messages": user_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_msg:
            body["system"] = system_msg

        url = f"{self.base_url}/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        try:
            with httpx.Client(timeout=self.timeout_sec) as client:
                response = client.post(url, headers=headers, json=body)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            # Convert to urllib.error.HTTPError for upstream retry logic
            raise urllib.error.HTTPError(
                url,
                exc.response.status_code,
                str(exc),
                dict(exc.response.headers),
                None,
            ) from exc
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise urllib.error.URLError(str(exc)) from exc

        # Convert Anthropic response to OpenAI format
        content = ""
        if "content" in data and data["content"]:
            content = data["content"][0].get("text", "")

        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": data.get("stop_reason", "stop"),
                }
            ],
            "usage": {
                "prompt_tokens": data.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": data.get("usage", {}).get("output_tokens", 0),
                "total_tokens": (
                    data.get("usage", {}).get("input_tokens", 0)
                    + data.get("usage", {}).get("output_tokens", 0)
                ),
            },
            "model": data.get("model", model),
        }
