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

# Map Anthropic stop_reason → OpenAI finish_reason
_STOP_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}


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
        self._client: httpx.Client | None = None

    def close(self) -> None:
        """BUG-DA8-09: Close the httpx connection pool to prevent fd leaks."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None

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
        # Extract and concatenate all system messages
        system_parts: list[str] = []
        user_messages: list[dict[str, str]] = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                user_messages.append(msg)

        system_msg = "\n\n".join(system_parts) if system_parts else None

        # Merge consecutive messages with the same role (Anthropic
        # requires strict user/assistant alternation)
        merged: list[dict[str, str]] = []
        for msg in user_messages:
            if merged and merged[-1]["role"] == msg["role"]:
                merged[-1] = {
                    "role": msg["role"],
                    "content": merged[-1]["content"] + "\n\n" + msg["content"],
                }
            else:
                merged.append(dict(msg))
        user_messages = merged

        # Ensure at least one user message and that it starts with "user"
        if not user_messages:
            user_messages = [{"role": "user", "content": "Hello."}]
        elif user_messages[0]["role"] != "user":
            user_messages.insert(0, {"role": "user", "content": "Continue."})

        # Prepend JSON instruction when json_mode is requested
        if json_mode:
            system_msg = (
                f"{_JSON_MODE_INSTRUCTION}\n\n{system_msg}"
                if system_msg
                else _JSON_MODE_INSTRUCTION
            )

        # BUG-DA8-05: Thinking-enabled Claude models require temperature=1.0
        # and do not accept other temperature values.
        _THINKING_MODELS = ("claude-3-7", "claude-4")
        _is_thinking = any(model.startswith(p) for p in _THINKING_MODELS)
        if _is_thinking:
            temperature = 1.0

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
            if self._client is None:
                self._client = httpx.Client(timeout=self.timeout_sec)
            response = self._client.post(url, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            # Convert to urllib.error.HTTPError for upstream retry logic.
            # Include Anthropic's error body so upstream logs show the
            # actual reason (e.g. "Prefilling not supported").
            detail = ""
            try:
                detail = exc.response.text[:500]
            except Exception:  # noqa: BLE001
                pass
            msg = f"{exc}: {detail}" if detail else str(exc)
            raise urllib.error.HTTPError(
                url,
                exc.response.status_code,
                msg,
                dict(exc.response.headers),
                None,
            ) from exc
        except httpx.HTTPError as exc:
            # Catch all transport errors (ConnectError, TimeoutException,
            # ReadError, RemoteProtocolError, PoolTimeout, etc.)
            raise urllib.error.URLError(str(exc)) from exc

        # Check for Anthropic error responses
        if data.get("type") == "error" or "error" in data:
            error_info = data.get("error", {})
            raise urllib.error.HTTPError(
                url,
                500,
                f"{error_info.get('type', 'api_error')}: {error_info.get('message', str(data))}",
                {},
                None,
            )

        # Extract ALL text content blocks (not just the first)
        content = ""
        if "content" in data and data["content"]:
            text_parts = [
                block.get("text", "")
                for block in data["content"]
                if block.get("type") == "text"
            ]
            content = "\n".join(text_parts)

        # Map Anthropic stop_reason to OpenAI finish_reason
        raw_stop_reason = data.get("stop_reason", "end_turn")
        finish_reason = _STOP_REASON_MAP.get(raw_stop_reason, "stop")

        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
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
