"""ChatGPT backend API adapter for Plus/Pro subscribers.

Converts standard Chat Completions requests into the Responses API format
used by ``chatgpt.com/backend-api/codex/responses``, allowing ChatGPT
subscription users to call OpenAI models without a separate API key.

The adapter follows the same protocol as :class:`AnthropicAdapter` —
it is injected into :class:`LLMClient` and returns an OpenAI-compatible
``dict`` that ``_raw_call`` can parse uniformly.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

CHATGPT_BACKEND_URL = "https://chatgpt.com/backend-api"
RESPONSES_PATH = "/codex/responses"

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

_JSON_MODE_INSTRUCTION = (
    "You MUST respond with valid JSON only. "
    "Do not include any text outside the JSON object."
)

# Valid models on the ChatGPT Codex backend and their normalized names.
_MODEL_MAP: dict[str, str] = {
    "gpt-5.2-codex": "gpt-5.2-codex",
    "gpt-5.2": "gpt-5.2",
    "gpt-5.1-codex-max": "gpt-5.1-codex-max",
    "gpt-5.1-codex-mini": "gpt-5.1-codex-mini",
    "gpt-5.1-codex": "gpt-5.1-codex",
    "gpt-5.1": "gpt-5.1",
    "gpt-5-codex": "gpt-5.1-codex",
    "gpt-5": "gpt-5.1",
    "codex-mini-latest": "gpt-5.1-codex-mini",
    "gpt-5.3-codex": "gpt-5.2-codex",
    "gpt-5.4": "gpt-5.2",
}

# Fallback model when the requested model is not in _MODEL_MAP
_DEFAULT_CODEX_MODEL = "gpt-5.1-codex"


def _normalize_model(model: str) -> str:
    """Map an arbitrary model name to a valid Codex backend model."""
    lower = model.lower().strip()
    if lower in _MODEL_MAP:
        return _MODEL_MAP[lower]
    # Pattern-based fallback
    if "codex" in lower:
        if "5.2" in lower:
            return "gpt-5.2-codex"
        if "mini" in lower:
            return "gpt-5.1-codex-mini"
        if "max" in lower:
            return "gpt-5.1-codex-max"
        return "gpt-5.1-codex"
    if "5.2" in lower:
        return "gpt-5.2"
    if "5.1" in lower:
        return "gpt-5.1"
    # Models not available on Codex backend (e.g. gpt-4o-mini) → use default
    logger.info("Model %r not available on ChatGPT backend, using %s", model, _DEFAULT_CODEX_MODEL)
    return _DEFAULT_CODEX_MODEL


def _parse_sse_response(raw_body: bytes) -> dict[str, Any]:
    """Parse an SSE stream and extract the final response object.

    The Codex backend returns ``stream: true`` SSE events.  The last
    ``response.done`` or ``response.completed`` event contains the
    complete Responses API object.
    """
    text = raw_body.decode("utf-8", errors="replace")
    final_response: dict[str, Any] | None = None

    for line in text.split("\n"):
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload.strip() == "[DONE]":
            break
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        event_type = data.get("type", "")
        if event_type in ("response.done", "response.completed"):
            final_response = data.get("response", data)

    if final_response is None:
        # Fallback: try to parse the whole body as plain JSON
        try:
            final_response = json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(
                f"No response.done event found in SSE stream. "
                f"Raw (first 500 chars): {text[:500]}"
            )

    return final_response


class ChatGPTAdapter:
    """Adapter that routes requests through the ChatGPT backend API.

    Uses the OAuth access token from ChatGPT Plus/Pro subscriptions
    and the Responses API format instead of Chat Completions.
    """

    def __init__(self, timeout_sec: int = 300) -> None:
        self.timeout_sec = timeout_sec
        self._tokens: Any = None  # Lazy-loaded AuthTokens

    def _get_tokens(self) -> Any:
        """Load and validate OAuth tokens (lazy, with auto-refresh)."""
        from researchclaw.llm.chatgpt_oauth import ensure_valid_token
        self._tokens = ensure_valid_token()
        return self._tokens

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        """Call ChatGPT backend Responses API and return OpenAI-compatible response.

        Raises ``urllib.error.HTTPError`` on API errors for upstream retry logic.
        """
        tokens = self._get_tokens()

        body = self._build_request_body(model, messages, temperature, json_mode)

        url = f"{CHATGPT_BACKEND_URL}{RESPONSES_PATH}"
        headers = {
            "Authorization": f"Bearer {tokens.access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": _DEFAULT_USER_AGENT,
            "chatgpt-account-id": tokens.account_id,
            "originator": "codex_cli_rs",
            "OpenAI-Beta": "responses=experimental",
        }

        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                raw = self._retry_with_refresh(url, payload, headers)
            else:
                raise

        data = _parse_sse_response(raw)
        return self._to_chat_completions(data, model)

    def _retry_with_refresh(
        self,
        url: str,
        payload: bytes,
        headers: dict[str, str],
    ) -> bytes:
        """Refresh token and retry the request once. Returns raw response body."""
        from researchclaw.llm.chatgpt_oauth import refresh_tokens, save_auth

        logger.info("Got 401, attempting token refresh...")
        if self._tokens is None:
            raise RuntimeError("No tokens available for refresh")

        new_tokens = refresh_tokens(self._tokens.refresh_token)
        save_auth(new_tokens)
        self._tokens = new_tokens

        headers = dict(headers)
        headers["Authorization"] = f"Bearer {new_tokens.access_token}"
        headers["chatgpt-account-id"] = new_tokens.account_id

        req = urllib.request.Request(url, data=payload, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            return resp.read()

    @staticmethod
    def _build_request_body(
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        json_mode: bool,
    ) -> dict[str, Any]:
        """Convert Chat Completions messages to Codex backend request body.

        Note: the Codex backend rejects ``temperature`` — it is intentionally
        omitted.  Reasoning effort is used to control output quality instead.
        """
        codex_model = _normalize_model(model)

        # Separate system messages (→ instructions) from conversation input
        instructions_parts: list[str] = []
        input_items: list[dict[str, Any]] = []

        for msg in messages:
            if msg["role"] == "system":
                instructions_parts.append(msg["content"])
            else:
                input_items.append({
                    "type": "message",
                    "role": msg["role"],
                    "content": [{"type": "input_text", "text": msg["content"]}],
                })

        if json_mode:
            instructions_parts.insert(0, _JSON_MODE_INSTRUCTION)

        if not input_items:
            input_items = [{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello."}],
            }]

        instructions = "\n\n".join(instructions_parts) if instructions_parts else ""

        # Determine reasoning effort based on model
        is_codex_max = "codex-max" in codex_model
        is_52 = "5.2" in codex_model
        effort = "high" if (is_codex_max or is_52) else "medium"

        body: dict[str, Any] = {
            "model": codex_model,
            "input": input_items,
            "instructions": instructions,
            "stream": True,
            "store": False,
            "reasoning": {
                "effort": effort,
                "summary": "auto",
            },
            "text": {
                "verbosity": "medium",
            },
            "include": ["reasoning.encrypted_content"],
        }

        if json_mode:
            body["text"]["format"] = {"type": "json_object"}

        return body

    @staticmethod
    def _to_chat_completions(data: dict[str, Any], original_model: str) -> dict[str, Any]:
        """Convert Responses API response to Chat Completions format."""
        if data.get("error"):
            error_info = data["error"]
            error_msg = error_info if isinstance(error_info, str) else error_info.get("message", str(error_info))
            raise urllib.error.HTTPError(
                "", 500, f"api_error: {error_msg}", {}, None
            )

        # Extract text content from output
        content = ""
        for item in data.get("output", []):
            if item.get("type") == "message":
                for block in item.get("content", []):
                    text_val = block.get("text", "")
                    if text_val:
                        content += text_val
            elif isinstance(item, dict) and "text" in item:
                content += item["text"]

        # Fallback for simple response shapes
        if not content and isinstance(data.get("output_text"), str):
            content = data["output_text"]
        if not content and isinstance(data.get("output"), str):
            content = data["output"]

        status = data.get("status", "completed")
        finish_reason = "stop" if status == "completed" else "length"
        if data.get("incomplete_details"):
            finish_reason = "length"

        usage = data.get("usage", {})
        return {
            "choices": [{
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            "model": data.get("model", original_model),
        }
