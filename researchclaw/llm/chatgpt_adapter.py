"""ChatGPT Backend API adapter for ResearchClaw.

Routes requests through the ChatGPT backend (Responses API) using an
OAuth access token from a ChatGPT Plus/Pro subscription. Returns
OpenAI-compatible response dicts so LLMClient can parse them uniformly.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_BACKEND_URL = "https://chatgpt.com/backend-api/v1/responses"


def _parse_sse_response(raw_body: bytes) -> dict[str, Any]:
    """Parse an SSE stream and extract the final response object.

    The backend returns ``stream: true`` SSE events. The last
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
        try:
            final_response = json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(
                f"No response.done event found in SSE stream. "
                f"Raw (first 500 chars): {text[:500]}"
            )

    return final_response


def _to_chat_completions(resp: dict[str, Any], model: str) -> dict[str, Any]:
    """Convert a Responses API object to an OpenAI Chat Completions dict."""
    output_text = ""
    for item in resp.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    output_text += part.get("text", "")

    usage = resp.get("usage", {})

    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": output_text},
                "finish_reason": "stop"
                if resp.get("status") == "completed"
                else "length",
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
        "model": resp.get("model", model),
    }


class ChatGPTAdapter:
    """Adapter that routes requests through the ChatGPT backend API.

    Requires a valid OAuth access token obtained via ``researchclaw login``.
    """

    def __init__(self, timeout_sec: int = 300) -> None:
        self.timeout_sec = timeout_sec

    def _get_access_token(self) -> str:
        from .chatgpt_oauth import get_valid_tokens

        return get_valid_tokens().access_token

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        """Call the ChatGPT backend and return an OpenAI-compatible dict.

        Raises ``urllib.error.HTTPError`` on API errors so that
        ``LLMClient._call_with_retry`` works unchanged.
        """
        access_token = self._get_access_token()

        input_parts: list[dict[str, Any]] = []
        system_msg = None
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                input_parts.append(
                    {"type": "message", "role": msg["role"],
                     "content": msg["content"]}
                )

        body: dict[str, Any] = {
            "model": model,
            "input": input_parts,
            "stream": True,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_msg:
            body["instructions"] = system_msg
        if json_mode:
            body["text"] = {"format": {"type": "json_object"}}

        payload = json.dumps(body).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        req = urllib.request.Request(
            _BACKEND_URL, data=payload, headers=headers
        )

        try:
            with urllib.request.urlopen(
                req, timeout=self.timeout_sec
            ) as resp:
                raw_body = resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                logger.info("Access token expired mid-request, refreshing...")
                from .chatgpt_oauth import get_valid_tokens, save_auth, refresh_tokens, load_auth

                tokens = load_auth()
                if tokens and tokens.refresh_token:
                    new_tokens = refresh_tokens(tokens.refresh_token)
                    save_auth(new_tokens)
                    headers["Authorization"] = f"Bearer {new_tokens.access_token}"
                    retry_req = urllib.request.Request(
                        _BACKEND_URL, data=payload, headers=headers
                    )
                    with urllib.request.urlopen(
                        retry_req, timeout=self.timeout_sec
                    ) as resp:
                        raw_body = resp.read()
                else:
                    raise
            else:
                raise

        parsed = _parse_sse_response(raw_body)
        return _to_chat_completions(parsed, model)
