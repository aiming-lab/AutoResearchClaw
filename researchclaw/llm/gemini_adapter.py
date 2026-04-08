"""Gemini Native API adapter for ResearchClaw."""

import json
import logging
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_JSON_MODE_INSTRUCTION = (
    "You MUST respond with valid JSON only. "
    "Do not include any text outside the JSON object."
)


class GeminiAdapter:
    """Adapter to call Gemini native API and return OpenAI-compatible response."""

    def __init__(self, base_url: str, api_key: str, timeout_sec: int = 300):
        # Allow trailing slash just in case
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_sec = timeout_sec

    def close(self) -> None:
        pass

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        """Call Gemini generateContent API and return OpenAI-compatible response.

        Raises urllib.error.HTTPError on API errors so the upstream retry
        logic in LLMClient._call_with_retry works unchanged.
        """
        system_parts: list[str] = []
        user_messages: list[dict[str, str]] = []

        for msg in messages:
            if msg.get("role") == "system":
                system_parts.append(msg["content"])
            else:
                user_messages.append(msg)

        system_msg = "\n\n".join(system_parts) if system_parts else None

        if json_mode:
            system_msg = (
                f"{_JSON_MODE_INSTRUCTION}\n\n{system_msg}"
                if system_msg
                else _JSON_MODE_INSTRUCTION
            )

        contents = []

        # Merge consecutive messages with same role, map "assistant" to "model"
        for msg in user_messages:
            role = "user" if msg["role"] == "user" else "model"
            content = msg["content"]

            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"][0]["text"] += "\n\n" + content
            else:
                contents.append({"role": role, "parts": [{"text": content}]})

        # Ensure it starts with user
        if not contents:
            contents = [{"role": "user", "parts": [{"text": "Hello."}]}]
        elif contents[0]["role"] != "user":
            contents.insert(0, {"role": "user", "parts": [{"text": "Continue."}]})

        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            }
        }

        if system_msg:
            body["systemInstruction"] = {
                "parts": [{"text": system_msg}]
            }

        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"

        # e.g. https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent
        url = f"{self.base_url}/models/{model}:generateContent"

        headers = {
            "x-goog-api-key": self.api_key,
            "content-type": "application/json",
        }

        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            # Attempt to extract detailed error from Gemini to aid debugging
            try:
                body_err = exc.read().decode("utf-8")
                logger.error("Gemini API Error: %s", body_err)
            except Exception: # noqa: BLE001
                pass
            raise

        # Map response back
        candidates = data.get("candidates", [])
        if not candidates:
            content = ""
            finish_reason = "stop"
        else:
            first_candidate = candidates[0]
            parts = first_candidate.get("content", {}).get("parts", [])
            content = "".join(
                part.get("text", "")
                for part in parts
                if isinstance(part, dict) and "text" in part
            )

            gemini_finish = first_candidate.get("finishReason", "STOP")
            if gemini_finish == "MAX_TOKENS":
                finish_reason = "length"
            else:
                finish_reason = "stop"

        usage = data.get("usageMetadata", {})

        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("promptTokenCount", 0),
                "completion_tokens": usage.get("candidatesTokenCount", 0),
                "total_tokens": usage.get("totalTokenCount", 0),
            },
            "model": model,
        }
