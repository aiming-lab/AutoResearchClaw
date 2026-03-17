"""测试 Anthropic Messages 兼容 API 是否可用。"""

from __future__ import annotations

from typing import Any

import os

import httpx

BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
API_KEY = os.environ["ANTHROPIC_API_KEY"]  # Required — set before running
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")


def _create_message() -> dict[str, Any]:
    url = f"{BASE_URL.rstrip('/')}/v1/messages"
    headers = {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
        "x-api-key": API_KEY,
    }
    payload = {
        "model": MODEL,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()


def test_anthropic_api() -> None:
    message = _create_message()
    usage = message.get("usage", {})
    content = message.get("content", [])
    text_blocks = [block.get("text", "") for block in content if block.get("type") == "text"]

    print(f"Status: stop_reason={message.get('stop_reason')}")
    print(f"Model: {message.get('model')}")
    print(f"Usage: input={usage.get('input_tokens')}, output={usage.get('output_tokens')}")
    print(f"Response: {' '.join(text_blocks)}")

    assert message.get("type") == "message"
    assert len(content) > 0
    print("\n✅ API 可用!")


if __name__ == "__main__":
    test_anthropic_api()
