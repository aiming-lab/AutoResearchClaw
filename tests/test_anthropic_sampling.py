"""Anthropic sampling parameters across model generations."""

from __future__ import annotations

import json

import httpx
import pytest

from researchclaw.llm.anthropic_adapter import AnthropicAdapter


def _call(
    model: str, temperature: float = 0.7
) -> tuple[dict[str, object], dict[str, object]]:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    adapter = AnthropicAdapter("https://api.anthropic.com", "test")
    adapter._client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        result = adapter.chat_completion(
            model,
            [{"role": "user", "content": "hello"}],
            128,
            temperature,
        )
    finally:
        adapter.close()
    return captured, result


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-4-7",
        "anthropic/claude-opus-4-8",
        "claude-sonnet-5",
        "claude-haiku-5-5-20260101",
        "claude-fable-5",
        "claude-mythos-5",
    ],
)
def test_omits_temperature_for_models_that_reject_sampling_params(
    model: str,
) -> None:
    body, result = _call(model)

    assert "temperature" not in body
    assert result["choices"][0]["message"]["content"] == "ok"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-opus-4-6", 0.7),
        ("claude-opus-4-20250514", 0.7),
        ("claude-sonnet-4-6", 0.7),
        ("claude-haiku-4-5", 0.7),
        ("claude-3-7-sonnet-20250219", 1.0),
    ],
)
def test_keeps_temperature_for_models_that_accept_it(
    model: str, expected: float
) -> None:
    body, result = _call(model)

    assert body["temperature"] == expected
    assert result["choices"][0]["message"]["content"] == "ok"
