"""Tests for MiniMax provider integration.

Covers: provider preset, CLI registration, factory wiring,
temperature clamping, and live API integration.
"""

from __future__ import annotations

import json
import os
import urllib.request
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from researchclaw.llm import PROVIDER_PRESETS, create_llm_client
from researchclaw.llm.client import LLMClient, LLMConfig, LLMResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _DummyHTTPResponse:
    """Minimal stub for ``urllib.request.urlopen`` results."""

    def __init__(self, payload: Mapping[str, Any]):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> _DummyHTTPResponse:
        return self

    def __exit__(self, *a: object) -> None:
        return None


def _make_minimax_client(
    *,
    api_key: str = "test-minimax-key",
    primary_model: str = "MiniMax-M2.5",
    fallback_models: list[str] | None = None,
) -> LLMClient:
    config = LLMConfig(
        base_url="https://api.minimax.io/v1",
        api_key=api_key,
        primary_model=primary_model,
        fallback_models=fallback_models or ["MiniMax-M2.5-highspeed"],
    )
    return LLMClient(config)


# ---------------------------------------------------------------------------
# Unit tests — provider preset
# ---------------------------------------------------------------------------


class TestMiniMaxPreset:
    """Verify MiniMax is registered in PROVIDER_PRESETS."""

    def test_minimax_in_provider_presets(self):
        assert "minimax" in PROVIDER_PRESETS

    def test_minimax_base_url(self):
        assert PROVIDER_PRESETS["minimax"]["base_url"] == "https://api.minimax.io/v1"


# ---------------------------------------------------------------------------
# Unit tests — from_rc_config wiring
# ---------------------------------------------------------------------------


class TestMiniMaxFromRCConfig:
    """Verify that LLMClient.from_rc_config resolves MiniMax preset."""

    def test_from_rc_config_sets_minimax_base_url(self):
        rc_config = SimpleNamespace(
            llm=SimpleNamespace(
                provider="minimax",
                base_url="",
                api_key="mk-test",
                api_key_env="",
                primary_model="MiniMax-M2.5",
                fallback_models=("MiniMax-M2.5-highspeed",),
            ),
        )
        client = LLMClient.from_rc_config(rc_config)
        assert client.config.base_url == "https://api.minimax.io/v1"
        assert client.config.api_key == "mk-test"
        assert client.config.primary_model == "MiniMax-M2.5"
        assert client.config.fallback_models == ["MiniMax-M2.5-highspeed"]

    def test_from_rc_config_reads_minimax_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "env-minimax-key")
        rc_config = SimpleNamespace(
            llm=SimpleNamespace(
                provider="minimax",
                base_url="",
                api_key="",
                api_key_env="MINIMAX_API_KEY",
                primary_model="MiniMax-M2.5",
                fallback_models=(),
            ),
        )
        client = LLMClient.from_rc_config(rc_config)
        assert client.config.api_key == "env-minimax-key"

    def test_from_rc_config_custom_base_url_overrides_preset(self):
        rc_config = SimpleNamespace(
            llm=SimpleNamespace(
                provider="minimax",
                base_url="https://custom-proxy.example/v1",
                api_key="mk-test",
                api_key_env="",
                primary_model="MiniMax-M2.5",
                fallback_models=(),
            ),
        )
        client = LLMClient.from_rc_config(rc_config)
        assert client.config.base_url == "https://custom-proxy.example/v1"


# ---------------------------------------------------------------------------
# Unit tests — temperature clamping
# ---------------------------------------------------------------------------


class TestMiniMaxTemperatureClamping:
    """MiniMax API requires temperature in [0, 1.0]."""

    def _capture_body(
        self,
        monkeypatch: pytest.MonkeyPatch,
        client: LLMClient,
        temperature: float,
    ) -> dict[str, Any]:
        captured: dict[str, Any] = {}

        def fake_urlopen(req: urllib.request.Request, timeout: int) -> _DummyHTTPResponse:
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _DummyHTTPResponse(
                {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
            )

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        client._raw_call(
            "MiniMax-M2.5",
            [{"role": "user", "content": "hi"}],
            1024,
            temperature,
            False,
        )
        return captured["body"]

    def test_temperature_above_one_clamped(self, monkeypatch):
        client = _make_minimax_client()
        body = self._capture_body(monkeypatch, client, 1.5)
        assert body["temperature"] == 1.0

    def test_temperature_within_range_unchanged(self, monkeypatch):
        client = _make_minimax_client()
        body = self._capture_body(monkeypatch, client, 0.7)
        assert body["temperature"] == 0.7

    def test_temperature_zero_allowed(self, monkeypatch):
        client = _make_minimax_client()
        body = self._capture_body(monkeypatch, client, 0.0)
        assert body["temperature"] == 0.0

    def test_temperature_negative_clamped_to_zero(self, monkeypatch):
        client = _make_minimax_client()
        body = self._capture_body(monkeypatch, client, -0.1)
        assert body["temperature"] == 0.0

    def test_non_minimax_url_no_clamping(self, monkeypatch):
        """Non-MiniMax URLs should not clamp temperature."""
        config = LLMConfig(
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            primary_model="gpt-4o",
        )
        client = LLMClient(config)
        captured: dict[str, Any] = {}

        def fake_urlopen(req: urllib.request.Request, timeout: int) -> _DummyHTTPResponse:
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _DummyHTTPResponse(
                {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
            )

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        client._raw_call("gpt-4o", [{"role": "user", "content": "hi"}], 1024, 1.5, False)
        assert captured["body"]["temperature"] == 1.5  # no clamping


# ---------------------------------------------------------------------------
# Unit tests — model chain
# ---------------------------------------------------------------------------


class TestMiniMaxModelChain:
    """Model fallback chain for MiniMax."""

    def test_model_chain_default(self):
        client = _make_minimax_client()
        assert client._model_chain == ["MiniMax-M2.5", "MiniMax-M2.5-highspeed"]

    def test_model_chain_custom_fallbacks(self):
        client = _make_minimax_client(
            primary_model="MiniMax-M2.7",
            fallback_models=["MiniMax-M2.5", "MiniMax-M2.5-highspeed"],
        )
        assert client._model_chain == [
            "MiniMax-M2.7",
            "MiniMax-M2.5",
            "MiniMax-M2.5-highspeed",
        ]


# ---------------------------------------------------------------------------
# Unit tests — raw call body structure
# ---------------------------------------------------------------------------


class TestMiniMaxRawCall:
    """Verify request body sent to MiniMax API."""

    def test_request_body_structure(self, monkeypatch):
        client = _make_minimax_client()
        captured: dict[str, Any] = {}

        def fake_urlopen(req: urllib.request.Request, timeout: int) -> _DummyHTTPResponse:
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode("utf-8"))
            captured["headers"] = {k.lower(): v for k, v in req.headers.items()}
            return _DummyHTTPResponse(
                {
                    "model": "MiniMax-M2.5",
                    "choices": [{"message": {"content": "pong"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                }
            )

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        resp = client._raw_call(
            "MiniMax-M2.5",
            [{"role": "user", "content": "ping"}],
            1024,
            0.5,
            False,
        )
        assert captured["url"] == "https://api.minimax.io/v1/chat/completions"
        assert captured["body"]["model"] == "MiniMax-M2.5"
        assert captured["body"]["temperature"] == 0.5
        assert captured["headers"]["authorization"] == "Bearer test-minimax-key"
        assert resp.content == "pong"
        assert resp.model == "MiniMax-M2.5"

    def test_json_mode_adds_response_format(self, monkeypatch):
        client = _make_minimax_client()
        captured: dict[str, Any] = {}

        def fake_urlopen(req: urllib.request.Request, timeout: int) -> _DummyHTTPResponse:
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _DummyHTTPResponse(
                {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}
            )

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        client._raw_call(
            "MiniMax-M2.5",
            [{"role": "user", "content": "json"}],
            1024,
            0.5,
            True,
        )
        assert captured["body"]["response_format"] == {"type": "json_object"}


# ---------------------------------------------------------------------------
# Unit tests — CLI provider registration
# ---------------------------------------------------------------------------


class TestMiniMaxCLI:
    """Verify MiniMax is in the CLI interactive provider menu."""

    def test_minimax_in_provider_choices(self):
        from researchclaw.cli import _PROVIDER_CHOICES

        found = any(v[0] == "minimax" for v in _PROVIDER_CHOICES.values())
        assert found, "minimax not found in _PROVIDER_CHOICES"

    def test_minimax_in_provider_urls(self):
        from researchclaw.cli import _PROVIDER_URLS

        assert _PROVIDER_URLS["minimax"] == "https://api.minimax.io/v1"

    def test_minimax_in_provider_models(self):
        from researchclaw.cli import _PROVIDER_MODELS

        primary, fallbacks = _PROVIDER_MODELS["minimax"]
        assert primary == "MiniMax-M2.5"
        assert "MiniMax-M2.5-highspeed" in fallbacks


# ---------------------------------------------------------------------------
# Unit tests — factory function
# ---------------------------------------------------------------------------


class TestMiniMaxFactory:
    """Verify create_llm_client dispatches correctly for MiniMax."""

    def test_create_llm_client_returns_llm_client(self):
        from researchclaw.config import LlmConfig, RCConfig

        rc_config = SimpleNamespace(
            llm=SimpleNamespace(
                provider="minimax",
                base_url="",
                api_key="mk-factory-test",
                api_key_env="",
                primary_model="MiniMax-M2.5",
                fallback_models=(),
            ),
        )
        client = create_llm_client(rc_config)
        assert isinstance(client, LLMClient)
        assert client.config.base_url == "https://api.minimax.io/v1"
        assert client._anthropic is None  # Not anthropic


# ---------------------------------------------------------------------------
# Unit tests — chat fallback with MiniMax models
# ---------------------------------------------------------------------------


class TestMiniMaxChatFallback:
    """Verify fallback works with MiniMax models."""

    def test_fallback_to_highspeed_on_primary_failure(self, monkeypatch):
        client = _make_minimax_client()
        calls: list[str] = []

        def fake_call_with_retry(
            self,
            model: str,
            messages: list[dict[str, str]],
            max_tokens: int,
            temperature: float,
            json_mode: bool,
        ) -> LLMResponse:
            calls.append(model)
            if model == "MiniMax-M2.5":
                raise RuntimeError("rate limited")
            return LLMResponse(content="ok", model=model)

        monkeypatch.setattr(LLMClient, "_call_with_retry", fake_call_with_retry)
        resp = client.chat([{"role": "user", "content": "test"}])
        assert calls == ["MiniMax-M2.5", "MiniMax-M2.5-highspeed"]
        assert resp.model == "MiniMax-M2.5-highspeed"


# ---------------------------------------------------------------------------
# Integration tests — live MiniMax API (skipped without key)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set",
)
class TestMiniMaxLiveAPI:
    """Integration tests against the real MiniMax API."""

    def _live_client(self) -> LLMClient:
        return LLMClient(
            LLMConfig(
                base_url="https://api.minimax.io/v1",
                api_key=os.environ["MINIMAX_API_KEY"],
                primary_model="MiniMax-M2.5",
                fallback_models=["MiniMax-M2.5-highspeed"],
                max_tokens=64,
                timeout_sec=60,
            )
        )

    def test_simple_chat_completion(self):
        client = self._live_client()
        resp = client.chat(
            [{"role": "user", "content": "Say 'hello' and nothing else."}],
            max_tokens=16,
            temperature=0.1,
        )
        assert resp.content.strip(), "empty response"
        assert "hello" in resp.content.lower()

    def test_json_mode(self):
        client = self._live_client()
        resp = client.chat(
            [
                {"role": "system", "content": "You are a helpful assistant that responds in JSON."},
                {"role": "user", "content": 'Return a JSON object with key "status" set to "ok".'},
            ],
            max_tokens=128,
            temperature=0.1,
            json_mode=True,
            strip_thinking=True,
        )
        # MiniMax M2.5 may wrap JSON in markdown code fences
        import re
        text = resp.content.strip()
        fence_match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()
        parsed = json.loads(text)
        assert "status" in parsed

    def test_preflight_check(self):
        client = self._live_client()
        ok, msg = client.preflight()
        assert ok, f"preflight failed: {msg}"
