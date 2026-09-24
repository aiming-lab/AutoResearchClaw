"""Tests for the Cheaper Inference OpenAI-compatible provider preset."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from researchclaw.cli import (
    _PROVIDER_CHOICES,
    _PROVIDER_MODELS,
    _PROVIDER_URLS,
    cmd_init,
)
from researchclaw.llm import PROVIDER_PRESETS, create_llm_client
from researchclaw.llm.client import LLMClient, LLMConfig


class _DummyHTTPResponse:
    def __init__(self, payload: Mapping[str, Any]):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> _DummyHTTPResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _rc_config(
    *,
    api_key: str = "",
    api_key_env: str = "CHEAPER_INFERENCE_API_KEY",
    base_url: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        llm=SimpleNamespace(
            provider="cheaperinference",
            base_url=base_url,
            api_key=api_key,
            api_key_env=api_key_env,
            wire_api="chat_completions",
            primary_model="gpt-5.4-mini",
            fallback_models=("gpt-5.4", "claude-sonnet-5"),
            timeout_sec=60,
        )
    )


def test_cheaperinference_preset_and_cli_registration() -> None:
    assert PROVIDER_PRESETS["cheaperinference"]["base_url"] == (
        "https://api.cheaperinference.com/v1"
    )
    assert any(value[0] == "cheaperinference" for value in _PROVIDER_CHOICES.values())
    assert _PROVIDER_URLS["cheaperinference"] == "https://api.cheaperinference.com/v1"
    assert _PROVIDER_MODELS["cheaperinference"] == (
        "gpt-5.4-mini",
        ["gpt-5.4", "claude-sonnet-5"],
    )


def test_from_rc_config_uses_preset_and_isolated_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHEAPER_INFERENCE_API_KEY", "test-cheaperinference-key")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-key")

    client = create_llm_client(_rc_config())

    assert isinstance(client, LLMClient)
    assert client.config.base_url == "https://api.cheaperinference.com/v1"
    assert client.config.api_key == "test-cheaperinference-key"
    assert client._model_chain == [
        "gpt-5.4-mini",
        "gpt-5.4",
        "claude-sonnet-5",
    ]


def test_custom_base_url_overrides_cheaperinference_preset() -> None:
    client = LLMClient.from_rc_config(
        _rc_config(
            api_key="test-key",
            api_key_env="",
            base_url="https://proxy.test/v1",
        )
    )

    assert client.config.base_url == "https://proxy.test/v1"


def test_cheaperinference_request_uses_openai_compatible_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(
        request: urllib.request.Request, timeout: int
    ) -> _DummyHTTPResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return _DummyHTTPResponse(
            {
                "model": "gpt-5.4-mini",
                "choices": [
                    {"message": {"content": "pong"}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 1,
                    "total_tokens": 6,
                },
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = LLMClient(
        LLMConfig(
            base_url="https://api.cheaperinference.com/v1",
            api_key="test-cheaperinference-key",
            primary_model="gpt-5.4-mini",
            fallback_models=[],
            timeout_sec=60,
        )
    )

    response = client._raw_call(
        "gpt-5.4-mini",
        [{"role": "user", "content": "ping"}],
        1024,
        0.2,
        False,
    )

    request = captured["request"]
    assert isinstance(request, urllib.request.Request)
    assert request.full_url == "https://api.cheaperinference.com/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer test-cheaperinference-key"
    assert captured["timeout"] == 60
    assert response.content == "pong"


def test_init_wizard_writes_cheaperinference_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _TTY:
        def isatty(self) -> bool:
            return True

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "stdin", _TTY())
    # Look the choice number up rather than hard-coding it, so adding
    # another provider preset can't silently break this test.
    choice = next(
        key
        for key, value in _PROVIDER_CHOICES.items()
        if value[0] == "cheaperinference"
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: choice)
    monkeypatch.setattr("researchclaw.cli._prompt_opencode_install", lambda: False)

    assert cmd_init(argparse.Namespace(force=False)) == 0
    content = (tmp_path / "config.arc.yaml").read_text(encoding="utf-8")

    assert 'provider: "cheaperinference"' in content
    assert 'base_url: "https://api.cheaperinference.com/v1"' in content
    assert 'api_key_env: "CHEAPER_INFERENCE_API_KEY"' in content
    assert 'primary_model: "gpt-5.4-mini"' in content
    assert '    - "gpt-5.4"' in content
    assert '    - "claude-sonnet-5"' in content
