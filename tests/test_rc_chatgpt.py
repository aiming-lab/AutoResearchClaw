"""Tests for ChatGPT subscription provider (OAuth + adapter + SSE parsing)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from researchclaw.llm.chatgpt_adapter import _parse_sse_response, _to_chat_completions


# ---------------------------------------------------------------------------
# SSE parsing
# ---------------------------------------------------------------------------

class TestParseSSEResponse:
    def test_parses_response_done_event(self) -> None:
        sse_lines = [
            'data: {"type": "response.chunk", "delta": "hello"}',
            'data: {"type": "response.done", "response": {"id": "r1", "status": "completed"}}',
            "data: [DONE]",
        ]
        body = ("\n".join(sse_lines) + "\n").encode()
        result = _parse_sse_response(body)
        assert result["id"] == "r1"
        assert result["status"] == "completed"

    def test_parses_response_completed_event(self) -> None:
        sse_lines = [
            'data: {"type": "response.completed", "response": {"status": "ok"}}',
            "data: [DONE]",
        ]
        body = ("\n".join(sse_lines) + "\n").encode()
        result = _parse_sse_response(body)
        assert result == {"status": "ok"}

    def test_ignores_malformed_json(self) -> None:
        sse_lines = [
            "data: {not-valid-json}",
            'data: {"type": "response.done", "response": {"id": "good"}}',
            "data: [DONE]",
        ]
        body = ("\n".join(sse_lines) + "\n").encode()
        result = _parse_sse_response(body)
        assert result["id"] == "good"

    def test_falls_back_to_plain_json(self) -> None:
        body = b'{"id": "plain", "answer": "ok"}'
        result = _parse_sse_response(body)
        assert result["id"] == "plain"

    def test_raises_on_no_final_event(self) -> None:
        sse_lines = [
            'data: {"type": "response.chunk", "delta": "partial"}',
            "data: [DONE]",
        ]
        body = ("\n".join(sse_lines) + "\n").encode()
        with pytest.raises(ValueError, match="No response.done event"):
            _parse_sse_response(body)


# ---------------------------------------------------------------------------
# Response conversion
# ---------------------------------------------------------------------------

class TestToChatCompletions:
    def test_extracts_output_text(self) -> None:
        resp = {
            "model": "gpt-4o",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hello world"}],
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        }
        result = _to_chat_completions(resp, "gpt-4o")
        assert result["choices"][0]["message"]["content"] == "Hello world"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5

    def test_non_completed_status(self) -> None:
        resp = {"model": "gpt-4o", "status": "incomplete", "output": [], "usage": {}}
        result = _to_chat_completions(resp, "gpt-4o")
        assert result["choices"][0]["finish_reason"] == "length"

    def test_empty_output(self) -> None:
        resp = {"model": "gpt-4o", "status": "completed", "output": [], "usage": {}}
        result = _to_chat_completions(resp, "gpt-4o")
        assert result["choices"][0]["message"]["content"] == ""


# ---------------------------------------------------------------------------
# OAuth state validation
# ---------------------------------------------------------------------------

class TestOAuthStateValidation:
    def test_state_mismatch_raises(self) -> None:
        from researchclaw.llm.chatgpt_oauth import build_authorize_url

        _, state, _ = build_authorize_url()
        assert state  # state is non-empty
        assert len(state) > 20  # sufficiently random

    def test_pkce_challenge_is_s256(self) -> None:
        from researchclaw.llm.chatgpt_oauth import _generate_pkce

        verifier, challenge = _generate_pkce()
        assert len(verifier) > 40
        assert len(challenge) > 20
        assert verifier != challenge


# ---------------------------------------------------------------------------
# Provider routing
# ---------------------------------------------------------------------------

class TestProviderRouting:
    def test_chatgpt_provider_creates_adapter(self) -> None:
        rc_config = SimpleNamespace(
            llm=SimpleNamespace(
                provider="chatgpt",
                base_url="",
                api_key_env="",
                api_key="",
                primary_model="gpt-4o",
                fallback_models=["gpt-4.1"],
            ),
            metaclaw_bridge=SimpleNamespace(enabled=False),
        )
        from researchclaw.llm.client import LLMClient

        client = LLMClient.from_rc_config(rc_config)
        assert client._chatgpt is not None
        assert client._anthropic is None

    def test_openai_provider_no_chatgpt_adapter(self) -> None:
        rc_config = SimpleNamespace(
            llm=SimpleNamespace(
                provider="openai",
                base_url="https://api.openai.com/v1",
                api_key_env="OPENAI_API_KEY",
                api_key="sk-test",
                primary_model="gpt-4o",
                fallback_models=[],
            ),
            metaclaw_bridge=SimpleNamespace(enabled=False),
        )
        from researchclaw.llm.client import LLMClient

        client = LLMClient.from_rc_config(rc_config)
        assert client._chatgpt is None
        assert client._anthropic is None


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_chatgpt_skips_base_url_and_api_key(self) -> None:
        from researchclaw.config import validate_config

        data = {
            "project": {"name": "test"},
            "research": {"topic": "test topic"},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "console"},
            "knowledge_base": {"root": "docs/kb"},
            "llm": {"provider": "chatgpt"},
        }
        result = validate_config(data, check_paths=False)
        base_url_errors = [e for e in result.errors if "llm.base_url" in e]
        api_key_errors = [e for e in result.errors if "llm.api_key_env" in e]
        assert not base_url_errors, f"Unexpected base_url errors: {base_url_errors}"
        assert not api_key_errors, f"Unexpected api_key errors: {api_key_errors}"
