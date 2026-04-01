from __future__ import annotations

import json
import urllib.error
import urllib.request
from http.client import HTTPMessage
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from researchclaw.llm.client import (
    LLMClient,
    LLMConfig,
    LLMResponse,
    _NEW_PARAM_MODELS,
    _NO_TEMPERATURE_MODELS,
)


class _DummyHTTPResponse:
    def __init__(self, payload: Mapping[str, Any]):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> _DummyHTTPResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


def _make_client(
    *,
    api_key: str = "test-key",
    primary_model: str = "gpt-5.2",
    fallback_models: list[str] | None = None,
    wire_api: str = "chat_completions",
    timeout_sec: int = 120,
) -> LLMClient:
    config = LLMConfig(
        base_url="https://api.example.com/v1",
        api_key=api_key,
        wire_api=wire_api,
        primary_model=primary_model,
        fallback_models=fallback_models or ["gpt-5.1", "gpt-4.1", "gpt-4o"],
        timeout_sec=timeout_sec,
    )
    return LLMClient(config)


def _capture_raw_call(
    monkeypatch: pytest.MonkeyPatch, *, model: str, response_data: Mapping[str, Any]
) -> tuple[dict[str, object], LLMResponse, dict[str, object]]:
    captured: dict[str, object] = {}

    def fake_urlopen(req: urllib.request.Request, timeout: int) -> _DummyHTTPResponse:
        captured["request"] = req
        captured["timeout"] = timeout
        return _DummyHTTPResponse(response_data)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = _make_client()
    resp = client._raw_call(
        model, [{"role": "user", "content": "hello"}], 123, 0.2, False
    )
    request = captured["request"]
    assert isinstance(request, urllib.request.Request)
    data = request.data
    assert isinstance(data, bytes)
    body = json.loads(data.decode("utf-8"))
    assert isinstance(body, dict)
    return body, resp, captured


def test_llm_config_defaults():
    config = LLMConfig(base_url="https://api.example.com/v1", api_key="k")
    assert config.primary_model == "gpt-4o"
    assert config.max_tokens == 4096
    assert config.temperature == 0.7


def test_llm_config_custom_values():
    config = LLMConfig(
        base_url="https://custom.example/v1",
        api_key="custom",
        primary_model="o3",
        fallback_models=["o3-mini"],
        max_tokens=2048,
        temperature=0.1,
        timeout_sec=30,
    )
    assert config.primary_model == "o3"
    assert config.fallback_models == ["o3-mini"]
    assert config.max_tokens == 2048
    assert config.temperature == 0.1
    assert config.timeout_sec == 30


def test_llm_response_dataclass_fields():
    response = LLMResponse(content="ok", model="gpt-5.2", completion_tokens=10)
    assert response.content == "ok"
    assert response.model == "gpt-5.2"
    assert response.completion_tokens == 10


def test_llm_response_defaults():
    response = LLMResponse(content="ok", model="gpt-5.2")
    assert response.prompt_tokens == 0
    assert response.completion_tokens == 0
    assert response.total_tokens == 0
    assert response.finish_reason == ""
    assert response.truncated is False
    assert response.raw == {}


def test_llm_client_initialization_stores_config():
    config = LLMConfig(base_url="https://api.example.com/v1", api_key="k")
    client = LLMClient(config)
    assert client.config is config


def test_llm_client_model_chain_is_primary_plus_fallbacks():
    client = _make_client(
        primary_model="gpt-5.4", fallback_models=["gpt-4.1", "gpt-4o"]
    )
    assert client._model_chain == ["gpt-5.4", "gpt-4.1", "gpt-4o"]


def test_needs_max_completion_tokens_for_new_models():
    model = "gpt-5.2"
    assert any(model.startswith(prefix) for prefix in _NEW_PARAM_MODELS)


def test_needs_max_completion_tokens_false_for_old_models():
    model = "gpt-4o"
    assert not any(model.startswith(prefix) for prefix in _NEW_PARAM_MODELS)


def test_build_request_body_structure_via_raw_call(monkeypatch: pytest.MonkeyPatch):
    response = {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}
    body, _, _ = _capture_raw_call(monkeypatch, model="gpt-4o", response_data=response)
    assert body["model"] == "gpt-4o"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["temperature"] == 0.2


def test_build_request_uses_max_completion_tokens_for_new_models(
    monkeypatch: pytest.MonkeyPatch,
):
    response = {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}
    body, _, _ = _capture_raw_call(monkeypatch, model="gpt-5.2", response_data=response)
    # Reasoning models enforce a minimum of 32768 tokens
    assert body["max_completion_tokens"] == 32768
    assert "max_tokens" not in body
    assert body["temperature"] == 0.2


def test_build_request_uses_max_tokens_for_old_models(monkeypatch: pytest.MonkeyPatch):
    response = {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}
    body, _, _ = _capture_raw_call(monkeypatch, model="gpt-4.1", response_data=response)
    assert body["max_tokens"] == 123
    assert "max_completion_tokens" not in body


def test_parse_response_with_valid_payload_via_raw_call(
    monkeypatch: pytest.MonkeyPatch,
):
    response = {
        "model": "gpt-5.2",
        "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    _, parsed, _ = _capture_raw_call(
        monkeypatch, model="gpt-5.2", response_data=response
    )
    assert parsed.content == "hello"
    assert parsed.model == "gpt-5.2"
    assert parsed.prompt_tokens == 1
    assert parsed.total_tokens == 3


def test_parse_response_truncated_when_finish_reason_length(
    monkeypatch: pytest.MonkeyPatch,
):
    response = {
        "choices": [{"message": {"content": "partial"}, "finish_reason": "length"}],
        "usage": {},
    }
    _, parsed, _ = _capture_raw_call(
        monkeypatch, model="gpt-5.2", response_data=response
    )
    assert parsed.finish_reason == "length"
    assert parsed.truncated is True


def test_parse_response_missing_optional_fields_graceful(
    monkeypatch: pytest.MonkeyPatch,
):
    response = {"choices": [{"message": {"content": None}}]}
    _, parsed, _ = _capture_raw_call(
        monkeypatch, model="gpt-5.2", response_data=response
    )
    assert parsed.content == ""
    assert parsed.prompt_tokens == 0
    assert parsed.completion_tokens == 0
    assert parsed.total_tokens == 0
    assert parsed.finish_reason == ""


def test_from_rc_config_builds_expected_llm_config():
    rc_config = SimpleNamespace(
        llm=SimpleNamespace(
            base_url="https://proxy.example/v1",
            api_key="inline-key",
            api_key_env="OPENAI_API_KEY",
            wire_api="responses",
            primary_model="o3",
            fallback_models=("o3-mini", "gpt-4o"),
        )
    )
    client = LLMClient.from_rc_config(rc_config)
    assert client.config.base_url == "https://proxy.example/v1"
    assert client.config.api_key == "inline-key"
    assert client.config.wire_api == "responses"
    assert client.config.primary_model == "o3"
    assert client.config.fallback_models == ["o3-mini", "gpt-4o"]


def test_responses_wire_api_uses_responses_endpoint(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_urlopen(req: urllib.request.Request, timeout: int) -> _DummyHTTPResponse:
        captured["request"] = req
        captured["timeout"] = timeout
        return _DummyHTTPResponse(
            {
                "model": "gpt-4.1",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "hello"}],
                    }
                ],
                "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
                "status": "completed",
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = _make_client(primary_model="gpt-4.1", wire_api="responses")
    resp = client._raw_call(
        "gpt-4.1", [{"role": "user", "content": "hello"}], 123, 0.2, False
    )

    request = captured["request"]
    assert isinstance(request, urllib.request.Request)
    assert request.full_url == "https://api.example.com/v1/responses"
    assert captured["timeout"] == 120

    data = request.data
    assert isinstance(data, bytes)
    body = json.loads(data.decode("utf-8"))
    assert body["model"] == "gpt-4.1"
    assert body["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "hello"}]}
    ]
    assert body["max_output_tokens"] == 123
    assert resp.content == "hello"
    assert resp.prompt_tokens == 11
    assert resp.completion_tokens == 7
    assert resp.total_tokens == 18


def test_responses_wire_api_includes_temperature_for_gpt5_models(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_urlopen(req: urllib.request.Request, timeout: int) -> _DummyHTTPResponse:
        captured["request"] = req
        return _DummyHTTPResponse(
            {
                "model": "gpt-5.2",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
                "usage": {},
                "status": "completed",
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = _make_client(primary_model="gpt-5.2", wire_api="responses")
    _ = client._raw_call(
        "gpt-5.2", [{"role": "user", "content": "hello"}], 55, 0.2, False
    )

    request = captured["request"]
    assert isinstance(request, urllib.request.Request)
    data = request.data
    assert isinstance(data, bytes)
    body = json.loads(data.decode("utf-8"))
    assert body["temperature"] == 0.2


def test_responses_wire_api_omits_temperature_for_o_series_models(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_urlopen(req: urllib.request.Request, timeout: int) -> _DummyHTTPResponse:
        captured["request"] = req
        return _DummyHTTPResponse(
            {
                "model": "o3",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
                "usage": {},
                "status": "completed",
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = _make_client(primary_model="o3", wire_api="responses")
    _ = client._raw_call("o3", [{"role": "user", "content": "hello"}], 55, 0.2, False)

    request = captured["request"]
    assert isinstance(request, urllib.request.Request)
    data = request.data
    assert isinstance(data, bytes)
    body = json.loads(data.decode("utf-8"))
    assert "temperature" not in body


def test_preflight_404_reports_responses_endpoint():
    client = _make_client(primary_model="gpt-4.1", wire_api="responses")

    def fake_chat(*args: Any, **kwargs: Any) -> LLMResponse:
        raise urllib.error.HTTPError(
            url="https://api.example.com/v1/responses",
            code=404,
            msg="Not Found",
            hdrs=HTTPMessage(),
            fp=None,
        )

    client.chat = fake_chat  # type: ignore[method-assign]
    ok, msg = client.preflight()

    assert ok is False
    assert msg == "Endpoint not found: https://api.example.com/v1/responses"


def test_from_rc_config_reads_api_key_from_env_when_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("RC_TEST_API_KEY", "env-key")
    rc_config = SimpleNamespace(
        llm=SimpleNamespace(
            base_url="https://proxy.example/v1",
            api_key="",
            api_key_env="RC_TEST_API_KEY",
            primary_model="gpt-5.2",
            fallback_models=(),
        )
    )
    client = LLMClient.from_rc_config(rc_config)
    assert client.config.api_key == "env-key"


def test_acp_large_prompt_uses_file_transport_before_cli_limit():
    from researchclaw.llm.acp_client import ACPClient, ACPConfig

    client = ACPClient(ACPConfig(agent="codex"))
    client._acpx = "acpx"
    client._session_ready = True
    original_limit = ACPClient._MAX_CLI_PROMPT_BYTES
    ACPClient._MAX_CLI_PROMPT_BYTES = 10
    client._ensure_session = lambda: None  # type: ignore[assignment]

    def fail_cli(acpx: str, prompt: str) -> str:
        raise AssertionError("CLI transport should not be used for oversized prompts")

    client._send_prompt_cli = fail_cli  # type: ignore[assignment]
    client._send_prompt_via_file = lambda acpx, prompt: "ok-from-file"  # type: ignore[assignment]

    try:
        result = client._send_prompt("x" * 11)
        assert result == "ok-from-file"
    finally:
        ACPClient._MAX_CLI_PROMPT_BYTES = original_limit


def test_acp_command_line_too_long_falls_back_to_file_transport():
    from researchclaw.llm.acp_client import ACPClient, ACPConfig

    client = ACPClient(ACPConfig(agent="codex"))
    client._acpx = "acpx"
    client._session_ready = True
    client._MAX_CLI_PROMPT_BYTES = 1000  # type: ignore[attr-defined]
    client._ensure_session = lambda: None  # type: ignore[assignment]

    call_count = 0

    def fail_cli(acpx: str, prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        raise RuntimeError("ACP prompt failed (exit 1): The command line is too long.")

    client._send_prompt_cli = fail_cli  # type: ignore[assignment]
    client._send_prompt_via_file = lambda acpx, prompt: "ok-from-file"  # type: ignore[assignment]

    result = client._send_prompt("short prompt")
    assert result == "ok-from-file"
    assert call_count == 1


def test_acp_windows_cmd_wrapper_uses_lower_inline_limit(monkeypatch: pytest.MonkeyPatch):
    from researchclaw.llm.acp_client import ACPClient

    monkeypatch.setattr("researchclaw.llm.acp_client.sys.platform", "win32")
    limit = ACPClient._cli_prompt_limit(r"C:\Users\test\AppData\Roaming\npm\acpx.CMD")
    assert limit == ACPClient._MAX_CMD_WRAPPER_PROMPT_BYTES


def test_new_param_models_contains_expected_models():
    expected = {"gpt-5", "gpt-5.1", "gpt-5.2", "gpt-5.4", "o3", "o3-mini", "o4-mini"}
    assert expected.issubset(_NEW_PARAM_MODELS)


def test_no_temperature_models_only_contains_o_series_models():
    assert _NO_TEMPERATURE_MODELS == frozenset({"o3", "o3-mini", "o4-mini"})


def test_raw_call_adds_json_mode_response_format(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_urlopen(req: urllib.request.Request, timeout: int) -> _DummyHTTPResponse:
        captured["request"] = req
        return _DummyHTTPResponse({"choices": [{"message": {"content": "{}"}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = _make_client()
    _ = client._raw_call(
        "gpt-5.2", [{"role": "user", "content": "json"}], 50, 0.1, True
    )
    request = captured["request"]
    assert isinstance(request, urllib.request.Request)
    data = request.data
    assert isinstance(data, bytes)
    body = json.loads(data.decode("utf-8"))
    assert isinstance(body, dict)
    assert body["response_format"] == {"type": "json_object"}


def test_raw_call_sets_auth_and_user_agent_headers(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_urlopen(req: urllib.request.Request, timeout: int) -> _DummyHTTPResponse:
        captured["request"] = req
        captured["timeout"] = timeout
        return _DummyHTTPResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = _make_client(api_key="secret", timeout_sec=77)
    _ = client._raw_call("gpt-5.2", [{"role": "user", "content": "hi"}], 20, 0.6, False)
    request = captured["request"]
    assert isinstance(request, urllib.request.Request)
    headers = {k.lower(): v for k, v in request.headers.items()}
    assert headers["authorization"] == "Bearer secret"
    assert "user-agent" in headers
    timeout = captured["timeout"]
    assert timeout == 77


def test_chat_prepends_system_message(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, list[dict[str, str]]] = {}

    def fake_raw_call(
        self: LLMClient,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> LLMResponse:
        captured["messages"] = messages
        return LLMResponse(content="ok", model=model)

    monkeypatch.setattr(LLMClient, "_raw_call", fake_raw_call)
    client = _make_client(primary_model="gpt-5.2", fallback_models=["gpt-4o"])
    client.chat([{"role": "user", "content": "q"}], system="sys")
    assert captured["messages"][0] == {"role": "system", "content": "sys"}


def test_chat_uses_fallback_after_first_model_error(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    def fake_call_with_retry(
        self: LLMClient,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> LLMResponse:
        _ = (self, messages, max_tokens, temperature, json_mode)
        calls.append(model)
        if model == "gpt-5.2":
            raise RuntimeError("first failed")
        return LLMResponse(content="ok", model=model)

    monkeypatch.setattr(LLMClient, "_call_with_retry", fake_call_with_retry)
    client = _make_client(primary_model="gpt-5.2", fallback_models=["gpt-5.1"])
    response = client.chat([{"role": "user", "content": "x"}])
    assert calls == ["gpt-5.2", "gpt-5.1"]
    assert response.model == "gpt-5.1"
