"""Fixed-model ACP routing checks with no agent or model requests."""

from __future__ import annotations

import copy
import json
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from researchclaw.llm.acp_client import ACPClient, ACPConfig


ASTRA = "gpt-6-astra"
SOL = "gpt-6.1-sol"


class FakeBridge:
    """Simulate the installed acpx command and native configuration schema."""

    def __init__(self) -> None:
        self.operations: list[tuple[str, str]] = []
        self.states: dict[str, dict[str, Any]] = {}
        self.accepted_modifier = lambda options: options
        self.record_modifier = lambda record: record
        self.failure_key: str | None = None
        self.invalid_json = False
        self.prompt_failure_once = False

    @staticmethod
    def options(model: str, effort: str) -> list[dict[str, Any]]:
        return [
            {
                "id": "model", "category": "model", "type": "select",
                "currentValue": model,
                "options": [{"value": ASTRA, "name": "Astra"}, {"value": SOL, "name": "Sol"}],
            },
            {
                "id": "reasoning_effort", "category": "thought_level", "type": "select",
                "currentValue": effort,
                "options": [{"value": "medium", "name": "Medium"}, {"value": "ultra", "name": "Ultra"}],
            },
        ]

    def run(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "sessions" in cmd:
            index = cmd.index("sessions")
            action = cmd[index + 1]
            name = cmd[cmd.index("--name") + 1] if "--name" in cmd else cmd[index + 2]
            self.operations.append((action, name))
            if action in ("ensure", "new"):
                self.states.setdefault(name, {"model": SOL, "effort": "medium"})
                payload: Any = {"action": "session_ensure"}
            elif action == "close":
                self.states.pop(name, None)
                payload = {"action": "session_close"}
            elif action == "show":
                state = self.states[name]
                record = {
                    "name": name, "closed": False,
                    "acpxRecordId": f"record-{name}", "acpSessionId": f"session-{name}",
                    "agentSessionId": f"thread-{name}",
                    "acpx": {"config_options": self.options(state["model"], state["effort"])},
                    "conversation": [{"prompt": "PRIVATE HISTORY MUST NOT ENTER SELECTION EVIDENCE"}],
                }
                payload = self.record_modifier(record)
            else:
                raise AssertionError(f"Unexpected session operation: {action}")
        elif "set" in cmd:
            name = cmd[cmd.index("-s") + 1]
            key, value = cmd[cmd.index("set") + 1:cmd.index("set") + 3]
            self.operations.append((key, name))
            if key == self.failure_key:
                return subprocess.CompletedProcess(cmd, 2, "", "Unsupported configuration")
            state = self.states[name]
            state["model" if key == "model" else "effort"] = value
            if self.invalid_json:
                return subprocess.CompletedProcess(cmd, 0, "PRIVATE INVALID RESPONSE", "")
            payload = {
                "action": "model_set" if key == "model" else "config_set",
                "acpxRecordId": f"record-{name}", "acpxSessionId": f"session-{name}",
                "agentSessionId": f"thread-{name}",
            }
            if key == "reasoning_effort":
                payload["configOptions"] = self.accepted_modifier(self.options(state["model"], state["effort"]))
        else:
            name = cmd[cmd.index("-s") + 1]
            self.operations.append(("warmup", name))
            payload = None
        return subprocess.CompletedProcess(cmd, 0, json.dumps(payload) if payload is not None else "OK", "")

    def prompt(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        name = cmd[cmd.index("-s") + 1]
        self.operations.append(("prompt", name))
        if self.prompt_failure_once:
            self.prompt_failure_once = False
            return subprocess.CompletedProcess(cmd, 1, "", "session not found")
        assert self.states[name]["effort"] == "ultra"
        assert "text-generation backend" in (kwargs.get("input_data") or cmd[-1])
        return subprocess.CompletedProcess(cmd, 0, "[client] ready\nanswer\n[done]", "")


@pytest.fixture
def bridge(monkeypatch: pytest.MonkeyPatch) -> FakeBridge:
    fake = FakeBridge()
    monkeypatch.setattr("researchclaw.llm.acp_client.subprocess.run", fake.run)
    monkeypatch.setattr(ACPClient, "_run_acp_with_heartbeat", lambda _self, cmd, **kwargs: fake.prompt(cmd, **kwargs))
    return fake


@pytest.fixture
def clients(bridge: FakeBridge):
    created: list[ACPClient] = []

    def create(model: str = ASTRA, *, name: str = "judge", **overrides: Any) -> ACPClient:
        config = dict(agent="codex", acpx_command="fake-acpx", session_name=name,
                      primary_model=model, reasoning_effort="ultra")
        config.update(overrides)
        client = ACPClient(ACPConfig(**config))
        created.append(client)
        return client

    yield create
    for client in created:
        client.close()


def test_prepare_uses_no_model_prompt_and_is_repeatable(clients, bridge: FakeBridge):
    client = clients()
    first = client.prepare()
    second = client.prepare()
    assert first == second
    assert first["selected_model"] == ASTRA
    assert first["selected_reasoning_effort"] == "ultra"
    assert first["verified"] is True
    assert first["model_advertised"] and first["effort_advertised"]
    assert bridge.operations == [
        ("ensure", "judge"), ("model", "judge"), ("reasoning_effort", "judge"), ("show", "judge"),
        ("model", "judge"), ("reasoning_effort", "judge"), ("show", "judge"),
    ]
    assert "PRIVATE HISTORY" not in json.dumps(first)


def test_two_models_have_independent_named_sessions(clients, bridge: FakeBridge):
    judge = clients(ASTRA, name="judge")
    executor = clients(SOL, name="executor")
    judge.prepare()
    executor.prepare()
    assert judge.session_selection["acpx_record_id"] != executor.session_selection["acpx_record_id"]
    assert bridge.states == {
        "judge": {"model": ASTRA, "effort": "ultra"},
        "executor": {"model": SOL, "effort": "ultra"},
    }
    judge.close()
    assert executor.session_selection["verified"] is True
    assert "executor" in bridge.states
    assert judge.session_selection["verified"] is False


def test_every_chat_revalidates_before_prompt(clients, bridge: FakeBridge):
    client = clients()
    client.prepare()
    bridge.states["judge"] = {"model": SOL, "effort": "medium"}
    response = client.chat([{"role": "user", "content": "write a conclusion"}], model=ASTRA)
    assert response.model == ASTRA
    assert response.content == "answer"
    assert response.raw["acp_selection"]["verified"] is True
    assert bridge.operations[-4:] == [
        ("model", "judge"), ("reasoning_effort", "judge"), ("show", "judge"), ("prompt", "judge"),
    ]


def test_wrong_chat_model_fails_without_touching_bridge(clients, bridge: FakeBridge):
    client = clients()
    with pytest.raises(RuntimeError, match="fixed to"):
        client.chat([{"role": "user", "content": "hello"}], model=SOL)
    assert client.selection_error
    assert bridge.operations == []
    client.prepare()
    assert client.selection_error is None


@pytest.mark.parametrize("problem", ["missing_model", "missing_effort", "unsupported_model", "unsupported_effort", "wrong_model", "wrong_effort", "duplicate_model", "missing_options"])
def test_bad_adapter_acknowledgement_blocks_prompt_and_latches_error(clients, bridge: FakeBridge, problem: str):
    def corrupt(options):
        options = copy.deepcopy(options)
        if problem == "missing_model":
            return options[1:]
        if problem == "missing_effort":
            return options[:1]
        if problem == "duplicate_model":
            return options + [options[0]]
        if problem == "missing_options":
            return None
        target = options[0] if problem.endswith("model") else options[1]
        if problem.startswith("unsupported"):
            target["options"] = []
        else:
            target["currentValue"] = SOL if problem.endswith("model") else "medium"
        return options

    bridge.accepted_modifier = corrupt
    client = clients()
    with pytest.raises(RuntimeError, match="ACP selection response"):
        client.chat([{"role": "user", "content": "hello"}])
    assert client.selection_error
    assert client.session_selection["verified"] is False
    assert "prompt" not in [action for action, _ in bridge.operations]
    bridge.accepted_modifier = lambda options: options
    client.prepare()
    assert client.selection_error is None


@pytest.mark.parametrize("problem", ["wrong_model", "wrong_effort", "missing_state", "wrong_session", "wrong_record", "closed_session"])
def test_readback_must_match_full_configuration_and_session(clients, bridge: FakeBridge, problem: str):
    def corrupt(record):
        if problem == "missing_state":
            record.pop("acpx")
        elif problem == "wrong_session":
            record["acpSessionId"] = "other-session"
        elif problem == "wrong_record":
            record["acpxRecordId"] = "other-record"
        elif problem == "closed_session":
            record["closed"] = True
        else:
            index = 0 if problem == "wrong_model" else 1
            record["acpx"]["config_options"][index]["currentValue"] = SOL if index == 0 else "medium"
        return record

    bridge.record_modifier = corrupt
    client = clients()
    with pytest.raises(RuntimeError):
        client.chat([{"role": "user", "content": "hello"}])
    assert client.selection_error
    assert not client.session_selection["verified"]
    assert "prompt" not in [action for action, _ in bridge.operations]


def test_unsupported_ultra_is_not_replaced_by_default_effort(clients, bridge: FakeBridge):
    bridge.failure_key = "reasoning_effort"
    client = clients()
    with pytest.raises(RuntimeError, match="Unsupported configuration"):
        client.prepare()
    assert bridge.states["judge"]["effort"] == "medium"
    assert client.selection_error
    assert bridge.operations == [("ensure", "judge"), ("model", "judge"), ("reasoning_effort", "judge")]


def test_invalid_json_does_not_expose_raw_bridge_response(clients, bridge: FakeBridge):
    bridge.invalid_json = True
    client = clients()
    with pytest.raises(RuntimeError, match="invalid JSON") as exc:
        client.prepare()
    assert "PRIVATE" not in str(exc.value)
    assert client.selection_error


def test_reconnect_reapplies_and_verifies_before_retry(clients, bridge: FakeBridge):
    bridge.prompt_failure_once = True
    client = clients()
    response = client.chat([{"role": "user", "content": "hello"}])
    assert response.content == "answer"
    assert bridge.operations == [
        ("ensure", "judge"), ("model", "judge"), ("reasoning_effort", "judge"), ("show", "judge"), ("prompt", "judge"),
        ("close", "judge"), ("ensure", "judge"), ("model", "judge"), ("reasoning_effort", "judge"), ("show", "judge"), ("prompt", "judge"),
    ]
    assert client.selection_error is None


def test_stdin_prompt_transport_keeps_verification_order(clients, bridge: FakeBridge):
    client = clients()
    client._MAX_CLI_PROMPT_BYTES = 1
    assert client.chat([{"role": "user", "content": "long prompt"}]).content == "answer"
    assert bridge.operations[-4:] == [
        ("model", "judge"), ("reasoning_effort", "judge"), ("show", "judge"), ("prompt", "judge"),
    ]


def test_legacy_session_keeps_warmup_and_agent_owned_model(clients, bridge: FakeBridge, monkeypatch: pytest.MonkeyPatch):
    client = clients("", reasoning_effort="")
    client.prepare()
    assert bridge.operations == [("ensure", "judge"), ("warmup", "judge")]
    monkeypatch.setattr(client, "_send_prompt_cli", lambda acpx, prompt: "legacy answer")
    response = client.chat([{"role": "user", "content": "hello"}], model=SOL)
    assert response.model == "acp:codex"
    assert response.raw == {}
    assert bridge.operations == [("ensure", "judge"), ("warmup", "judge")]


@pytest.mark.parametrize("enabled", [True, False])
def test_from_config_only_enables_selection_with_explicit_opt_in(enabled: bool):
    acp = SimpleNamespace(agent="codex", cwd=".", model_routing=enabled, reasoning_effort="ultra")
    rc = SimpleNamespace(llm=SimpleNamespace(acp=acp, primary_model=ASTRA))
    client = ACPClient.from_rc_config(rc)
    assert client.config.primary_model == (ASTRA if enabled else "")
    assert client.config.reasoning_effort == ("ultra" if enabled else "")


def test_opt_in_requires_complete_model_configuration():
    acp = SimpleNamespace(agent="codex", cwd=".", model_routing=True, reasoning_effort="")
    rc = SimpleNamespace(llm=SimpleNamespace(acp=acp, primary_model=ASTRA))
    with pytest.raises(ValueError, match="requires"):
        ACPClient.from_rc_config(rc)


def test_grouped_advertised_values_are_supported(clients, bridge: FakeBridge):
    def group(options):
        for option in options:
            option["options"] = [{"group": "Supported", "options": option["options"]}]
        return options

    bridge.accepted_modifier = group
    assert clients().prepare()["verified"] is True
