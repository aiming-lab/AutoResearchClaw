# pyright: reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnusedCallResult=false, reportAttributeAccessIssue=false, reportUnknownLambdaType=false, reportMissingImports=false, reportUntypedNamedTuple=false, reportMissingTypeArgument=false, reportArgumentType=false
from __future__ import annotations

import json
import socket
import urllib.error
from pathlib import Path
from typing import NamedTuple, cast
from unittest.mock import patch

import pytest

from researchclaw import health


class _VersionInfo(NamedTuple):
    major: int
    minor: int
    micro: int
    releaselevel: str
    serial: int


class _DummyHTTPResponse:
    status: int
    _payload: dict[str, object]

    def __init__(
        self, *, status: int = 200, payload: dict[str, object] | None = None
    ) -> None:
        self.status = status
        self._payload = payload if payload is not None else {}

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> _DummyHTTPResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


def _write_valid_config(path: Path) -> None:
    _ = path.write_text(
        """
project:
  name: demo
research:
  topic: Doctor checks
runtime:
  timezone: UTC
notifications:
  channel: test
knowledge_base:
  root: kb
llm:
  base_url: https://api.example.com/v1
  api_key_env: OPENAI_API_KEY
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_check_python_version_pass() -> None:
    with patch("sys.version_info", _VersionInfo(3, 11, 0, "final", 0)):
        result = health.check_python_version()
    assert result.status == "pass"


def test_check_python_version_fail() -> None:
    with patch("sys.version_info", _VersionInfo(3, 10, 9, "final", 0)):
        result = health.check_python_version()
    assert result.status == "fail"
    assert "Install Python 3.11 or newer" == result.fix


def test_check_yaml_import_pass() -> None:
    with patch("importlib.import_module", return_value=object()):
        result = health.check_yaml_import()
    assert result.status == "pass"


def test_check_yaml_import_fail() -> None:
    with patch("importlib.import_module", side_effect=ImportError):
        result = health.check_yaml_import()
    assert result.status == "fail"
    assert result.fix == "pip install pyyaml"


def test_check_config_valid_pass(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_valid_config(config_path)
    result = health.check_config_valid(config_path)
    assert result.status == "pass"


def test_check_config_invalid(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _ = config_path.write_text("project: {}\n", encoding="utf-8")
    result = health.check_config_valid(config_path)
    assert result.status == "fail"
    assert "Missing required field:" in result.detail


def test_check_config_missing_file(tmp_path: Path) -> None:
    result = health.check_config_valid(tmp_path / "missing.yaml")
    assert result.status == "fail"
    assert "Config file not found" in result.detail


def test_check_llm_connectivity_pass() -> None:
    with patch("urllib.request.urlopen", return_value=_DummyHTTPResponse(status=200)):
        result = health.check_llm_connectivity("https://api.example.com/v1")
    assert result.status == "pass"


def test_check_llm_connectivity_timeout() -> None:
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError(socket.timeout("timed out")),
    ):
        result = health.check_llm_connectivity("https://api.example.com/v1")
    assert result.status == "fail"
    assert result.detail == "LLM endpoint unreachable"


def test_check_llm_connectivity_http_error() -> None:
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.HTTPError(
            "https://api.example.com/v1/models", 503, "unavailable", {}, None
        ),
    ):
        result = health.check_llm_connectivity("https://api.example.com/v1")
    assert result.status == "fail"
    assert "503" in result.detail


def test_check_api_key_valid() -> None:
    with patch(
        "urllib.request.urlopen",
        return_value=_DummyHTTPResponse(status=200, payload={"data": []}),
    ):
        result = health.check_api_key_valid("https://api.example.com/v1", "sk-test")
    assert result.status == "pass"


def test_check_api_key_invalid_401() -> None:
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.HTTPError(
            "https://api.example.com/v1/models", 401, "unauthorized", {}, None
        ),
    ):
        result = health.check_api_key_valid("https://api.example.com/v1", "bad")
    assert result.status == "fail"
    assert result.detail == "Invalid API key"


def test_check_model_available_pass() -> None:
    payload = {"data": [{"id": "gpt-5.2"}, {"id": "gpt-4o"}]}
    with patch(
        "urllib.request.urlopen",
        return_value=_DummyHTTPResponse(status=200, payload=payload),
    ):
        result = health.check_model_available(
            "https://api.example.com/v1", "sk-test", "gpt-5.2"
        )
    assert result.status == "pass"


def test_check_model_not_available() -> None:
    payload = {"data": [{"id": "gpt-4o"}]}
    with patch(
        "urllib.request.urlopen",
        return_value=_DummyHTTPResponse(status=200, payload=payload),
    ):
        result = health.check_model_available(
            "https://api.example.com/v1", "sk-test", "gpt-5.2"
        )
    assert result.status == "fail"
    assert result.detail == "Model gpt-5.2 not available"


def test_check_model_chain_all_available() -> None:
    payload = {"data": [{"id": "gpt-4o"}, {"id": "gpt-4.1"}]}
    with patch(
        "urllib.request.urlopen",
        return_value=_DummyHTTPResponse(status=200, payload=payload),
    ):
        result = health.check_model_chain(
            "https://api.example.com/v1", "sk-test", "gpt-4o", ("gpt-4.1",)
        )
    assert result.status == "pass"
    assert "All models available" in result.detail


def test_check_model_chain_primary_missing_fallback_ok() -> None:
    payload = {"data": [{"id": "gpt-4.1"}, {"id": "gpt-4o-mini"}]}
    with patch(
        "urllib.request.urlopen",
        return_value=_DummyHTTPResponse(status=200, payload=payload),
    ):
        result = health.check_model_chain(
            "https://api.example.com/v1", "sk-test",
            "gpt-5.2", ("gpt-4.1", "gpt-4o-mini")
        )
    assert result.status == "pass"
    assert "unavailable" in result.detail
    assert "gpt-5.2" in result.detail


def test_check_model_chain_all_missing() -> None:
    payload = {"data": [{"id": "gpt-4o"}]}
    with patch(
        "urllib.request.urlopen",
        return_value=_DummyHTTPResponse(status=200, payload=payload),
    ):
        result = health.check_model_chain(
            "https://api.example.com/v1", "sk-test",
            "gpt-5.2", ("gpt-5.1",)
        )
    assert result.status == "fail"
    assert "No models available" in result.detail


def test_check_model_chain_no_models() -> None:
    result = health.check_model_chain(
        "https://api.example.com/v1", "sk-test", "", ()
    )
    assert result.status == "warn"
    assert "No models configured" in result.detail


def test_check_sandbox_python_exists() -> None:
    with (
        patch.object(Path, "exists", return_value=True),
        patch("os.access", return_value=True),
    ):
        result = health.check_sandbox_python(".venv_arc/bin/python3")
    assert result.status == "pass"


def test_check_sandbox_python_missing() -> None:
    with (
        patch.object(Path, "exists", return_value=False),
        patch("os.access", return_value=False),
    ):
        result = health.check_sandbox_python(".venv_arc/bin/python3")
    assert result.status == "warn"


def test_check_matplotlib_available() -> None:
    with patch("importlib.import_module", return_value=object()):
        result = health.check_matplotlib()
    assert result.status == "pass"


def test_check_matplotlib_missing() -> None:
    with patch("importlib.import_module", side_effect=ImportError):
        result = health.check_matplotlib()
    assert result.status == "warn"
    assert result.detail == "Not installed; charts will be skipped"


def test_check_experiment_mode_simulated() -> None:
    result = health.check_experiment_mode("simulated")
    assert result.status == "warn"


def test_check_experiment_mode_sandbox() -> None:
    result = health.check_experiment_mode("sandbox")
    assert result.status == "pass"


def test_run_doctor_all_pass_openai(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _ = config_path.write_text("project: {}\n", encoding="utf-8")
    with (
        patch.object(
            health,
            "check_python_version",
            return_value=health.CheckResult("python_version", "pass", "ok"),
        ),
        patch.object(
            health,
            "check_yaml_import",
            return_value=health.CheckResult("yaml_import", "pass", "ok"),
        ),
        patch.object(
            health,
            "check_config_valid",
            return_value=health.CheckResult("config_valid", "pass", "ok"),
        ),
        patch.object(
            health,
            "check_llm_connectivity",
            return_value=health.CheckResult("llm_connectivity", "pass", "ok"),
        ),
        patch.object(
            health,
            "check_api_key_valid",
            return_value=health.CheckResult("api_key_valid", "pass", "ok"),
        ),
        patch.object(
            health,
            "check_model_chain",
            return_value=health.CheckResult("model_chain", "pass", "ok"),
        ),
        patch.object(
            health,
            "check_sandbox_python",
            return_value=health.CheckResult("sandbox_python", "pass", "ok"),
        ),
        patch.object(
            health,
            "check_matplotlib",
            return_value=health.CheckResult("matplotlib", "pass", "ok"),
        ),
        patch.object(
            health,
            "check_experiment_mode",
            return_value=health.CheckResult("experiment_mode", "pass", "ok"),
        ),
    ):
        report = health.run_doctor(config_path)
    assert report.overall == "pass"
    assert len(report.checks) == 9


def test_run_doctor_with_failures(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _ = config_path.write_text("project: {}\n", encoding="utf-8")
    with (
        patch.object(
            health,
            "check_python_version",
            return_value=health.CheckResult("python_version", "pass", "ok"),
        ),
        patch.object(
            health,
            "check_yaml_import",
            return_value=health.CheckResult("yaml_import", "pass", "ok"),
        ),
        patch.object(
            health,
            "check_config_valid",
            return_value=health.CheckResult("config_valid", "fail", "bad", "fix it"),
        ),
        patch.object(
            health,
            "check_llm_connectivity",
            return_value=health.CheckResult("llm_connectivity", "pass", "ok"),
        ),
        patch.object(
            health,
            "check_api_key_valid",
            return_value=health.CheckResult("api_key_valid", "warn", "warn", "later"),
        ),
        patch.object(
            health,
            "check_model_chain",
            return_value=health.CheckResult("model_chain", "pass", "ok"),
        ),
        patch.object(
            health,
            "check_sandbox_python",
            return_value=health.CheckResult("sandbox_python", "pass", "ok"),
        ),
        patch.object(
            health,
            "check_matplotlib",
            return_value=health.CheckResult("matplotlib", "pass", "ok"),
        ),
        patch.object(
            health,
            "check_experiment_mode",
            return_value=health.CheckResult("experiment_mode", "pass", "ok"),
        ),
    ):
        report = health.run_doctor(config_path)
    assert report.overall == "fail"
    assert "fix it" in report.actionable_fixes


def test_doctor_report_json_structure(tmp_path: Path) -> None:
    report = health.DoctorReport(
        timestamp="2026-01-01T00:00:00+00:00",
        checks=[
            health.CheckResult("python_version", "pass", "ok"),
            health.CheckResult(
                "matplotlib", "warn", "missing", "pip install matplotlib"
            ),
        ],
        overall="pass",
    )
    output_path = tmp_path / "reports" / "doctor.json"
    health.write_doctor_report(report, output_path)

    raw = cast(dict[str, object], json.loads(output_path.read_text(encoding="utf-8")))
    assert raw["timestamp"] == "2026-01-01T00:00:00+00:00"
    assert raw["overall"] == "pass"
    assert isinstance(raw["checks"], list)
    assert raw["actionable_fixes"] == ["pip install matplotlib"]


def test_doctor_report_overall_logic() -> None:
    passing = health.DoctorReport(
        timestamp="2026-01-01T00:00:00+00:00",
        checks=[health.CheckResult("x", "pass", "ok")],
        overall="pass",
    )
    failing = health.DoctorReport(
        timestamp="2026-01-01T00:00:00+00:00",
        checks=[health.CheckResult("x", "fail", "bad", "fix")],
        overall="fail",
    )

    assert passing.overall == "pass"
    assert failing.overall == "fail"
    assert failing.actionable_fixes == ["fix"]


def test_print_doctor_report_pass(capsys: pytest.CaptureFixture[str]) -> None:
    report = health.DoctorReport(
        timestamp="2026-01-01T00:00:00+00:00",
        checks=[health.CheckResult("python_version", "pass", "ok")],
        overall="pass",
    )
    health.print_doctor_report(report)
    out = capsys.readouterr().out
    assert "✅" in out
    assert "Result: PASS" in out


def test_print_doctor_report_fail(capsys: pytest.CaptureFixture[str]) -> None:
    report = health.DoctorReport(
        timestamp="2026-01-01T00:00:00+00:00",
        checks=[
            health.CheckResult("config_valid", "fail", "bad config", "fix config"),
            health.CheckResult(
                "matplotlib", "warn", "missing", "pip install matplotlib"
            ),
        ],
        overall="fail",
    )
    health.print_doctor_report(report)
    out = capsys.readouterr().out
    assert "❌" in out
    assert "⚠️" in out
    assert "Result: FAIL (1 errors, 1 warnings)" in out


# --- ACP agent checks ---


def test_check_acp_agent_found() -> None:
    with patch("shutil.which", return_value="/usr/local/bin/claude"):
        result = health.check_acp_agent("claude")
    assert result.status == "pass"
    assert "/usr/local/bin/claude" in result.detail


def test_check_acp_agent_missing() -> None:
    with patch("shutil.which", return_value=None):
        result = health.check_acp_agent("claude")
    assert result.status == "fail"
    assert "'claude' not found" in result.detail
    assert "Install claude" in result.fix


def _write_acp_config(path: Path) -> None:
    _ = path.write_text(
        """\
project:
  name: demo
research:
  topic: ACP test
runtime:
  timezone: UTC
notifications:
  channel: test
knowledge_base:
  root: kb
llm:
  provider: acp
  acp:
    agent: claude
""",
        encoding="utf-8",
    )


def test_run_doctor_acp_skips_http_checks(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_acp_config(config_path)
    with (
        patch.object(
            health, "check_python_version",
            return_value=health.CheckResult("python_version", "pass", "ok"),
        ),
        patch.object(
            health, "check_yaml_import",
            return_value=health.CheckResult("yaml_import", "pass", "ok"),
        ),
        patch.object(
            health, "check_config_valid",
            return_value=health.CheckResult("config_valid", "pass", "ok"),
        ),
        patch.object(
            health, "check_acp_agent",
            return_value=health.CheckResult("acp_agent", "pass", "ok"),
        ),
        patch.object(
            health, "check_sandbox_python",
            return_value=health.CheckResult("sandbox_python", "pass", "ok"),
        ),
        patch.object(
            health, "check_matplotlib",
            return_value=health.CheckResult("matplotlib", "pass", "ok"),
        ),
        patch.object(
            health, "check_experiment_mode",
            return_value=health.CheckResult("experiment_mode", "pass", "ok"),
        ),
    ):
        report = health.run_doctor(config_path)

    check_names = [c.name for c in report.checks]
    assert "llm_connectivity" not in check_names
    assert "api_key_valid" not in check_names
    assert "model_chain" not in check_names


def test_run_doctor_acp_includes_agent_check(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_acp_config(config_path)
    with (
        patch.object(
            health, "check_python_version",
            return_value=health.CheckResult("python_version", "pass", "ok"),
        ),
        patch.object(
            health, "check_yaml_import",
            return_value=health.CheckResult("yaml_import", "pass", "ok"),
        ),
        patch.object(
            health, "check_config_valid",
            return_value=health.CheckResult("config_valid", "pass", "ok"),
        ),
        patch.object(
            health, "check_acp_agent",
            return_value=health.CheckResult("acp_agent", "pass", "ok"),
        ),
        patch.object(
            health, "check_sandbox_python",
            return_value=health.CheckResult("sandbox_python", "pass", "ok"),
        ),
        patch.object(
            health, "check_matplotlib",
            return_value=health.CheckResult("matplotlib", "pass", "ok"),
        ),
        patch.object(
            health, "check_experiment_mode",
            return_value=health.CheckResult("experiment_mode", "pass", "ok"),
        ),
    ):
        report = health.run_doctor(config_path)

    check_names = [c.name for c in report.checks]
    assert "acp_agent" in check_names
    assert report.overall == "pass"
    assert len(report.checks) == 7


def test_print_doctor_report_ascii_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    report = health.DoctorReport(
        timestamp="2026-01-01T00:00:00+00:00",
        checks=[health.CheckResult("python_version", "pass", "ok")],
        overall="pass",
    )

    class _AsciiStdout:
        encoding = "ascii"

        def __init__(self) -> None:
            self.parts: list[str] = []

        def write(self, text: str) -> int:
            text.encode(self.encoding)
            self.parts.append(text)
            return len(text)

        def flush(self) -> None:
            return None

    fake_stdout = _AsciiStdout()
    monkeypatch.setattr(health.sys, "stdout", fake_stdout)

    health.print_doctor_report(report)

    out = "".join(fake_stdout.parts)
    assert "[OK] python_version: ok" in out
    assert "Result: PASS" in out
