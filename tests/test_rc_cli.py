# pyright: reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnusedCallResult=false, reportAttributeAccessIssue=false, reportUnknownLambdaType=false
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from researchclaw import cli as rc_cli
from researchclaw.config import resolve_config_path


def _write_valid_config(path: Path) -> None:
    path.write_text(
        """
project:
  name: demo
  mode: docs-first
research:
  topic: Synthetic benchmark research
runtime:
  timezone: UTC
notifications:
  channel: test
knowledge_base:
  backend: markdown
  root: kb
openclaw_bridge: {}
llm:
  provider: openai-compatible
  base_url: http://localhost:1234/v1
  api_key_env: TEST_KEY
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_main_with_no_args_returns_zero_and_prints_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = rc_cli.main([])
    assert code == 0
    captured = capsys.readouterr()
    assert "ResearchClaw" in captured.out
    assert "usage:" in captured.out


@pytest.mark.parametrize("argv", [["run", "--help"], ["validate", "--help"]])
def test_help_subcommands_exit_zero(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        rc_cli.main(argv)
    assert exc_info.value.code == 0


def test_generate_run_id_format() -> None:
    run_id = rc_cli._generate_run_id("my topic")
    assert run_id.startswith("rc-")
    assert re.fullmatch(r"rc-\d{8}-\d{6}-[0-9a-f]{6}", run_id)


def test_cmd_run_missing_config_returns_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = argparse.Namespace(
        config=str(tmp_path / "missing.yaml"),
        topic=None,
        output=None,
        from_stage=None,
        auto_approve=False,
        skip_preflight=True,
        resume=False,
        skip_noncritical_stage=False,
    )
    code = rc_cli.cmd_run(args)
    assert code == 1
    assert "config file not found" in capsys.readouterr().err


def test_cmd_validate_missing_config_returns_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = argparse.Namespace(
        config=str(tmp_path / "missing.yaml"), no_check_paths=False
    )
    code = rc_cli.cmd_validate(args)
    assert code == 1
    assert "config file not found" in capsys.readouterr().err


def test_cmd_validate_valid_config_returns_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_valid_config(config_path)
    args = argparse.Namespace(config=str(config_path), no_check_paths=True)
    code = rc_cli.cmd_validate(args)
    assert code == 0
    assert "Config validation passed" in capsys.readouterr().out


def test_main_dispatches_run_command(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_cmd_run(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(rc_cli, "cmd_run", fake_cmd_run)
    code = rc_cli.main(
        [
            "run",
            "--topic",
            "new topic",
            "--config",
            "cfg.yaml",
            "--output",
            "out-dir",
            "--from-stage",
            "PAPER_OUTLINE",
            "--auto-approve",
        ]
    )
    assert code == 0
    parsed = captured["args"]
    assert parsed.topic == "new topic"
    assert parsed.config == "cfg.yaml"
    assert parsed.output == "out-dir"
    assert parsed.from_stage == "PAPER_OUTLINE"
    assert parsed.auto_approve is True


def test_main_dispatches_validate_command(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_cmd_validate(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(rc_cli, "cmd_validate", fake_cmd_validate)
    code = rc_cli.main(["validate", "--config", "cfg.yaml", "--no-check-paths"])
    assert code == 0
    parsed = captured["args"]
    assert parsed.config == "cfg.yaml"
    assert parsed.no_check_paths is True


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--topic", "x", "--config", "c.yaml"],
        ["run", "--output", "out", "--config", "c.yaml"],
        ["run", "--from-stage", "TOPIC_INIT", "--config", "c.yaml"],
        ["run", "--auto-approve", "--config", "c.yaml"],
    ],
)
def test_run_parser_accepts_required_flags(
    argv: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rc_cli, "cmd_run", lambda args: 0)
    assert rc_cli.main(argv) == 0


def test_validate_parser_accepts_config_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rc_cli, "cmd_validate", lambda args: 0)
    assert rc_cli.main(["validate", "--config", "cfg.yaml"]) == 0


# --- resolve_config_path tests ---


def test_resolve_config_finds_arc_yaml_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.arc.yaml").write_text("x: 1\n")
    (tmp_path / "config.yaml").write_text("x: 2\n")
    result = resolve_config_path(None)
    assert result is not None
    assert result.name == "config.arc.yaml"


def test_resolve_config_falls_back_to_config_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("x: 1\n")
    result = resolve_config_path(None)
    assert result is not None
    assert result.name == "config.yaml"


def test_resolve_config_returns_none_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = resolve_config_path(None)
    assert result is None


def test_resolve_config_explicit_path_no_search() -> None:
    result = resolve_config_path("/some/explicit/path.yaml")
    assert result is not None
    assert str(result) == "/some/explicit/path.yaml"


# --- cmd_init tests ---


def _write_example_config(path: Path) -> None:
    path.write_text(
        """\
project:
  name: "my-research"
llm:
  provider: "openai"
  base_url: "https://api.openai.com/v1"
  api_key_env: "OPENAI_API_KEY"
  primary_model: "gpt-4o"
  fallback_models:
    - "gpt-4.1"
    - "gpt-4o-mini"
""",
        encoding="utf-8",
    )


def test_cmd_init_creates_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_example_config(tmp_path / "config.researchclaw.example.yaml")
    # Simulate non-TTY (stdin not a tty) → defaults to openai
    monkeypatch.setattr("sys.stdin", type("FakeStdin", (), {"isatty": lambda self: False})())
    args = argparse.Namespace(force=False)
    code = rc_cli.cmd_init(args)
    assert code == 0
    created = tmp_path / "config.arc.yaml"
    assert created.exists()
    content = created.read_text()
    assert 'provider: "openai"' in content
    assert "Created config.arc.yaml" in capsys.readouterr().out


def test_cmd_init_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_example_config(tmp_path / "config.researchclaw.example.yaml")
    (tmp_path / "config.arc.yaml").write_text("existing\n")
    args = argparse.Namespace(force=False)
    code = rc_cli.cmd_init(args)
    assert code == 0
    assert "already exists" in capsys.readouterr().out
    assert (tmp_path / "config.arc.yaml").read_text() == "existing\n"


def test_cmd_init_force_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_example_config(tmp_path / "config.researchclaw.example.yaml")
    (tmp_path / "config.arc.yaml").write_text("old\n")
    monkeypatch.setattr("sys.stdin", type("FakeStdin", (), {"isatty": lambda self: False})())
    args = argparse.Namespace(force=True)
    code = rc_cli.cmd_init(args)
    assert code == 0
    assert (tmp_path / "config.arc.yaml").read_text() != "old\n"


def test_cmd_run_missing_config_shows_init_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(
        config=None,
        topic=None,
        output=None,
        from_stage=None,
        auto_approve=False,
        skip_preflight=True,
        resume=False,
        skip_noncritical_stage=False,
    )
    code = rc_cli.cmd_run(args)
    assert code == 1
    assert "researchclaw init" in capsys.readouterr().err


def test_main_dispatches_init(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_cmd_init(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(rc_cli, "cmd_init", fake_cmd_init)
    code = rc_cli.main(["init", "--force"])
    assert code == 0
    assert captured["args"].force is True
