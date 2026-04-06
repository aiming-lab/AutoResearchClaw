# pyright: reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnusedCallResult=false, reportAttributeAccessIssue=false, reportUnknownLambdaType=false
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pytest

from researchclaw import cli as rc_cli
from researchclaw.config import resolve_config_path
from researchclaw.pipeline.executor import StageResult
from researchclaw.pipeline.stages import Stage, StageStatus


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


def test_cmd_run_reports_paused_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_valid_config(config_path)
    output_dir = tmp_path / "artifacts" / "paused-run"

    from researchclaw.pipeline import runner as rc_runner

    monkeypatch.setattr(
        rc_runner,
        "execute_pipeline",
        lambda **kwargs: [
            StageResult(
                stage=Stage.TOPIC_INIT,
                status=StageStatus.DONE,
                artifacts=("goal.md",),
            ),
            StageResult(
                stage=Stage.PROBLEM_DECOMPOSE,
                status=StageStatus.PAUSED,
                artifacts=("refinement_log.json",),
                error="ACP prompt timed out after 1800s",
                decision="resume",
            ),
        ],
    )
    monkeypatch.setattr(rc_runner, "read_checkpoint", lambda run_dir: None)

    args = argparse.Namespace(
        config=str(config_path),
        topic=None,
        output=str(output_dir),
        from_stage=None,
        auto_approve=False,
        skip_preflight=True,
        resume=False,
        skip_noncritical_stage=False,
        no_graceful_degradation=False,
    )
    code = rc_cli.cmd_run(args)
    captured = capsys.readouterr()
    assert code == 0
    assert "Pipeline paused:" in captured.out
    assert "1 paused" in captured.out


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


def test_main_dispatches_chat_command(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_cmd_chat(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(rc_cli, "cmd_chat", fake_cmd_chat)
    code = rc_cli.main(
        [
            "chat",
            "--config",
            "cfg.yaml",
            "--check",
            "--message",
            "hello",
            "--system",
            "you are concise",
            "--max-turns",
            "3",
            "--json",
        ]
    )
    assert code == 0
    parsed = captured["args"]
    assert parsed.config == "cfg.yaml"
    assert parsed.check is True
    assert parsed.message == "hello"
    assert parsed.system == "you are concise"
    assert parsed.max_turns == 3
    assert parsed.json is True


def test_cmd_chat_check_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_valid_config(config_path)

    class _FakeClient:
        def preflight(self) -> tuple[bool, str]:
            return True, "OK - test"

    import researchclaw.llm as llm_mod

    monkeypatch.setattr(llm_mod, "create_llm_client", lambda cfg: _FakeClient())

    args = argparse.Namespace(
        config=str(config_path),
        check=True,
        message=None,
        system=None,
        max_turns=0,
    )
    code = rc_cli.cmd_chat(args)
    assert code == 0
    assert "API check passed" in capsys.readouterr().out


def test_cmd_chat_check_json_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_valid_config(config_path)

    class _FakeClient:
        def preflight(self) -> tuple[bool, str]:
            return True, "OK - test"

    import researchclaw.llm as llm_mod

    monkeypatch.setattr(llm_mod, "create_llm_client", lambda cfg: _FakeClient())

    args = argparse.Namespace(
        config=str(config_path),
        check=True,
        message=None,
        system=None,
        max_turns=0,
        json=True,
    )
    code = rc_cli.cmd_chat(args)
    assert code == 0
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["ok"] is True
    assert payload["mode"] == "check"
    assert "message" in payload


def test_cmd_chat_one_shot_prints_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_valid_config(config_path)

    class _FakeResp:
        content = "pong"

    class _FakeClient:
        def preflight(self) -> tuple[bool, str]:
            return True, "OK"

        def chat(self, messages, **kwargs):  # noqa: ANN001
            return _FakeResp()

    import researchclaw.llm as llm_mod

    monkeypatch.setattr(llm_mod, "create_llm_client", lambda cfg: _FakeClient())

    args = argparse.Namespace(
        config=str(config_path),
        check=False,
        message="ping",
        system=None,
        max_turns=0,
        json=False,
    )
    code = rc_cli.cmd_chat(args)
    assert code == 0
    assert "pong" in capsys.readouterr().out


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
    assert result == Path("/some/explicit/path.yaml")


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
    content = created.read_text(encoding="utf-8")
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
    assert code == 1
    assert "already exists" in capsys.readouterr().err
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
    assert (tmp_path / "config.arc.yaml").read_text(encoding="utf-8") != "old\n"


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


def test_resume_finds_existing_checkpoint_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """BUG-119: --resume without --output should find the latest checkpoint dir."""
    import hashlib
    import json

    monkeypatch.chdir(tmp_path)

    # Write a valid config
    config_path = tmp_path / "config.arc.yaml"
    _write_valid_config(config_path)

    # Create a fake previous run directory with a checkpoint
    topic = "Synthetic benchmark research"  # matches _write_valid_config
    topic_hash = hashlib.sha256(topic.encode()).hexdigest()[:6]
    old_run_dir = tmp_path / "artifacts" / f"rc-20260319-100000-{topic_hash}"
    old_run_dir.mkdir(parents=True)
    (old_run_dir / "checkpoint.json").write_text(
        json.dumps({"last_completed_stage": 5, "last_completed_name": "HYPOTHESIS_GEN",
                     "run_id": old_run_dir.name, "timestamp": "2026-03-19T10:00:00Z"})
    )

    # Mock execute_pipeline so we don't actually run
    import researchclaw.pipeline.runner as runner_mod
    monkeypatch.setattr(runner_mod, "execute_pipeline", lambda **kw: [])

    # Also mock preflight
    from unittest.mock import MagicMock
    mock_client = MagicMock()
    mock_client.preflight.return_value = (True, "OK")
    import researchclaw.llm as llm_mod
    monkeypatch.setattr(llm_mod, "create_llm_client", lambda cfg: mock_client)

    args = argparse.Namespace(
        config=str(config_path),
        topic=None,
        output=None,
        from_stage=None,
        auto_approve=False,
        skip_preflight=True,
        resume=True,
        skip_noncritical_stage=False,
        no_graceful_degradation=False,
    )
    rc_cli.cmd_run(args)
    captured = capsys.readouterr()
    assert "Found existing run to resume" in captured.out
    assert old_run_dir.name in captured.out


def test_resume_no_checkpoint_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """BUG-119: --resume with no matching checkpoint should warn and start new."""
    monkeypatch.chdir(tmp_path)

    config_path = tmp_path / "config.arc.yaml"
    _write_valid_config(config_path)

    # Create empty artifacts dir (no checkpoints)
    (tmp_path / "artifacts").mkdir()

    import researchclaw.pipeline.runner as runner_mod
    monkeypatch.setattr(runner_mod, "execute_pipeline", lambda **kw: [])

    from unittest.mock import MagicMock
    mock_client = MagicMock()
    mock_client.preflight.return_value = (True, "OK")
    import researchclaw.llm as llm_mod
    monkeypatch.setattr(llm_mod, "create_llm_client", lambda cfg: mock_client)

    args = argparse.Namespace(
        config=str(config_path),
        topic=None,
        output=None,
        from_stage=None,
        auto_approve=False,
        skip_preflight=True,
        resume=True,
        skip_noncritical_stage=False,
        no_graceful_degradation=False,
    )
    rc_cli.cmd_run(args)
    captured = capsys.readouterr()
    assert "no checkpoint found" in captured.err


def test_main_dispatches_init(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_cmd_init(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(rc_cli, "cmd_init", fake_cmd_init)
    code = rc_cli.main(["init", "--force"])
    assert code == 0
    assert captured["args"].force is True
