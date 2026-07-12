import json
from pathlib import Path
from typing import cast

import pytest
import yaml

from researchclaw.config import (
    ExperimentConfig,
    PaperRevisionConfig,
    RCConfig,
    SandboxConfig,
    SecurityConfig,
    ValidationResult,
    _parse_paper_revision_config,
    load_config,
    validate_config,
)


def _write_valid_config(tmp_path: Path) -> Path:
    kb_root = tmp_path / "docs" / "kb"
    for name in (
        "questions",
        "literature",
        "experiments",
        "findings",
        "decisions",
        "reviews",
    ):
        (kb_root / name).mkdir(parents=True, exist_ok=True)

    config_path = tmp_path / "config.rc.yaml"
    _ = config_path.write_text(
        """
project:
  name: demo
  mode: docs-first
research:
  topic: Test topic
  domains: [ml, agents]
runtime:
  timezone: America/New_York
notifications:
  channel: discord
knowledge_base:
  backend: markdown
  root: docs/kb
openclaw_bridge:
  use_cron: true
  use_message: true
  use_memory: true
  use_sessions_spawn: true
  use_web_fetch: true
  use_browser: false
llm:
  provider: openai-compatible
  base_url: https://example.invalid/v1
  api_key_env: OPENAI_API_KEY
security:
  hitl_required_stages: [5, 9, 20]
experiment:
  mode: simulated
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return config_path


def _valid_config_data() -> dict[str, dict[str, object]]:
    return {
        "project": {"name": "demo", "mode": "docs-first"},
        "research": {"topic": "Test topic", "domains": ["ml", "agents"]},
        "runtime": {"timezone": "America/New_York"},
        "notifications": {"channel": "discord"},
        "knowledge_base": {"backend": "markdown", "root": "docs/kb"},
        "openclaw_bridge": {
            "use_cron": True,
            "use_message": True,
            "use_memory": True,
            "use_sessions_spawn": True,
            "use_web_fetch": True,
            "use_browser": False,
        },
        "llm": {
            "provider": "openai-compatible",
            "base_url": "https://example.invalid/v1",
            "api_key_env": "OPENAI_API_KEY",
            "primary_model": "gpt-4.1",
            "fallback_models": ["gpt-4o-mini", "gpt-4o"],
        },
        "security": {"hitl_required_stages": [5, 9, 20]},
        "experiment": {
            "mode": "simulated",
            "metric_direction": "minimize",
        },
    }


def test_valid_config_data_helper_returns_expected_baseline_shape():
    data = _valid_config_data()
    assert data["project"]["name"] == "demo"
    assert data["knowledge_base"]["root"] == "docs/kb"
    assert data["security"]["hitl_required_stages"] == [5, 9, 20]


def test_validate_config_with_valid_data_returns_ok_true(tmp_path: Path):
    result = validate_config(
        _valid_config_data(), project_root=tmp_path, check_paths=False
    )

    assert isinstance(result, ValidationResult)
    assert result.ok is True
    assert result.errors == ()


def test_code_agent_long_call_max_tokens_is_parsed(tmp_path: Path):
    data = _valid_config_data()
    data["experiment"]["code_agent"] = {"long_call_max_tokens": 6144}

    config = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

    assert config.experiment.code_agent.long_call_max_tokens == 6144


def test_opencode_fallback_to_code_agent_is_parsed(tmp_path: Path):
    data = _valid_config_data()
    data["experiment"]["opencode"] = {"fallback_to_code_agent": False}

    config = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

    assert config.experiment.opencode.fallback_to_code_agent is False


def test_paper_revision_defaults_disabled_and_parses_bounded_values(
    tmp_path: Path,
) -> None:
    config = RCConfig.from_dict(
        _valid_config_data(), project_root=tmp_path, check_paths=False
    )
    assert config.paper_revision == PaperRevisionConfig()

    data = _valid_config_data()
    data["paper_revision"] = {
        "sectional_enabled": True,
        "max_section_retries": 2,
        "min_length_ratio": 0.75,
        "max_length_ratio": 1.5,
        "critic_model": "critic-model",
    }
    parsed = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)
    assert parsed.paper_revision.sectional_enabled is True
    assert parsed.paper_revision.max_section_retries == 2
    assert parsed.paper_revision.min_length_ratio == 0.75
    assert parsed.paper_revision.max_length_ratio == 1.5


@pytest.mark.parametrize(
    ("paper_revision", "expected"),
    (
        ({"max_section_retries": -1}, "max_section_retries"),
        ({"max_section_retries": 4}, "max_section_retries"),
        ({"min_length_ratio": 0.49}, "min_length_ratio"),
        ({"max_length_ratio": 3.01}, "max_length_ratio"),
        ({"sectional_enabled": True}, "critic_model is required"),
        ({"sectional_enabled": "false"}, "sectional_enabled must be a boolean"),
        ({"critic_model": ["critic"]}, "critic_model must be a string"),
        ({"unexpected": True}, "Unknown paper_revision fields"),
    ),
)
def test_paper_revision_config_rejects_unsafe_or_unknown_values(
    tmp_path: Path,
    paper_revision: dict[str, object],
    expected: str,
) -> None:
    data = _valid_config_data()
    data["paper_revision"] = paper_revision

    result = validate_config(data, project_root=tmp_path, check_paths=False)

    assert result.ok is False
    assert any(expected in error for error in result.errors)


def test_only_dedicated_dry_run_config_enables_sectional_revision() -> None:
    root = Path(__file__).resolve().parents[1]
    enabled: list[str] = []
    for path in root.glob("config*.yaml"):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        revision = payload.get("paper_revision") or {}
        if revision.get("sectional_enabled", False) is True:
            enabled.append(path.name)

    assert enabled == ["config.deepseek.sectional-dry-run.yaml"]


def test_sectional_dry_run_config_is_explicitly_non_release() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "config.deepseek.sectional-dry-run.yaml"
    config = RCConfig.load(path, project_root=root, check_paths=False)

    assert config.project.name == "hwsec-sectional-dry-run"
    assert config.experiment.claim_scope == "pipeline_validation"
    assert config.experiment.dataset_origin == "synthetic"
    assert config.experiment.allow_legacy_experiment_path is False
    assert config.paper_revision.sectional_enabled is True
    assert config.paper_revision.critic_model == config.llm.critic_model
    assert config.paper_revision.critic_model != config.llm.primary_model
    assert config.experiment.opencode.fallback_to_code_agent is False


def test_private_paper_revision_parser_does_not_coerce_non_boolean_flag() -> None:
    with pytest.raises(ValueError, match="sectional_enabled must be a boolean"):
        _parse_paper_revision_config({"sectional_enabled": "yes"})


def test_validate_config_missing_required_fields_returns_errors(tmp_path: Path):
    data = _valid_config_data()
    data["research"] = {}

    result = validate_config(data, project_root=tmp_path, check_paths=False)

    assert result.ok is False
    assert "Missing required field: research.topic" in result.errors


def test_validate_config_rejects_invalid_project_mode(tmp_path: Path):
    data = _valid_config_data()
    data["project"]["mode"] = "invalid-mode"

    result = validate_config(data, project_root=tmp_path, check_paths=False)

    assert result.ok is False
    assert "Invalid project.mode: invalid-mode" in result.errors


def test_validate_config_rejects_invalid_knowledge_base_backend(tmp_path: Path):
    data = _valid_config_data()
    data["knowledge_base"]["backend"] = "sqlite"

    result = validate_config(data, project_root=tmp_path, check_paths=False)

    assert result.ok is False
    assert "Invalid knowledge_base.backend: sqlite" in result.errors


def test_validate_config_accepts_llm_wire_api_responses(tmp_path: Path):
    data = _valid_config_data()
    data["llm"]["wire_api"] = "responses"

    result = validate_config(data, project_root=tmp_path, check_paths=False)

    assert result.ok is True


@pytest.mark.parametrize("provider", ["claude-cli", "codex-cli"])
def test_validate_config_accepts_local_cli_provider_without_url_or_key_env(
    tmp_path: Path, provider: str
):
    data = _valid_config_data()
    data["llm"]["provider"] = provider
    data["llm"]["base_url"] = ""
    data["llm"]["api_key_env"] = ""

    result = validate_config(data, project_root=tmp_path, check_paths=False)

    assert result.ok is True


def test_validate_config_rejects_invalid_llm_wire_api(tmp_path: Path):
    data = _valid_config_data()
    data["llm"]["wire_api"] = "responses_only"

    result = validate_config(data, project_root=tmp_path, check_paths=False)

    assert result.ok is False
    assert "Invalid llm.wire_api: responses_only" in result.errors


def test_validate_config_rejects_invalid_llm_provider(tmp_path: Path):
    data = _valid_config_data()
    data["llm"]["provider"] = "claude-clii"

    result = validate_config(data, project_root=tmp_path, check_paths=False)

    assert result.ok is False
    assert "Invalid llm.provider: claude-clii" in result.errors


@pytest.mark.parametrize("entry", [0, 24, "5", 9.1])
def test_validate_config_rejects_invalid_hitl_required_stages_entries(
    tmp_path: Path, entry: object
):
    data = _valid_config_data()
    data["security"]["hitl_required_stages"] = [5, entry, 20]

    result = validate_config(data, project_root=tmp_path, check_paths=False)

    assert result.ok is False
    assert f"Invalid security.hitl_required_stages entry: {entry}" in result.errors


def test_validate_config_rejects_non_list_hitl_required_stages(tmp_path: Path):
    data = _valid_config_data()
    data["security"]["hitl_required_stages"] = "5,9,20"

    result = validate_config(data, project_root=tmp_path, check_paths=False)

    assert result.ok is False
    assert "security.hitl_required_stages must be a list" in result.errors


def test_validate_config_rejects_invalid_experiment_mode(tmp_path: Path):
    data = _valid_config_data()
    data["experiment"]["mode"] = "kubernetes"

    result = validate_config(data, project_root=tmp_path, check_paths=False)

    assert result.ok is False
    assert "Invalid experiment.mode: kubernetes" in result.errors


def test_validate_config_accepts_docker_mode(tmp_path: Path):
    data = _valid_config_data()
    data["experiment"]["mode"] = "docker"

    result = validate_config(data, project_root=tmp_path, check_paths=False)

    assert result.ok is True


def test_validate_config_rejects_invalid_metric_direction(tmp_path: Path):
    data = _valid_config_data()
    data["experiment"]["metric_direction"] = "upward"

    result = validate_config(data, project_root=tmp_path, check_paths=False)

    assert result.ok is False
    assert "Invalid experiment.metric_direction: upward" in result.errors


def test_rcconfig_from_dict_happy_path(tmp_path: Path):
    config = RCConfig.from_dict(
        _valid_config_data(),
        project_root=tmp_path,
        check_paths=False,
    )

    assert isinstance(config, RCConfig)
    assert config.project.name == "demo"
    assert config.research.domains == ("ml", "agents")
    assert config.llm.fallback_models == ("gpt-4o-mini", "gpt-4o")


def test_rcconfig_from_dict_parses_llm_wire_api(tmp_path: Path):
    data = _valid_config_data()
    data["llm"]["wire_api"] = "responses"

    config = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

    assert config.llm.wire_api == "responses"


def test_rcconfig_from_dict_parses_partial_run_and_q1_gate_controls(
    tmp_path: Path,
):
    data = _valid_config_data()
    data["runtime"]["skip_stages"] = [9, 13]
    data["runtime"]["inject_artifacts"] = {"stage-14/analysis.md": "existing"}
    data["security"]["q1_spine_hard_gate"] = True
    data["security"]["q1_spine_max_rollbacks"] = 2

    config = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

    assert config.runtime.skip_stages == (9, 13)
    assert config.runtime.inject_artifacts == {"stage-14/analysis.md": "existing"}
    assert config.security.q1_spine_hard_gate is True
    assert config.security.q1_spine_max_rollbacks == 2


@pytest.mark.parametrize("stage", [4, 5])
def test_validate_config_rejects_skipping_evidence_authority_stages(
    tmp_path: Path, stage: int
) -> None:
    data = _valid_config_data()
    data["runtime"]["skip_stages"] = [stage]

    result = validate_config(data, project_root=tmp_path, check_paths=False)

    assert result.ok is False
    assert (
        f"runtime.skip_stages cannot skip evidence-authority stage: {stage}"
        in result.errors
    )


def test_rcconfig_from_dict_missing_fields_raises_value_error(tmp_path: Path):
    data = _valid_config_data()
    del data["runtime"]

    with pytest.raises(ValueError, match="Missing required field: runtime.timezone"):
        _ = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)


def test_rcconfig_load_from_yaml_file(tmp_path: Path):
    config_path = _write_valid_config(tmp_path)
    config = RCConfig.load(config_path, project_root=tmp_path)

    assert isinstance(config, RCConfig)
    assert config.project.name == "demo"
    assert config.knowledge_base.root == "docs/kb"


def test_load_config_wrapper_returns_rcconfig(tmp_path: Path):
    config_path = _write_valid_config(tmp_path)
    config = load_config(config_path, project_root=tmp_path)

    assert isinstance(config, RCConfig)
    assert config.security.hitl_required_stages == (5, 9, 20)


def test_security_config_defaults_match_expected_values():
    defaults = SecurityConfig()

    assert defaults.hitl_required_stages == (5, 9, 20)
    assert defaults.allow_publish_without_approval is False
    assert defaults.redact_sensitive_logs is True


def test_experiment_config_defaults_mode_is_simulated():
    defaults = ExperimentConfig()

    assert defaults.mode == "simulated"
    assert defaults.metric_direction == "minimize"


def test_sandbox_config_defaults_match_expected_values():
    from researchclaw.config import DEFAULT_PYTHON_PATH

    defaults = SandboxConfig()

    assert defaults.python_path == DEFAULT_PYTHON_PATH
    assert defaults.gpu_required is False
    assert defaults.max_memory_mb == 4096
    assert "numpy" in defaults.allowed_imports


def test_to_dict_roundtrip_rehydrates_equivalent_rcconfig(tmp_path: Path):
    original = RCConfig.from_dict(
        _valid_config_data(),
        project_root=tmp_path,
        check_paths=False,
    )

    normalized = cast(dict[str, object], json.loads(json.dumps(original.to_dict())))

    rehydrated = RCConfig.from_dict(
        normalized,
        project_root=tmp_path,
        check_paths=False,
    )

    assert rehydrated == original
    assert isinstance(original.to_dict()["security"]["hitl_required_stages"], tuple)


def test_check_paths_false_skips_missing_kb_root_validation(tmp_path: Path):
    data = _valid_config_data()
    data["knowledge_base"]["root"] = "docs/missing-kb"

    result = validate_config(data, project_root=tmp_path, check_paths=False)

    assert result.ok is True
    assert not any(error.startswith("Missing path:") for error in result.errors)


def test_path_validation_missing_kb_root_is_error(tmp_path: Path):
    result = validate_config(
        _valid_config_data(), project_root=tmp_path, check_paths=True
    )

    assert result.ok is False
    assert any(error.startswith("Missing path:") for error in result.errors)


def test_validate_config_missing_kb_subdirs_emits_warnings(tmp_path: Path):
    data = _valid_config_data()
    _ = (tmp_path / "docs" / "kb").mkdir(parents=True)

    result = validate_config(data, project_root=tmp_path, check_paths=True)

    assert result.ok is True
    assert len(result.warnings) == 6
    assert all(
        warning.startswith("Missing recommended kb subdir:")
        for warning in result.warnings
    )


def test_rcconfig_from_dict_uses_default_security_when_missing(tmp_path: Path):
    data = _valid_config_data()
    del data["security"]

    config = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)
    assert config.security.hitl_required_stages == (5, 9, 20)


def test_load_uses_file_parent_as_default_project_root(tmp_path: Path):
    config_path = _write_valid_config(tmp_path)
    config = RCConfig.load(config_path)

    assert config.project.name == "demo"
    assert config.knowledge_base.root == "docs/kb"
