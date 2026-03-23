# pyright: reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnusedCallResult=false, reportAttributeAccessIssue=false, reportUnknownLambdaType=false
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.pipeline import executor as rc_executor
from researchclaw.pipeline.stages import Stage, StageStatus


class FakeLLMClient:
    def __init__(self, response_text: str = "mock response"):
        self.response_text: str = response_text
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], **kwargs: object):
        _ = kwargs
        self.calls.append(messages)
        from researchclaw.llm.client import LLMResponse

        return LLMResponse(content=self.response_text, model="fake-model")


class FakeLLMClientWithConfig(FakeLLMClient):
    def __init__(self, response_text: str = "mock response"):
        super().__init__(response_text=response_text)
        self.config: SimpleNamespace = SimpleNamespace(
            base_url="http://fake", api_key="fake-key"
        )


@pytest.fixture()
def rc_config(tmp_path: Path) -> RCConfig:
    data = {
        "project": {"name": "rc-test", "mode": "docs-first"},
        "research": {
            "topic": "test-driven science",
            "domains": ["ml", "systems"],
            "daily_paper_count": 2,
            "quality_threshold": 8.2,
        },
        "runtime": {"timezone": "UTC"},
        "notifications": {
            "channel": "local",
            "on_stage_start": True,
            "on_stage_fail": False,
            "on_gate_required": True,
        },
        "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
        "openclaw_bridge": {"use_memory": True, "use_message": True},
        "llm": {
            "provider": "openai-compatible",
            "base_url": "http://localhost:1234/v1",
            "api_key_env": "RC_TEST_KEY",
            "api_key": "inline-test-key",
            "primary_model": "fake-model",
            "fallback_models": [],
        },
        "security": {"hitl_required_stages": [5, 9, 20]},
        "experiment": {"mode": "sandbox"},
    }
    return RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)


@pytest.fixture()
def adapters() -> AdapterBundle:
    return AdapterBundle()


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    path = tmp_path / "run"
    path.mkdir()
    return path


def _write_prior_artifact(
    run_dir: Path, stage_num: int, filename: str, content: str
) -> None:
    stage_dir = run_dir / f"stage-{stage_num:02d}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / filename).write_text(content, encoding="utf-8")


def test_executor_map_has_23_entries() -> None:
    executor_map = getattr(rc_executor, "EXECUTOR_MAP", rc_executor._STAGE_EXECUTORS)
    assert len(executor_map) == 23


def test_every_stage_member_has_matching_executor() -> None:
    executor_map = getattr(rc_executor, "EXECUTOR_MAP", rc_executor._STAGE_EXECUTORS)
    assert set(executor_map.keys()) == set(Stage)


def test_stage_result_dataclass_fields() -> None:
    result = rc_executor.StageResult(
        stage=Stage.TOPIC_INIT, status=StageStatus.DONE, artifacts=("goal.md",)
    )
    assert result.stage == Stage.TOPIC_INIT
    assert result.status == StageStatus.DONE
    assert result.artifacts == ("goal.md",)
    assert result.error is None
    assert result.decision == "proceed"
    assert result.evidence_refs == ()


def test_utcnow_iso_returns_valid_iso_timestamp() -> None:
    ts = rc_executor._utcnow_iso()
    assert ts.endswith("+00:00")
    assert "T" in ts


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("before\n```yaml\na: 1\n```\nafter", "a: 1"),
        ("```yml\nkey: value\n```", "key: value"),
        ("```\nplain: true\n```", "plain: true"),
        ("  x: y  ", "x: y"),
    ],
)
def test_extract_yaml_block_variants(text: str, expected: str) -> None:
    assert rc_executor._extract_yaml_block(text) == expected


@pytest.mark.parametrize(
    ("payload", "default", "expected"),
    [
        ('{"ok": true}', {"fallback": True}, {"ok": True}),
        ("[1, 2, 3]", {"fallback": True}, [1, 2, 3]),
        ("not-json", {"fallback": True}, {"fallback": True}),
    ],
)
def test_safe_json_loads_valid_and_invalid(payload: str, default, expected) -> None:
    assert rc_executor._safe_json_loads(payload, default) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a/b", "a_b"),
        ("a\\b", "a_b"),
        ("../secret", "__secret"),
        ("name with spaces!.md", "name_with_spaces_.md"),
        ("", "unnamed"),
    ],
)
def test_safe_filename_sanitization(raw: str, expected: str) -> None:
    assert rc_executor._safe_filename(raw) == expected


def test_safe_filename_truncates_to_100_chars() -> None:
    raw = "x" * 120
    cleaned = rc_executor._safe_filename(raw)
    assert len(cleaned) == 100
    assert cleaned == "x" * 100


def test_build_context_preamble_basic_fields(
    rc_config: RCConfig, run_dir: Path
) -> None:
    text = rc_executor._build_context_preamble(rc_config, run_dir)
    assert "## Research Context" in text
    assert "test-driven science" in text
    assert "ml, systems" in text


def test_build_context_preamble_includes_selected_prior_artifacts(
    rc_config: RCConfig, run_dir: Path
) -> None:
    _write_prior_artifact(run_dir, 1, "goal.md", "goal content")
    _write_prior_artifact(run_dir, 8, "hypotheses.md", "hyp content")
    _write_prior_artifact(run_dir, 7, "synthesis.md", "synth content")
    text = rc_executor._build_context_preamble(
        rc_config,
        run_dir,
        include_goal=True,
        include_hypotheses=True,
        include_synthesis=True,
    )
    assert "### Goal" in text
    assert "goal content" in text
    assert "### Hypotheses" in text
    assert "hyp content" in text
    assert "### Synthesis" in text
    assert "synth content" in text


def test_read_prior_artifact_finds_newest_file(run_dir: Path) -> None:
    _write_prior_artifact(run_dir, 1, "goal.md", "old")
    _write_prior_artifact(run_dir, 3, "goal.md", "new")
    found = rc_executor._read_prior_artifact(run_dir, "goal.md")
    assert found == "new"


def test_read_prior_artifact_finds_directory_path(run_dir: Path) -> None:
    cards_dir = run_dir / "stage-06" / "cards"
    cards_dir.mkdir(parents=True)
    (cards_dir / "card-1.json").write_text("{}", encoding="utf-8")
    found = rc_executor._read_prior_artifact(run_dir, "cards/")
    assert found == str(cards_dir)


def test_read_prior_artifact_returns_none_when_not_found(run_dir: Path) -> None:
    assert rc_executor._read_prior_artifact(run_dir, "missing.md") is None


def test_read_best_analysis_prefers_best_file(run_dir: Path) -> None:
    """BUG-225: _read_best_analysis prefers analysis_best.md at run root."""
    from researchclaw.pipeline._helpers import _read_best_analysis

    # Create degenerate analysis in stage-14 and best at run root
    s14 = run_dir / "stage-14"
    s14.mkdir(parents=True)
    (s14 / "analysis.md").write_text("Degenerate analysis", encoding="utf-8")
    (run_dir / "analysis_best.md").write_text("Best analysis", encoding="utf-8")

    result = _read_best_analysis(run_dir)
    assert result == "Best analysis"


def test_read_best_analysis_falls_back_to_prior_artifact(run_dir: Path) -> None:
    """BUG-225: Falls back to _read_prior_artifact when no analysis_best.md."""
    from researchclaw.pipeline._helpers import _read_best_analysis

    s14 = run_dir / "stage-14"
    s14.mkdir(parents=True)
    (s14 / "analysis.md").write_text("Only analysis", encoding="utf-8")

    result = _read_best_analysis(run_dir)
    assert result == "Only analysis"


def test_read_best_analysis_returns_empty_when_none(run_dir: Path) -> None:
    """BUG-225: Returns empty string when no analysis exists at all."""
    from researchclaw.pipeline._helpers import _read_best_analysis

    result = _read_best_analysis(run_dir)
    assert result == ""


def test_write_stage_meta_writes_expected_json(run_dir: Path) -> None:
    stage_dir = run_dir / "stage-01"
    stage_dir.mkdir()
    result = rc_executor.StageResult(
        stage=Stage.TOPIC_INIT,
        status=StageStatus.DONE,
        artifacts=("goal.md",),
        decision="proceed",
        evidence_refs=("stage-01/goal.md",),
    )
    rc_executor._write_stage_meta(stage_dir, Stage.TOPIC_INIT, "run-abc", result)
    payload = cast(
        dict[str, Any],
        json.loads((stage_dir / "decision.json").read_text(encoding="utf-8")),
    )
    assert payload["stage_id"] == "01-topic_init"
    assert payload["run_id"] == "run-abc"
    assert payload["status"] == "done"
    assert payload["decision"] == "proceed"
    assert payload["output_artifacts"] == ["goal.md"]
    assert payload["evidence_refs"] == ["stage-01/goal.md"]
    assert payload["next_stage"] == 2
    assert re.match(r"\d{4}-\d{2}-\d{2}T", payload["ts"])


def test_execute_stage_creates_stage_dir_writes_artifacts_and_meta(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    fake_llm = FakeLLMClientWithConfig("# Goal\n\nMocked goal body")
    monkeypatch.setattr(
        "researchclaw.pipeline.executor.LLMClient.from_rc_config",
        lambda _config: fake_llm,
    )

    result = rc_executor.execute_stage(
        Stage.TOPIC_INIT,
        run_dir=run_dir,
        run_id="run-1",
        config=rc_config,
        adapters=adapters,
        auto_approve_gates=True,
    )

    assert result.status == StageStatus.DONE
    assert "goal.md" in result.artifacts
    assert "hardware_profile.json" in result.artifacts
    assert (run_dir / "stage-01").is_dir()
    assert (
        (run_dir / "stage-01" / "goal.md")
        .read_text(encoding="utf-8")
        .startswith("# Goal")
    )
    assert (run_dir / "stage-01" / "hardware_profile.json").exists()
    assert len(fake_llm.calls) == 1

    decision = cast(
        dict[str, Any],
        json.loads(
            (run_dir / "stage-01" / "decision.json").read_text(encoding="utf-8")
        ),
    )
    assert decision["run_id"] == "run-1"
    assert decision["status"] == "done"
    assert decision["output_artifacts"] == ["goal.md", "hardware_profile.json"]


def test_execute_stage_contract_validation_missing_output_file_marks_failed(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    def bad_executor(
        _stage_dir: Path,
        _run_dir: Path,
        _config: RCConfig,
        _adapters: AdapterBundle,
        *,
        llm: object = None,
    ):
        _ = llm
        return rc_executor.StageResult(
            stage=Stage.TOPIC_INIT, status=StageStatus.DONE, artifacts=("goal.md",)
        )

    monkeypatch.setitem(rc_executor._STAGE_EXECUTORS, Stage.TOPIC_INIT, bad_executor)
    result = rc_executor.execute_stage(
        Stage.TOPIC_INIT,
        run_dir=run_dir,
        run_id="run-2",
        config=rc_config,
        adapters=adapters,
        auto_approve_gates=True,
    )
    assert result.status == StageStatus.FAILED
    assert "Missing or empty output: goal.md" in (result.error or "")


def test_execute_stage_contract_validation_missing_output_directory_marks_failed(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    _write_prior_artifact(run_dir, 5, "shortlist.jsonl", '{"title": "x"}')

    def bad_executor(
        _stage_dir: Path,
        _run_dir: Path,
        _config: RCConfig,
        _adapters: AdapterBundle,
        *,
        llm: object = None,
    ):
        _ = llm
        return rc_executor.StageResult(
            stage=Stage.KNOWLEDGE_EXTRACT,
            status=StageStatus.DONE,
            artifacts=("cards/",),
        )

    monkeypatch.setitem(
        rc_executor._STAGE_EXECUTORS, Stage.KNOWLEDGE_EXTRACT, bad_executor
    )
    result = rc_executor.execute_stage(
        Stage.KNOWLEDGE_EXTRACT,
        run_dir=run_dir,
        run_id="run-3",
        config=rc_config,
        adapters=adapters,
        auto_approve_gates=True,
    )
    assert result.status == StageStatus.FAILED
    assert "Missing output directory: cards/" in (result.error or "")


def test_execute_stage_missing_required_input_returns_failed(
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    result = rc_executor.execute_stage(
        Stage.PROBLEM_DECOMPOSE,
        run_dir=run_dir,
        run_id="run-4",
        config=rc_config,
        adapters=adapters,
        auto_approve_gates=True,
    )
    assert result.status == StageStatus.FAILED
    assert "Missing input: goal.md" in (result.error or "")


def test_execute_stage_gate_behavior_auto_approve_true_keeps_done(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    _write_prior_artifact(run_dir, 4, "candidates.jsonl", '{"title": "paper"}')

    def good_executor(
        stage_dir: Path,
        _run_dir: Path,
        _config: RCConfig,
        _adapters: AdapterBundle,
        *,
        llm: object = None,
        **_kwargs: object,
    ):
        _ = llm
        (stage_dir / "shortlist.jsonl").write_text(
            '{"title": "paper"}\n', encoding="utf-8"
        )
        return rc_executor.StageResult(
            stage=Stage.LITERATURE_SCREEN,
            status=StageStatus.DONE,
            artifacts=("shortlist.jsonl",),
        )

    monkeypatch.setitem(
        rc_executor._STAGE_EXECUTORS, Stage.LITERATURE_SCREEN, good_executor
    )
    result = rc_executor.execute_stage(
        Stage.LITERATURE_SCREEN,
        run_dir=run_dir,
        run_id="run-5",
        config=rc_config,
        adapters=adapters,
        auto_approve_gates=True,
    )
    assert result.status == StageStatus.DONE
    memory_entries = getattr(adapters.memory, "entries", [])
    assert any(
        ns == "gates" and "auto-approved" in content for ns, content in memory_entries
    )


def test_execute_stage_gate_behavior_auto_approve_false_blocks(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    _write_prior_artifact(run_dir, 4, "candidates.jsonl", '{"title": "paper"}')

    def good_executor(
        stage_dir: Path,
        _run_dir: Path,
        _config: RCConfig,
        _adapters: AdapterBundle,
        *,
        llm: object = None,
        **_kwargs: object,
    ):
        _ = llm
        (stage_dir / "shortlist.jsonl").write_text(
            '{"title": "paper"}\n', encoding="utf-8"
        )
        return rc_executor.StageResult(
            stage=Stage.LITERATURE_SCREEN,
            status=StageStatus.DONE,
            artifacts=("shortlist.jsonl",),
        )

    monkeypatch.setitem(
        rc_executor._STAGE_EXECUTORS, Stage.LITERATURE_SCREEN, good_executor
    )
    result = rc_executor.execute_stage(
        Stage.LITERATURE_SCREEN,
        run_dir=run_dir,
        run_id="run-6",
        config=rc_config,
        adapters=adapters,
        auto_approve_gates=False,
    )
    assert result.status == StageStatus.BLOCKED_APPROVAL
    assert result.decision == "block"
    message_calls = getattr(adapters.message, "calls", [])
    assert message_calls
    assert "Approval required" in message_calls[-1][2]


def test_execute_stage_llm_client_creation_error_falls_back_without_crash(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    def boom(_config: RCConfig):
        raise RuntimeError("llm init failed")

    monkeypatch.setattr("researchclaw.pipeline.executor.LLMClient.from_rc_config", boom)
    result = rc_executor.execute_stage(
        Stage.TOPIC_INIT,
        run_dir=run_dir,
        run_id="run-7",
        config=rc_config,
        adapters=adapters,
        auto_approve_gates=True,
    )
    assert result.status == StageStatus.DONE
    assert (run_dir / "stage-01" / "goal.md").exists()


def test_execute_stage_executor_exception_returns_failed(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    def raising_executor(
        _stage_dir: Path,
        _run_dir: Path,
        _config: RCConfig,
        _adapters: AdapterBundle,
        *,
        llm: object = None,
        **_kwargs: object,
    ):
        _ = llm
        raise RuntimeError("stage exploded")

    monkeypatch.setitem(
        rc_executor._STAGE_EXECUTORS, Stage.TOPIC_INIT, raising_executor
    )
    result = rc_executor.execute_stage(
        Stage.TOPIC_INIT,
        run_dir=run_dir,
        run_id="run-8",
        config=rc_config,
        adapters=adapters,
        auto_approve_gates=True,
    )
    assert result.status == StageStatus.FAILED
    assert result.decision == "retry"
    assert "stage exploded" in (result.error or "")


@pytest.mark.parametrize(
    "stage",
    [
        Stage.TOPIC_INIT,
        Stage.PROBLEM_DECOMPOSE,
        Stage.SEARCH_STRATEGY,
        Stage.LITERATURE_COLLECT,
        Stage.LITERATURE_SCREEN,
        Stage.KNOWLEDGE_EXTRACT,
        Stage.SYNTHESIS,
        Stage.HYPOTHESIS_GEN,
        Stage.EXPERIMENT_DESIGN,
        Stage.CODE_GENERATION,
    ],
)
def test_stage_executor_mapping_values_are_callable(stage: Stage) -> None:
    assert callable(rc_executor._STAGE_EXECUTORS[stage])


class TestStageHealth:
    def test_stage_health_json_written(self, tmp_path: Path) -> None:
        from researchclaw.pipeline.executor import execute_stage
        from researchclaw.pipeline.stages import Stage

        config = RCConfig.load(
            Path(__file__).parent.parent / "config.researchclaw.example.yaml",
            check_paths=False,
        )
        result = execute_stage(
            Stage.TOPIC_INIT,
            run_dir=tmp_path,
            run_id="test-health",
            config=config,
            adapters=AdapterBundle(),
            auto_approve_gates=True,
        )
        health_path = tmp_path / "stage-01" / "stage_health.json"
        assert result is not None
        assert health_path.exists()

    def test_stage_health_has_required_fields(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock, patch

        from researchclaw.pipeline.executor import execute_stage
        from researchclaw.pipeline.stages import Stage

        config = RCConfig.load(
            Path(__file__).parent.parent / "config.researchclaw.example.yaml",
            check_paths=False,
        )

        with patch("researchclaw.pipeline.executor.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.chat.return_value = MagicMock(
                content='{"topic": "test", "research_questions": ["q1"]}'
            )
            mock_llm_cls.from_rc_config.return_value = mock_client

            execute_stage(
                Stage.TOPIC_INIT,
                run_dir=tmp_path,
                run_id="test-health-fields",
                config=config,
                adapters=AdapterBundle(),
                auto_approve_gates=True,
            )

        health_path = tmp_path / "stage-01" / "stage_health.json"
        if health_path.exists():
            data = json.loads(health_path.read_text(encoding="utf-8"))
            assert "stage_id" in data
            assert "run_id" in data
            assert "duration_sec" in data
            assert "status" in data
            assert "timestamp" in data
            assert data["duration_sec"] >= 0


    def test_stage_health_duration_positive(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock, patch

        from researchclaw.pipeline.executor import execute_stage
        from researchclaw.pipeline.stages import Stage

        config = RCConfig.load(
            Path(__file__).parent.parent / "config.researchclaw.example.yaml",
            check_paths=False,
        )

        with patch("researchclaw.pipeline.executor.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.chat.return_value = MagicMock(
                content='{"topic": "test", "sub_problems": []}'
            )
            mock_llm_cls.from_rc_config.return_value = mock_client

            execute_stage(
                Stage.TOPIC_INIT,
                run_dir=tmp_path,
                run_id="test-duration",
                config=config,
                adapters=AdapterBundle(),
                auto_approve_gates=True,
            )

        health_path = tmp_path / "stage-01" / "stage_health.json"
        if health_path.exists():
            data = json.loads(health_path.read_text(encoding="utf-8"))
            assert data["duration_sec"] >= 0

# Contracts import for Stage 13/22 preservation features.
from researchclaw.pipeline.contracts import CONTRACTS


class TestIterativeRefine:
    def _prepare_refine_inputs(self, run_dir: Path) -> None:
        _write_prior_artifact(
            run_dir,
            10,
            "experiment.py",
            (
                "import random\n"
                "random.seed(42)\n"
                "for i in range(5):\n"
                "    print(f'val_loss: {0.5 - i*0.05:.4f}')\n"
            ),
        )
        (run_dir / "stage-12" / "runs").mkdir(parents=True, exist_ok=True)
        _write_prior_artifact(
            run_dir,
            12,
            "runs/run-1.json",
            json.dumps(
                {
                    "run_id": "run-1",
                    "status": "completed",
                    "metrics": {"val_loss": 0.35},
                }
            ),
        )

    def test_refine_simulated_mode_skips(
        self,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
    ) -> None:
        """R10-Fix3: Simulated mode should skip iterative refinement entirely."""
        self._prepare_refine_inputs(run_dir)
        stage_dir = run_dir / "stage-13"
        stage_dir.mkdir(parents=True, exist_ok=True)
        # Force simulated mode to test the skip behavior
        import copy
        sim_cfg = copy.deepcopy(rc_config)
        object.__setattr__(sim_cfg.experiment, "mode", "simulated")

        result = rc_executor._execute_iterative_refine(
            stage_dir,
            run_dir,
            sim_cfg,
            adapters,
            llm=None,
        )

        payload = json.loads(
            (stage_dir / "refinement_log.json").read_text(encoding="utf-8")
        )
        assert payload["skipped"] is True
        assert payload["mode"] == "simulated"
        assert result.status == StageStatus.DONE
        # Original code should be copied as final
        assert (stage_dir / "experiment_final.py").exists()

    def test_refine_no_llm_saves_original_as_final(
        self,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
    ) -> None:
        self._prepare_refine_inputs(run_dir)
        stage_dir = run_dir / "stage-13"
        stage_dir.mkdir(parents=True, exist_ok=True)

        result = rc_executor._execute_iterative_refine(
            stage_dir,
            run_dir,
            rc_config,
            adapters,
            llm=None,
        )

        original_code = (run_dir / "stage-10" / "experiment.py").read_text(
            encoding="utf-8"
        )
        final_code = (stage_dir / "experiment_final.py").read_text(encoding="utf-8")
        assert original_code == final_code
        payload = json.loads(
            (stage_dir / "refinement_log.json").read_text(encoding="utf-8")
        )
        assert payload["stop_reason"] == "llm_unavailable"
        assert result.status == StageStatus.DONE

    def test_refine_with_llm_generates_improved_code(
        self,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
    ) -> None:
        self._prepare_refine_inputs(run_dir)
        stage_dir = run_dir / "stage-13"
        stage_dir.mkdir(parents=True, exist_ok=True)
        llm = FakeLLMClient(
            "```python\n"
            "import random\n"
            "random.seed(42)\n"
            "for i in range(10):\n"
            "    print(f'val_loss: {0.4 - i*0.03:.4f}')\n"
            "```"
        )

        rc_executor._execute_iterative_refine(
            stage_dir, run_dir, rc_config, adapters, llm=llm
        )

        assert (stage_dir / "experiment_v1").is_dir()
        assert (stage_dir / "experiment_final.py").exists()
        payload = json.loads(
            (stage_dir / "refinement_log.json").read_text(encoding="utf-8")
        )
        assert isinstance(payload.get("iterations"), list)
        assert payload["iterations"]

    def test_refine_converges_after_no_improvement(
        self,
        tmp_path: Path,
        run_dir: Path,
        adapters: AdapterBundle,
    ) -> None:
        import sys

        self._prepare_refine_inputs(run_dir)
        stage_dir = run_dir / "stage-13"
        stage_dir.mkdir(parents=True, exist_ok=True)

        sandbox_data = {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {
                "topic": "test-driven science",
                "domains": ["ml", "systems"],
                "daily_paper_count": 2,
                "quality_threshold": 8.2,
            },
            "runtime": {"timezone": "UTC"},
            "notifications": {
                "channel": "local",
                "on_stage_start": True,
                "on_stage_fail": False,
                "on_gate_required": True,
            },
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": True, "use_message": True},
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:1234/v1",
                "api_key_env": "RC_TEST_KEY",
                "api_key": "inline-test-key",
                "primary_model": "fake-model",
                "fallback_models": [],
            },
            "security": {"hitl_required_stages": [5, 9, 20]},
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 30,
                "max_iterations": 3,
                "metric_key": "val_loss",
                "metric_direction": "minimize",
                "sandbox": {
                    "python_path": sys.executable,
                    "gpu_required": False,
                    "max_memory_mb": 1024,
                },
            },
        }
        sandbox_config = RCConfig.from_dict(
            sandbox_data,
            project_root=tmp_path,
            check_paths=False,
        )
        llm = FakeLLMClient(
            "```python\nfor _ in range(3):\n    print('val_loss: 0.5000')\n```"
        )

        rc_executor._execute_iterative_refine(
            stage_dir,
            run_dir,
            sandbox_config,
            adapters,
            llm=llm,
        )

        payload = json.loads(
            (stage_dir / "refinement_log.json").read_text(encoding="utf-8")
        )
        assert payload["converged"] is True
        assert payload["stop_reason"] == "no_improvement_for_2_iterations"

    def test_refine_artifacts_include_version_files(
        self,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
    ) -> None:
        self._prepare_refine_inputs(run_dir)
        stage_dir = run_dir / "stage-13"
        stage_dir.mkdir(parents=True, exist_ok=True)
        llm = FakeLLMClient(
            "```python\n"
            "import random\n"
            "random.seed(42)\n"
            "for i in range(10):\n"
            "    print(f'val_loss: {0.4 - i*0.03:.4f}')\n"
            "```"
        )

        result = rc_executor._execute_iterative_refine(
            stage_dir,
            run_dir,
            rc_config,
            adapters,
            llm=llm,
        )

        assert "refinement_log.json" in result.artifacts
        assert "experiment_final/" in result.artifacts
        assert any(
            artifact.startswith("experiment_v") and artifact.endswith("/")
            for artifact in result.artifacts
        )

    def test_refine_sandbox_mode_runs_code(
        self,
        tmp_path: Path,
        run_dir: Path,
        adapters: AdapterBundle,
    ) -> None:
        import sys

        self._prepare_refine_inputs(run_dir)
        stage_dir = run_dir / "stage-13"
        stage_dir.mkdir(parents=True, exist_ok=True)

        sandbox_data = {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {
                "topic": "test-driven science",
                "domains": ["ml", "systems"],
                "daily_paper_count": 2,
                "quality_threshold": 8.2,
            },
            "runtime": {"timezone": "UTC"},
            "notifications": {
                "channel": "local",
                "on_stage_start": True,
                "on_stage_fail": False,
                "on_gate_required": True,
            },
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": True, "use_message": True},
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:1234/v1",
                "api_key_env": "RC_TEST_KEY",
                "api_key": "inline-test-key",
                "primary_model": "fake-model",
                "fallback_models": [],
            },
            "security": {"hitl_required_stages": [5, 9, 20]},
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 30,
                "max_iterations": 3,
                "metric_key": "val_loss",
                "metric_direction": "minimize",
                "sandbox": {
                    "python_path": sys.executable,
                    "gpu_required": False,
                    "max_memory_mb": 1024,
                },
            },
        }
        sandbox_config = RCConfig.from_dict(
            sandbox_data,
            project_root=tmp_path,
            check_paths=False,
        )
        llm = FakeLLMClient(
            "```python\n"
            "import random\n"
            "random.seed(42)\n"
            "for i in range(10):\n"
            "    print(f'val_loss: {0.4 - i*0.03:.4f}')\n"
            "```"
        )

        rc_executor._execute_iterative_refine(
            stage_dir,
            run_dir,
            sandbox_config,
            adapters,
            llm=llm,
        )

        payload = json.loads(
            (stage_dir / "refinement_log.json").read_text(encoding="utf-8")
        )
        assert any(
            "sandbox" in iteration for iteration in payload.get("iterations", [])
        )


class TestExportPublishCodePackage:
    def test_export_packages_experiment_final(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
    ) -> None:
        _write_prior_artifact(
            run_dir, 19, "paper_revised.md", "# Test Paper\n\nSome content..."
        )
        _write_prior_artifact(
            run_dir,
            13,
            "experiment_final.py",
            'import numpy\nprint("val_loss: 0.1")\n',
        )
        stage_dir = tmp_path / "run" / "stage-22"
        stage_dir.mkdir(parents=True, exist_ok=True)

        rc_executor._execute_export_publish(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )

        assert (stage_dir / "code" / "experiment.py").exists()
        assert (stage_dir / "code" / "README.md").exists()
        req_text = (stage_dir / "code" / "requirements.txt").read_text(encoding="utf-8")
        assert "numpy" in req_text

    def test_export_falls_back_to_experiment_py(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
    ) -> None:
        _write_prior_artifact(
            run_dir, 19, "paper_revised.md", "# Test Paper\n\nSome content..."
        )
        _write_prior_artifact(
            run_dir,
            10,
            "experiment.py",
            'import numpy\nprint("val_loss: 0.1")\n',
        )
        stage_dir = tmp_path / "run" / "stage-22"
        stage_dir.mkdir(parents=True, exist_ok=True)

        rc_executor._execute_export_publish(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )

        code_text = (stage_dir / "code" / "experiment.py").read_text(encoding="utf-8")
        assert "val_loss: 0.1" in code_text

    def test_export_no_experiment_skips_code_dir(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
    ) -> None:
        _write_prior_artifact(
            run_dir, 19, "paper_revised.md", "# Test Paper\n\nSome content..."
        )
        stage_dir = tmp_path / "run" / "stage-22"
        stage_dir.mkdir(parents=True, exist_ok=True)

        result = rc_executor._execute_export_publish(
            stage_dir,
            run_dir,
            rc_config,
            adapters,
            llm=None,
        )

        assert not (stage_dir / "code").exists()
        assert "code/" not in result.artifacts

    def test_export_detects_multiple_dependencies(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
    ) -> None:
        _write_prior_artifact(
            run_dir, 19, "paper_revised.md", "# Test Paper\n\nSome content..."
        )
        _write_prior_artifact(
            run_dir,
            13,
            "experiment_final.py",
            (
                "import numpy\n"
                "import torch\n"
                "from sklearn.metrics import accuracy_score\n"
                "print(accuracy_score([1], [1]))\n"
            ),
        )
        stage_dir = tmp_path / "run" / "stage-22"
        stage_dir.mkdir(parents=True, exist_ok=True)

        rc_executor._execute_export_publish(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )

        requirements = (stage_dir / "code" / "requirements.txt").read_text(
            encoding="utf-8"
        )
        assert "numpy" in requirements
        assert "torch" in requirements
        assert "scikit-learn" in requirements

    def test_export_code_readme_contains_title(
        self,
        tmp_path: Path,
        run_dir: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
    ) -> None:
        _write_prior_artifact(
            run_dir, 19, "paper_revised.md", "# My Great Paper\n\nSome content..."
        )
        _write_prior_artifact(
            run_dir,
            13,
            "experiment_final.py",
            'print("val_loss: 0.1")\n',
        )
        stage_dir = tmp_path / "run" / "stage-22"
        stage_dir.mkdir(parents=True, exist_ok=True)

        rc_executor._execute_export_publish(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )

        readme = (stage_dir / "code" / "README.md").read_text(encoding="utf-8")
        assert "My Great Paper" in readme


def test_contracts_stage13_includes_experiment_final() -> None:
    assert "experiment_final/" in CONTRACTS[Stage.ITERATIVE_REFINE].output_files


def test_contracts_stage22_includes_code_dir() -> None:
    assert "code/" in CONTRACTS[Stage.EXPORT_PUBLISH].output_files


# ── P1-1: Topic keyword extraction tests ──


class TestExtractTopicKeywords:
    def test_basic_extraction(self) -> None:
        keywords = rc_executor._extract_topic_keywords(
            "Agent-based Reinforcement Learning for Automated Scientific Discovery"
        )
        assert "agent-based" in keywords
        assert "reinforcement" in keywords
        assert "learning" in keywords
        assert "automated" in keywords
        assert "scientific" in keywords
        assert "discovery" in keywords
        # Stop words excluded
        # Stop words excluded
        assert "for" not in keywords

    def test_includes_domain_keywords(self) -> None:
        keywords = rc_executor._extract_topic_keywords(
            "Neural network pruning", domains=("ml", "optimization")
        )
        assert "neural" in keywords
        assert "network" in keywords
        assert "pruning" in keywords
        assert "ml" in keywords
        assert "optimization" in keywords

    def test_deduplication(self) -> None:
        keywords = rc_executor._extract_topic_keywords(
            "Learning to learn meta-learning", domains=("learning",)
        )
        assert keywords.count("learning") == 1

    def test_empty_topic(self) -> None:
        keywords = rc_executor._extract_topic_keywords("")
        assert keywords == []


# ── P1-2: Topic constraint block test ──


class TestTopicConstraintBlock:
    def test_contains_topic(self) -> None:
        block = rc_executor._topic_constraint_block("Transformer attention for time series")
        assert "Transformer attention for time series" in block

    def test_contains_prohibition(self) -> None:
        block = rc_executor._topic_constraint_block("anything")
        assert "PROHIBITED" in block
        assert "environment" in block.lower()
        assert "infrastructure" in block.lower()

    def test_hard_constraint_markers(self) -> None:
        block = rc_executor._topic_constraint_block("test")
        assert "HARD TOPIC CONSTRAINT" in block
        assert "END CONSTRAINT" in block


# ── Multi-perspective debate tests ──


class TestParseDecision:
    def test_proceed_default(self) -> None:
        assert rc_executor._parse_decision("Some random text") == "proceed"

    def test_proceed_explicit(self) -> None:
        text = "## Decision\nPROCEED\n## Justification\nGood results."
        assert rc_executor._parse_decision(text) == "proceed"

    def test_pivot_detected(self) -> None:
        text = "## Decision\nPIVOT\n## Justification\nHypotheses flawed."
        assert rc_executor._parse_decision(text) == "pivot"

    def test_refine_detected(self) -> None:
        text = "## Decision\nREFINE\n## Justification\nNeed more tuning."
        assert rc_executor._parse_decision(text) == "refine"

    def test_pivot_case_insensitive(self) -> None:
        text = "## Decision\npivot\n## Justification\nBad approach."
        assert rc_executor._parse_decision(text) == "pivot"

    def test_pivot_takes_priority_over_proceed(self) -> None:
        text = "## Decision\nPIVOT\nWe should not PROCEED."
        assert rc_executor._parse_decision(text) == "pivot"

    def test_decision_in_body_not_heading(self) -> None:
        text = "The results suggest we should PIVOT to a new approach."
        assert rc_executor._parse_decision(text) == "pivot"


class TestResearchDecisionStructured:
    def test_decision_produces_structured_json(
        self, tmp_path: Path, rc_config: RCConfig, adapters: AdapterBundle
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stage_dir = run_dir / "stage-15"
        stage_dir.mkdir(parents=True)
        _write_prior_artifact(run_dir, 14, "analysis.md", "# Analysis\nResults ok.")
        fake_llm = FakeLLMClient("## Decision\nPROCEED\n## Justification\nGood.")
        result = rc_executor._execute_research_decision(
            stage_dir, run_dir, rc_config, adapters, llm=fake_llm
        )
        assert result.decision == "proceed"
        assert "decision_structured.json" in result.artifacts
        import json
        data = json.loads((stage_dir / "decision_structured.json").read_text())
        assert data["decision"] == "proceed"

    def test_pivot_decision_from_llm(
        self, tmp_path: Path, rc_config: RCConfig, adapters: AdapterBundle
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stage_dir = run_dir / "stage-15"
        stage_dir.mkdir(parents=True)
        _write_prior_artifact(run_dir, 14, "analysis.md", "# Analysis\nBad results.")
        fake_llm = FakeLLMClient("## Decision\nPIVOT\n## Justification\nFlawed.")
        result = rc_executor._execute_research_decision(
            stage_dir, run_dir, rc_config, adapters, llm=fake_llm
        )
        assert result.decision == "pivot"

    def test_no_llm_defaults_to_proceed(
        self, tmp_path: Path, rc_config: RCConfig, adapters: AdapterBundle
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stage_dir = run_dir / "stage-15"
        stage_dir.mkdir(parents=True)
        result = rc_executor._execute_research_decision(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )
        assert result.decision == "proceed"


class TestMultiPerspectiveGenerate:
    def test_generates_all_perspectives(self, tmp_path: Path) -> None:
        roles = {
            "role_a": {"system": "You are A.", "user": "Do A for {topic}."},
            "role_b": {"system": "You are B.", "user": "Do B for {topic}."},
        }
        fake_llm = FakeLLMClient("perspective output")
        perspectives_dir = tmp_path / "perspectives"
        result = rc_executor._multi_perspective_generate(
            fake_llm, roles, {"topic": "test"}, perspectives_dir
        )
        assert set(result.keys()) == {"role_a", "role_b"}
        assert (perspectives_dir / "role_a.md").exists()
        assert (perspectives_dir / "role_b.md").exists()
        assert len(fake_llm.calls) == 2

    def test_saves_perspective_content(self, tmp_path: Path) -> None:
        roles = {"critic": {"system": "Be critical.", "user": "Criticize {topic}."}}
        fake_llm = FakeLLMClient("critical analysis here")
        perspectives_dir = tmp_path / "perspectives"
        rc_executor._multi_perspective_generate(
            fake_llm, roles, {"topic": "ml"}, perspectives_dir
        )
        content = (perspectives_dir / "critic.md").read_text()
        assert content == "critical analysis here"

    def test_renders_variables_in_prompts(self, tmp_path: Path) -> None:
        roles = {"r1": {"system": "Sys for {topic}.", "user": "User for {topic}."}}
        fake_llm = FakeLLMClient("ok")
        rc_executor._multi_perspective_generate(
            fake_llm, roles, {"topic": "RL"}, tmp_path / "p"
        )
        call = fake_llm.calls[0]
        assert "RL" in call[0]["content"]


class TestSynthesizePerspectives:
    def test_combines_perspectives(self) -> None:
        fake_llm = FakeLLMClient("synthesized result")
        pm = rc_executor.PromptManager()
        perspectives = {"innovator": "idea A", "contrarian": "idea B"}
        result = rc_executor._synthesize_perspectives(
            fake_llm, perspectives, "hypothesis_synthesize", pm
        )
        assert result == "synthesized result"
        # Check the user prompt contained both perspectives
        call_content = fake_llm.calls[0][0]["content"]
        assert "innovator" in call_content
        assert "contrarian" in call_content


class TestHypothesisGenDebate:
    def test_hypothesis_gen_with_llm_creates_perspectives(
        self, tmp_path: Path, rc_config: RCConfig, adapters: AdapterBundle
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stage_dir = run_dir / "stage-08"
        stage_dir.mkdir(parents=True)
        _write_prior_artifact(run_dir, 7, "synthesis.md", "# Synthesis\nGap found.")
        fake_llm = FakeLLMClient("## H1\nTest hypothesis")
        result = rc_executor._execute_hypothesis_gen(
            stage_dir, run_dir, rc_config, adapters, llm=fake_llm
        )
        assert result.status == StageStatus.DONE
        assert "hypotheses.md" in result.artifacts
        perspectives_dir = stage_dir / "perspectives"
        assert perspectives_dir.exists()
        # Should have 3 perspective files (innovator, pragmatist, contrarian)
        perspective_files = list(perspectives_dir.glob("*.md"))
        assert len(perspective_files) == 3

    def test_hypothesis_gen_without_llm_no_perspectives(
        self, tmp_path: Path, rc_config: RCConfig, adapters: AdapterBundle
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stage_dir = run_dir / "stage-08"
        stage_dir.mkdir(parents=True)
        _write_prior_artifact(run_dir, 7, "synthesis.md", "# Synthesis\nGap found.")
        result = rc_executor._execute_hypothesis_gen(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )
        assert result.status == StageStatus.DONE
        assert "hypotheses.md" in result.artifacts
        # No perspectives directory when no LLM
        assert not (stage_dir / "perspectives").exists()


class TestResultAnalysisDebate:
    def test_result_analysis_with_llm_creates_perspectives(
        self, tmp_path: Path, rc_config: RCConfig, adapters: AdapterBundle
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stage_dir = run_dir / "stage-14"
        stage_dir.mkdir(parents=True)
        _write_prior_artifact(run_dir, 1, "goal.md", "# Goal\nTest")
        _write_prior_artifact(run_dir, 8, "hypotheses.md", "# H1\nTest")
        fake_llm = FakeLLMClient("## Analysis\nResults look good.")
        result = rc_executor._execute_result_analysis(
            stage_dir, run_dir, rc_config, adapters, llm=fake_llm
        )
        assert result.status == StageStatus.DONE
        assert "analysis.md" in result.artifacts
        perspectives_dir = stage_dir / "perspectives"
        assert perspectives_dir.exists()
        # Should have 3 perspective files (optimist, skeptic, methodologist)
        perspective_files = list(perspectives_dir.glob("*.md"))
        assert len(perspective_files) == 3

    def test_result_analysis_without_llm_no_perspectives(
        self, tmp_path: Path, rc_config: RCConfig, adapters: AdapterBundle
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stage_dir = run_dir / "stage-14"
        stage_dir.mkdir(parents=True)
        result = rc_executor._execute_result_analysis(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )
        assert result.status == StageStatus.DONE
        assert "analysis.md" in result.artifacts
        assert not (stage_dir / "perspectives").exists()


class TestParseMetricsFromStdout:
    """Tests for _parse_metrics_from_stdout() helper."""

    def test_parses_simple_name_value(self) -> None:
        from researchclaw.pipeline.executor import _parse_metrics_from_stdout

        stdout = "loss: 0.0042\naccuracy: 0.95"
        metrics = _parse_metrics_from_stdout(stdout)
        assert metrics["loss"] == pytest.approx(0.0042)
        assert metrics["accuracy"] == pytest.approx(0.95)

    def test_parses_compound_names(self) -> None:
        from researchclaw.pipeline.executor import _parse_metrics_from_stdout

        stdout = "UCB (Stochastic) cumulative_regret: 361.9233\nEXP3 (Adversarial) total_rewards: 13368.4811"
        metrics = _parse_metrics_from_stdout(stdout)
        assert "UCB (Stochastic) cumulative_regret" in metrics
        assert metrics["UCB (Stochastic) cumulative_regret"] == pytest.approx(361.9233)

    def test_ignores_non_numeric_lines(self) -> None:
        from researchclaw.pipeline.executor import _parse_metrics_from_stdout

        stdout = "Running experiment...\nloss: 0.5\nDone."
        metrics = _parse_metrics_from_stdout(stdout)
        assert len(metrics) == 1
        assert metrics["loss"] == pytest.approx(0.5)

    def test_empty_stdout_returns_empty_dict(self) -> None:
        from researchclaw.pipeline.executor import _parse_metrics_from_stdout

        assert _parse_metrics_from_stdout("") == {}

    def test_handles_negative_values(self) -> None:
        from researchclaw.pipeline.executor import _parse_metrics_from_stdout

        stdout = "UCB (Adversarial) cumulative_regret: -3877.5323"
        metrics = _parse_metrics_from_stdout(stdout)
        assert metrics["UCB (Adversarial) cumulative_regret"] == pytest.approx(-3877.5323)

    def test_filters_log_lines(self) -> None:
        from researchclaw.pipeline.executor import _parse_metrics_from_stdout

        stdout = (
            "Running experiments for support set size: 1\n"
            "Loading model weights: 42\n"
            "Training epoch: 5\n"
            "loss: 0.123\n"
            "accuracy: 0.95\n"
        )
        metrics = _parse_metrics_from_stdout(stdout)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert len(metrics) == 2  # log lines should be excluded

    def test_filters_long_name_lines(self) -> None:
        from researchclaw.pipeline.executor import _parse_metrics_from_stdout

        stdout = "this is a very long status message that should not be a metric: 42\n"
        metrics = _parse_metrics_from_stdout(stdout)
        assert len(metrics) == 0


class TestDetectRuntimeIssues:
    """Tests for _detect_runtime_issues() helper."""

    def _make_sandbox_result(
        self,
        metrics: dict | None = None,
        stdout: str = "",
        stderr: str = "",
    ):
        from types import SimpleNamespace

        return SimpleNamespace(
            metrics=metrics or {},
            stdout=stdout,
            stderr=stderr,
            returncode=0,
            elapsed_sec=1.0,
            timed_out=False,
        )

    def test_no_issues_returns_empty_string(self) -> None:
        r = self._make_sandbox_result(metrics={"loss": 0.5}, stdout="loss: 0.5")
        assert rc_executor._detect_runtime_issues(r) == ""

    def test_detects_nan_in_metrics(self) -> None:
        r = self._make_sandbox_result(metrics={"loss": float("nan")})
        result = rc_executor._detect_runtime_issues(r)
        assert "NaN" in result
        assert "loss" in result

    def test_detects_inf_in_metrics(self) -> None:
        r = self._make_sandbox_result(metrics={"loss": float("inf")})
        result = rc_executor._detect_runtime_issues(r)
        assert "Inf" in result

    def test_detects_nan_in_stdout(self) -> None:
        r = self._make_sandbox_result(stdout="accuracy: nan\nloss: 0.5")
        result = rc_executor._detect_runtime_issues(r)
        assert "NaN" in result or "nan" in result

    def test_detects_runtime_warning_in_stderr(self) -> None:
        stderr = (
            "optimizers.py:76: RuntimeWarning: invalid value encountered in divide\n"
            "  directions = np.vstack((directions[1:], new_direction / norm))\n"
        )
        r = self._make_sandbox_result(stderr=stderr)
        result = rc_executor._detect_runtime_issues(r)
        assert "RuntimeWarning" in result
        assert "invalid value" in result

    def test_detects_division_error_in_stderr(self) -> None:
        stderr = "ZeroDivisionError: division by zero\n"
        r = self._make_sandbox_result(stderr=stderr)
        result = rc_executor._detect_runtime_issues(r)
        assert "Error" in result

    def test_ignores_benign_stderr(self) -> None:
        # Non-warning stderr should be ignored
        r = self._make_sandbox_result(stderr="Loading module...\nDone.\n")
        assert rc_executor._detect_runtime_issues(r) == ""

    def test_combined_nan_and_stderr(self) -> None:
        r = self._make_sandbox_result(
            metrics={"accuracy": float("nan")},
            stderr="RuntimeWarning: invalid value\n",
        )
        result = rc_executor._detect_runtime_issues(r)
        assert "NaN" in result
        assert "RuntimeWarning" in result

    def test_detects_dummy_metric_identical_values(self) -> None:
        stdout = (
            "UCB (Stochastic) convergence_rate: 1.0000\n"
            "UCB (Adversarial) convergence_rate: 1.0000\n"
            "Thompson (Stochastic) convergence_rate: 1.0000\n"
            "Thompson (Adversarial) convergence_rate: 1.0000\n"
        )
        r = self._make_sandbox_result(stdout=stdout)
        result = rc_executor._detect_runtime_issues(r)
        assert "DUMMY" in result
        assert "convergence_rate" in result

    def test_no_dummy_metric_when_values_differ(self) -> None:
        stdout = (
            "UCB (Stochastic) regret: 78.5\n"
            "Thompson (Stochastic) regret: 121.0\n"
            "EpsilonGreedy (Stochastic) regret: 42.1\n"
        )
        r = self._make_sandbox_result(stdout=stdout)
        result = rc_executor._detect_runtime_issues(r)
        assert "DUMMY" not in result


class TestRemoveBibtexEntries:
    """Tests for _remove_bibtex_entries() helper."""

    def test_removes_specified_keys(self) -> None:
        bib = (
            '@article{smith2024,\n  title={Good Paper},\n  author={Smith},\n}\n\n'
            '@article{venus2024,\n  title={Venus Exploration},\n  author={NASA},\n}\n'
        )
        result = rc_executor._remove_bibtex_entries(bib, {"venus2024"})
        assert "smith2024" in result
        assert "venus2024" not in result

    def test_keeps_all_when_no_match(self) -> None:
        bib = '@article{smith2024,\n  title={Paper},\n}\n'
        result = rc_executor._remove_bibtex_entries(bib, {"other_key"})
        assert "smith2024" in result

    def test_empty_bib(self) -> None:
        assert rc_executor._remove_bibtex_entries("", {"key"}) == ""


class TestRemoveCitationsFromText:
    """Tests for _remove_citations_from_text() helper."""

    def test_removes_latex_cite(self) -> None:
        text = r"As shown in \cite{venus2024}, the results are..."
        result = rc_executor._remove_citations_from_text(text, {"venus2024"})
        assert "venus2024" not in result
        assert "results are" in result

    def test_removes_markdown_cite(self) -> None:
        text = "Prior work [venus2024] explored this topic."
        result = rc_executor._remove_citations_from_text(text, {"venus2024"})
        assert "venus2024" not in result

    def test_cleans_multi_cite_comma(self) -> None:
        text = r"\cite{good2024,venus2024}"
        result = rc_executor._remove_citations_from_text(text, {"venus2024"})
        assert r"\cite{good2024}" in result


class TestCollectRawExperimentMetrics:
    """Tests for _collect_raw_experiment_metrics() helper."""

    def test_returns_empty_when_no_runs(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        block, has_parsed = rc_executor._collect_raw_experiment_metrics(run_dir)
        assert block == ""
        assert not has_parsed

    def test_extracts_metrics_from_stdout(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True)
        payload = {
            "metrics": {},
            "stdout": "UCB regret: 361.92\nThompson regret: 576.24\n",
        }
        (runs_dir / "run-1.json").write_text(json.dumps(payload))
        result, has_parsed = rc_executor._collect_raw_experiment_metrics(run_dir)
        assert "361.92" in result
        assert "576.24" in result
        assert "1 run(s)" in result
        assert not has_parsed

    def test_extracts_from_metrics_dict(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True)
        payload = {"metrics": {"loss": 0.042, "accuracy": 0.95}, "stdout": ""}
        (runs_dir / "run-1.json").write_text(json.dumps(payload))
        result, has_parsed = rc_executor._collect_raw_experiment_metrics(run_dir)
        assert "loss" in result
        assert "0.042" in result
        assert has_parsed

    def test_deduplicates_metrics(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True)
        payload = {
            "metrics": {"loss": 0.5},
            "stdout": "loss: 0.5\nloss: 0.5\n",
        }
        (runs_dir / "run-1.json").write_text(json.dumps(payload))
        result, _ = rc_executor._collect_raw_experiment_metrics(run_dir)
        # "loss: 0.5" should appear only once (deduplicated)
        assert result.count("loss: 0.5") == 1


class TestCollectExperimentEvidence:
    """Tests for _collect_experiment_evidence() helper."""

    def test_returns_empty_when_no_artifacts(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        assert rc_executor._collect_experiment_evidence(run_dir) == ""

    def test_includes_main_py_code(self, run_dir: Path) -> None:
        exp_dir = run_dir / "stage-10" / "experiment"
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "main.py").write_text("print('hello')", encoding="utf-8")
        result = rc_executor._collect_experiment_evidence(run_dir)
        assert "main.py" in result
        assert "hello" in result

    def test_includes_run_metrics(self, run_dir: Path) -> None:
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({"metrics": {"loss": 0.5}, "elapsed_sec": 3.2}),
            encoding="utf-8",
        )
        result = rc_executor._collect_experiment_evidence(run_dir)
        assert "loss" in result
        assert "0.5" in result

    def test_includes_stderr_excerpt(self, run_dir: Path) -> None:
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({
                "metrics": {"loss": 0.5},
                "stderr": "RuntimeWarning: divide by zero",
            }),
            encoding="utf-8",
        )
        result = rc_executor._collect_experiment_evidence(run_dir)
        assert "divide by zero" in result

    def test_includes_refinement_summary(self, run_dir: Path) -> None:
        refine_dir = run_dir / "stage-13"
        refine_dir.mkdir(parents=True, exist_ok=True)
        (refine_dir / "refinement_log.json").write_text(
            json.dumps({
                "iterations": [{"iteration": 1}, {"iteration": 2}],
                "converged": True,
                "stop_reason": "no_improvement_for_2_iterations",
                "best_metric": 0.3,
            }),
            encoding="utf-8",
        )
        result = rc_executor._collect_experiment_evidence(run_dir)
        assert "iterations_executed" in result
        assert "2" in result

    def test_includes_actual_trial_count(self, run_dir: Path) -> None:
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({"metrics": {"loss": 0.5}}), encoding="utf-8"
        )
        result = rc_executor._collect_experiment_evidence(run_dir)
        assert "1 time(s)" in result
        assert "CRITICAL" in result


class TestWritePaperSections:
    """Tests for _write_paper_sections() multi-call writing."""

    def test_produces_three_part_draft(self) -> None:
        call_count = {"n": 0}
        parts = [
            "# Test Title\n\n## Abstract\nTest abstract.\n\n## Introduction\nTest intro.\n\n## Related Work\nTest related.",
            "## Method\nTest method.\n\n## Experiments\nTest experiments.",
            "## Results\nTest results.\n\n## Discussion\nTest discussion.\n\n## Limitations\nTest limits.\n\n## Conclusion\nTest conclusion.",
        ]

        class MultiCallLLM:
            def __init__(self):
                self.calls: list = []

            def chat(self, messages, **kwargs):
                self.calls.append(messages)
                from researchclaw.llm.client import LLMResponse
                idx = len(self.calls) - 1
                return LLMResponse(content=parts[min(idx, 2)], model="fake")

        llm = MultiCallLLM()
        from researchclaw.prompts import PromptManager
        pm = PromptManager()

        draft = rc_executor._write_paper_sections(
            llm=llm,
            pm=pm,
            preamble="Test preamble",
            topic_constraint="",
            exp_metrics_instruction="",
            citation_instruction="",
            outline="Test outline",
        )

        assert llm.calls is not None
        assert len(llm.calls) == 3
        assert "## Abstract" in draft
        assert "## Method" in draft
        assert "## Results" in draft
        assert "## Conclusion" in draft

    def test_each_call_receives_prior_context(self) -> None:
        class ContextTrackingLLM:
            def __init__(self):
                self.user_prompts: list[str] = []

            def chat(self, messages, **kwargs):
                for m in messages:
                    if m.get("role") == "user":
                        self.user_prompts.append(m["content"])
                from researchclaw.llm.client import LLMResponse
                return LLMResponse(content="## Section\nContent here.", model="fake")

        llm = ContextTrackingLLM()
        from researchclaw.prompts import PromptManager
        pm = PromptManager()

        rc_executor._write_paper_sections(
            llm=llm,
            pm=pm,
            preamble="Preamble",
            topic_constraint="",
            exp_metrics_instruction="",
            citation_instruction="",
            outline="Outline",
        )

        assert len(llm.user_prompts) == 3
        # Call 2 and 3 should contain "sections written so far"
        assert "sections written so far" in llm.user_prompts[1]
        assert "completing a paper" in llm.user_prompts[2]


class TestLoadHardwareProfile:
    """Tests for _load_hardware_profile()."""

    @pytest.fixture()
    def run_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "run"
        d.mkdir()
        return d

    def test_loads_valid_profile(self, run_dir: Path) -> None:
        stage = run_dir / "stage-01"
        stage.mkdir()
        profile = {"has_gpu": True, "gpu_type": "mps", "tier": "limited"}
        (stage / "hardware_profile.json").write_text(
            json.dumps(profile), encoding="utf-8"
        )
        result = rc_executor._load_hardware_profile(run_dir)
        assert result is not None
        assert result["gpu_type"] == "mps"

    def test_returns_none_when_missing(self, run_dir: Path) -> None:
        assert rc_executor._load_hardware_profile(run_dir) is None

    def test_returns_none_on_invalid_json(self, run_dir: Path) -> None:
        stage = run_dir / "stage-01"
        stage.mkdir()
        (stage / "hardware_profile.json").write_text("not json", encoding="utf-8")
        assert rc_executor._load_hardware_profile(run_dir) is None


class TestExpandSearchQueries:
    """Tests for _expand_search_queries()."""

    def test_adds_broader_queries(self) -> None:
        queries = ["gradient descent optimization algorithms"]
        topic = "Comparing gradient descent optimization algorithms on benchmark functions"
        result = rc_executor._expand_search_queries(queries, topic)
        assert len(result) > len(queries)

    def test_deduplicates(self) -> None:
        queries = ["gradient descent survey"]
        topic = "gradient descent optimization"
        result = rc_executor._expand_search_queries(queries, topic)
        lowered = [q.lower().strip() for q in result]
        assert len(lowered) == len(set(lowered))

    def test_preserves_original_queries(self) -> None:
        queries = ["query A", "query B"]
        topic = "some research topic about machine learning methods"
        result = rc_executor._expand_search_queries(queries, topic)
        assert result[0] == "query A"
        assert result[1] == "query B"

    def test_adds_survey_benchmark_variants(self) -> None:
        queries = ["deep learning"]
        topic = "deep learning for image classification with limited data"
        result = rc_executor._expand_search_queries(queries, topic)
        has_survey = any("survey" in q.lower() for q in result)
        has_benchmark = any("benchmark" in q.lower() for q in result)
        assert has_survey
        assert has_benchmark


# ── R4-1: Experiment Budget Guard Tests ──────────────────────────────


class TestComputeBudgetBlock:
    """Test compute_budget prompt block injection (R4-1a)."""

    def test_compute_budget_block_exists_in_prompt_manager(self) -> None:
        from researchclaw.prompts import PromptManager

        pm = PromptManager()
        block = pm.block("compute_budget")
        assert "time_budget_sec" in block or "Compute Budget" in block

    def test_compute_budget_injected_into_code_generation(
        self, tmp_path: Path, run_dir: Path, adapters: AdapterBundle
    ) -> None:
        import sys

        data = {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {
                "topic": "optimizer comparison",
                "domains": ["ml"],
                "daily_paper_count": 2,
                "quality_threshold": 8.2,
            },
            "runtime": {"timezone": "UTC"},
            "notifications": {
                "channel": "local",
                "on_stage_start": True,
                "on_stage_fail": False,
                "on_gate_required": True,
            },
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": True, "use_message": True},
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:1234/v1",
                "api_key_env": "RC_TEST_KEY",
                "api_key": "inline-test-key",
                "primary_model": "fake-model",
                "fallback_models": [],
            },
            "security": {"hitl_required_stages": [5, 9, 20]},
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 60,
                "metric_key": "best_loss",
                "metric_direction": "minimize",
                "sandbox": {
                    "python_path": sys.executable,
                    "gpu_required": False,
                    "max_memory_mb": 1024,
                },
            },
        }
        cfg = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

        # Write exp_plan prior artifact
        _write_prior_artifact(run_dir, 10, "exp_plan.yaml", "objectives: test")

        # Capture what the LLM receives
        llm = FakeLLMClient(
            "```filename:main.py\nimport numpy as np\nprint('best_loss: 0.1')\n```"
        )
        stage_dir = run_dir / "stage-11"
        stage_dir.mkdir(parents=True, exist_ok=True)

        rc_executor._execute_code_generation(
            stage_dir, run_dir, cfg, adapters, llm=llm
        )

        # The LLM should have received compute budget info in some call
        # (may be first call in legacy mode, or second call with CodeAgent)
        assert len(llm.calls) >= 1
        all_user_msgs = " ".join(
            call[-1]["content"] for call in llm.calls if call
        )
        assert "60" in all_user_msgs or "Compute Budget" in all_user_msgs


class TestPartialTimeoutStatus:
    """Test partial status for timed-out experiments with data (R4-1c)."""

    def test_timed_out_with_metrics_sets_partial_status(
        self, tmp_path: Path, run_dir: Path, adapters: AdapterBundle
    ) -> None:
        import sys

        data = {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {
                "topic": "test topic",
                "domains": ["ml"],
                "daily_paper_count": 2,
                "quality_threshold": 8.2,
            },
            "runtime": {"timezone": "UTC"},
            "notifications": {
                "channel": "local",
                "on_stage_start": True,
                "on_stage_fail": False,
                "on_gate_required": True,
            },
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": True, "use_message": True},
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:1234/v1",
                "api_key_env": "RC_TEST_KEY",
                "api_key": "inline-test-key",
                "primary_model": "fake-model",
                "fallback_models": [],
            },
            "security": {"hitl_required_stages": [5, 9, 20]},
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 2,
                "metric_key": "best_loss",
                "metric_direction": "minimize",
                "sandbox": {
                    "python_path": sys.executable,
                    "gpu_required": False,
                    "max_memory_mb": 1024,
                },
            },
        }
        cfg = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

        # Write experiment code that prints some metrics then sleeps
        exp_dir = run_dir / "stage-11" / "experiment"
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "main.py").write_text(
            "import time, sys\n"
            "print('best_loss: 0.5', flush=True)\n"
            "sys.stdout.flush()\n"
            "time.sleep(10)\n",
            encoding="utf-8",
        )

        stage_dir = run_dir / "stage-12"
        stage_dir.mkdir(parents=True, exist_ok=True)

        rc_executor._execute_experiment_run(
            stage_dir, run_dir, cfg, adapters
        )

        run_file = stage_dir / "runs" / "run-1.json"
        assert run_file.exists()
        payload = json.loads(run_file.read_text(encoding="utf-8"))
        # Should be "partial" since metrics were captured before timeout
        assert payload["timed_out"] is True
        # Status should be "partial" if metrics captured, "failed" if not
        if payload["metrics"]:
            assert payload["status"] == "partial"
        else:
            # Subprocess stdout may not flush before kill on some platforms
            assert payload["status"] == "failed"


class TestTimeoutAwareRefine:
    """Test timeout-aware prompt injection in iterative refine (R4-1b)."""

    def _prepare_timed_out_run(self, run_dir: Path) -> None:
        """Create a prior run that timed out with no metrics."""
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({
                "run_id": "run-1",
                "task_id": "sandbox-main",
                "status": "failed",
                "metrics": {},
                "timed_out": True,
                "elapsed_sec": 120.0,
            }),
            encoding="utf-8",
        )
        # Write experiment code
        exp_dir = run_dir / "stage-11" / "experiment"
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "main.py").write_text(
            "print('best_loss: 0.1')\n",
            encoding="utf-8",
        )

    def test_timeout_refine_injects_scale_reduction_prompt(
        self, tmp_path: Path, run_dir: Path, adapters: AdapterBundle
    ) -> None:
        self._prepare_timed_out_run(run_dir)
        stage_dir = run_dir / "stage-13"
        stage_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {
                "topic": "test topic",
                "domains": ["ml"],
                "daily_paper_count": 2,
                "quality_threshold": 8.2,
            },
            "runtime": {"timezone": "UTC"},
            "notifications": {
                "channel": "local",
                "on_stage_start": True,
                "on_stage_fail": False,
                "on_gate_required": True,
            },
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": True, "use_message": True},
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:1234/v1",
                "api_key_env": "RC_TEST_KEY",
                "api_key": "inline-test-key",
                "primary_model": "fake-model",
                "fallback_models": [],
            },
            "security": {"hitl_required_stages": [5, 9, 20]},
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 120,
                "max_iterations": 1,
                "metric_key": "best_loss",
                "metric_direction": "minimize",
            },
        }
        cfg = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

        llm = FakeLLMClient(
            "```python\nimport numpy as np\nprint('best_loss: 0.1')\n```"
        )

        rc_executor._execute_iterative_refine(
            stage_dir, run_dir, cfg, adapters, llm=llm
        )

        # The LLM should have received the timeout-aware prompt
        assert len(llm.calls) >= 1
        user_msg = llm.calls[0][-1]["content"]
        assert "TIMED OUT" in user_msg
        assert "120" in user_msg


# ── R4-2: Data Integrity Enforcement Tests ───────────────────────────


class TestDataIntegrityBlock:
    """Test paper draft blocked when no metrics exist (R4-2a)."""

    def test_paper_draft_blocked_with_no_metrics(
        self, tmp_path: Path, run_dir: Path, rc_config: RCConfig, adapters: AdapterBundle
    ) -> None:
        # Write prior artifacts with NO metrics
        _write_prior_artifact(run_dir, 16, "outline.md", "# Outline\n## Abstract\n")
        # No experiment_summary.json, no run files with metrics
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({"run_id": "run-1", "status": "failed", "metrics": {}, "timed_out": True}),
            encoding="utf-8",
        )

        stage_dir = run_dir / "stage-17"
        stage_dir.mkdir(parents=True, exist_ok=True)

        llm = FakeLLMClient("should not be called")
        result = rc_executor._execute_paper_draft(
            stage_dir, run_dir, rc_config, adapters, llm=llm
        )

        assert result.status == StageStatus.FAILED
        draft = (stage_dir / "paper_draft.md").read_text(encoding="utf-8")
        assert "Blocked" in draft or "BLOCKED" in draft or "no metrics" in draft.lower()
        # LLM should NOT have been called
        assert len(llm.calls) == 0

    def test_paper_draft_proceeds_with_metrics(
        self, tmp_path: Path, run_dir: Path, rc_config: RCConfig, adapters: AdapterBundle
    ) -> None:
        _write_prior_artifact(run_dir, 16, "outline.md", "# Outline\n## Abstract\n")
        # Write experiment data with real metrics
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({
                "run_id": "run-1",
                "status": "completed",
                "metrics": {"best_loss": 0.123},
                "stdout": "best_loss: 0.123\n",
            }),
            encoding="utf-8",
        )

        stage_dir = run_dir / "stage-17"
        stage_dir.mkdir(parents=True, exist_ok=True)

        llm = FakeLLMClient("# Paper Title\n## Abstract\nSome abstract text.")
        result = rc_executor._execute_paper_draft(
            stage_dir, run_dir, rc_config, adapters, llm=llm
        )

        # Should proceed (LLM was called)
        assert len(llm.calls) >= 1
        # The prompt should contain anti-fabrication instructions
        all_prompts = " ".join(
            msg["content"] for call in llm.calls for msg in call
        )
        assert "Data Integrity" in all_prompts or "ONLY report numbers" in all_prompts


# ── R4-3: Conference-Grade Title Guidelines Tests ────────────────────


class TestTitleGuidelines:
    """Test title_guidelines and abstract_structure blocks (R4-3)."""

    def test_title_guidelines_block_exists(self) -> None:
        from researchclaw.prompts import PromptManager

        pm = PromptManager()
        block = pm.block("title_guidelines")
        assert "novelty" in block.lower() or "TITLE RULES" in block
        assert "14 words" in block or "15 words" in block or "concrete" in block.lower()

    def test_abstract_structure_block_exists(self) -> None:
        from researchclaw.prompts import PromptManager

        pm = PromptManager()
        block = pm.block("abstract_structure")
        assert "5-sentence" in block or "problem" in block.lower()

    def test_title_guidelines_injected_into_paper_draft(
        self, tmp_path: Path, run_dir: Path, rc_config: RCConfig, adapters: AdapterBundle
    ) -> None:
        _write_prior_artifact(run_dir, 16, "outline.md", "# Outline\n")
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({"run_id": "run-1", "status": "completed",
                        "metrics": {"best_loss": 0.1}, "stdout": "best_loss: 0.1\n"}),
            encoding="utf-8",
        )

        stage_dir = run_dir / "stage-17"
        stage_dir.mkdir(parents=True, exist_ok=True)

        llm = FakeLLMClient("# Paper Title\n## Abstract\nText.")
        rc_executor._execute_paper_draft(
            stage_dir, run_dir, rc_config, adapters, llm=llm
        )

        all_prompts = " ".join(
            msg["content"] for call in llm.calls for msg in call
        )
        assert "Title" in all_prompts or "TITLE" in all_prompts


# ── R4-4: Conference-Grade Writing Quality Tests ─────────────────────


class TestConferenceWritingQuality:
    """Test enhanced writing prompts and writing_guide.py (R4-4)."""

    def test_writing_guide_format_all(self) -> None:
        from researchclaw.writing_guide import format_writing_tips

        result = format_writing_tips()
        assert "Conference Writing Best Practices" in result
        assert "Title" in result
        assert "Common Rejections" in result

    def test_writing_guide_format_subset(self) -> None:
        from researchclaw.writing_guide import format_writing_tips

        result = format_writing_tips(["title", "abstract"])
        assert "Title" in result
        assert "Abstract" in result
        assert "Common Rejections" not in result

    def test_paper_draft_system_includes_principles(self) -> None:
        from researchclaw.prompts import PromptManager

        pm = PromptManager()
        sp = pm.for_stage(
            "paper_draft",
            preamble="test",
            topic_constraint="test",
            exp_metrics_instruction="test",
            citation_instruction="test",
            outline="test",
        )
        # System prompt should mention key principles
        assert "NOVELTY" in sp.system or "novelty" in sp.system.lower()
        assert "fabricate" in sp.system.lower() or "real experimental" in sp.system.lower()


# ── R5-1 & R5-2: Bug Fixes Tests ────────────────────────────────────


class TestRefineTimeoutAndIterationCap:
    """Test R5-1 (no 120s cap) and R5-2 (iteration cap raised to 10)."""

    def test_refine_timeout_uses_full_budget(self) -> None:
        """R5-1: Refine sandbox should NOT cap at 120s."""
        import ast
        import inspect

        source = inspect.getsource(rc_executor._execute_iterative_refine)
        tree = ast.parse(source)
        source_text = inspect.getsource(rc_executor._execute_iterative_refine)
        # Should NOT contain min(..., 120)
        assert "min(config.experiment.time_budget_sec, 120)" not in source_text

    def test_iteration_cap_is_10(self) -> None:
        """R5-2: Max iterations should be capped at 10, not 3."""
        import inspect

        source = inspect.getsource(rc_executor._execute_iterative_refine)
        assert "min(requested_iterations, 10)" in source
        assert "min(requested_iterations, 3)" not in source

    def test_refine_respects_high_iteration_count(
        self, tmp_path: Path, run_dir: Path, adapters: AdapterBundle
    ) -> None:
        """R5-2: Setting max_iterations=7 should actually allow 7 iterations."""
        # Write prior run artifacts
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({"run_id": "run-1", "status": "completed",
                        "metrics": {"best_loss": 0.5}}),
            encoding="utf-8",
        )
        exp_dir = run_dir / "stage-11" / "experiment"
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "main.py").write_text("print('best_loss: 0.5')\n", encoding="utf-8")

        stage_dir = run_dir / "stage-13"
        stage_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {"topic": "test", "domains": ["ml"],
                         "daily_paper_count": 2, "quality_threshold": 8.2},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local", "on_stage_start": True,
                              "on_stage_fail": False, "on_gate_required": True},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": True, "use_message": True},
            "llm": {"provider": "openai-compatible", "base_url": "http://localhost:1234/v1",
                    "api_key_env": "RC_TEST_KEY", "api_key": "inline-test-key",
                    "primary_model": "fake-model", "fallback_models": []},
            "security": {"hitl_required_stages": [5, 9, 20]},
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 300,
                "max_iterations": 7,
                "metric_key": "best_loss",
                "metric_direction": "minimize",
            },
        }
        cfg = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

        # LLM always returns same code — will trigger no_improvement early stop
        llm = FakeLLMClient("```python\nprint('best_loss: 0.5')\n```")

        rc_executor._execute_iterative_refine(
            stage_dir, run_dir, cfg, adapters, llm=llm
        )

        log = json.loads((stage_dir / "refinement_log.json").read_text(encoding="utf-8"))
        # Should have been allowed more than 3 iterations (capped at 7)
        assert log["max_iterations_executed"] == 7
        # But may have stopped early due to no_improvement_for_2_iterations
        assert len(log["iterations"]) >= 2


# ── R5-3: NaN/Divergence Fast-Fail Tests ────────────────────────────


class TestNaNDivergenceDetection:
    """Test NaN/Inf filtering and divergence detection (R5-3)."""

    def test_parse_metrics_filters_nan(self) -> None:
        from researchclaw.experiment.sandbox import parse_metrics

        stdout = "best_loss: 0.5\nbad_metric: nan\ngood_metric: 1.23\n"
        metrics = parse_metrics(stdout)
        assert "best_loss" in metrics
        assert "good_metric" in metrics
        assert "bad_metric" not in metrics  # NaN should be filtered

    def test_parse_metrics_filters_inf(self) -> None:
        from researchclaw.experiment.sandbox import parse_metrics

        stdout = "metric_a: inf\nmetric_b: -inf\nmetric_c: 0.42\n"
        metrics = parse_metrics(stdout)
        assert "metric_c" in metrics
        assert "metric_a" not in metrics
        assert "metric_b" not in metrics

    def test_detect_nan_divergence_finds_nan(self) -> None:
        from researchclaw.experiment.sandbox import detect_nan_divergence

        result = detect_nan_divergence("loss: nan\nstep 5 done", "")
        assert result is not None
        assert "NaN" in result or "nan" in result.lower()

    def test_detect_nan_divergence_finds_diverging_loss(self) -> None:
        from researchclaw.experiment.sandbox import detect_nan_divergence

        result = detect_nan_divergence("best_loss: 999.5\n", "")
        assert result is not None
        assert "loss" in result.lower() or "999" in result

    def test_detect_nan_divergence_returns_none_for_clean(self) -> None:
        from researchclaw.experiment.sandbox import detect_nan_divergence

        result = detect_nan_divergence("best_loss: 0.123\naccuracy: 0.95\n", "")
        assert result is None

    def test_runtime_issues_detects_diverging_loss(self) -> None:
        from types import SimpleNamespace

        fake_result = SimpleNamespace(
            metrics={"best_loss": 500.0},
            stdout="best_loss: 500.0\n",
            stderr="",
        )
        issues = rc_executor._detect_runtime_issues(fake_result)
        assert "DIVERGING" in issues or "diverging" in issues.lower()

    def test_compute_budget_includes_nan_guard(self) -> None:
        from researchclaw.prompts import PromptManager

        pm = PromptManager()
        block = pm.block("compute_budget")
        assert "NaN" in block or "nan" in block.lower() or "divergence" in block.lower()


# ── R5-4: Experiment Harness Template Tests ──────────────────────────


class TestExperimentHarness:
    """Test the immutable experiment harness (R5-4)."""

    def test_harness_should_stop(self) -> None:
        from researchclaw.experiment.harness_template import ExperimentHarness

        h = ExperimentHarness(time_budget=1)
        assert not h.should_stop()  # Just created, not at 80% yet
        import time
        time.sleep(0.9)
        assert h.should_stop()  # Should be past 80% of 1s

    def test_harness_report_metric(self, capsys: pytest.CaptureFixture[str]) -> None:
        from researchclaw.experiment.harness_template import ExperimentHarness

        h = ExperimentHarness(time_budget=60)
        h.report_metric("best_loss", 0.123)
        captured = capsys.readouterr()
        assert "best_loss: 0.123" in captured.out
        assert h._metrics["best_loss"] == 0.123

    def test_harness_rejects_nan(self, capsys: pytest.CaptureFixture[str]) -> None:
        from researchclaw.experiment.harness_template import ExperimentHarness

        h = ExperimentHarness(time_budget=60)
        h.report_metric("bad", float("nan"))
        captured = capsys.readouterr()
        assert "bad" not in h._metrics
        assert "non-finite" in captured.err.lower() or "WARNING" in captured.err

    def test_harness_rejects_inf(self, capsys: pytest.CaptureFixture[str]) -> None:
        from researchclaw.experiment.harness_template import ExperimentHarness

        h = ExperimentHarness(time_budget=60)
        h.report_metric("bad", float("inf"))
        assert "bad" not in h._metrics

    def test_harness_finalize(self, tmp_path: Path) -> None:
        import os
        from researchclaw.experiment.harness_template import ExperimentHarness

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            h = ExperimentHarness(time_budget=60)
            h.report_metric("accuracy", 0.95)
            h.report_metric("loss", 0.05)
            h.log_result({"condition": "A", "value": 1.0})
            h.finalize()

            results = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
            assert results["metrics"]["accuracy"] == 0.95
            assert results["metrics"]["loss"] == 0.05
            assert len(results["results"]) == 1
        finally:
            os.chdir(old_cwd)

    def test_harness_progress(self) -> None:
        from researchclaw.experiment.harness_template import ExperimentHarness

        h = ExperimentHarness(time_budget=1000)
        assert h.progress < 0.01  # Just started
        assert 0.0 <= h.progress <= 1.0

    def test_harness_injected_into_sandbox(self, tmp_path: Path) -> None:
        import sys
        from researchclaw.config import SandboxConfig
        from researchclaw.experiment.sandbox import ExperimentSandbox

        config = SandboxConfig(python_path=sys.executable)
        sandbox = ExperimentSandbox(config, tmp_path / "sandbox")

        # Create a project dir
        project = tmp_path / "project"
        project.mkdir()
        (project / "main.py").write_text("print('test: 1.0')\n", encoding="utf-8")

        sandbox.run_project(project, timeout_sec=5)

        # Check that harness was injected (BUG-DA8-06: dir is now _project_{N})
        project_dirs = list((tmp_path / "sandbox").glob("_project_*"))
        assert project_dirs, "No _project_N directory found"
        harness_path = project_dirs[0] / "experiment_harness.py"
        assert harness_path.exists()
        content = harness_path.read_text(encoding="utf-8")
        assert "ExperimentHarness" in content

    def test_harness_not_overwritten_by_project(self, tmp_path: Path) -> None:
        import sys
        from researchclaw.config import SandboxConfig
        from researchclaw.experiment.sandbox import ExperimentSandbox

        config = SandboxConfig(python_path=sys.executable)
        sandbox = ExperimentSandbox(config, tmp_path / "sandbox")

        # Create a project with a fake experiment_harness.py
        project = tmp_path / "project"
        project.mkdir()
        (project / "main.py").write_text("print('test: 1.0')\n", encoding="utf-8")
        (project / "experiment_harness.py").write_text("# FAKE HARNESS", encoding="utf-8")

        sandbox.run_project(project, timeout_sec=5)

        # The real harness should be there, not the fake one (BUG-DA8-06)
        project_dirs = list((tmp_path / "sandbox").glob("_project_*"))
        assert project_dirs
        harness_path = project_dirs[0] / "experiment_harness.py"
        content = harness_path.read_text(encoding="utf-8")
        assert "ExperimentHarness" in content
        assert "FAKE HARNESS" not in content

    def test_prompt_mentions_harness(self) -> None:
        from researchclaw.prompts import PromptManager

        pm = PromptManager()
        block = pm.block("compute_budget")
        assert "experiment_harness" in block or "ExperimentHarness" in block


# ── R5-5: Stdout Truncation Tests ────────────────────────────────────


class TestStdoutTruncation:
    """Test stdout/stderr truncation in refine run summaries (R5-5)."""

    def test_long_stdout_truncated_in_refine(
        self, tmp_path: Path, run_dir: Path, adapters: AdapterBundle
    ) -> None:
        # Create a run with very long stdout
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        long_stdout = "\n".join(f"step {i}: loss={0.5 - i * 0.001:.6f}" for i in range(200))
        (runs_dir / "run-1.json").write_text(
            json.dumps({
                "run_id": "run-1",
                "status": "completed",
                "metrics": {"best_loss": 0.3},
                "stdout": long_stdout,
            }),
            encoding="utf-8",
        )

        exp_dir = run_dir / "stage-11" / "experiment"
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "main.py").write_text("print('best_loss: 0.3')\n", encoding="utf-8")

        stage_dir = run_dir / "stage-13"
        stage_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {"topic": "test", "domains": ["ml"],
                         "daily_paper_count": 2, "quality_threshold": 8.2},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local", "on_stage_start": True,
                              "on_stage_fail": False, "on_gate_required": True},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": True, "use_message": True},
            "llm": {"provider": "openai-compatible", "base_url": "http://localhost:1234/v1",
                    "api_key_env": "RC_TEST_KEY", "api_key": "inline-test-key",
                    "primary_model": "fake-model", "fallback_models": []},
            "security": {"hitl_required_stages": [5, 9, 20]},
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 30,
                "max_iterations": 1,
                "metric_key": "best_loss",
                "metric_direction": "minimize",
            },
        }
        cfg = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

        llm = FakeLLMClient("```python\nprint('best_loss: 0.3')\n```")
        rc_executor._execute_iterative_refine(
            stage_dir, run_dir, cfg, adapters, llm=llm
        )

        # The LLM should have received truncated stdout, not all 200 lines
        assert len(llm.calls) >= 1
        user_msg = llm.calls[0][-1]["content"]
        # Should contain truncation indicator
        assert "truncated" in user_msg or len(user_msg) < len(long_stdout)


# ===================================================================
# R6 Tests — Post-E2E Failure Analysis Fixes
# ===================================================================


class TestNoImproveStreakFix:
    """R6-1: no_improve_streak should only count iterations with real metrics."""

    def test_empty_metrics_dont_increment_streak(
        self, tmp_path: Path, run_dir: Path, adapters: AdapterBundle
    ) -> None:
        """When metrics are empty (None), the streak should NOT increment."""
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({
                "run_id": "run-1",
                "status": "failed",
                "metrics": {},
                "stdout": "FAIL: NaN/divergence detected",
            }),
            encoding="utf-8",
        )
        exp_dir = run_dir / "stage-11" / "experiment"
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "main.py").write_text("print('hello')\n", encoding="utf-8")

        stage_dir = run_dir / "stage-13"
        stage_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {"topic": "test", "domains": ["ml"],
                         "daily_paper_count": 2, "quality_threshold": 8.2},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local", "on_stage_start": True,
                              "on_stage_fail": False, "on_gate_required": True},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": True, "use_message": True},
            "llm": {"provider": "openai-compatible", "base_url": "http://localhost:1234/v1",
                    "api_key_env": "RC_TEST_KEY", "api_key": "inline-test-key",
                    "primary_model": "fake-model", "fallback_models": []},
            "security": {"hitl_required_stages": [5, 9, 20]},
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 30,
                "max_iterations": 4,
                "metric_key": "primary_metric",
                "metric_direction": "minimize",
            },
        }
        cfg = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

        # LLM returns code that won't produce metrics in simulated mode
        llm = FakeLLMClient("```python\nprint('no metrics here')\n```")
        result = rc_executor._execute_iterative_refine(
            stage_dir, run_dir, cfg, adapters, llm=llm
        )

        # Should abort after 3 consecutive no-metrics iterations
        log_path = stage_dir / "refinement_log.json"
        log_data = json.loads(log_path.read_text())
        # consecutive_no_metrics triggers early abort after 3 iterations
        assert len(log_data["iterations"]) == 3
        assert log_data.get("stop_reason") == "consecutive_no_metrics"


class TestStdoutFailureDetection:
    """R6-2: Detect stdout failure signals even when exit code is 0."""

    def test_fail_signal_in_stdout_marks_failed(self, tmp_path: Path) -> None:
        """Exit code 0 + 'FAIL:' in stdout + no metrics → status='failed'."""
        from researchclaw.pipeline.executor import _execute_experiment_run

        # Create necessary structure
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "stage-10").mkdir()
        exp_dir = run_dir / "stage-10" / "experiment"
        exp_dir.mkdir()
        # Simple code that prints FAIL but exits 0
        (exp_dir / "main.py").write_text(
            "print('FAIL: NaN/divergence detected')\n", encoding="utf-8"
        )
        (run_dir / "stage-11").mkdir()
        (run_dir / "stage-11" / "schedule.json").write_text("{}", encoding="utf-8")

        stage_dir = run_dir / "stage-12"
        stage_dir.mkdir()

        data = {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {"topic": "test", "domains": ["ml"],
                         "daily_paper_count": 2, "quality_threshold": 8.2},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local", "on_stage_start": True,
                              "on_stage_fail": False, "on_gate_required": True},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": True, "use_message": True},
            "llm": {"provider": "openai-compatible", "base_url": "http://localhost:1234/v1",
                    "api_key_env": "RC_TEST_KEY", "api_key": "inline-test-key",
                    "primary_model": "fake-model", "fallback_models": []},
            "security": {"hitl_required_stages": [5, 9, 20]},
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 30,
                "max_iterations": 1,
                "metric_key": "primary_metric",
                "metric_direction": "minimize",
                "sandbox": {
                    "python_path": sys.executable,
                    "gpu_required": False,
                    "max_memory_mb": 512,
                    "allowed_imports": ["json"],
                },
            },
        }
        cfg = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)
        adapters = AdapterBundle()

        result = _execute_experiment_run(
            stage_dir, run_dir, cfg, adapters
        )

        # Check the run payload
        runs_dir = stage_dir / "runs"
        run_file = runs_dir / "run-1.json"
        assert run_file.exists()
        payload = json.loads(run_file.read_text())
        assert payload["status"] == "failed"

    def test_clean_exit_no_fail_signal_marks_completed(self, tmp_path: Path) -> None:
        """Exit code 0 + valid metrics + no FAIL signal → status='completed'."""
        from researchclaw.pipeline.executor import _execute_experiment_run

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "stage-10").mkdir()
        exp_dir = run_dir / "stage-10" / "experiment"
        exp_dir.mkdir()
        (exp_dir / "main.py").write_text(
            "print('primary_metric: 0.95')\n", encoding="utf-8"
        )
        (run_dir / "stage-11").mkdir()
        (run_dir / "stage-11" / "schedule.json").write_text("{}", encoding="utf-8")

        stage_dir = run_dir / "stage-12"
        stage_dir.mkdir()

        data = {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {"topic": "test", "domains": ["ml"],
                         "daily_paper_count": 2, "quality_threshold": 8.2},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local", "on_stage_start": True,
                              "on_stage_fail": False, "on_gate_required": True},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": True, "use_message": True},
            "llm": {"provider": "openai-compatible", "base_url": "http://localhost:1234/v1",
                    "api_key_env": "RC_TEST_KEY", "api_key": "inline-test-key",
                    "primary_model": "fake-model", "fallback_models": []},
            "security": {"hitl_required_stages": [5, 9, 20]},
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 30,
                "max_iterations": 1,
                "metric_key": "primary_metric",
                "metric_direction": "minimize",
                "sandbox": {
                    "python_path": sys.executable,
                    "gpu_required": False,
                    "max_memory_mb": 512,
                    "allowed_imports": ["json"],
                },
            },
        }
        cfg = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)
        adapters = AdapterBundle()

        result = _execute_experiment_run(
            stage_dir, run_dir, cfg, adapters
        )

        runs_dir = stage_dir / "runs"
        payload = json.loads((runs_dir / "run-1.json").read_text())
        assert payload["status"] == "completed"


class TestMetricValUndefined:
    """R6-3: metric_val should be initialized to None before conditional block."""

    def test_metric_val_initialized_before_use(self) -> None:
        """Verify the code pattern: metric_val = None before if block."""
        import inspect
        source = inspect.getsource(rc_executor._execute_iterative_refine)
        # Find that metric_val = None appears before the sandbox block
        init_pos = source.find("metric_val = None")
        sandbox_pos = source.find("if validation.ok and config.experiment.mode")
        assert init_pos != -1, "metric_val = None not found"
        assert sandbox_pos != -1, "sandbox block not found"
        assert init_pos < sandbox_pos, "metric_val = None should come before sandbox block"


class TestConsecutiveEmptyMetrics:
    """R6-4: Pipeline should detect consecutive empty-metrics REFINE cycles."""

    def test_detects_consecutive_empty(self, tmp_path: Path) -> None:
        """Two cycles with empty metrics should return True."""
        from researchclaw.pipeline.runner import _consecutive_empty_metrics

        run_dir = tmp_path / "run"
        # Current cycle (stage-14)
        s14 = run_dir / "stage-14"
        s14.mkdir(parents=True)
        (s14 / "experiment_summary.json").write_text(json.dumps({
            "metrics_summary": {},
            "best_run": {"metrics": {}},
        }))
        # Previous cycle (stage-14_v1)
        s14v1 = run_dir / "stage-14_v1"
        s14v1.mkdir(parents=True)
        (s14v1 / "experiment_summary.json").write_text(json.dumps({
            "metrics_summary": {},
            "best_run": {"metrics": {}},
        }))

        assert _consecutive_empty_metrics(run_dir, pivot_count=1) is True

    def test_not_empty_when_metrics_exist(self, tmp_path: Path) -> None:
        """If any cycle has real metrics, return False."""
        from researchclaw.pipeline.runner import _consecutive_empty_metrics

        run_dir = tmp_path / "run"
        s14 = run_dir / "stage-14"
        s14.mkdir(parents=True)
        (s14 / "experiment_summary.json").write_text(json.dumps({
            "metrics_summary": {},
            "best_run": {"metrics": {"loss": 0.5}},
        }))
        s14v1 = run_dir / "stage-14_v1"
        s14v1.mkdir(parents=True)
        (s14v1 / "experiment_summary.json").write_text(json.dumps({
            "metrics_summary": {},
            "best_run": {"metrics": {}},
        }))

        assert _consecutive_empty_metrics(run_dir, pivot_count=1) is False

    def test_false_when_no_previous_cycle(self, tmp_path: Path) -> None:
        """First cycle (no v1) should return False."""
        from researchclaw.pipeline.runner import _consecutive_empty_metrics

        run_dir = tmp_path / "run"
        s14 = run_dir / "stage-14"
        s14.mkdir(parents=True)
        (s14 / "experiment_summary.json").write_text(json.dumps({
            "metrics_summary": {},
            "best_run": {"metrics": {}},
        }))

        # No stage-14_v1 exists
        assert _consecutive_empty_metrics(run_dir, pivot_count=1) is False


# ===================================================================
# R7 Tests — Experiment-Paper Quality Alignment
# ===================================================================


class TestMultiConditionEnforcement:
    """R7-1: Code generation prompt must enforce multi-condition experiments."""

    def test_code_generation_prompt_has_multi_condition_block(self) -> None:
        """The code_generation prompt should contain multi-condition instructions."""
        from researchclaw.prompts import PromptManager
        pm = PromptManager()
        sp = pm.for_stage(
            "code_generation",
            topic="test topic",
            metric="primary_metric",
            pkg_hint="",
            exp_plan="conditions:\n  - echo_chamber\n  - bridge_building\n  - random",
        )
        assert "MULTI-CONDITION REQUIREMENT" in sp.user
        assert "condition=" in sp.user
        assert "SUMMARY" in sp.user

    def test_multi_condition_labels_required(self) -> None:
        """Prompt must mention per-condition labeled output format."""
        from researchclaw.prompts import PromptManager
        pm = PromptManager()
        sp = pm.for_stage(
            "code_generation",
            topic="test",
            metric="loss",
            pkg_hint="",
            exp_plan="treatments: [A, B, C]",
        )
        assert "condition=<name>" in sp.user


class TestEvidenceBoundedWriting:
    """R7-2: Paper draft prompt must enforce evidence-bounded claims."""

    def test_paper_draft_has_evidence_bounding_rules(self) -> None:
        """System prompt should contain evidence-bounding rules."""
        from researchclaw.prompts import PromptManager
        pm = PromptManager()
        sp = pm.for_stage(
            "paper_draft",
            preamble="test preamble",
            topic_constraint="",
            exp_metrics_instruction="",
            citation_instruction="",
            outline="# Outline",
        )
        assert "EVIDENCE-BOUNDING RULES" in sp.system
        assert "title" in sp.system.lower()
        assert "causal claim" in sp.system.lower() or "causal claims" in sp.system.lower()

    def test_hedging_language_guidance(self) -> None:
        """Should suggest hedged alternatives like 'Toward...' for partial data."""
        from researchclaw.prompts import PromptManager
        pm = PromptManager()
        sp = pm.for_stage(
            "paper_draft",
            preamble="",
            topic_constraint="",
            exp_metrics_instruction="",
            citation_instruction="",
            outline="",
        )
        assert "Toward" in sp.system or "Investigating" in sp.system


class TestConditionCoverageDetection:
    """R7-3: REFINE should detect condition coverage gaps."""

    def test_coverage_hint_injected_when_no_labels(
        self, tmp_path: Path, run_dir: Path, adapters: AdapterBundle
    ) -> None:
        """If stdout has no 'condition=' labels, a coverage hint should be injected."""
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({
                "run_id": "run-1",
                "status": "completed",
                "metrics": {"primary_metric": 0.5},
                "stdout": "primary_metric: 0.5\nprimary_metric: 0.3\n",
            }),
            encoding="utf-8",
        )

        exp_plan_dir = run_dir / "stage-09"
        exp_plan_dir.mkdir(parents=True, exist_ok=True)
        (exp_plan_dir / "exp_plan.yaml").write_text(
            "conditions:\n  - echo_chamber\n  - bridge_building\n  - random\n",
            encoding="utf-8",
        )

        exp_dir = run_dir / "stage-11" / "experiment"
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "main.py").write_text("print('primary_metric: 0.5')\n", encoding="utf-8")

        stage_dir = run_dir / "stage-13"
        stage_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {"topic": "test", "domains": ["ml"],
                         "daily_paper_count": 2, "quality_threshold": 8.2},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local", "on_stage_start": True,
                              "on_stage_fail": False, "on_gate_required": True},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": True, "use_message": True},
            "llm": {"provider": "openai-compatible", "base_url": "http://localhost:1234/v1",
                    "api_key_env": "RC_TEST_KEY", "api_key": "inline-test-key",
                    "primary_model": "fake-model", "fallback_models": []},
            "security": {"hitl_required_stages": [5, 9, 20]},
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 30,
                "max_iterations": 1,
                "metric_key": "primary_metric",
                "metric_direction": "minimize",
            },
        }
        cfg = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

        llm = FakeLLMClient("```python\nprint('primary_metric: 0.3')\n```")
        rc_executor._execute_iterative_refine(
            stage_dir, run_dir, cfg, adapters, llm=llm
        )

        assert len(llm.calls) >= 1
        user_msg = llm.calls[0][-1]["content"]
        assert "CONDITION COVERAGE GAP" in user_msg

    def test_no_hint_when_labels_present(
        self, tmp_path: Path, run_dir: Path, adapters: AdapterBundle
    ) -> None:
        """If stdout already has 'condition=' labels, no hint should be injected."""
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({
                "run_id": "run-1",
                "status": "completed",
                "metrics": {"primary_metric": 0.5},
                "stdout": "condition=echo primary_metric: 0.5\ncondition=bridge primary_metric: 0.3\n",
            }),
            encoding="utf-8",
        )

        exp_plan_dir = run_dir / "stage-09"
        exp_plan_dir.mkdir(parents=True, exist_ok=True)
        (exp_plan_dir / "exp_plan.yaml").write_text(
            "conditions:\n  - echo\n  - bridge\n",
            encoding="utf-8",
        )

        exp_dir = run_dir / "stage-11" / "experiment"
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "main.py").write_text("print('primary_metric: 0.5')\n", encoding="utf-8")

        stage_dir = run_dir / "stage-13"
        stage_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {"topic": "test", "domains": ["ml"],
                         "daily_paper_count": 2, "quality_threshold": 8.2},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local", "on_stage_start": True,
                              "on_stage_fail": False, "on_gate_required": True},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": True, "use_message": True},
            "llm": {"provider": "openai-compatible", "base_url": "http://localhost:1234/v1",
                    "api_key_env": "RC_TEST_KEY", "api_key": "inline-test-key",
                    "primary_model": "fake-model", "fallback_models": []},
            "security": {"hitl_required_stages": [5, 9, 20]},
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 30,
                "max_iterations": 1,
                "metric_key": "primary_metric",
                "metric_direction": "minimize",
            },
        }
        cfg = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

        llm = FakeLLMClient("```python\nprint('primary_metric: 0.3')\n```")
        rc_executor._execute_iterative_refine(
            stage_dir, run_dir, cfg, adapters, llm=llm
        )

        assert len(llm.calls) >= 1
        user_msg = llm.calls[0][-1]["content"]
        assert "CONDITION COVERAGE GAP" not in user_msg


# ===================================================================
# R8 Tests — AutoBench Round 1 Fixes
# ===================================================================


class TestBreadthFirstPrompt:
    """R8-1: Code generation prompt should require breadth-first condition ordering."""

    def test_breadth_first_in_code_generation(self) -> None:
        from researchclaw.prompts import PromptManager
        pm = PromptManager()
        sp = pm.for_stage(
            "code_generation",
            topic="test",
            metric="primary_metric",
            pkg_hint="",
            exp_plan="conditions: [A, B, C]",
        )
        assert "BREADTH-FIRST" in sp.user
        assert "ONE representative" in sp.user


class TestRefineFilePreservation:
    """R8-2: Refine should preserve supporting files when LLM only returns main.py."""

    def test_supporting_files_preserved_in_refine(
        self, tmp_path: Path, run_dir: Path, adapters: AdapterBundle
    ) -> None:
        """When LLM returns only main.py, other project files should be preserved."""
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({
                "run_id": "run-1",
                "status": "completed",
                "metrics": {"primary_metric": 0.5},
                "stdout": "primary_metric: 0.5",
            }),
            encoding="utf-8",
        )

        # Multi-file experiment project
        exp_dir = run_dir / "stage-11" / "experiment"
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "main.py").write_text("from helpers import foo\nprint('primary_metric: 0.5')\n")
        (exp_dir / "helpers.py").write_text("def foo(): return 42\n")
        (exp_dir / "utils.py").write_text("def bar(): return 99\n")

        stage_dir = run_dir / "stage-13"
        stage_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {"topic": "test", "domains": ["ml"],
                         "daily_paper_count": 2, "quality_threshold": 8.2},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local", "on_stage_start": True,
                              "on_stage_fail": False, "on_gate_required": True},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": True, "use_message": True},
            "llm": {"provider": "openai-compatible", "base_url": "http://localhost:1234/v1",
                    "api_key_env": "RC_TEST_KEY", "api_key": "inline-test-key",
                    "primary_model": "fake-model", "fallback_models": []},
            "security": {"hitl_required_stages": [5, 9, 20]},
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 30,
                "max_iterations": 1,
                "metric_key": "primary_metric",
                "metric_direction": "minimize",
            },
        }
        cfg = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

        # LLM returns only main.py in multi-file format
        llm = FakeLLMClient("```filename:main.py\nfrom helpers import foo\nprint('primary_metric: 0.3')\n```")
        rc_executor._execute_iterative_refine(
            stage_dir, run_dir, cfg, adapters, llm=llm
        )

        # Check that experiment_v1 has ALL files, not just main.py
        v1_dir = stage_dir / "experiment_v1"
        assert v1_dir.exists()
        v1_files = {f.name for f in v1_dir.glob("*.py")}
        assert "main.py" in v1_files
        assert "helpers.py" in v1_files, "Supporting file helpers.py should be preserved"
        assert "utils.py" in v1_files, "Supporting file utils.py should be preserved"


# ===================================================================
# R9 Tests — AutoBench Round 2 Fixes
# ===================================================================


class TestCodeGenTopicNeutral:
    """R9-1: Code generation prompt should be topic-neutral, not optimization-biased."""

    def test_no_gradient_descent_bias(self) -> None:
        from researchclaw.prompts import PromptManager
        pm = PromptManager()
        sp = pm.for_stage(
            "code_generation",
            topic="multi-agent simulation",
            metric="primary_metric",
            pkg_hint="",
            exp_plan="conditions: [L1, L2, L3, L4]",
        )
        # Should NOT contain optimization-specific examples as recommended approaches
        assert "Adam" not in sp.user
        assert "SGD" not in sp.user
        assert "Rosenbrock" not in sp.user
        # "gradient descent" may appear as anti-pattern warning but not as example
        assert "e.g., gradient descent" not in sp.user

    def test_topic_relevant_guidance(self) -> None:
        from researchclaw.prompts import PromptManager
        pm = PromptManager()
        sp = pm.for_stage(
            "code_generation",
            topic="multi-agent simulation",
            metric="primary_metric",
            pkg_hint="",
            exp_plan="conditions: [L1, L2, L3, L4]",
        )
        # Should contain generic guidance that works for any topic
        assert "simulation" in sp.user.lower() or "appropriate" in sp.user.lower()
        assert "ACTUAL experiment" in sp.user or "relevant to the TOPIC" in sp.user


class TestRefineTopicAlignment:
    """R9-2: Refine prompt should include topic-code alignment check."""

    def test_topic_alignment_in_refine_prompt(self) -> None:
        from researchclaw.prompts import PromptManager
        pm = PromptManager()
        sp = pm.sub_prompt(
            "iterative_improve",
            metric_key="primary_metric",
            metric_direction="maximize",
            files_context="# main.py\nprint('hello')",
            run_summaries="{}",
            condition_coverage_hint="",
            topic="multi-agent diversity scaling",
            exp_plan_anchor="",
        )
        assert "EXPERIMENT PLAN ANCHOR" in sp.user
        assert "multi-agent diversity scaling" in sp.user
        assert "NEVER rename" in sp.user


# =====================================================================
# _validate_draft_quality tests
# =====================================================================


def _make_prose(word_count: int) -> str:  # noqa: E302
    """Generate flowing prose text of approximately *word_count* words."""
    sentence = (
        "This is a flowing academic prose sentence "
        "that demonstrates our research findings. "
    )
    words_per = len(sentence.split())
    return sentence * (word_count // words_per + 1)


def _make_bullets(word_count: int) -> str:
    """Generate bullet-point text of approximately *word_count* words."""
    line = "- This is a bullet point about a research finding\n"
    words_per = len(line.split())
    return line * (word_count // words_per + 1)


def _make_comparative_prose(word_count: int) -> str:
    """Generate related-work style prose with comparative language."""
    sentence = (
        "Unlike prior work that focuses on simple baselines, "
        "our approach differs by incorporating novel techniques. "
        "In contrast to existing methods, we address key limitations. "
        "However, while previous approaches rely on heuristics, "
        "our method provides theoretical guarantees. "
    )
    words_per = len(sentence.split())
    return sentence * (word_count // words_per + 1)


def _make_results_prose(word_count: int) -> str:
    """Generate results prose with statistical measures."""
    sentence = (
        "Our method achieves 85.3 ± 1.2 accuracy averaged over 5 seeds. "
        "The baseline comparison yields a p-value of 0.003, confirming "
        "statistical significance with 95% confidence interval. "
    )
    words_per = len(sentence.split())
    return sentence * (word_count // words_per + 1)


def _build_draft(**section_overrides: str) -> str:
    """Build a paper draft with default prose sections."""
    defaults = {
        "Abstract": _make_prose(200),
        "Introduction": _make_prose(900),
        "Related Work": _make_comparative_prose(700),
        "Method": _make_prose(1200),
        "Experiments": _make_prose(1000),
        "Results": _make_results_prose(700),
        "Discussion": _make_prose(500),
        "Limitations": _make_prose(250),
        "Conclusion": _make_prose(250),
    }
    defaults.update(section_overrides)
    parts = ["# My Research Title\n"]
    for heading, body in defaults.items():
        parts.append(f"# {heading}\n{body}\n")
    return "\n".join(parts)


class TestValidateDraftQuality:
    """Tests for _validate_draft_quality()."""

    def test_short_section_triggers_warning(self) -> None:
        """Short Method section triggers expand warning."""
        draft = _build_draft(Method=_make_prose(200))
        result = rc_executor._validate_draft_quality(draft)
        assert any("Method" in w for w in result["overall_warnings"])
        assert any("EXPAND" in d or "Expand" in d
                    for d in result["revision_directives"])

    def test_bullet_density_triggers_warning(self) -> None:
        """Bullet-heavy Method section triggers rewrite warning."""
        draft = _build_draft(Method=_make_bullets(1200))
        result = rc_executor._validate_draft_quality(draft)
        assert any(
            "bullet" in w.lower() or "density" in w.lower()
            for w in result["overall_warnings"]
        )
        assert any("REWRITE" in d for d in result["revision_directives"])

    def test_clean_draft_no_warnings(self) -> None:
        """Balanced prose draft produces zero warnings."""
        draft = _build_draft()
        result = rc_executor._validate_draft_quality(draft)
        assert len(result["overall_warnings"]) == 0
        assert len(result["revision_directives"]) == 0

    def test_balance_warning(self) -> None:
        """Large imbalance between sections triggers balance warning."""
        draft = _build_draft(
            Introduction=_make_prose(1500),
            Results=_make_prose(100),
        )
        result = rc_executor._validate_draft_quality(draft)
        bal = [w for w in result["overall_warnings"]
               if "imbalance" in w.lower()]
        assert len(bal) >= 1, (
            f"Expected balance warning, got: {result['overall_warnings']}"
        )

    def test_writes_json_to_stage_dir(self, tmp_path: Path) -> None:
        """Quality report is written as draft_quality.json."""
        draft = _build_draft(Method=_make_prose(200))
        rc_executor._validate_draft_quality(draft, stage_dir=tmp_path)
        assert (tmp_path / "draft_quality.json").exists()
        data = json.loads(
            (tmp_path / "draft_quality.json").read_text()
        )
        assert "section_analysis" in data
        assert "overall_warnings" in data
        assert "revision_directives" in data
