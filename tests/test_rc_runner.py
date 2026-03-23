# pyright: reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnusedCallResult=false, reportAttributeAccessIssue=false, reportUnknownLambdaType=false
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.pipeline import runner as rc_runner
from researchclaw.pipeline.executor import StageResult
from researchclaw.pipeline.stages import STAGE_SEQUENCE, Stage, StageStatus


@pytest.fixture()
def rc_config(tmp_path: Path) -> RCConfig:
    data = {
        "project": {"name": "rc-runner-test", "mode": "docs-first"},
        "research": {"topic": "pipeline testing"},
        "runtime": {"timezone": "UTC"},
        "notifications": {"channel": "local"},
        "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
        "openclaw_bridge": {},
        "llm": {
            "provider": "openai-compatible",
            "base_url": "http://localhost:1234/v1",
            "api_key_env": "RC_TEST_KEY",
            "api_key": "inline",
        },
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


def _done(stage: Stage, artifacts: tuple[str, ...] = ("out.md",)) -> StageResult:
    return StageResult(stage=stage, status=StageStatus.DONE, artifacts=artifacts)


def _failed(stage: Stage, msg: str = "boom") -> StageResult:
    return StageResult(stage=stage, status=StageStatus.FAILED, artifacts=(), error=msg)


def _blocked(stage: Stage) -> StageResult:
    return StageResult(
        stage=stage,
        status=StageStatus.BLOCKED_APPROVAL,
        artifacts=("gate.md",),
        decision="block",
    )


def test_execute_pipeline_runs_stages_in_sequence(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    seen: list[Stage] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        seen.append(stage)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    results = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-seq",
        config=rc_config,
        adapters=adapters,
    )
    assert seen == list(STAGE_SEQUENCE)
    assert len(results) == 23
    assert all(r.status == StageStatus.DONE for r in results)


def test_execute_pipeline_stops_on_failed_stage(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    fail_stage = Stage.SEARCH_STRATEGY

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        if stage == fail_stage:
            return _failed(stage, "forced failure")
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    results = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-fail",
        config=rc_config,
        adapters=adapters,
    )
    assert results[-1].stage == fail_stage
    assert results[-1].status == StageStatus.FAILED
    assert len(results) == int(fail_stage)


def test_execute_pipeline_stops_on_gate_when_stop_on_gate_enabled(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    gate_stage = Stage.LITERATURE_SCREEN

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        if stage == gate_stage:
            return _blocked(stage)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    results = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-gate-stop",
        config=rc_config,
        adapters=adapters,
        stop_on_gate=True,
    )
    assert results[-1].stage == gate_stage
    assert results[-1].status == StageStatus.BLOCKED_APPROVAL
    assert len(results) == int(gate_stage)


def test_execute_pipeline_continues_after_gate_when_stop_on_gate_disabled(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    gate_stage = Stage.LITERATURE_SCREEN

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        if stage == gate_stage:
            return _blocked(stage)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    results = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-gate-continue",
        config=rc_config,
        adapters=adapters,
        stop_on_gate=False,
    )
    assert len(results) == 23
    assert any(item.status == StageStatus.BLOCKED_APPROVAL for item in results)


def test_execute_pipeline_writes_pipeline_summary_json(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-summary",
        config=rc_config,
        adapters=adapters,
    )
    summary_path = run_dir / "pipeline_summary.json"
    assert summary_path.exists()


def test_pipeline_summary_has_expected_fields_and_values(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        if stage == Stage.LITERATURE_SCREEN:
            return _blocked(stage)
        if stage == Stage.HYPOTHESIS_GEN:
            return _failed(stage)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    results = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-summary-fields",
        config=rc_config,
        adapters=adapters,
    )
    summary = cast(
        dict[str, Any],
        json.loads((run_dir / "pipeline_summary.json").read_text(encoding="utf-8")),
    )
    assert summary["run_id"] == "run-summary-fields"
    assert summary["stages_executed"] == len(results)
    assert summary["stages_done"] == sum(
        1 for r in results if r.status == StageStatus.DONE
    )
    assert summary["stages_blocked"] == 1
    assert summary["stages_failed"] == 1
    assert summary["from_stage"] == 1
    assert summary["final_stage"] == int(Stage.HYPOTHESIS_GEN)
    assert summary["final_status"] == "failed"
    assert "generated" in summary


def test_execute_pipeline_from_stage_skips_earlier_stages(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    seen: list[Stage] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        seen.append(stage)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    results = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-from-stage",
        config=rc_config,
        adapters=adapters,
        from_stage=Stage.PAPER_OUTLINE,
    )
    assert seen[0] == Stage.PAPER_OUTLINE
    assert len(seen) == len(STAGE_SEQUENCE) - (int(Stage.PAPER_OUTLINE) - 1)
    assert len(results) == len(seen)


def test_execute_pipeline_writes_kb_entries_when_kb_root_provided(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
    tmp_path: Path,
) -> None:
    calls: list[tuple[int, str, str]] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        stage_dir = run_dir / f"stage-{int(stage):02d}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "out.md").write_text(f"stage {int(stage)}", encoding="utf-8")
        return _done(stage)

    def mock_write_stage_to_kb(
        kb_root: Path,
        stage_id: int,
        stage_name: str,
        run_id: str,
        artifacts: list[str],
        stage_dir: Path,
        **kwargs,
    ):
        _ = kb_root, artifacts, stage_dir, kwargs
        calls.append((stage_id, stage_name, run_id))
        return []

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    monkeypatch.setattr(rc_runner, "write_stage_to_kb", mock_write_stage_to_kb)

    kb_root = tmp_path / "kb-out"
    results = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-kb",
        config=rc_config,
        adapters=adapters,
        kb_root=kb_root,
    )
    assert len(results) == 23
    assert len(calls) == 23
    assert calls[0] == (1, "topic_init", "run-kb")


def test_execute_pipeline_passes_auto_approve_flag_to_execute_stage(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    received: list[bool] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        received.append(kwargs["auto_approve_gates"])
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-auto-approve",
        config=rc_config,
        adapters=adapters,
        auto_approve_gates=True,
    )
    assert received
    assert all(received)


@pytest.mark.parametrize(
    ("stage", "started", "expected"),
    [
        (Stage.TOPIC_INIT, False, True),
        (Stage.PROBLEM_DECOMPOSE, False, False),
        (Stage.PAPER_DRAFT, True, True),
    ],
)
def test_should_start_logic(stage: Stage, started: bool, expected: bool) -> None:
    assert rc_runner._should_start(stage, Stage.TOPIC_INIT, started) is expected


@pytest.mark.parametrize(
    ("results", "expected_status", "expected_final_stage"),
    [
        ([], "no_stages", int(Stage.TOPIC_INIT)),
        ([_done(Stage.TOPIC_INIT)], "done", int(Stage.TOPIC_INIT)),
        (
            [_done(Stage.TOPIC_INIT), _failed(Stage.PROBLEM_DECOMPOSE)],
            "failed",
            int(Stage.PROBLEM_DECOMPOSE),
        ),
    ],
)
def test_build_pipeline_summary_core_fields(
    results, expected_status: str, expected_final_stage: int
) -> None:
    summary = rc_runner._build_pipeline_summary(
        run_id="run-core",
        results=results,
        from_stage=Stage.TOPIC_INIT,
    )
    assert summary["run_id"] == "run-core"
    assert summary["final_status"] == expected_status
    assert summary["final_stage"] == expected_final_stage


def test_pipeline_prints_stage_progress(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mock_results = [
        StageResult(
            stage=Stage.TOPIC_INIT, status=StageStatus.DONE, artifacts=("topic.json",)
        ),
        StageResult(
            stage=Stage.PROBLEM_DECOMPOSE,
            status=StageStatus.DONE,
            artifacts=("tree.json",),
        ),
        StageResult(
            stage=Stage.SEARCH_STRATEGY,
            status=StageStatus.FAILED,
            artifacts=(),
            error="LLM timeout",
        ),
    ]

    call_idx = 0

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = stage, kwargs
        nonlocal call_idx
        idx = call_idx
        call_idx += 1
        return mock_results[min(idx, len(mock_results) - 1)]

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    monkeypatch.setattr(rc_runner, "write_stage_to_kb", lambda *args, **kwargs: [])

    _ = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="rc-test-001",
        config=rc_config,
        adapters=adapters,
    )

    captured = capsys.readouterr()
    assert "TOPIC_INIT — running..." in captured.out
    assert "TOPIC_INIT — done" in captured.out
    assert "SEARCH_STRATEGY — FAILED" in captured.out
    assert "LLM timeout" in captured.out


def test_pipeline_prints_elapsed_time(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mock_result = StageResult(
        stage=Stage.TOPIC_INIT,
        status=StageStatus.DONE,
        artifacts=("topic.json",),
    )
    mock_fail = StageResult(
        stage=Stage.PROBLEM_DECOMPOSE,
        status=StageStatus.FAILED,
        artifacts=(),
        error="test",
    )
    results_iter = iter([mock_result, mock_fail])

    monkeypatch.setattr(
        rc_runner, "execute_stage", lambda *args, **kwargs: next(results_iter)
    )
    monkeypatch.setattr(rc_runner, "write_stage_to_kb", lambda *args, **kwargs: [])

    _ = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="rc-test-002",
        config=rc_config,
        adapters=adapters,
    )

    captured = capsys.readouterr()
    import re

    assert re.search(r"\d+\.\d+s\)", captured.out), (
        f"No elapsed time found in: {captured.out}"
    )


# ── PIVOT/PROCEED/REFINE decision loop tests ──


def _pivot_result(stage: Stage) -> StageResult:
    return StageResult(
        stage=stage, status=StageStatus.DONE, artifacts=("decision.md",), decision="pivot"
    )


def _refine_result(stage: Stage) -> StageResult:
    return StageResult(
        stage=stage, status=StageStatus.DONE, artifacts=("decision.md",), decision="refine"
    )


def test_pivot_decision_triggers_rollback_to_hypothesis_gen(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    seen: list[Stage] = []
    pivot_count = 0

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        seen.append(stage)
        nonlocal pivot_count
        if stage == Stage.RESEARCH_DECISION and pivot_count == 0:
            pivot_count += 1
            return _pivot_result(stage)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    results = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-pivot",
        config=rc_config,
        adapters=adapters,
    )
    # Should have seen HYPOTHESIS_GEN at least twice (original + rollback)
    hyp_gen_count = sum(1 for s in seen if s == Stage.HYPOTHESIS_GEN)
    assert hyp_gen_count >= 2
    # Decision history should be recorded
    history_path = run_dir / "decision_history.json"
    assert history_path.exists()
    history = json.loads(history_path.read_text())
    assert len(history) == 1
    assert history[0]["decision"] == "pivot"


def test_refine_decision_triggers_rollback_to_iterative_refine(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    seen: list[Stage] = []
    refine_count = 0

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        seen.append(stage)
        nonlocal refine_count
        if stage == Stage.RESEARCH_DECISION and refine_count == 0:
            refine_count += 1
            return _refine_result(stage)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    results = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-refine",
        config=rc_config,
        adapters=adapters,
    )
    # Should have seen ITERATIVE_REFINE at least twice
    refine_stage_count = sum(1 for s in seen if s == Stage.ITERATIVE_REFINE)
    assert refine_stage_count >= 2


def test_max_pivot_count_prevents_infinite_loop(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    seen: list[Stage] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        seen.append(stage)
        # Always PIVOT — should be limited by MAX_DECISION_PIVOTS
        if stage == Stage.RESEARCH_DECISION:
            return _pivot_result(stage)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    results = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-max-pivot",
        config=rc_config,
        adapters=adapters,
    )
    # RESEARCH_DECISION should appear at most MAX_DECISION_PIVOTS + 1 times
    from researchclaw.pipeline.stages import MAX_DECISION_PIVOTS
    decision_count = sum(1 for s in seen if s == Stage.RESEARCH_DECISION)
    assert decision_count <= MAX_DECISION_PIVOTS + 1


def test_proceed_decision_does_not_trigger_rollback(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    seen: list[Stage] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        seen.append(stage)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    results = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-proceed",
        config=rc_config,
        adapters=adapters,
    )
    # Should be exactly 23 stages, no rollback
    assert len(seen) == 23
    assert not (run_dir / "decision_history.json").exists()


def test_read_pivot_count_returns_zero_for_no_history(run_dir: Path) -> None:
    assert rc_runner._read_pivot_count(run_dir) == 0


def test_record_decision_history_appends(run_dir: Path) -> None:
    rc_runner._record_decision_history(run_dir, "pivot", Stage.HYPOTHESIS_GEN, 1)
    rc_runner._record_decision_history(run_dir, "refine", Stage.ITERATIVE_REFINE, 2)
    history = json.loads((run_dir / "decision_history.json").read_text())
    assert len(history) == 2
    assert history[0]["decision"] == "pivot"
    assert history[1]["decision"] == "refine"


# ── Deliverables packaging tests ──


def _setup_stage_artifacts(run_dir: Path) -> None:
    """Create typical stage-22 and stage-23 output files for testing."""
    s22 = run_dir / "stage-22"
    s22.mkdir(parents=True, exist_ok=True)
    (s22 / "paper_final.md").write_text("# My Paper\nContent here.", encoding="utf-8")
    (s22 / "paper.tex").write_text("\\documentclass{article}\n\\begin{document}\nHello\n\\end{document}", encoding="utf-8")
    (s22 / "references.bib").write_text("@article{smith2024,\n  title={Test}\n}", encoding="utf-8")
    code_dir = s22 / "code"
    code_dir.mkdir()
    (code_dir / "main.py").write_text("print('hello')", encoding="utf-8")
    (code_dir / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    (code_dir / "README.md").write_text("# Code\n", encoding="utf-8")

    s23 = run_dir / "stage-23"
    s23.mkdir(parents=True, exist_ok=True)
    (s23 / "paper_final_verified.md").write_text("# My Paper (verified)\nContent.", encoding="utf-8")
    (s23 / "references_verified.bib").write_text("@article{smith2024,\n  title={Test}\n}", encoding="utf-8")
    (s23 / "verification_report.json").write_text(
        json.dumps({"summary": {"total": 5, "verified": 4}}), encoding="utf-8"
    )


def test_package_deliverables_collects_all_artifacts(
    run_dir: Path, rc_config: RCConfig
) -> None:
    _setup_stage_artifacts(run_dir)
    dest = rc_runner._package_deliverables(run_dir, "run-pkg-test", rc_config)
    assert dest is not None
    assert dest == run_dir / "deliverables"
    assert (dest / "paper_final.md").exists()
    assert (dest / "paper.tex").exists()
    assert (dest / "references.bib").exists()
    assert (dest / "code" / "main.py").exists()
    assert (dest / "verification_report.json").exists()
    assert (dest / "manifest.json").exists()
    manifest = json.loads((dest / "manifest.json").read_text())
    assert manifest["run_id"] == "run-pkg-test"
    assert "paper_final.md" in manifest["files"]


def test_package_deliverables_prefers_verified_versions(
    run_dir: Path, rc_config: RCConfig
) -> None:
    _setup_stage_artifacts(run_dir)
    rc_runner._package_deliverables(run_dir, "run-verified", rc_config)
    dest = run_dir / "deliverables"
    # Should contain verified content (from stage 23), not base (from stage 22)
    paper = (dest / "paper_final.md").read_text(encoding="utf-8")
    assert "verified" in paper
    bib = (dest / "references.bib").read_text(encoding="utf-8")
    assert "smith2024" in bib


def test_package_deliverables_falls_back_to_stage22(
    run_dir: Path, rc_config: RCConfig
) -> None:
    """When stage 23 outputs are missing, falls back to stage 22 versions."""
    s22 = run_dir / "stage-22"
    s22.mkdir(parents=True, exist_ok=True)
    (s22 / "paper_final.md").write_text("# Base Paper", encoding="utf-8")
    (s22 / "references.bib").write_text("@article{a,title={A}}", encoding="utf-8")

    dest = rc_runner._package_deliverables(run_dir, "run-fallback", rc_config)
    assert dest is not None
    paper = (dest / "paper_final.md").read_text(encoding="utf-8")
    assert "Base Paper" in paper


def test_package_deliverables_returns_none_when_no_stage_artifacts(
    run_dir: Path, tmp_path: Path,
) -> None:
    """Returns None when no stage artifacts exist and no style files found."""
    # Use a config with an unknown conference so style files aren't bundled
    data = {
        "project": {"name": "empty-test", "mode": "docs-first"},
        "research": {"topic": "empty"},
        "runtime": {"timezone": "UTC"},
        "notifications": {"channel": "local"},
        "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
        "openclaw_bridge": {},
        "llm": {
            "provider": "openai-compatible",
            "base_url": "http://localhost:1234/v1",
            "api_key_env": "RC_TEST_KEY",
            "api_key": "inline",
        },
        "export": {"target_conference": "unknown_conf_9999"},
    }
    cfg = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)
    result = rc_runner._package_deliverables(run_dir, "run-empty", cfg)
    assert result is None
    assert not (run_dir / "deliverables").exists()


def test_package_deliverables_includes_style_files(
    run_dir: Path, rc_config: RCConfig
) -> None:
    """Style files (.sty, .bst) for the target conference are bundled."""
    _setup_stage_artifacts(run_dir)
    dest = rc_runner._package_deliverables(run_dir, "run-styles", rc_config)
    assert dest is not None
    # Default config uses neurips_2025 → should have neurips_2025.sty
    assert (dest / "neurips_2025.sty").exists()
    manifest = json.loads((dest / "manifest.json").read_text())
    assert "neurips_2025.sty" in manifest["files"]


# ── Atomic checkpoint write tests ──


def test_write_checkpoint_uses_atomic_rename(run_dir: Path) -> None:
    """Checkpoint must be written via temp file + rename, not direct write"""
    rc_runner._write_checkpoint(run_dir, Stage.TOPIC_INIT, "run-atomic")
    cp = run_dir / "checkpoint.json"
    assert cp.exists()
    data = json.loads(cp.read_text(encoding="utf-8"))
    assert data["last_completed_stage"] == int(Stage.TOPIC_INIT)
    assert data["run_id"] == "run-atomic"


def test_write_checkpoint_leaves_no_temp_files(run_dir: Path) -> None:
    """Atomic write must clean up temp files on success"""
    rc_runner._write_checkpoint(run_dir, Stage.TOPIC_INIT, "run-clean")
    temps = list(run_dir.glob("*.tmp"))
    assert temps == [], f"Leftover temp files: {temps}"


def test_write_checkpoint_preserves_old_on_write_failure(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the temp-file write fails, the existing checkpoint must survive"""
    import builtins

    rc_runner._write_checkpoint(run_dir, Stage.TOPIC_INIT, "run-ok")

    original_open = builtins.open

    def _exploding_open(path, *args, **kwargs):
        # After os.close(fd), _write_checkpoint opens via path string —
        # intercept temp-file opens (checkpoint_*.tmp)
        if isinstance(path, (str, Path)) and "checkpoint_" in str(path):
            raise OSError("disk full")
        if isinstance(path, int):
            raise OSError("disk full")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _exploding_open)
    with pytest.raises(OSError):
        rc_runner._write_checkpoint(run_dir, Stage.PROBLEM_DECOMPOSE, "run-ok")

    # Original checkpoint must be intact
    data = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert data["last_completed_stage"] == int(Stage.TOPIC_INIT)
    # Temp file must be cleaned up
    assert list(run_dir.glob("checkpoint_*.tmp")) == []


def test_write_checkpoint_overwrites_previous(run_dir: Path) -> None:
    """A second checkpoint call must fully replace the first"""
    rc_runner._write_checkpoint(run_dir, Stage.TOPIC_INIT, "run-1")
    rc_runner._write_checkpoint(run_dir, Stage.PROBLEM_DECOMPOSE, "run-1")
    data = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert data["last_completed_stage"] == int(Stage.PROBLEM_DECOMPOSE)
    assert data["last_completed_name"] == Stage.PROBLEM_DECOMPOSE.name


def _degraded(stage: Stage) -> StageResult:
    return StageResult(
        stage=stage,
        status=StageStatus.DONE,
        artifacts=("quality_report.json",),
        decision="degraded",
    )


def test_degraded_quality_gate_continues_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When quality gate returns decision='degraded', pipeline continues to completion."""
    seen: list[Stage] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        seen.append(stage)
        if stage == Stage.QUALITY_GATE:
            return _degraded(stage)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    results = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-degraded",
        config=rc_config,
        adapters=adapters,
    )
    # All 23 stages should execute (not stopped at quality gate)
    assert len(results) == 23
    assert seen == list(STAGE_SEQUENCE)
    # Quality gate result should have decision="degraded"
    qg_result = [r for r in results if r.stage == Stage.QUALITY_GATE][0]
    assert qg_result.decision == "degraded"
    assert qg_result.status == StageStatus.DONE
    # Pipeline summary should have degraded=True
    summary = json.loads((run_dir / "pipeline_summary.json").read_text())
    assert summary["degraded"] is True
    # Output should show DEGRADED message
    captured = capsys.readouterr()
    assert "DEGRADED" in captured.out


def test_package_deliverables_called_after_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Deliverables packaging is called at end of execute_pipeline."""
    _setup_stage_artifacts(run_dir)

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-with-deliverables",
        config=rc_config,
        adapters=adapters,
    )
    captured = capsys.readouterr()
    assert "Deliverables packaged" in captured.out
    assert (run_dir / "deliverables" / "manifest.json").exists()


# ---------------------------------------------------------------------------
# BUG-223: _promote_best_stage14 must always write experiment_summary_best.json
# ---------------------------------------------------------------------------

def _make_stage14_summary(run_dir: Path, suffix: str, pm_value: float) -> None:
    """Helper: create a stage-14{suffix}/experiment_summary.json."""
    d = run_dir / f"stage-14{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    data = {
        "metrics_summary": {
            "primary_metric": {"min": pm_value, "max": pm_value, "mean": pm_value, "count": 1}
        },
        "condition_summaries": {"cond_a": {"metrics": {"primary_metric": pm_value}}},
    }
    (d / "experiment_summary.json").write_text(json.dumps(data), encoding="utf-8")


class TestPromoteBestStage14BestJson:
    """BUG-223: experiment_summary_best.json must be written even when
    stage-14/ already has the best result (early-return path)."""

    @pytest.fixture()
    def max_config(self, rc_config: RCConfig) -> RCConfig:
        """Config with metric_direction=maximize (accuracy-like metrics)."""
        object.__setattr__(rc_config.experiment, "metric_direction", "maximize")
        return rc_config

    def test_best_json_written_when_current_is_best(
        self, run_dir: Path, max_config: RCConfig
    ) -> None:
        """stage-14/ already best → should still write best.json."""
        _make_stage14_summary(run_dir, "", 90.0)
        _make_stage14_summary(run_dir, "_v1", 80.0)
        _make_stage14_summary(run_dir, "_v2", 70.0)

        rc_runner._promote_best_stage14(run_dir, max_config)  # type: ignore[attr-defined]

        best_path = run_dir / "experiment_summary_best.json"
        assert best_path.exists(), "experiment_summary_best.json must always be written"
        data = json.loads(best_path.read_text(encoding="utf-8"))
        pm = data["metrics_summary"]["primary_metric"]
        assert pm["mean"] == 90.0

    def test_best_json_written_when_promotion_needed(
        self, run_dir: Path, max_config: RCConfig
    ) -> None:
        """stage-14/ is NOT best → promote + write best.json."""
        _make_stage14_summary(run_dir, "", 70.0)
        _make_stage14_summary(run_dir, "_v1", 95.0)

        rc_runner._promote_best_stage14(run_dir, max_config)  # type: ignore[attr-defined]

        best_path = run_dir / "experiment_summary_best.json"
        assert best_path.exists()
        data = json.loads(best_path.read_text(encoding="utf-8"))
        pm = data["metrics_summary"]["primary_metric"]
        assert pm["mean"] == 95.0

    def test_best_json_written_with_equal_values(
        self, run_dir: Path, max_config: RCConfig
    ) -> None:
        """BUG-223 exact scenario: stage-14 and stage-14_v1 have equal
        metrics, stage-14_v2 is regressed."""
        _make_stage14_summary(run_dir, "", 64.46)
        _make_stage14_summary(run_dir, "_v1", 64.46)
        _make_stage14_summary(run_dir, "_v2", 26.80)

        rc_runner._promote_best_stage14(run_dir, max_config)  # type: ignore[attr-defined]

        best_path = run_dir / "experiment_summary_best.json"
        assert best_path.exists(), "BUG-223: best.json missing when current is tied-best"
        data = json.loads(best_path.read_text(encoding="utf-8"))
        pm = data["metrics_summary"]["primary_metric"]
        assert pm["mean"] == 64.46


class TestPromoteBestStage14AnalysisBest:
    """BUG-225: analysis_best.md must be written from best stage-14 iteration."""

    @pytest.fixture()
    def max_config(self, rc_config: RCConfig) -> RCConfig:
        object.__setattr__(rc_config.experiment, "metric_direction", "maximize")
        return rc_config

    def test_analysis_best_written_from_best_iteration(
        self, run_dir: Path, max_config: RCConfig
    ) -> None:
        """analysis_best.md should come from the best stage-14 iteration."""
        _make_stage14_summary(run_dir, "", 70.0)
        _make_stage14_summary(run_dir, "_v1", 95.0)
        # Write analysis.md in each
        (run_dir / "stage-14" / "analysis.md").write_text("Degenerate analysis", encoding="utf-8")
        (run_dir / "stage-14_v1" / "analysis.md").write_text("Best analysis v1", encoding="utf-8")

        rc_runner._promote_best_stage14(run_dir, max_config)  # type: ignore[attr-defined]

        best_analysis = run_dir / "analysis_best.md"
        assert best_analysis.exists(), "BUG-225: analysis_best.md must be written"
        assert best_analysis.read_text(encoding="utf-8") == "Best analysis v1"

    def test_analysis_best_written_when_current_is_best(
        self, run_dir: Path, max_config: RCConfig
    ) -> None:
        """Even when stage-14 is already best, analysis_best.md should be written."""
        _make_stage14_summary(run_dir, "", 90.0)
        _make_stage14_summary(run_dir, "_v1", 80.0)
        (run_dir / "stage-14" / "analysis.md").write_text("Best analysis current", encoding="utf-8")
        (run_dir / "stage-14_v1" / "analysis.md").write_text("Worse analysis", encoding="utf-8")

        rc_runner._promote_best_stage14(run_dir, max_config)  # type: ignore[attr-defined]

        best_analysis = run_dir / "analysis_best.md"
        assert best_analysis.exists()
        assert best_analysis.read_text(encoding="utf-8") == "Best analysis current"

    def test_no_analysis_best_when_no_analysis_md(
        self, run_dir: Path, max_config: RCConfig
    ) -> None:
        """If best stage-14 has no analysis.md, no analysis_best.md is written."""
        _make_stage14_summary(run_dir, "", 90.0)

        rc_runner._promote_best_stage14(run_dir, max_config)  # type: ignore[attr-defined]

        assert not (run_dir / "analysis_best.md").exists()


class TestPromoteBestStage14DegenerateDetection:
    """BUG-226: Degenerate near-zero metrics must not be promoted as best."""

    def test_degenerate_minimize_skipped(self, run_dir: Path, rc_config: RCConfig) -> None:
        """When minimize, a value 1000x smaller than second-best is degenerate."""
        # metric_direction defaults to "minimize"
        _make_stage14_summary(run_dir, "", 7.26e-8)   # degenerate (broken normalization)
        _make_stage14_summary(run_dir, "_v2", 0.37)   # valid

        rc_runner._promote_best_stage14(run_dir, rc_config)  # type: ignore[attr-defined]

        best_path = run_dir / "experiment_summary_best.json"
        assert best_path.exists()
        data = json.loads(best_path.read_text(encoding="utf-8"))
        pm = data["metrics_summary"]["primary_metric"]
        assert pm["mean"] == 0.37, "Degenerate value should be skipped, valid v2 promoted"

    def test_legitimate_minimize_not_skipped(self, run_dir: Path, rc_config: RCConfig) -> None:
        """When values are within normal range, smaller is legitimately best."""
        _make_stage14_summary(run_dir, "", 0.15)
        _make_stage14_summary(run_dir, "_v1", 0.37)

        rc_runner._promote_best_stage14(run_dir, rc_config)  # type: ignore[attr-defined]

        best_path = run_dir / "experiment_summary_best.json"
        data = json.loads(best_path.read_text(encoding="utf-8"))
        pm = data["metrics_summary"]["primary_metric"]
        assert pm["mean"] == 0.15, "Legitimate lower value should be promoted"

    def test_single_candidate_not_affected(self, run_dir: Path, rc_config: RCConfig) -> None:
        """Single candidate is never skipped regardless of value."""
        _make_stage14_summary(run_dir, "", 1e-10)

        rc_runner._promote_best_stage14(run_dir, rc_config)  # type: ignore[attr-defined]

        best_path = run_dir / "experiment_summary_best.json"
        data = json.loads(best_path.read_text(encoding="utf-8"))
        pm = data["metrics_summary"]["primary_metric"]
        assert pm["mean"] == 1e-10
