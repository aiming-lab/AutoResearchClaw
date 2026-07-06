# pyright: reportPrivateUsage=false
"""Tests for the evolution (self-learning) system."""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from researchclaw.evolution import (
    EvolutionStore,
    LessonCategory,
    LessonEntry,
    TrajectoryStore,
    extract_lessons,
    get_trajectory_signal,
    record_refine_trajectory,
    _classify_error,
    _time_weight,
)


# ── LessonEntry tests ──


class TestLessonEntry:
    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        entry = LessonEntry(
            stage_name="hypothesis_gen",
            stage_num=8,
            category="experiment",
            severity="error",
            description="Code validation failed",
            timestamp="2026-03-10T12:00:00+00:00",
            run_id="run-1",
        )
        data = entry.to_dict()
        restored = LessonEntry.from_dict(data)
        assert restored.stage_name == "hypothesis_gen"
        assert restored.stage_num == 8
        assert restored.category == "experiment"
        assert restored.severity == "error"

    def test_from_dict_handles_missing_fields(self) -> None:
        entry = LessonEntry.from_dict({})
        assert entry.stage_name == ""
        assert entry.stage_num == 0
        assert entry.category == "pipeline"


# ── Classification tests ──


class TestClassifyError:
    def test_timeout_classified_as_system(self) -> None:
        assert _classify_error("experiment_run", "Connection timeout after 30s") == "system"

    def test_validation_classified_as_experiment(self) -> None:
        assert _classify_error("code_generation", "Syntax error in code") == "experiment"

    def test_citation_classified_as_literature(self) -> None:
        assert _classify_error("citation_verify", "Hallucinated reference") == "literature"

    def test_paper_classified_as_writing(self) -> None:
        assert _classify_error("paper_draft", "Draft quality too low") == "writing"

    def test_unknown_defaults_to_pipeline(self) -> None:
        assert _classify_error("unknown_stage", "something random") == "pipeline"


# ── Time weight tests ──


class TestTimeWeight:
    def test_recent_lesson_has_high_weight(self) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        assert _time_weight(now) > 0.9

    def test_30_day_old_has_half_weight(self) -> None:
        ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(timespec="seconds")
        weight = _time_weight(ts)
        assert 0.4 < weight < 0.6  # Should be ~0.5

    def test_90_day_old_returns_zero(self) -> None:
        ts = (datetime.now(timezone.utc) - timedelta(days=91)).isoformat(timespec="seconds")
        assert _time_weight(ts) == 0.0

    def test_invalid_timestamp_returns_zero(self) -> None:
        assert _time_weight("not-a-date") == 0.0

    def test_empty_timestamp_returns_zero(self) -> None:
        assert _time_weight("") == 0.0


# ── Extract lessons tests ──


class TestExtractLessons:
    def _make_result(self, stage_num, status, error=None, decision="proceed"):
        from types import SimpleNamespace
        from researchclaw.pipeline.stages import Stage, StageStatus

        stage = Stage(stage_num)
        return SimpleNamespace(
            stage=stage,
            status=StageStatus(status),
            error=error,
            decision=decision,
        )

    def test_extracts_lesson_from_failed_stage(self) -> None:
        results = [self._make_result(4, "failed", error="API rate limited")]
        lessons = extract_lessons(results, run_id="test-run")
        assert len(lessons) == 1
        assert lessons[0].severity == "error"
        assert "rate limited" in lessons[0].description

    def test_extracts_lesson_from_blocked_stage(self) -> None:
        results = [self._make_result(5, "blocked_approval")]
        lessons = extract_lessons(results, run_id="test-run")
        assert len(lessons) == 1
        assert lessons[0].severity == "warning"
        assert "blocked" in lessons[0].description

    def test_extracts_lesson_from_pivot_decision(self) -> None:
        results = [self._make_result(15, "done", decision="pivot")]
        lessons = extract_lessons(results, run_id="test-run")
        assert len(lessons) == 1
        assert "PIVOT" in lessons[0].description

    def test_no_lessons_from_successful_proceed(self) -> None:
        results = [self._make_result(1, "done", decision="proceed")]
        lessons = extract_lessons(results)
        assert len(lessons) == 0

    def test_multiple_results_multiple_lessons(self) -> None:
        results = [
            self._make_result(4, "failed", error="timeout"),
            self._make_result(5, "blocked_approval"),
            self._make_result(15, "done", decision="refine"),
        ]
        lessons = extract_lessons(results)
        assert len(lessons) == 3

    def test_extracts_decision_rationale(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        stage_dir = run_dir / "stage-15"
        stage_dir.mkdir(parents=True)
        (stage_dir / "decision_structured.json").write_text(
            json.dumps({"decision": "pivot", "rationale": "NaN in metrics"}),
            encoding="utf-8",
        )
        results = [self._make_result(15, "done", decision="pivot")]
        lessons = extract_lessons(results, run_id="test", run_dir=run_dir)
        assert any("NaN in metrics" in l.description for l in lessons)

    def test_extracts_rationale_from_raw_text_excerpt(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        stage_dir = run_dir / "stage-15"
        stage_dir.mkdir(parents=True)
        (stage_dir / "decision_structured.json").write_text(
            json.dumps({
                "decision": "refine",
                "raw_text_excerpt": (
                    "## Decision\n**REFINE**\n\n"
                    "## Justification\n"
                    "The analysis provides promising evidence but lacks statistical rigor."
                ),
                "generated": "2026-03-11T05:15:43+00:00",
            }),
            encoding="utf-8",
        )
        results = [self._make_result(15, "done", decision="refine")]
        lessons = extract_lessons(results, run_id="test", run_dir=run_dir)
        assert any("statistical rigor" in l.description for l in lessons)

    def test_extracts_stderr_runtime_lesson(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({
                "metrics": {"loss": 0.5},
                "stderr": "RuntimeWarning: invalid value encountered in divide",
            }),
            encoding="utf-8",
        )
        results = [self._make_result(12, "done")]
        lessons = extract_lessons(results, run_dir=run_dir)
        assert any("RuntimeWarning" in l.description for l in lessons)

    def test_extracts_nan_metric_lesson(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({"metrics": {"accuracy": "nan"}}),
            encoding="utf-8",
        )
        results = [self._make_result(12, "done")]
        lessons = extract_lessons(results, run_dir=run_dir)
        assert any("accuracy" in l.description and "nan" in l.description.lower()
                    for l in lessons)

    def test_no_runtime_lessons_without_run_dir(self) -> None:
        results = [self._make_result(12, "done")]
        lessons = extract_lessons(results)
        assert len(lessons) == 0


# ── EvolutionStore tests ──


class TestEvolutionStore:
    def test_append_and_load(self, tmp_path: Path) -> None:
        store = EvolutionStore(tmp_path / "evo")
        lesson = LessonEntry(
            stage_name="hypothesis_gen",
            stage_num=8,
            category="pipeline",
            severity="warning",
            description="PIVOT triggered",
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        store.append(lesson)
        loaded = store.load_all()
        assert len(loaded) == 1
        assert loaded[0].stage_name == "hypothesis_gen"

    def test_append_many(self, tmp_path: Path) -> None:
        store = EvolutionStore(tmp_path / "evo")
        lessons = [
            LessonEntry("s1", 1, "system", "error", "err1",
                        datetime.now(timezone.utc).isoformat()),
            LessonEntry("s2", 2, "pipeline", "info", "info1",
                        datetime.now(timezone.utc).isoformat()),
        ]
        store.append_many(lessons)
        assert store.count() == 2

    def test_append_many_empty_is_noop(self, tmp_path: Path) -> None:
        store = EvolutionStore(tmp_path / "evo")
        store.append_many([])
        assert store.count() == 0

    def test_load_all_empty_store(self, tmp_path: Path) -> None:
        store = EvolutionStore(tmp_path / "evo")
        assert store.load_all() == []

    def test_query_for_stage_returns_relevant_lessons(self, tmp_path: Path) -> None:
        store = EvolutionStore(tmp_path / "evo")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        store.append(LessonEntry("hypothesis_gen", 8, "pipeline", "error",
                                 "Failed hypothesis", now))
        store.append(LessonEntry("paper_draft", 17, "writing", "warning",
                                 "Draft too short", now))
        result = store.query_for_stage("hypothesis_gen", max_lessons=5)
        # hypothesis_gen lesson should be boosted
        assert len(result) >= 1
        assert result[0].stage_name == "hypothesis_gen"

    def test_query_respects_max_lessons(self, tmp_path: Path) -> None:
        store = EvolutionStore(tmp_path / "evo")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for i in range(10):
            store.append(LessonEntry("stage_1", 1, "system", "error",
                                     f"Error {i}", now))
        result = store.query_for_stage("stage_1", max_lessons=3)
        assert len(result) == 3

    def test_build_overlay_returns_empty_for_no_lessons(self, tmp_path: Path) -> None:
        store = EvolutionStore(tmp_path / "evo")
        assert store.build_overlay("hypothesis_gen") == ""

    def test_build_overlay_returns_formatted_text(self, tmp_path: Path) -> None:
        store = EvolutionStore(tmp_path / "evo")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        store.append(LessonEntry("hypothesis_gen", 8, "experiment", "error",
                                 "Code syntax error in experiment", now))
        overlay = store.build_overlay("hypothesis_gen")
        assert "Lessons from Prior Runs" in overlay
        assert "Code syntax error" in overlay
        assert "❌" in overlay

    def test_old_lessons_filtered_by_time_weight(self, tmp_path: Path) -> None:
        store = EvolutionStore(tmp_path / "evo")
        old_ts = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        store.append(LessonEntry("stage_1", 1, "system", "error", "Old error", old_ts))
        result = store.query_for_stage("stage_1")
        assert len(result) == 0  # Filtered out due to age > 90 days

    def test_creates_directory_if_not_exists(self, tmp_path: Path) -> None:
        store_dir = tmp_path / "nested" / "evo"
        store = EvolutionStore(store_dir)
        assert store_dir.exists()


class TestTrajectoryStore:
    def test_record_refine_trajectory_and_signal_detects_plateau(
        self, tmp_path: Path
    ) -> None:
        log1 = {
            "metric_key": "loss",
            "metric_direction": "minimize",
            "iterations": [{"metric": 1.0}, {"metric": 0.9}],
        }
        log2 = {
            "metric_key": "loss",
            "metric_direction": "minimize",
            "iterations": [{"metric": 0.88}, {"metric": 0.87}],
        }
        log3 = {
            "metric_key": "loss",
            "metric_direction": "minimize",
            "iterations": [{"metric": 0.86}, {"metric": 0.855}],
        }

        points = record_refine_trajectory(tmp_path, "run-a", log1, cycle=1)
        record_refine_trajectory(tmp_path, "run-a", log2, cycle=2)
        record_refine_trajectory(tmp_path, "run-a", log3, cycle=3)

        assert len(points) == 2
        assert TrajectoryStore(tmp_path).path.exists()
        signal = get_trajectory_signal(tmp_path, "run-a", current_cycle=3)
        assert signal["stagnating"] is True
        assert signal["recommendation"] == "pivot"
        assert "Cycle 3" in str(signal["summary"])


# ── PromptManager evolution overlay integration ──


class TestPromptManagerEvolutionOverlay:
    def test_overlay_appended_to_user_prompt(self) -> None:
        from researchclaw.prompts import PromptManager

        pm = PromptManager()
        overlay = "## Lessons\n1. Avoid timeout errors."
        sp = pm.for_stage(
            "topic_init",
            evolution_overlay=overlay,
            topic="test",
            domains="ml",
            project_name="p1",
            quality_threshold="8.0",
        )
        assert "Avoid timeout errors" in sp.user

    def test_no_overlay_when_empty(self) -> None:
        from researchclaw.prompts import PromptManager

        pm = PromptManager()
        sp1 = pm.for_stage(
            "topic_init",
            topic="test",
            domains="ml",
            project_name="p1",
            quality_threshold="8.0",
        )
        sp2 = pm.for_stage(
            "topic_init",
            evolution_overlay="",
            topic="test",
            domains="ml",
            project_name="p1",
            quality_threshold="8.0",
        )
        assert sp1.user == sp2.user
