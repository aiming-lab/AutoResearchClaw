# pyright: reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnusedCallResult=false
"""Tests for checkpoint/resume and content metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from researchclaw.pipeline.executor import StageResult
from researchclaw.pipeline.runner import (
    _build_pipeline_summary,
    _collect_content_metrics,
    _write_checkpoint,
    read_checkpoint,
    resume_from_checkpoint,
)
from researchclaw.pipeline.stages import (
    NONCRITICAL_STAGES,
    STAGE_SEQUENCE,
    Stage,
    StageStatus,
)


class TestCheckpoint:
    def test_write_checkpoint(self, tmp_path: Path):
        _write_checkpoint(tmp_path, Stage.LITERATURE_COLLECT, "test-run")
        cp = json.loads((tmp_path / "checkpoint.json").read_text())
        assert cp["last_completed_stage"] == 4
        assert cp["last_completed_name"] == "LITERATURE_COLLECT"
        assert cp["run_id"] == "test-run"
        assert "timestamp" in cp

    def test_read_checkpoint_returns_next_stage(self, tmp_path: Path):
        _write_checkpoint(tmp_path, Stage.LITERATURE_COLLECT, "test-run")
        next_stage = read_checkpoint(tmp_path)
        assert next_stage == Stage.LITERATURE_SCREEN

    def test_read_checkpoint_no_file(self, tmp_path: Path):
        assert read_checkpoint(tmp_path) is None

    def test_read_checkpoint_last_stage(self, tmp_path: Path):
        _write_checkpoint(tmp_path, Stage.CITATION_VERIFY, "test-run")
        assert read_checkpoint(tmp_path) is None

    def test_read_checkpoint_corrupted(self, tmp_path: Path):
        (tmp_path / "checkpoint.json").write_text("not json", encoding="utf-8")
        assert read_checkpoint(tmp_path) is None

    def test_read_checkpoint_invalid_stage(self, tmp_path: Path):
        (tmp_path / "checkpoint.json").write_text(
            json.dumps({"last_completed_stage": 999}), encoding="utf-8"
        )
        assert read_checkpoint(tmp_path) is None

    def test_resume_from_checkpoint_uses_default(self, tmp_path: Path):
        assert resume_from_checkpoint(tmp_path) == Stage.TOPIC_INIT

    def test_resume_from_checkpoint_uses_next_stage(self, tmp_path: Path):
        _write_checkpoint(tmp_path, Stage.SEARCH_STRATEGY, "run-x")
        assert resume_from_checkpoint(tmp_path) == Stage.LITERATURE_COLLECT


class TestNoncriticalStages:
    def test_knowledge_archive_is_noncritical(self):
        assert Stage.KNOWLEDGE_ARCHIVE in NONCRITICAL_STAGES

    def test_citation_verify_is_critical(self):
        # T3.4: CITATION_VERIFY is now critical — hallucinated refs must block export
        assert Stage.CITATION_VERIFY not in NONCRITICAL_STAGES

    def test_topic_init_is_critical(self):
        assert Stage.TOPIC_INIT not in NONCRITICAL_STAGES

    def test_paper_draft_is_critical(self):
        assert Stage.PAPER_DRAFT not in NONCRITICAL_STAGES

    def test_stage_sequence_still_ends_with_citation_verify(self):
        assert STAGE_SEQUENCE[-1] == Stage.CITATION_VERIFY


class TestContentMetrics:
    def test_metrics_empty_run_dir(self, tmp_path: Path):
        metrics = _collect_content_metrics(tmp_path)
        assert metrics["template_ratio"] is None
        assert metrics["citation_verify_score"] is None
        assert metrics["total_citations"] is None
        assert metrics["degraded_sources"] == []

    def test_metrics_with_draft(self, tmp_path: Path):
        draft_dir = tmp_path / "stage-17"
        draft_dir.mkdir()
        (draft_dir / "paper_draft.md").write_text(
            "This is a real academic paper about transformers and attention mechanisms. We propose a novel method for improving efficiency.",
            encoding="utf-8",
        )
        metrics = _collect_content_metrics(tmp_path)
        assert metrics["template_ratio"] is not None
        assert cast(float, metrics["template_ratio"]) < 0.5

    def test_metrics_with_verification(self, tmp_path: Path):
        verify_dir = tmp_path / "stage-23"
        verify_dir.mkdir()
        (verify_dir / "verification_report.json").write_text(
            json.dumps(
                {
                    "summary": {
                        "total": 10,
                        "verified": 8,
                        "suspicious": 1,
                        "hallucinated": 1,
                        "skipped": 0,
                        "integrity_score": 0.8
                    },
                    "results": []
                }
            ),
            encoding="utf-8",
        )
        metrics = _collect_content_metrics(tmp_path)
        assert metrics["total_citations"] == 10
        assert metrics["verified_citations"] == 8
        assert metrics["citation_verify_score"] == 0.8

    def test_metrics_no_stage23(self, tmp_path: Path):
        metrics = _collect_content_metrics(tmp_path)
        assert metrics["citation_verify_score"] is None

    def test_metrics_with_non_dict_summary(self, tmp_path: Path):
        """Must not raise NameError when 'summary' is not a dict."""
        verify_dir = tmp_path / "stage-23"
        verify_dir.mkdir()
        (verify_dir / "verification_report.json").write_text(
            json.dumps({"summary": "unexpected string"}),
            encoding="utf-8",
        )
        metrics = _collect_content_metrics(tmp_path)
        assert metrics["total_citations"] is None
        assert metrics["verified_citations"] is None
        assert metrics["citation_verify_score"] is None

    def test_metrics_with_summary_missing_fields(self, tmp_path: Path):
        """summary dict without total/verified should not crash."""
        verify_dir = tmp_path / "stage-23"
        verify_dir.mkdir()
        (verify_dir / "verification_report.json").write_text(
            json.dumps({"summary": {"notes": "incomplete"}}),
            encoding="utf-8",
        )
        metrics = _collect_content_metrics(tmp_path)
        assert metrics["total_citations"] == 0
        assert metrics["verified_citations"] == 0
        assert metrics["citation_verify_score"] is None

    def test_summary_includes_content_metrics(self, tmp_path: Path):
        results = [
            StageResult(
                stage=Stage.TOPIC_INIT,
                status=StageStatus.DONE,
                artifacts=("topic.json",),
            ),
        ]
        summary = _build_pipeline_summary(
            run_id="test",
            results=results,
            from_stage=Stage.TOPIC_INIT,
            run_dir=tmp_path,
        )
        assert "content_metrics" in summary
        assert isinstance(summary["content_metrics"], dict)
