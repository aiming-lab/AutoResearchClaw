from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from researchclaw.config import PaperRevisionConfig
from researchclaw.pipeline.sectional_execution import (
    ResolutionAssessment,
    SectionProposal,
    build_validation_context,
    execute_sectional_revision,
)


DRAFT = """## Title

Example Study

## Method

The detector scored 0.475 using three seeds \\cite{smith2024}.

## Results

The recorded score was 0.475.
"""

REVIEWS = """## Reviewer A

### Strengths
The method is concise.

### Weaknesses
The metric wording is unclear.

### Actionable Revisions
1. Clarify how the recorded metric is reported.
"""


class _FakeProvider:
    writer_model = "writer-model"
    critic_model = "critic-model"

    def __init__(self, *, verdict: str = "resolved", fail_proposal: bool = False):
        self.verdict = verdict
        self.fail_proposal = fail_proposal

    def build_plan(self, *, ledger, document):
        method = next(section for section in document.sections if section.title == "Method")
        return {
            "schema_version": 1,
            "planner_version": 1,
            "source_paper_sha256": document.source_sha256,
            "source_reviews_sha256": ledger.source_reviews_sha256,
            "section_model_version": 1,
            "assignments": [
                {
                    "comment_id": comment.comment_id,
                    "target_section_ids": [method.section_id],
                    "disposition": "assigned",
                    "reason": None,
                }
                for comment in ledger.comments
            ],
        }

    def propose(self, *, section, comments, attempt, context):
        _ = attempt, context
        if self.fail_proposal:
            raise RuntimeError("synthetic transport failure")
        return SectionProposal(
            section_id=section.section_id,
            revised_body=(
                "\nThe recorded detector score was 0.475 across three seeds "
                "\\cite{smith2024}. This sentence clarifies the reporting basis.\n\n"
            ),
            resolution_comment_ids=tuple(comment.comment_id for comment in comments),
        )

    def assess(
        self,
        *,
        comment,
        section,
        original_body,
        revised_body,
        attempt_id,
        validator_codes,
    ):
        _ = original_body, revised_body, validator_codes
        return ResolutionAssessment(
            comment_id=comment.comment_id,
            section_id=section.section_id,
            attempt_id=attempt_id,
            critic_model=self.critic_model,
            context_isolated=True,
            verdict=self.verdict,
            reason=(
                "The requested wording is explicit."
                if self.verdict == "resolved"
                else "Still unclear."
            ),
        )


def _prepare_run(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-19"
    stage_dir.mkdir(parents=True)
    for stage, name, text in (
        (17, "paper_draft.md", DRAFT),
        (18, "reviews.md", REVIEWS),
        (4, "references.bib", "@article{smith2024, title={Example}, year={2024}}\n"),
    ):
        path = run_dir / f"stage-{stage:02d}" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    metrics = run_dir / "stage-12" / "runs" / "results.json"
    metrics.parent.mkdir(parents=True)
    metrics.write_text(
        json.dumps(
            {
                "evaluator_owner": "scaffold",
                "metrics": {"detection_f1": 0.475},
            }
        ),
        encoding="utf-8",
    )
    stage10 = run_dir / "stage-10" / "smoke" / "smoke_results.json"
    stage10.parent.mkdir(parents=True)
    stage10.write_text(json.dumps({"metrics": {"fabricated": 0.99}}), encoding="utf-8")
    return run_dir, stage_dir


def _config() -> PaperRevisionConfig:
    return PaperRevisionConfig(
        sectional_enabled=True,
        max_section_retries=1,
        min_length_ratio=0.5,
        max_length_ratio=2.0,
        critic_model="critic-model",
    )


def test_context_builder_binds_canonical_sources_and_excludes_stage10(
    tmp_path: Path,
) -> None:
    run_dir, _ = _prepare_run(tmp_path)
    from researchclaw.pipeline.manuscript_sections import parse_manuscript

    bundle = build_validation_context(
        run_dir=run_dir,
        document=parse_manuscript(DRAFT),
        config=_config(),
    )
    payload = json.loads(bundle.text)

    assert payload["allowed_citation_keys"] == ["smith2024"]
    assert payload["grounded_numeric_values"] == [0.475]
    assert all(not source["path"].startswith("stage-10/") for source in payload["sources"])
    assert {source["kind"] for source in payload["sources"]} == {
        "citations",
        "config",
        "metrics",
    }
    metric_source = next(
        source for source in payload["sources"] if source["kind"] == "metrics"
    )
    assert metric_source["sha256"] == hashlib.sha256(
        (run_dir / metric_source["path"]).read_bytes()
    ).hexdigest()


def test_context_builder_fails_when_canonical_bib_omits_draft_key(
    tmp_path: Path,
) -> None:
    run_dir, _ = _prepare_run(tmp_path)
    (run_dir / "stage-04" / "references.bib").write_text(
        "@article{other2024, title={Other}, year={2024}}\n",
        encoding="utf-8",
    )
    from researchclaw.pipeline.manuscript_sections import parse_manuscript

    with pytest.raises(RuntimeError, match="missing draft citation keys"):
        build_validation_context(
            run_dir=run_dir,
            document=parse_manuscript(DRAFT),
            config=_config(),
        )


def test_provider_must_match_configured_isolated_critic(tmp_path: Path) -> None:
    run_dir, stage_dir = _prepare_run(tmp_path)
    provider = _FakeProvider()
    provider.critic_model = provider.writer_model

    with pytest.raises(RuntimeError, match="distinct nonempty writer and critic"):
        execute_sectional_revision(
            stage_dir=stage_dir,
            run_dir=run_dir,
            config=_config(),
            claim_scope="pipeline_validation",
            provider=provider,
        )


def test_pipeline_validation_writes_complete_hash_bound_artifacts(tmp_path: Path) -> None:
    run_dir, stage_dir = _prepare_run(tmp_path)

    result = execute_sectional_revision(
        stage_dir=stage_dir,
        run_dir=run_dir,
        config=_config(),
        claim_scope="pipeline_validation",
        provider=_FakeProvider(),
    )

    assert result.completed is True
    assert "clarifies the reporting basis" in (result.paper_text or "")
    manifest = json.loads(
        (stage_dir / "section_revision_manifest.json").read_text(encoding="utf-8")
    )
    context_bytes = (stage_dir / "validation_context.json").read_bytes()
    assert manifest["completed"] is True
    assert manifest["validation_context_path"] == "stage-19/validation_context.json"
    assert manifest["validation_context_sha256"] == hashlib.sha256(context_bytes).hexdigest()
    assert (stage_dir / "paper_revised.md").is_file()


def test_pipeline_validation_preserves_original_when_critic_rejects(
    tmp_path: Path,
) -> None:
    run_dir, stage_dir = _prepare_run(tmp_path)

    result = execute_sectional_revision(
        stage_dir=stage_dir,
        run_dir=run_dir,
        config=replace(_config(), max_section_retries=0),
        claim_scope="pipeline_validation",
        provider=_FakeProvider(verdict="unresolved"),
    )

    assert result.completed is True
    assert result.paper_text == DRAFT
    unresolved = json.loads(
        (stage_dir / "unresolved_comments.json").read_text(encoding="utf-8")
    )
    assert len(unresolved["comments"]) == 1
    manifest = json.loads(
        (stage_dir / "section_revision_manifest.json").read_text(encoding="utf-8")
    )
    assert any(
        section["final_status"] == "unresolved_original_preserved"
        for section in manifest["sections"]
    )


def test_research_release_does_not_expose_paper_with_unresolved_required_comment(
    tmp_path: Path,
) -> None:
    run_dir, stage_dir = _prepare_run(tmp_path)

    result = execute_sectional_revision(
        stage_dir=stage_dir,
        run_dir=run_dir,
        config=replace(_config(), max_section_retries=0),
        claim_scope="research_release",
        provider=_FakeProvider(verdict="unresolved"),
    )

    assert result.completed is False
    assert result.paper_text is None
    assert not (stage_dir / "paper_revised.md").exists()
    manifest = json.loads(
        (stage_dir / "section_revision_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["completed"] is False


def test_transport_failures_are_bounded_and_stale_outputs_are_removed(
    tmp_path: Path,
) -> None:
    run_dir, stage_dir = _prepare_run(tmp_path)
    (stage_dir / "paper_revised.md").write_text("stale", encoding="utf-8")
    (stage_dir / "paper_revised.md.tmp").write_text("stale temp", encoding="utf-8")
    unrelated_temp = stage_dir / "unrelated.tmp"
    unrelated_temp.write_text("preserve", encoding="utf-8")
    stale_dir = stage_dir / "sections"
    stale_dir.mkdir()
    (stale_dir / "stale.md").write_text("stale", encoding="utf-8")

    result = execute_sectional_revision(
        stage_dir=stage_dir,
        run_dir=run_dir,
        config=replace(_config(), max_section_retries=1),
        claim_scope="pipeline_validation",
        provider=_FakeProvider(fail_proposal=True),
    )

    assert result.completed is True
    attempts = [
        json.loads(line)
        for line in (stage_dir / "section_attempts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [attempt["status"] for attempt in attempts] == [
        "transport_failed",
        "transport_failed",
    ]
    assert not (stage_dir / "sections" / "stale.md").exists()
    assert not (stage_dir / "paper_revised.md.tmp").exists()
    assert unrelated_temp.read_text(encoding="utf-8") == "preserve"
    assert (stage_dir / "paper_revised.md").read_text(encoding="utf-8") == DRAFT


def test_context_source_mutation_during_provider_execution_fails_closed(
    tmp_path: Path,
) -> None:
    run_dir, stage_dir = _prepare_run(tmp_path)

    class MutatingProvider(_FakeProvider):
        def propose(self, **kwargs):
            result = super().propose(**kwargs)
            (run_dir / "stage-12" / "runs" / "results.json").write_text(
                json.dumps({"metrics": {"detection_f1": 0.999}}),
                encoding="utf-8",
            )
            return result

    with pytest.raises(RuntimeError, match="validation context source changed"):
        execute_sectional_revision(
            stage_dir=stage_dir,
            run_dir=run_dir,
            config=replace(_config(), max_section_retries=0),
            claim_scope="pipeline_validation",
            provider=MutatingProvider(),
        )

    assert not (stage_dir / "section_revision_manifest.json").exists()
    assert not (stage_dir / "paper_revised.md").exists()


def test_provider_programming_error_is_not_misclassified_as_transport_failure(
    tmp_path: Path,
) -> None:
    run_dir, stage_dir = _prepare_run(tmp_path)

    class BrokenProvider(_FakeProvider):
        def propose(self, **kwargs):
            _ = kwargs
            raise KeyError("provider implementation bug")

    with pytest.raises(KeyError, match="provider implementation bug"):
        execute_sectional_revision(
            stage_dir=stage_dir,
            run_dir=run_dir,
            config=replace(_config(), max_section_retries=0),
            claim_scope="pipeline_validation",
            provider=BrokenProvider(),
        )

    assert not (stage_dir / "section_attempts.jsonl").exists()
    assert not (stage_dir / "section_revision_manifest.json").exists()
