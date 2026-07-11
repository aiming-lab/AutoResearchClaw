"""Tests for release_check v2 gates and the release_artifacts contracts.

Builds a synthetic complete run directory that passes every v2 gate, then
breaks one gate at a time and asserts the corresponding failure code.
"""

from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import release_check  # noqa: E402
from researchclaw.pipeline import release_artifacts as ra  # noqa: E402
from researchclaw.config import PaperRevisionConfig  # noqa: E402
from researchclaw.pipeline.sectional_execution import (  # noqa: E402
    ResolutionAssessment,
    SectionProposal,
    execute_sectional_revision,
)
from researchclaw.pipeline.stages import FINAL_STAGE, Stage  # noqa: E402


PAPER = (
    "# Result\n\nOur method reaches 0.1234 loss, beating the baseline "
    "[smith2024deep]. Related work [jones2023survey] surveys the field.\n"
)

SECTIONAL_DRAFT = """## Title

Release Fixture

## Method

The detector scored 0.1234 using three seeds \\cite{smith2024deep}.

## Results

The recorded score was 0.1234.
"""

SECTIONAL_REVIEWS = """## Reviewer A

### Strengths
The method is concise.

### Weaknesses
The reporting basis is terse.

### Actionable Revisions
1. Clarify how the recorded metric is reported.
"""


class _SectionalFixtureProvider:
    writer_model = "writer-model-y"
    critic_model = "section-critic-z"

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
        return SectionProposal(
            section_id=section.section_id,
            revised_body=(
                "\nThe recorded detector score was 0.1234 across three seeds "
                "\\cite{smith2024deep}. This sentence clarifies the reporting basis.\n\n"
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
            verdict="resolved",
            reason="The reporting basis is explicit in the revised section.",
        )


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, str):
        path.write_text(obj, encoding="utf-8")
    else:
        path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


@pytest.fixture()
def good_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir()

    # --- experiment evidence ---
    _write(
        run / "stage-14" / "experiment_summary.json",
        {"metrics_summary": {"loss": {"mean": 0.1234}}},
    )
    evidence_rel = "stage-14/experiment_summary.json"
    evidence_sha = ra.sha256_file(run / evidence_rel)

    _write(
        run / "stage-09" / "experiment_contract.yaml",
        "\n".join(
            [
                "schema_version: 1",
                "topic: release check fixture",
                "claim_scope: research_release",
                "dataset_origin: public",
                "primary_metric:",
                "  key: loss",
                "  direction: minimize",
                "smoke_budget_sec: 60",
                "run_budget_sec: 300",
                "evaluator:",
                "  owner: scaffold",
                "  required_result_keys:",
                "    - dataset_origin",
                "    - metrics",
            ]
        ),
    )
    contract_sha = ra.sha256_file(run / "stage-09" / "experiment_contract.yaml")
    _write(
        run / "stage-10" / "selected_candidate_manifest.json",
        {
            "schema_version": 1,
            "generated": "2026-07-11T00:00:00+00:00",
            "contract_path": "stage-09/experiment_contract.yaml",
            "contract_sha256": contract_sha,
            "scaffold_sha256": "a" * 64,
            "entry_point": "main.py",
            "files": {"main.py": {"sha256": "b" * 64}},
        },
    )

    # --- v1 artifacts ---
    _write(
        run / "pipeline_summary.json",
        {
            "final_stage": int(FINAL_STAGE),
            "final_status": "done",
            "stages_failed": 0,
            "degraded": False,
        },
    )
    _write(
        run / "stage-20" / "quality_report.json",
        {"score_1_to_10": 8.0, "verdict": "proceed", "generated": "2026-07-06T00:00:00+00:00"},
    )
    _write(
        run / "stage-20" / "fabrication_flags.json",
        {"fabrication_suspected": False, "has_real_data": True},
    )
    _write(run / "stage-22" / "paper_final.md", PAPER)
    _write(run / "stage-23" / "paper_final_verified.md", PAPER)
    _write(
        run / "stage-23" / "verification_report.json",
        {"summary": {"total": 2, "verified": 2, "suspicious": 0, "hallucinated": 0, "skipped": 0}},
    )
    _write(
        run / "stage-23" / "references_verified.bib",
        "@article{smith2024deep, title={Deep}}\n@article{jones2023survey, title={Survey}}\n",
    )
    _write(
        run / "stage-04" / "references.bib",
        "@article{smith2024deep, title={Deep}, year={2024}}\n",
    )
    _write(
        run / "stage-12" / "sandbox_metadata.json",
        {"requested_backend": "docker", "actual_backend": "docker", "fallback_used": False},
    )
    _write(
        run / "stage-12" / "environment_policy.json",
        {"policy": "container_isolated"},
    )
    _write(run / "deliverables" / "manifest.json", {"files": ["paper_final.md"]})
    _write(run / "deliverables" / "paper_final.md", PAPER)
    _write(
        run / "stage-23" / "canonical_source.json",
        {
            "source_id": "stage-22/paper_final.md",
            "markdown_source_id": "stage-22/paper_final.md",
            "latex_source_id": "stage-22/paper_final.md",
            "markdown_path": "stage-23/paper_final_verified.md",
            "markdown_sha256": ra.sha256_file(run / "stage-23" / "paper_final_verified.md"),
        },
    )
    _write(run / "cost_log.jsonl", json.dumps({"stage": 1, "cost_usd": 0.1}) + "\n")

    # --- critique (stage 15) ---
    _write(
        run / "stage-15" / "critique.json",
        {
            "schema_version": ra.SCHEMA_VERSION,
            "recommend_only": True,
            "critic_source": "model",
            "critic_model": "critic-model-x",
            "writer_model": "writer-model-y",
            "shared_context": False,
            "findings": [
                {
                    "id": "sc-01",
                    "severity": "P1",
                    "category": "statistics",
                    "question": "Single seed?",
                    "finding": "No variance reported.",
                    "falsification_criterion": "Different seed changes ranking.",
                }
            ],
        },
    )

    # --- stage 24 (truth audit) ---
    claims = [
        {
            "id": "clm-0000",
            "text": "Our method reaches 0.1234 loss, beating the baseline [smith2024deep].",
            "type": "quantitative",
            "values": [0.1234],
            "cited_keys": ["smith2024deep"],
            "evidence": [
                {"path": evidence_rel, "sha256": evidence_sha, "matched_value": 0.1234}
            ],
            "status": "supported",
        }
    ]
    paper_hash = ra.paper_sha256(PAPER)
    _write(
        run / "stage-24" / "claims.json",
        {"schema_version": ra.SCHEMA_VERSION, "claims": claims},
    )
    _write(
        run / "stage-24" / "citations.json",
        {
            "schema_version": ra.SCHEMA_VERSION,
            # Background citations must be explicitly whitelisted (auditable),
            # not silently escape claim binding.
            "background_whitelist": {"cite_keys": ["jones2023survey"], "sections": []},
            "instances": [
                {
                    "instance_id": "cit-0000",
                    "cite_key": "smith2024deep",
                    "role": "claim_support",
                    "supported_claim_id": "clm-0000",
                    # A real substring of the supported claim's text.
                    "support_excerpt": "Our method reaches 0.1234 loss, beating the baseline",
                },
                {
                    "instance_id": "cit-0001",
                    "cite_key": "jones2023survey",
                    "role": "background",
                    "supported_claim_id": None,
                    "support_excerpt": "",
                },
            ],
        },
    )
    _write(
        run / "stage-24" / "critique_resolution.json",
        {
            "resolutions": [
                {"finding_id": "sc-01", "severity": "P1", "resolution": "fixed", "note": "Added 3 seeds."}
            ]
        },
    )
    _write(
        run / "stage-24" / "truth_audit.json",
        {
            "schema_version": ra.SCHEMA_VERSION,
            "paper_path": "stage-23/paper_final_verified.md",
            "paper_sha256": paper_hash,
            "claims_digest": ra.claims_digest(claims),
        },
    )

    # --- stage 25 (de-AI audit) ---
    _write(
        run / "stage-25" / "deai_audit.json",
        {
            "schema_version": ra.SCHEMA_VERSION,
            "recommend_only": True,
            "applied": False,
            "paper_sha256": paper_hash,
            "truth_audit_sha256": paper_hash,
            "suggestions": [],
        },
    )

    # --- Stage 19 sectional bundle, generated through the real producer ---
    _write(run / "stage-17" / "paper_draft.md", SECTIONAL_DRAFT)
    _write(run / "stage-18" / "reviews.md", SECTIONAL_REVIEWS)
    sectional_result = execute_sectional_revision(
        stage_dir=run / "stage-19",
        run_dir=run,
        config=PaperRevisionConfig(
            sectional_enabled=True,
            max_section_retries=0,
            min_length_ratio=0.5,
            max_length_ratio=2.0,
            critic_model="section-critic-z",
        ),
        claim_scope="research_release",
        provider=_SectionalFixtureProvider(),
    )
    assert sectional_result.completed is True

    # --- run manifest ---
    _write(
        run / "run_manifest.json",
        {
            "schema_version": ra.SCHEMA_VERSION,
            "expected_final_stage": int(FINAL_STAGE),
            "reviewer": {
                "writer_model": "writer-model-y",
                "critic_model": "critic-model-x",
                "critic_source": "model",
                "sectional_writer_model": "writer-model-y",
                "sectional_critic_model": "section-critic-z",
                "shared_context": False,
            },
        },
    )
    return run


def _check(run: Path) -> release_check.ReleaseChecker:
    checker = release_check.ReleaseChecker(
        run, quality_threshold=5.0, allow_suspicious=False
    )
    checker.run()
    return checker


def _codes(checker: release_check.ReleaseChecker) -> set[str]:
    return {f.code for f in checker.findings if f.severity == "error"}


def _stage19_manifest(run: Path) -> dict:
    return json.loads(
        (run / "stage-19" / "section_revision_manifest.json").read_text(
            encoding="utf-8"
        )
    )


def _write_stage19_manifest(run: Path, manifest: dict) -> None:
    _write(run / "stage-19" / "section_revision_manifest.json", manifest)


def _rewrite_jsonl_record(run: Path, name: str, mutate) -> str:
    path = run / "stage-19" / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    text = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def test_good_run_passes(good_run: Path) -> None:
    checker = _check(good_run)
    assert _codes(checker) == set(), _codes(checker)
    assert checker.exit_code() == release_check.EXIT_PASS


def test_sectional_manifest_is_required(good_run: Path) -> None:
    (good_run / "stage-19" / "section_revision_manifest.json").unlink()
    assert "sectional_revision_manifest_missing" in _codes(_check(good_run))


def test_legacy_stage19_is_not_release_eligible(good_run: Path) -> None:
    stage19 = good_run / "stage-19"
    for path in tuple(stage19.iterdir()):
        if path.name == "paper_revised.md":
            continue
        if path.is_dir():
            import shutil

            shutil.rmtree(path)
        else:
            path.unlink()
    assert "sectional_revision_manifest_missing" in _codes(_check(good_run))


@pytest.mark.parametrize(
    ("completed", "expected"),
    ((False, "sectional_revision_incomplete"), ("true", "sectional_revision_artifact_invalid")),
)
def test_sectional_completed_is_strict_bool(
    good_run: Path, completed, expected: str
) -> None:
    manifest = _stage19_manifest(good_run)
    manifest["completed"] = completed
    _write_stage19_manifest(good_run, manifest)
    assert expected in _codes(_check(good_run))


def test_sectional_required_artifact_missing(good_run: Path) -> None:
    (good_run / "stage-19" / "revision_plan.json").unlink()
    assert "sectional_revision_artifact_missing" in _codes(_check(good_run))


def test_sectional_contract_edit_after_stage19_is_blocked(good_run: Path) -> None:
    contract = good_run / "stage-09" / "experiment_contract.yaml"
    contract.write_text(
        contract.read_text(encoding="utf-8") + "\n# post-stage-19 edit\n",
        encoding="utf-8",
    )
    assert "sectional_contract_hash_mismatch" in _codes(_check(good_run))


def test_sectional_manifest_claim_scope_cannot_drift(good_run: Path) -> None:
    manifest = _stage19_manifest(good_run)
    manifest["claim_scope"] = "exploratory"
    _write_stage19_manifest(good_run, manifest)
    assert "sectional_claim_scope_mismatch" in _codes(_check(good_run))


def test_sectional_sealed_candidate_anchor_is_required(good_run: Path) -> None:
    (good_run / "stage-10" / "selected_candidate_manifest.json").unlink()
    assert "sectional_sealed_candidate_missing" in _codes(_check(good_run))


def test_sectional_sealed_candidate_contract_hash_must_match(good_run: Path) -> None:
    path = good_run / "stage-10" / "selected_candidate_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["contract_sha256"] = "0" * 64
    _write(path, payload)
    assert "sectional_contract_hash_mismatch" in _codes(_check(good_run))


def test_sectional_unresolved_artifact_cannot_hide_comment(good_run: Path) -> None:
    path = good_run / "stage-19" / "unresolved_comments.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["comments"] = [
        {"comment_id": "forged", "final_status": "unresolved", "reason": "x"}
    ]
    _write(path, payload)
    assert "sectional_unresolved_artifact_mismatch" in _codes(_check(good_run))


def test_sectional_numeric_whitelist_recomputes_from_sources(good_run: Path) -> None:
    path = good_run / "stage-19" / "validation_context.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["grounded_numeric_values"].append(0.999)
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    manifest = _stage19_manifest(good_run)
    manifest["validation_context_sha256"] = ra.sha256_text(text)
    _write_stage19_manifest(good_run, manifest)
    assert "sectional_validation_context_invalid" in _codes(_check(good_run))


def test_sectional_ledger_comment_edit_is_blocked(good_run: Path) -> None:
    path = good_run / "stage-19" / "review_comment_ledger.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["comments"][0]["exact_text"] = "forged review text"
    _write(path, payload)
    assert "sectional_ledger_recompute_mismatch" in _codes(_check(good_run))


def test_sectional_plan_comment_omission_is_blocked(good_run: Path) -> None:
    path = good_run / "stage-19" / "revision_plan.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["assignments"] = []
    _write(path, payload)
    assert "sectional_revision_plan_invalid" in _codes(_check(good_run))


def test_sectional_candidate_missing_and_extra_are_blocked(good_run: Path) -> None:
    candidate = next((good_run / "stage-19" / "sections").glob("*.md"))
    candidate.unlink()
    assert "sectional_attempt_artifact_missing" in _codes(_check(good_run))


def test_sectional_unmanifested_temp_file_is_blocked(good_run: Path) -> None:
    _write(good_run / "stage-19" / "sections" / "forged.tmp", "stale")
    assert "sectional_attempt_artifact_unmanifested" in _codes(_check(good_run))


def test_sectional_resolution_ids_are_replayed(good_run: Path) -> None:
    attempts_text = _rewrite_jsonl_record(
        good_run,
        "section_attempts.jsonl",
        lambda payload: payload.update(resolution_comment_ids=[]),
    )
    manifest = _stage19_manifest(good_run)
    manifest["attempts_sha256"] = ra.sha256_text(attempts_text)
    _write_stage19_manifest(good_run, manifest)
    assert "sectional_validation_recompute_mismatch" in _codes(_check(good_run))


def test_sectional_validation_report_edit_is_blocked(good_run: Path) -> None:
    attempt = json.loads(
        (good_run / "stage-19" / "section_attempts.jsonl").read_text(
            encoding="utf-8"
        )
    )
    path = good_run / attempt["validation_report_path"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["candidate_sha256"] = "0" * 64
    text = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    path.write_text(text, encoding="utf-8")
    attempts_text = _rewrite_jsonl_record(
        good_run,
        "section_attempts.jsonl",
        lambda row: row.update(validation_report_sha256=ra.sha256_text(text)),
    )
    manifest = _stage19_manifest(good_run)
    manifest["attempts_sha256"] = ra.sha256_text(attempts_text)
    _write_stage19_manifest(good_run, manifest)
    assert "sectional_validation_recompute_mismatch" in _codes(_check(good_run))


def test_sectional_third_critic_identity_is_blocked(good_run: Path) -> None:
    assessments_text = _rewrite_jsonl_record(
        good_run,
        "resolution_assessments.jsonl",
        lambda payload: payload.update(critic_model="third-critic"),
    )
    manifest = _stage19_manifest(good_run)
    manifest["assessments_sha256"] = ra.sha256_text(assessments_text)
    _write_stage19_manifest(good_run, manifest)
    assert "sectional_run_manifest_model_mismatch" in _codes(_check(good_run))


def test_sectional_assessment_missing_reason_is_blocked(good_run: Path) -> None:
    _rewrite_jsonl_record(
        good_run,
        "resolution_assessments.jsonl",
        lambda payload: payload.update(reason=""),
    )
    assert "sectional_assessment_log_invalid" in _codes(_check(good_run))


def test_sectional_orphan_assessment_is_blocked(good_run: Path) -> None:
    def orphan(payload: dict) -> None:
        payload["attempt_id"] = payload["attempt_id"].rsplit("-a", 1)[0] + "-a2"
        payload["assessment_id"] = (
            f"ra-{payload['comment_id']}-{payload['attempt_id']}"
        )

    assessments_text = _rewrite_jsonl_record(
        good_run, "resolution_assessments.jsonl", orphan
    )
    manifest = _stage19_manifest(good_run)
    manifest["assessments_sha256"] = ra.sha256_text(assessments_text)
    _write_stage19_manifest(good_run, manifest)
    assert "sectional_assessment_identity_mismatch" in _codes(_check(good_run))


def test_sectional_revised_paper_must_equal_replay(good_run: Path) -> None:
    path = good_run / "stage-19" / "paper_revised.md"
    path.write_text(path.read_text(encoding="utf-8") + "forged\n", encoding="utf-8")
    assert "sectional_merge_body_mismatch" in _codes(_check(good_run))


def test_sectional_manifest_sections_are_recomputed(good_run: Path) -> None:
    manifest = _stage19_manifest(good_run)
    manifest["sections"] = list(reversed(manifest["sections"]))
    _write_stage19_manifest(good_run, manifest)
    assert "sectional_manifest_sections_mismatch" in _codes(_check(good_run))


def test_sectional_root_model_binding_is_required(good_run: Path) -> None:
    path = good_run / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["reviewer"].pop("sectional_critic_model")
    _write(path, manifest)
    assert "sectional_run_manifest_binding_missing" in _codes(_check(good_run))


def test_pipeline_validation_sectional_bundle_never_releases(good_run: Path) -> None:
    contract_path = good_run / "stage-09" / "experiment_contract.yaml"
    contract = contract_path.read_text(encoding="utf-8")
    contract = contract.replace(
        "claim_scope: research_release", "claim_scope: pipeline_validation"
    ).replace("dataset_origin: public", "dataset_origin: synthetic")
    _write(contract_path, contract)
    sealed_path = good_run / "stage-10" / "selected_candidate_manifest.json"
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    sealed["contract_sha256"] = ra.sha256_file(contract_path)
    _write(sealed_path, sealed)
    result = execute_sectional_revision(
        stage_dir=good_run / "stage-19",
        run_dir=good_run,
        config=PaperRevisionConfig(
            sectional_enabled=True,
            max_section_retries=0,
            min_length_ratio=0.5,
            max_length_ratio=2.0,
            critic_model="section-critic-z",
        ),
        claim_scope="pipeline_validation",
        provider=_SectionalFixtureProvider(),
    )
    assert result.completed is True

    checker = _check(good_run)
    assert "non_release_claim_scope" in _codes(checker)
    assert "sectional_non_release_claim_scope" in _codes(checker)
    assert checker.exit_code() == release_check.EXIT_FAIL


def test_sectional_error_plus_degradation_is_exit_one(good_run: Path) -> None:
    (good_run / "stage-19" / "section_revision_manifest.json").unlink()
    _write(good_run / "degradation_signal.json", {"generated": "2026-07-11T00:00:00+00:00"})
    checker = _check(good_run)
    assert checker.exit_code() == release_check.EXIT_FAIL


def test_final_stage_is_manifest_driven(good_run: Path) -> None:
    summary = json.loads((good_run / "pipeline_summary.json").read_text())
    summary["final_stage"] = 23  # legacy terminal stage
    _write(good_run / "pipeline_summary.json", summary)
    assert "incomplete_run" in _codes(_check(good_run))


def test_missing_manifest_fails_closed(good_run: Path) -> None:
    (good_run / "run_manifest.json").unlink()
    assert "missing_artifact" in _codes(_check(good_run))


def test_compile_toolchain_missing_has_distinct_blocker(good_run: Path) -> None:
    _write(good_run / "stage-22" / "paper.tex", "\\documentclass{article}\\begin{document}x\\end{document}")
    _write(
        good_run / "stage-22" / "compile_status.json",
        {
            "success": False,
            "attempts": 0,
            "status": "toolchain_missing",
            "tooling_available": False,
            "errors": ["pdflatex not installed"],
        },
    )
    codes = _codes(_check(good_run))
    assert "compile_toolchain_missing" in codes
    assert "compile_failed" not in codes


def test_canonical_source_hash_mismatch_fails(good_run: Path) -> None:
    metadata = json.loads((good_run / "stage-23" / "canonical_source.json").read_text())
    metadata["markdown_path"] = "stage-23/paper_final_verified.md"
    metadata["markdown_sha256"] = "0" * 64
    _write(good_run / "stage-23" / "canonical_source.json", metadata)
    assert "canonical_source_hash_mismatch" in _codes(_check(good_run))


def test_canonical_source_hash_missing_fails(good_run: Path) -> None:
    _write(good_run / "stage-22" / "paper.tex", "\\documentclass{article}\\begin{document}x\\end{document}")
    _write(good_run / "stage-22" / "compile_status.json", {"success": True})
    metadata = json.loads((good_run / "stage-23" / "canonical_source.json").read_text())
    metadata["latex_path"] = "stage-22/paper.tex"
    metadata.pop("latex_sha256", None)
    _write(good_run / "stage-23" / "canonical_source.json", metadata)
    assert "canonical_source_hash_missing" in _codes(_check(good_run))


def test_reviewer_not_isolated(good_run: Path) -> None:
    manifest = json.loads((good_run / "run_manifest.json").read_text())
    manifest["reviewer"]["critic_model"] = "writer-model-y"
    _write(good_run / "run_manifest.json", manifest)
    assert "reviewer_not_isolated" in _codes(_check(good_run))


def test_reviewer_shared_context(good_run: Path) -> None:
    manifest = json.loads((good_run / "run_manifest.json").read_text())
    manifest["reviewer"]["shared_context"] = True
    _write(good_run / "run_manifest.json", manifest)
    assert "reviewer_shared_context" in _codes(_check(good_run))


def test_external_reviewer_route_passes(good_run: Path) -> None:
    manifest = json.loads((good_run / "run_manifest.json").read_text())
    manifest["reviewer"].update(
        {
            "critic_model": "",
            "critic_source": "external",
            "external_review_path": "reviews_external/claude_review.md",
        }
    )
    _write(good_run / "run_manifest.json", manifest)
    _write(
        good_run / "reviews_external" / "claude_review.md",
        "# External review\n\nMethod is sound; single-seed concern addressed.",
    )
    critique = json.loads((good_run / "stage-15" / "critique.json").read_text())
    critique.update({"critic_source": "external", "critic_model": ""})
    _write(good_run / "stage-15" / "critique.json", critique)
    assert _codes(_check(good_run)) == set()


def test_external_reviewer_requires_nonempty_artifact(good_run: Path) -> None:
    manifest = json.loads((good_run / "run_manifest.json").read_text())
    manifest["reviewer"].update(
        {
            "critic_source": "external",
            "external_review_path": "reviews_external/claude_review.md",
        }
    )
    _write(good_run / "run_manifest.json", manifest)
    # missing artifact
    assert "external_review_artifact_missing" in _codes(_check(good_run))
    # empty artifact is also not a review
    _write(good_run / "reviews_external" / "claude_review.md", "")
    assert "external_review_artifact_missing" in _codes(_check(good_run))


def test_external_review_path_must_stay_in_run_dir(good_run: Path) -> None:
    manifest = json.loads((good_run / "run_manifest.json").read_text())
    manifest["reviewer"].update(
        {"critic_source": "external", "external_review_path": "../outside.md"}
    )
    _write(good_run / "run_manifest.json", manifest)
    _write(good_run.parent / "outside.md", "not a run artifact")
    assert "external_review_artifact_missing" in _codes(_check(good_run))


# ---------------------------------------------------------------------------
# Regression tests for the 5 gate-hole fixes (Codex review round 2)
# ---------------------------------------------------------------------------

def test_supported_claim_with_empty_evidence_fails(good_run: Path) -> None:
    # Hole 1a: status=supported but evidence list empty must NOT pass.
    claims_data = json.loads((good_run / "stage-24" / "claims.json").read_text())
    claims_data["claims"][0]["evidence"] = []
    _write(good_run / "stage-24" / "claims.json", claims_data)
    codes = _codes(_check(good_run))
    assert "claims_supported_without_evidence" in codes


def test_citation_claim_supported_by_cited_keys_only_fails(good_run: Path) -> None:
    # Hole 1b: a citation-type claim marked supported with only cited_keys
    # (no run-internal evidence pointer) must NOT pass.
    claims_data = json.loads((good_run / "stage-24" / "claims.json").read_text())
    claims_data["claims"].append(
        {
            "id": "clm-0001",
            "text": "Prior work [jones2023survey] surveys the field.",
            "type": "citation",
            "values": [],
            "cited_keys": ["jones2023survey"],
            "evidence": [],  # cited_keys only, no run-internal evidence
            "status": "supported",
        }
    )
    _write(good_run / "stage-24" / "claims.json", claims_data)
    # digest will also mismatch, but the provenance hole is the point:
    assert "claims_supported_without_evidence" in _codes(_check(good_run))


def test_unpinned_evidence_pointer_fails(good_run: Path) -> None:
    # Hole 1c: evidence pointer without a sha256 is not verifiable → orphan.
    claims_data = json.loads((good_run / "stage-24" / "claims.json").read_text())
    claims_data["claims"][0]["evidence"] = [
        {"path": "stage-14/experiment_summary.json"}  # no sha256
    ]
    _write(good_run / "stage-24" / "claims.json", claims_data)
    codes = _codes(_check(good_run))
    assert "claims_orphan_evidence" in codes or "claims_supported_without_evidence" in codes


def test_disallowed_in_run_evidence_path_fails(good_run: Path) -> None:
    # A forged file inside the run dir is not release evidence merely because
    # it exists and its sha256 matches. It must be in the evidence allowlist.
    fake_rel = "stage-24/fake_evidence.json"
    _write(good_run / fake_rel, {"loss": 0.1234})
    claims_data = json.loads((good_run / "stage-24" / "claims.json").read_text())
    claims_data["claims"][0]["evidence"] = [
        {
            "path": fake_rel,
            "sha256": ra.sha256_file(good_run / fake_rel),
            "matched_value": 0.1234,
        }
    ]
    _write(good_run / "stage-24" / "claims.json", claims_data)
    assert "claims_disallowed_evidence_path" in _codes(_check(good_run))


def test_evidence_allowlist_does_not_cross_directory_segments(good_run: Path) -> None:
    # The stage-12/runs allowlist permits one JSON file segment only; nested
    # files must not pass through a glob-style "*" that crosses '/'.
    fake_rel = "stage-12/runs/nested/fake.json"
    _write(good_run / fake_rel, {"loss": 0.1234})
    claims_data = json.loads((good_run / "stage-24" / "claims.json").read_text())
    claims_data["claims"][0]["evidence"] = [
        {
            "path": fake_rel,
            "sha256": ra.sha256_file(good_run / fake_rel),
            "matched_value": 0.1234,
        }
    ]
    _write(good_run / "stage-24" / "claims.json", claims_data)
    assert "claims_disallowed_evidence_path" in _codes(_check(good_run))


def test_evidence_allowlist_rejects_stage_prefix_spoofing(good_run: Path) -> None:
    fake_rel = "stage-14evil/experiment_summary.json"
    _write(good_run / fake_rel, {"loss": 0.1234})
    claims_data = json.loads((good_run / "stage-24" / "claims.json").read_text())
    claims_data["claims"][0]["evidence"] = [
        {
            "path": fake_rel,
            "sha256": ra.sha256_file(good_run / fake_rel),
            "matched_value": 0.1234,
        }
    ]
    _write(good_run / "stage-24" / "claims.json", claims_data)
    assert "claims_disallowed_evidence_path" in _codes(_check(good_run))


def test_background_without_whitelist_fails(good_run: Path) -> None:
    # Hole 2: role=background must NOT be a blanket escape.
    citations = json.loads((good_run / "stage-24" / "citations.json").read_text())
    citations.pop("background_whitelist", None)
    _write(good_run / "stage-24" / "citations.json", citations)
    assert "citation_background_not_whitelisted" in _codes(_check(good_run))


def test_background_whitelist_by_section_passes(good_run: Path) -> None:
    # The escape is allowed ONLY via an explicit, auditable whitelist.
    citations = json.loads((good_run / "stage-24" / "citations.json").read_text())
    citations["background_whitelist"] = {"cite_keys": [], "sections": ["related work"]}
    citations["instances"][1]["section"] = "Related Work"
    _write(good_run / "stage-24" / "citations.json", citations)
    assert "citation_background_not_whitelisted" not in _codes(_check(good_run))


def test_truth_audit_empty_paper_path_fails(good_run: Path) -> None:
    # Hole 3a: empty paper_path must fail, not silently skip.
    truth = json.loads((good_run / "stage-24" / "truth_audit.json").read_text())
    truth["paper_path"] = ""
    _write(good_run / "stage-24" / "truth_audit.json", truth)
    assert "truth_audit_paper_path_missing" in _codes(_check(good_run))


def test_truth_audit_missing_paper_file_fails(good_run: Path) -> None:
    # Hole 3b: paper_path points at a non-existent file must fail, not skip.
    truth = json.loads((good_run / "stage-24" / "truth_audit.json").read_text())
    truth["paper_path"] = "stage-23/does_not_exist.md"
    _write(good_run / "stage-24" / "truth_audit.json", truth)
    assert "truth_audit_paper_missing" in _codes(_check(good_run))


def test_cost_log_delta_sum_not_double_counted() -> None:
    # Hole 4: CostGuard must use cumulative_usd (last), never sum cumulative.
    import tempfile
    from researchclaw.hitl.cost_guard import CostGuard
    from researchclaw.pipeline import release_artifacts as _ra

    with tempfile.TemporaryDirectory() as d:
        run = Path(d)
        # Simulate 3 stages: cumulative grows 0.10 → 0.30 → 0.55.
        for stage, (delta, cum) in enumerate(
            [(0.10, 0.10), (0.20, 0.30), (0.25, 0.55)], start=1
        ):
            _ra.append_cost_entry(
                run, stage=stage, stage_name=f"S{stage}", model="m",
                attempt_id=f"a{stage}", cost_usd=delta, cumulative_usd=cum,
            )
        guard = CostGuard(budget_usd=0.0)
        total = guard._get_total_cost(run)
        # Correct total is 0.55 (last cumulative), NOT 0.10+0.30+0.55=0.95.
        assert abs(total - 0.55) < 1e-9, total


def test_cost_log_delta_only_sums_when_no_cumulative() -> None:
    import tempfile
    from researchclaw.hitl.cost_guard import CostGuard
    from researchclaw.pipeline import release_artifacts as _ra

    with tempfile.TemporaryDirectory() as d:
        run = Path(d)
        for stage, delta in enumerate([0.10, 0.20, 0.25], start=1):
            _ra.append_cost_entry(
                run, stage=stage, stage_name=f"S{stage}", model="m",
                attempt_id=f"a{stage}", cost_usd=delta, cumulative_usd=None,
            )
        guard = CostGuard(budget_usd=0.0)
        assert abs(guard._get_total_cost(run) - 0.55) < 1e-9


def test_degraded_deliverables_must_flag_not_release_ready(good_run: Path) -> None:
    # Hole 5: a degraded run whose deliverables don't declare not_release_ready
    # (or falsely claim release_ready) must fail.
    _write(good_run / "degradation_signal.json", {"score": 3.0, "threshold": 5.0})
    _write(
        good_run / "deliverables" / "manifest.json",
        {"files": ["paper_final.md"], "release_ready": True},
    )
    assert "deliverables_not_flagged_not_release_ready" in _codes(_check(good_run))


def test_degraded_deliverables_pass_when_flagged(good_run: Path) -> None:
    # Correctly flagged deliverables for a degraded run do not trip THIS gate
    # (the run still fails elsewhere on degradation — that's separate).
    _write(good_run / "degradation_signal.json", {"score": 3.0, "threshold": 5.0})
    _write(
        good_run / "deliverables" / "manifest.json",
        {"files": ["paper_final.md"], "release_ready": False, "not_release_ready": True},
    )
    assert "deliverables_not_flagged_not_release_ready" not in _codes(_check(good_run))


# ---------------------------------------------------------------------------
# Regression tests for the 2 P0 fixes (Codex review round 3)
# ---------------------------------------------------------------------------

def test_quant_claim_with_unrelated_evidence_fails(good_run: Path) -> None:
    # P0-1: a quantitative claim whose only evidence pointer carries a
    # matched_value that does NOT correspond to any number in the claim
    # (values or text) must fail numeric closure.
    claims_data = json.loads((good_run / "stage-24" / "claims.json").read_text())
    ev_path = claims_data["claims"][0]["evidence"][0]["path"]
    ev_sha = claims_data["claims"][0]["evidence"][0]["sha256"]
    # Claim states 0.1234; attach evidence matched_value 9.9999 (unrelated).
    claims_data["claims"][0]["evidence"] = [
        {"path": ev_path, "sha256": ev_sha, "matched_value": 9.9999}
    ]
    _write(good_run / "stage-24" / "claims.json", claims_data)
    codes = _codes(_check(good_run))
    assert "claims_numeric_not_closed" in codes


def test_quant_claim_empty_values_generic_evidence_fails(good_run: Path) -> None:
    # P0-1: a quantitative claim with values=[] pointed at generic evidence
    # that carries NO matched_value must not be waved through.
    claims_data = json.loads((good_run / "stage-24" / "claims.json").read_text())
    ev_path = claims_data["claims"][0]["evidence"][0]["path"]
    ev_sha = claims_data["claims"][0]["evidence"][0]["sha256"]
    claims_data["claims"][0]["values"] = []
    # generic pointer, no matched_value (as attempt_log/best_summary would be)
    claims_data["claims"][0]["evidence"] = [{"path": ev_path, "sha256": ev_sha}]
    _write(good_run / "stage-24" / "claims.json", claims_data)
    codes = _codes(_check(good_run))
    # text still contains 0.1234, so the claim has numbers, but no evidence
    # matched_value closes them → numeric-not-closed.
    assert "claims_numeric_not_closed" in codes


def test_quant_claim_text_number_closure_passes(good_run: Path) -> None:
    # Positive: values=[] but the claim TEXT number (0.1234) is matched by an
    # evidence pointer's matched_value → closure holds via text extraction.
    claims_data = json.loads((good_run / "stage-24" / "claims.json").read_text())
    ev_path = claims_data["claims"][0]["evidence"][0]["path"]
    ev_sha = claims_data["claims"][0]["evidence"][0]["sha256"]
    claims_data["claims"][0]["values"] = []
    claims_data["claims"][0]["evidence"] = [
        {"path": ev_path, "sha256": ev_sha, "matched_value": 0.1234}
    ]
    _write(good_run / "stage-24" / "claims.json", claims_data)
    assert "claims_numeric_not_closed" not in _codes(_check(good_run))


def test_fabricated_support_excerpt_fails(good_run: Path) -> None:
    # P0-2: a non-empty but fabricated excerpt (not a substring of context or
    # claim text) must fail, not pass merely on non-emptiness.
    citations = json.loads((good_run / "stage-24" / "citations.json").read_text())
    citations["instances"][0]["support_excerpt"] = (
        "This sentence appears nowhere in the paper or the claim."
    )
    _write(good_run / "stage-24" / "citations.json", citations)
    assert "citation_support_excerpt_fabricated" in _codes(_check(good_run))


def test_support_excerpt_from_context_passes(good_run: Path) -> None:
    # Excerpt drawn from the instance context (not the claim text) is valid.
    citations = json.loads((good_run / "stage-24" / "citations.json").read_text())
    citations["instances"][0]["context"] = (
        "As shown in Table 2, the detector attains strong recall on held-out traces."
    )
    citations["instances"][0]["support_excerpt"] = "the detector attains strong recall"
    _write(good_run / "stage-24" / "citations.json", citations)
    assert "citation_support_excerpt_fabricated" not in _codes(_check(good_run))


def test_matched_value_absent_from_evidence_file_fails(good_run: Path) -> None:
    # R4 P0: claim states 0.1234; evidence points at an attempt_log that does
    # NOT contain 0.1234. Even with matched_value=0.1234 and a correct sha256,
    # release_check must fail — a matched_value asserted only in claims.json
    # is not evidence.
    attempt_log = good_run / "attempts" / "attempt_log.jsonl"
    attempt_log.parent.mkdir(parents=True, exist_ok=True)
    # Contains numbers, but NOT 0.1234.
    attempt_log.write_text(
        json.dumps({"stage": 12, "status": "ok", "elapsed_sec": 3.5, "seed": 7}) + "\n",
        encoding="utf-8",
    )
    att_sha = ra.sha256_file(attempt_log)
    claims_data = json.loads((good_run / "stage-24" / "claims.json").read_text())
    claims_data["claims"][0]["evidence"] = [
        {"path": "attempts/attempt_log.jsonl", "sha256": att_sha, "matched_value": 0.1234}
    ]
    _write(good_run / "stage-24" / "claims.json", claims_data)
    codes = _codes(_check(good_run))
    assert "claims_numeric_evidence_value_missing" in codes
    assert "claims_numeric_not_closed" in codes


def test_matched_value_present_in_evidence_json_passes(good_run: Path) -> None:
    # R4 positive: evidence JSON really contains 0.1234 → closure holds.
    ev = good_run / "stage-14" / "extra_metrics.json"
    ev.write_text(json.dumps({"results": {"loss": 0.1234, "acc": 0.9}}), encoding="utf-8")
    ev_sha = ra.sha256_file(ev)
    claims_data = json.loads((good_run / "stage-24" / "claims.json").read_text())
    claims_data["claims"][0]["evidence"] = [
        {"path": "stage-14/extra_metrics.json", "sha256": ev_sha, "matched_value": 0.1234}
    ]
    _write(good_run / "stage-24" / "claims.json", claims_data)
    codes = _codes(_check(good_run))
    assert "claims_numeric_evidence_value_missing" not in codes
    assert "claims_numeric_not_closed" not in codes


def test_matched_value_in_text_artifact_passes(good_run: Path) -> None:
    # Deterministic extraction must also work on non-JSON (text) artifacts.
    ev = good_run / "stage-14" / "log.txt"
    ev.write_text("Epoch 3: validation loss = 0.1234 (best so far)\n", encoding="utf-8")
    ev_sha = ra.sha256_file(ev)
    claims_data = json.loads((good_run / "stage-24" / "claims.json").read_text())
    claims_data["claims"][0]["evidence"] = [
        {"path": "stage-14/log.txt", "sha256": ev_sha, "matched_value": 0.1234}
    ]
    _write(good_run / "stage-24" / "claims.json", claims_data)
    assert "claims_numeric_evidence_value_missing" not in _codes(_check(good_run))


def test_numbers_in_artifact_json_and_text() -> None:
    import tempfile
    from researchclaw.pipeline import release_artifacts as _ra

    with tempfile.TemporaryDirectory() as d:
        j = Path(d) / "m.json"
        j.write_text(json.dumps({"a": {"f1": 0.947}, "b": [1, 2.5]}), encoding="utf-8")
        nums = _ra.numbers_in_artifact(j)
        assert any(abs(n - 0.947) < 1e-9 for n in nums)
        assert any(abs(n - 2.5) < 1e-9 for n in nums)
        t = Path(d) / "m.txt"
        t.write_text("f1=0.947, fpr=2.1%", encoding="utf-8")
        tnums = _ra.numbers_in_artifact(t)
        assert any(abs(n - 0.947) < 1e-9 for n in tnums)


def test_extract_numbers_deterministic() -> None:
    from researchclaw.pipeline import release_artifacts as _ra
    nums = _ra.extract_numbers("F1 of 0.947 (±0.011), FPR 2.1%, over 1,200 windows and 5 seeds")
    assert 0.947 in nums and 0.011 in nums and 2.1 in nums and 1200.0 in nums and 5.0 in nums
    assert _ra.extract_numbers("no digits here at all") == []


def test_unsupported_claim_fails(good_run: Path) -> None:
    claims_data = json.loads((good_run / "stage-24" / "claims.json").read_text())
    claims_data["claims"][0]["status"] = "unsupported"
    claims_data["claims"][0]["evidence"] = []
    _write(good_run / "stage-24" / "claims.json", claims_data)
    codes = _codes(_check(good_run))
    assert "claims_unsupported" in codes
    # digest changed too — invariance must also trip
    assert "claims_digest_mismatch" in codes


def test_orphan_evidence_fails(good_run: Path) -> None:
    # Tamper with the evidence file after the audit froze its digest.
    _write(
        good_run / "stage-14" / "experiment_summary.json",
        {"metrics_summary": {"loss": {"mean": 0.9999}}},
    )
    assert "claims_orphan_evidence" in _codes(_check(good_run))


def test_citation_existence_is_not_support(good_run: Path) -> None:
    # Keys exist in the verified bib, but an instance is unmapped:
    # existence alone must NOT pass the support gate.
    citations = json.loads((good_run / "stage-24" / "citations.json").read_text())
    citations["instances"][0]["role"] = "unmapped"
    _write(good_run / "stage-24" / "citations.json", citations)
    assert "citation_support_unmapped" in _codes(_check(good_run))


def test_citation_keys_with_empty_instances_fails(good_run: Path) -> None:
    citations = json.loads((good_run / "stage-24" / "citations.json").read_text())
    citations["instances"] = []
    _write(good_run / "stage-24" / "citations.json", citations)
    assert "citation_support_instances_missing" in _codes(_check(good_run))


def test_claim_support_requires_excerpt(good_run: Path) -> None:
    citations = json.loads((good_run / "stage-24" / "citations.json").read_text())
    citations["instances"][0]["support_excerpt"] = ""
    _write(good_run / "stage-24" / "citations.json", citations)
    assert "citation_support_invalid" in _codes(_check(good_run))


def test_prose_edit_after_truth_audit_fails(good_run: Path) -> None:
    # Simulate adopting a de-AI rewrite without re-running the truth audit.
    tampered = PAPER.replace("beating", "delving past")
    _write(good_run / "stage-23" / "paper_final_verified.md", tampered)
    assert "paper_hash_mismatch" in _codes(_check(good_run))


def test_deai_applied_fails(good_run: Path) -> None:
    deai = json.loads((good_run / "stage-25" / "deai_audit.json").read_text())
    deai["applied"] = True
    _write(good_run / "stage-25" / "deai_audit.json", deai)
    assert "deai_audit_applied" in _codes(_check(good_run))


def test_unresolved_critique_fails(good_run: Path) -> None:
    _write(
        good_run / "stage-24" / "critique_resolution.json",
        {"resolutions": [{"finding_id": "sc-01", "resolution": "unresolved"}]},
    )
    assert "critique_findings_unresolved" in _codes(_check(good_run))


def test_no_real_data_fails_without_waiver(good_run: Path) -> None:
    _write(
        good_run / "stage-20" / "fabrication_flags.json",
        {"fabrication_suspected": False, "has_real_data": False},
    )
    assert "no_real_data" in _codes(_check(good_run))


def test_no_real_data_waiver_downgrades_to_warning(good_run: Path) -> None:
    _write(
        good_run / "stage-20" / "fabrication_flags.json",
        {"fabrication_suspected": False, "has_real_data": False},
    )
    _write(
        good_run / "waivers" / "no_real_data.json",
        {"reason": "theory-only run; no experiment applicable", "approved_by": "marcel"},
    )
    checker = _check(good_run)
    assert "no_real_data" not in _codes(checker)
    assert any(f.code == "no_real_data_waived" for f in checker.findings)


def test_release_check_blocks_synthetic_research_release(good_run: Path) -> None:
    contract = (good_run / "stage-09" / "experiment_contract.yaml").read_text(
        encoding="utf-8"
    )
    contract = contract.replace(
        "dataset_origin: public", "dataset_origin: synthetic"
    )
    _write(good_run / "stage-09" / "experiment_contract.yaml", contract)
    assert "experiment_contract_invalid" in _codes(_check(good_run))


def test_synthetic_research_release_waiver_is_ineffective(good_run: Path) -> None:
    contract = (good_run / "stage-09" / "experiment_contract.yaml").read_text(
        encoding="utf-8"
    )
    contract = contract.replace(
        "dataset_origin: public", "dataset_origin: synthetic"
    )
    _write(good_run / "stage-09" / "experiment_contract.yaml", contract)
    _write(
        good_run / "waivers" / "synthetic_research_release.json",
        {"reason": "benchmark-only methods paper", "approved_by": "human-reviewer"},
    )
    checker = _check(good_run)
    assert "experiment_contract_invalid" in _codes(checker)
    assert all(
        f.code != "synthetic_research_release_waived" for f in checker.findings
    )


def test_release_check_requires_experiment_contract(good_run: Path) -> None:
    (good_run / "stage-09" / "experiment_contract.yaml").unlink()
    assert "experiment_contract_missing" in _codes(_check(good_run))


def test_release_check_blocks_pipeline_validation_claim_scope(good_run: Path) -> None:
    contract = (good_run / "stage-09" / "experiment_contract.yaml").read_text(
        encoding="utf-8"
    )
    contract = contract.replace(
        "claim_scope: research_release", "claim_scope: pipeline_validation"
    )
    _write(good_run / "stage-09" / "experiment_contract.yaml", contract)
    assert "non_release_claim_scope" in _codes(_check(good_run))


def test_release_check_blocks_exploratory_claim_scope(good_run: Path) -> None:
    contract = (good_run / "stage-09" / "experiment_contract.yaml").read_text(
        encoding="utf-8"
    )
    contract = contract.replace(
        "claim_scope: research_release", "claim_scope: exploratory"
    )
    _write(good_run / "stage-09" / "experiment_contract.yaml", contract)
    assert "non_release_claim_scope" in _codes(_check(good_run))


def test_release_check_passes_research_release_claim_scope(good_run: Path) -> None:
    # good_run fixture already has claim_scope: research_release
    checker = _check(good_run)
    assert "non_release_claim_scope" not in _codes(checker)


def test_release_check_uses_runtime_contract_validation(good_run: Path) -> None:
    contract_path = good_run / "stage-09" / "experiment_contract.yaml"
    contract = contract_path.read_text(encoding="utf-8").replace(
        "owner: scaffold", "owner: model"
    )
    _write(contract_path, contract)
    assert "experiment_contract_invalid" in _codes(_check(good_run))


def test_contract_validator_import_failure_is_explicit_finding(
    good_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fail_runtime_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "researchclaw.experiment_runtime":
            raise ImportError("package unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_runtime_import)
    checker = release_check.ReleaseChecker(
        good_run, quality_threshold=5.0, allow_suspicious=False
    )
    checker.check_experiment_contract()
    assert "experiment_contract_unverifiable" in _codes(checker)


def test_invalid_utf8_contract_is_explicit_finding(good_run: Path) -> None:
    (good_run / "stage-09" / "experiment_contract.yaml").write_bytes(b"\xff\xfe")
    assert "experiment_contract_invalid" in _codes(_check(good_run))


def test_local_hardware_research_release_contract_passes(good_run: Path) -> None:
    contract_path = good_run / "stage-09" / "experiment_contract.yaml"
    contract = contract_path.read_text(encoding="utf-8").replace(
        "dataset_origin: public", "dataset_origin: local_hardware"
    )
    _write(contract_path, contract)
    assert "experiment_contract_invalid" not in _codes(_check(good_run))
    assert "non_release_claim_scope" not in _codes(_check(good_run))


def test_contract_selector_prefers_direct_stage09(good_run: Path) -> None:
    direct = good_run / "stage-09" / "experiment_contract.yaml"
    versioned = direct.read_text(encoding="utf-8").replace(
        "claim_scope: research_release", "claim_scope: exploratory"
    )
    _write(good_run / "stage-09_v2" / "experiment_contract.yaml", versioned)
    assert "non_release_claim_scope" not in _codes(_check(good_run))


def test_contract_selector_uses_latest_version_when_direct_missing(
    good_run: Path,
) -> None:
    direct = good_run / "stage-09" / "experiment_contract.yaml"
    release_contract = direct.read_text(encoding="utf-8")
    direct.unlink()
    _write(good_run / "stage-09_v1" / "experiment_contract.yaml", release_contract)
    exploratory = release_contract.replace(
        "claim_scope: research_release", "claim_scope: exploratory"
    )
    _write(good_run / "stage-09_v2" / "experiment_contract.yaml", exploratory)
    assert "non_release_claim_scope" in _codes(_check(good_run))


def test_contract_selector_orders_versions_numerically(good_run: Path) -> None:
    direct = good_run / "stage-09" / "experiment_contract.yaml"
    release_contract = direct.read_text(encoding="utf-8")
    direct.unlink()
    exploratory = release_contract.replace(
        "claim_scope: research_release", "claim_scope: exploratory"
    )
    _write(good_run / "stage-09_v2" / "experiment_contract.yaml", exploratory)
    _write(good_run / "stage-09_v10" / "experiment_contract.yaml", release_contract)
    assert "non_release_claim_scope" not in _codes(_check(good_run))


def test_evidence_path_rejects_stage10_smoke() -> None:
    assert not release_check.is_allowed_claim_evidence_path(
        "stage-10/smoke/smoke_results.json", "quantitative"
    )


def test_evidence_path_rejects_stage10_candidates() -> None:
    assert not release_check.is_allowed_claim_evidence_path(
        "stage-10/candidates/cand-001/attempt.json", "quantitative"
    )


def test_placeholder_paper_fails(good_run: Path) -> None:
    _write(good_run / "stage-22" / "paper_final.md", "# Final Paper\n\nNo content generated.")
    assert "paper_artifact_placeholder" in _codes(_check(good_run))


def test_cost_log_missing_is_warning_not_error(good_run: Path) -> None:
    (good_run / "cost_log.jsonl").unlink()
    checker = _check(good_run)
    assert "cost_log_missing" not in _codes(checker)
    assert any(f.code == "cost_log_missing" for f in checker.findings)


# ---------------------------------------------------------------------------
# release_artifacts unit tests
# ---------------------------------------------------------------------------

def test_paper_hash_whitespace_insensitive() -> None:
    a = ra.paper_sha256("Hello   world\n\n")
    b = ra.paper_sha256("Hello world")
    c = ra.paper_sha256("Hello worlds")
    assert a == b and a != c


def test_claims_digest_ignores_notes_but_not_evidence() -> None:
    base = [{"id": "c1", "text": "x", "type": "result", "status": "supported",
             "evidence": [{"path": "p", "sha256": "s"}]}]
    with_note = [dict(base[0], note="commentary")]
    changed_ev = [dict(base[0], evidence=[{"path": "p2", "sha256": "s"}])]
    assert ra.claims_digest(base) == ra.claims_digest(with_note)
    assert ra.claims_digest(base) != ra.claims_digest(changed_ev)


def test_attempt_log_append_only(tmp_path: Path) -> None:
    e1 = ra.append_attempt(tmp_path, run_id="r", stage=12, stage_name="EXPERIMENT_RUN",
                           status="failed", error="boom")
    e2 = ra.append_attempt(tmp_path, run_id="r", stage=12, stage_name="EXPERIMENT_RUN",
                           status="done")
    assert e1["attempt"] == 1 and e2["attempt"] == 2
    lines = (tmp_path / "attempts" / "attempt_log.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2  # the failed attempt is preserved


def test_citation_instance_extraction() -> None:
    text = r"A result \cite{smith2024deep,jones2023survey} and [smith2024deep]."
    instances = ra.extract_citation_instances(text)
    keys = [i["cite_key"] for i in instances]
    assert keys.count("smith2024deep") == 2
    assert "jones2023survey" in keys


def test_final_stage_is_25() -> None:
    assert int(FINAL_STAGE) == 25
    assert Stage.TRUTH_AUDIT == 24 and Stage.DEAI_AUDIT == 25
