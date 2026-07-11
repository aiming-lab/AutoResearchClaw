"""Read-only disk replay for the Stage 19 sectional release boundary.

The audit proves structural and provenance closure from run-local artifacts.
It cannot prove that a recorded critic verdict is semantically truthful without
an external signature; that remains part of the trusted run-production boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from researchclaw.config import PaperRevisionConfig
from researchclaw.experiment_runtime.contract import ExperimentContract, sha256_file
from researchclaw.pipeline.manuscript_sections import parse_manuscript
from researchclaw.pipeline.sectional_revision import (
    ReviewLedger,
    RevisionPlan,
    SectionalRevisionContractError,
    extract_review_ledger,
    validate_review_ledger,
    validate_revision_plan,
)
from researchclaw.pipeline.sectional_execution import (
    SectionalExecutionError,
    build_validation_context,
)
from researchclaw.pipeline.sectional_validation import (
    ResolutionAssessmentRecord,
    SectionAttemptRecord,
    SectionManifestMetadata,
    SectionRevisionManifest,
    SectionValidationContext,
    SectionValidationResult,
    ValidatedSectionReplacement,
    _validate_validation_context_payload,
    build_section_revision_manifest,
    build_unresolved_comments_artifact,
    merge_validated_sections,
    parse_resolution_assessments_jsonl,
    parse_section_attempts_jsonl,
    validate_section_candidate,
    validate_section_revision_manifest,
)


_STAGE19_FILES = (
    "review_comment_ledger.json",
    "revision_plan.json",
    "validation_context.json",
    "section_attempts.jsonl",
    "resolution_assessments.jsonl",
    "unresolved_comments.json",
    "paper_revised.md",
)


@dataclass(frozen=True)
class SectionalAuditError(RuntimeError):
    code: str
    message: str
    path: str = ""

    def __str__(self) -> str:
        return self.message


def audit_sectional_revision(
    run_dir: Path,
    *,
    run_manifest: Mapping[str, Any] | None,
    contract_path: Path,
    contract: ExperimentContract,
) -> None:
    """Replay a complete sectional bundle without trusting stored conclusions."""

    stage19 = run_dir / "stage-19"
    manifest_path = stage19 / "section_revision_manifest.json"
    if not manifest_path.is_file():
        _raise(
            "sectional_revision_manifest_missing",
            "stage-19/section_revision_manifest.json is required for release",
            "stage-19/section_revision_manifest.json",
        )
    for name in _STAGE19_FILES:
        if not (stage19 / name).is_file():
            _raise(
                "sectional_revision_artifact_missing",
                f"required Stage 19 sectional artifact is missing: {name}",
                f"stage-19/{name}",
            )
    for name in ("sections", "section_validation"):
        if not (stage19 / name).is_dir():
            _raise(
                "sectional_revision_artifact_missing",
                f"required Stage 19 sectional directory is missing: {name}",
                f"stage-19/{name}",
            )

    try:
        manifest = SectionRevisionManifest.from_dict(_read_json(manifest_path))
    except (OSError, ValueError, SectionalRevisionContractError) as exc:
        _raise(
            "sectional_revision_artifact_invalid",
            f"section revision manifest is invalid: {exc}",
            "stage-19/section_revision_manifest.json",
        )
    if not manifest.completed:
        _raise(
            "sectional_revision_incomplete",
            "section revision manifest is not completed",
            "stage-19/section_revision_manifest.json",
        )

    contract_rel = _relative_path(contract_path, run_dir)
    contract_sha = sha256_file(contract_path)
    if manifest.experiment_contract_path != contract_rel:
        _raise(
            "sectional_contract_binding_missing",
            "Stage 19 manifest does not name the canonical Stage 9 contract",
            "stage-19/section_revision_manifest.json",
        )
    if manifest.experiment_contract_sha256 != contract_sha:
        _raise(
            "sectional_contract_hash_mismatch",
            "Stage 19 contract hash differs from the canonical Stage 9 contract",
            contract_rel,
        )
    _check_sealed_contract_anchor(run_dir, contract_sha)
    if manifest.claim_scope != contract.claim_scope:
        _raise(
            "sectional_claim_scope_mismatch",
            "Stage 19 and Stage 9 claim scopes differ",
            "stage-19/section_revision_manifest.json",
        )
    if contract.claim_scope != "research_release":
        _raise(
            "sectional_non_release_claim_scope",
            f"sectional release requires research_release, got {contract.claim_scope!r}",
            contract_rel,
        )

    paper_path = run_dir / "stage-17" / "paper_draft.md"
    reviews_path = run_dir / "stage-18" / "reviews.md"
    if not paper_path.is_file() or not reviews_path.is_file():
        _raise(
            "sectional_revision_artifact_missing",
            "canonical Stage 17 paper and Stage 18 reviews are required",
        )
    paper = _read_text(paper_path, "sectional_revision_artifact_invalid")
    reviews = _read_text(reviews_path, "sectional_revision_artifact_invalid")
    try:
        document = parse_manuscript(paper, strict=True)
    except Exception as exc:  # tokenizer/structure errors are release blockers
        _raise(
            "sectional_merge_structure_mismatch",
            f"Stage 17 manuscript structure is invalid: {exc}",
            "stage-17/paper_draft.md",
        )

    ledger_path = stage19 / "review_comment_ledger.json"
    plan_path = stage19 / "revision_plan.json"
    try:
        ledger = ReviewLedger.from_dict(_read_json(ledger_path))
        fresh_ledger = extract_review_ledger(
            reviews, source_path="stage-18/reviews.md"
        )
        validate_review_ledger(
            ledger,
            reviews=reviews,
            source_path="stage-18/reviews.md",
            require_final=True,
        )
        if not ledger.comments or not fresh_ledger.comments:
            _raise(
                "sectional_review_comments_empty",
                "release review ledger must contain at least one comment",
                "stage-19/review_comment_ledger.json",
            )
        plan = RevisionPlan.from_dict(_read_json(plan_path))
        validate_revision_plan(plan, ledger, document, reviews=reviews)
    except SectionalAuditError:
        raise
    except (OSError, ValueError, SectionalRevisionContractError) as exc:
        code = (
            "sectional_revision_plan_invalid"
            if "plan" in str(exc).casefold()
            else "sectional_ledger_recompute_mismatch"
        )
        _raise(code, f"ledger/plan replay failed: {exc}")
    if manifest.source_paper_path != "stage-17/paper_draft.md":
        _raise(
            "sectional_source_hash_mismatch",
            "manifest source paper path is not canonical",
            "stage-19/section_revision_manifest.json",
        )
    if manifest.source_paper_sha256 != document.source_sha256:
        _raise(
            "sectional_source_hash_mismatch",
            "manifest source paper hash differs from Stage 17",
            "stage-17/paper_draft.md",
        )

    unresolved_path = stage19 / "unresolved_comments.json"
    unresolved_text = _read_text(
        unresolved_path, "sectional_revision_artifact_invalid"
    )
    try:
        unresolved = _read_json_text(unresolved_text)
    except ValueError as exc:
        _raise(
            "sectional_unresolved_artifact_mismatch",
            f"unresolved comments artifact is invalid: {exc}",
            "stage-19/unresolved_comments.json",
        )
    if unresolved != build_unresolved_comments_artifact(ledger):
        _raise(
            "sectional_unresolved_artifact_mismatch",
            "unresolved comments artifact does not recompute from the ledger",
            "stage-19/unresolved_comments.json",
        )
    if unresolved.get("comments") or any(
        comment.final_status != "resolved" for comment in ledger.comments
    ):
        _raise(
            "sectional_unresolved_comments",
            "release bundle contains unresolved or non-actionable review comments",
            "stage-19/unresolved_comments.json",
        )
    expected_counts = {
        "input": len(ledger.comments),
        "resolved": len(ledger.comments),
        "unresolved": 0,
        "not_actionable_with_reason": 0,
    }
    if dict(manifest.comment_counts) != expected_counts:
        _raise(
            "sectional_comment_counts_mismatch",
            "manifest comment counts do not describe complete resolution",
            "stage-19/section_revision_manifest.json",
        )

    context_path = stage19 / "validation_context.json"
    context_text = _read_text(
        context_path, "sectional_validation_context_invalid"
    )
    if _text_sha256(context_text) != manifest.validation_context_sha256:
        _raise(
            "sectional_validation_context_hash_mismatch",
            "validation context hash differs from the Stage 19 manifest",
            "stage-19/validation_context.json",
        )
    try:
        context_payload = _read_json_text(context_text)
        _validate_validation_context_payload(
            context_payload, document=document, section_metadata={}
        )
    except SectionalRevisionContractError as exc:
        if any(
            issue.code == "validation_context_stage10_source"
            for issue in exc.issues
        ):
            _raise(
                "sectional_validation_source_disallowed",
                f"validation context uses a forbidden Stage 10 source: {exc}",
                "stage-19/validation_context.json",
            )
        _raise(
            "sectional_validation_context_invalid",
            f"validation context is invalid: {exc}",
            "stage-19/validation_context.json",
        )
    except ValueError as exc:
        _raise(
            "sectional_validation_context_invalid",
            f"validation context is invalid: {exc}",
            "stage-19/validation_context.json",
        )
    _verify_context_sources(run_dir, context_payload)
    try:
        rebuilt_context = build_validation_context(
            run_dir=run_dir,
            document=document,
            config=PaperRevisionConfig(
                sectional_enabled=True,
                max_section_retries=int(context_payload["max_section_retries"]),
                min_length_ratio=float(context_payload["min_length_ratio"]),
                max_length_ratio=float(context_payload["max_length_ratio"]),
                critic_model=manifest.critic_model,
            ),
        )
        rebuilt_payload = _read_json_text(rebuilt_context.text)
    except (SectionalExecutionError, ValueError) as exc:
        _raise(
            "sectional_validation_context_invalid",
            f"validation context cannot be rebuilt from canonical sources: {exc}",
            "stage-19/validation_context.json",
        )
    if rebuilt_payload != context_payload:
        _raise(
            "sectional_validation_context_invalid",
            "validation context differs from canonical source reconstruction",
            "stage-19/validation_context.json",
        )

    attempts_text = _read_text(
        stage19 / "section_attempts.jsonl", "sectional_attempt_log_invalid"
    )
    assessments_text = _read_text(
        stage19 / "resolution_assessments.jsonl",
        "sectional_assessment_log_invalid",
    )
    try:
        attempts = parse_section_attempts_jsonl(attempts_text)
    except SectionalRevisionContractError as exc:
        _raise(
            "sectional_attempt_log_invalid",
            f"attempt log is invalid: {exc}",
            "stage-19/section_attempts.jsonl",
        )
    try:
        assessments = parse_resolution_assessments_jsonl(assessments_text)
    except SectionalRevisionContractError as exc:
        _raise(
            "sectional_assessment_log_invalid",
            f"assessment log is invalid: {exc}",
            "stage-19/resolution_assessments.jsonl",
        )
    if _text_sha256(attempts_text) != manifest.attempts_sha256:
        _raise(
            "sectional_manifest_hash_mismatch",
            "attempt log hash differs from the Stage 19 manifest",
            "stage-19/section_attempts.jsonl",
        )
    if _text_sha256(assessments_text) != manifest.assessments_sha256:
        _raise(
            "sectional_manifest_hash_mismatch",
            "assessment log hash differs from the Stage 19 manifest",
            "stage-19/resolution_assessments.jsonl",
        )
    _verify_model_binding(run_manifest, manifest, attempts, assessments)

    comments_by_id = {comment.comment_id: comment for comment in ledger.comments}
    assigned_by_section: dict[str, list[str]] = {}
    for assignment in plan.assignments:
        if assignment.disposition == "assigned":
            for section_id in assignment.target_section_ids:
                assigned_by_section.setdefault(section_id, []).append(
                    assignment.comment_id
                )
    section_by_id = {section.section_id: section for section in document.sections}
    attempts_by_id = {attempt.attempt_id: attempt for attempt in attempts}
    attempts_by_section: dict[str, list[SectionAttemptRecord]] = {}
    for attempt in attempts:
        if attempt.section_id not in section_by_id:
            _raise(
                "sectional_attempt_log_invalid",
                f"attempt names unknown section {attempt.section_id}",
            )
        attempts_by_section.setdefault(attempt.section_id, []).append(attempt)
    for section_id, records in attempts_by_section.items():
        if [record.attempt for record in records] != list(range(1, len(records) + 1)):
            _raise(
                "sectional_attempt_log_invalid",
                f"attempt ordinals are not contiguous for {section_id}",
            )
        if len(records) > int(context_payload["max_section_retries"]) + 1:
            _raise(
                "sectional_attempt_log_invalid",
                f"attempt count exceeds bounded retries for {section_id}",
            )

    assessments_by_attempt: dict[str, list[ResolutionAssessmentRecord]] = {}
    for assessment in assessments:
        attempt = attempts_by_id.get(assessment.attempt_id)
        if (
            attempt is None
            or attempt.status == "transport_failed"
            or assessment.section_id != attempt.section_id
            or assessment.comment_id not in attempt.comment_ids
        ):
            _raise(
                "sectional_assessment_identity_mismatch",
                f"assessment {assessment.assessment_id} is orphaned or cross-linked",
                "stage-19/resolution_assessments.jsonl",
            )
        assessments_by_attempt.setdefault(assessment.attempt_id, []).append(assessment)

    expected_candidate_paths = {
        attempt.candidate_path for attempt in attempts if attempt.candidate_path
    }
    expected_validation_paths = {
        attempt.validation_report_path
        for attempt in attempts
        if attempt.validation_report_path
    }
    _verify_default_deny_directory(
        run_dir,
        stage19 / "sections",
        expected_candidate_paths,
        "sectional_attempt_artifact_unmanifested",
    )
    _verify_default_deny_directory(
        run_dir,
        stage19 / "section_validation",
        expected_validation_paths,
        "sectional_attempt_artifact_unmanifested",
    )

    replacements: dict[str, ValidatedSectionReplacement] = {}
    metadata: dict[str, SectionManifestMetadata] = {}
    accepted_attempts: dict[str, SectionAttemptRecord] = {}
    for section_id, records in attempts_by_section.items():
        assigned = tuple(assigned_by_section.get(section_id, ()))
        source_section = section_by_id[section_id]
        if any(record.comment_ids != assigned for record in records):
            _raise(
                "sectional_attempt_log_invalid",
                f"attempt comments do not match the validated plan for {section_id}",
            )
        accepted = [record for record in records if record.status == "accepted"]
        if len(accepted) > 1 or (accepted and accepted[0] is not records[-1]):
            _raise(
                "sectional_attempt_reused",
                f"accepted attempt is not unique and final for {section_id}",
            )
        for record in records:
            if record.source_section_sha256 != source_section.original_sha256:
                _raise(
                    "sectional_source_hash_mismatch",
                    f"attempt source hash differs for {record.attempt_id}",
                )
            if record.status == "transport_failed":
                if assessments_by_attempt.get(record.attempt_id):
                    _raise(
                        "sectional_assessment_identity_mismatch",
                        "transport attempt cannot own an assessment",
                    )
                continue
            candidate_path = _run_path(run_dir, record.candidate_path or "")
            validation_path = _run_path(run_dir, record.validation_report_path or "")
            if not candidate_path.is_file() or not validation_path.is_file():
                _raise(
                    "sectional_attempt_artifact_missing",
                    f"attempt artifacts are missing for {record.attempt_id}",
                )
            candidate = _read_text(candidate_path, "sectional_attempt_artifact_missing")
            validation_text = _read_text(
                validation_path, "sectional_attempt_artifact_missing"
            )
            if (
                _text_sha256(candidate) != record.candidate_body_sha256
                or _text_sha256(validation_text) != record.validation_report_sha256
            ):
                _raise(
                    "sectional_attempt_hash_mismatch",
                    f"attempt artifact hash differs for {record.attempt_id}",
                )
            try:
                stored_validation = SectionValidationResult.from_dict(
                    _read_json_text(validation_text)
                )
            except (ValueError, SectionalRevisionContractError) as exc:
                _raise(
                    "sectional_validation_recompute_mismatch",
                    f"validation report is invalid: {exc}",
                )
            context = SectionValidationContext(
                document=document,
                section_id=section_id,
                attempt=record.attempt,
                allowed_citation_keys=frozenset(
                    context_payload["allowed_citation_keys"]
                ),
                grounded_numeric_values=tuple(
                    float(value) for value in context_payload["grounded_numeric_values"]
                ),
                required_comment_ids=tuple(
                    comment_id
                    for comment_id in record.comment_ids
                    if comments_by_id[comment_id].required
                ),
                resolution_comment_ids=record.resolution_comment_ids,
                min_length_ratio=float(context_payload["min_length_ratio"]),
                max_length_ratio=float(context_payload["max_length_ratio"]),
            )
            recomputed = validate_section_candidate(context, candidate)
            failed_codes = tuple(
                check.code for check in recomputed.checks if check.status == "failed"
            )
            if recomputed != stored_validation or failed_codes != record.validator_codes:
                _raise(
                    "sectional_validation_recompute_mismatch",
                    f"validation does not replay for {record.attempt_id}",
                )
            attempt_assessments = assessments_by_attempt.get(record.attempt_id, [])
            if record.status == "accepted":
                by_comment = {item.comment_id: item for item in attempt_assessments}
                if set(by_comment) != set(record.comment_ids) or any(
                    item.verdict != "resolved" for item in by_comment.values()
                ):
                    _raise(
                        "sectional_resolved_without_evidence",
                        f"accepted attempt lacks resolved critic evidence: {record.attempt_id}",
                    )
                if not recomputed.accepted:
                    _raise(
                        "sectional_validation_recompute_mismatch",
                        f"accepted attempt fails deterministic validation: {record.attempt_id}",
                    )
                accepted_attempts[section_id] = record
                replacements[section_id] = ValidatedSectionReplacement(
                    section_id=section_id,
                    body=candidate,
                    validation=recomputed,
                    context=context,
                )
        if accepted:
            record = accepted[0]
            replacement = replacements[section_id]
            metadata[section_id] = SectionManifestMetadata(
                comment_ids=assigned,
                attempt_ids=tuple(item.attempt_id for item in records),
                final_status="accepted",
                validation_result=replacement.validation,
                validation_context=replacement.context,
            )
        else:
            metadata[section_id] = SectionManifestMetadata(
                comment_ids=assigned,
                attempt_ids=tuple(item.attempt_id for item in records),
                final_status="unresolved_original_preserved",
                validation_result=None,
                validation_context=None,
            )

    for section_id in assigned_by_section:
        if section_id not in attempts_by_section:
            _raise(
                "sectional_resolved_without_evidence",
                f"assigned section has no attempts: {section_id}",
            )
    for comment in ledger.comments:
        if comment.final_status == "resolved":
            targets = assigned_by_section_for_comment(plan, comment.comment_id)
            if not targets or any(target not in accepted_attempts for target in targets):
                _raise(
                    "sectional_resolved_without_evidence",
                    f"resolved comment lacks accepted section evidence: {comment.comment_id}",
                )

    merge_result = merge_validated_sections(document, replacements)
    revised_text = _read_text(
        stage19 / "paper_revised.md", "sectional_merge_body_mismatch"
    )
    if merge_result.merged_text != revised_text:
        _raise(
            "sectional_merge_body_mismatch",
            "paper_revised.md differs from the deterministic sectional merge",
            "stage-19/paper_revised.md",
        )
    if merge_result.merged_paper_sha256 != manifest.merged_paper_sha256:
        _raise(
            "sectional_merge_hash_mismatch",
            "merged paper hash differs from the Stage 19 manifest",
            "stage-19/section_revision_manifest.json",
        )
    manifest_args = {
        "document": document,
        "merge_result": merge_result,
        "ledger": ledger,
        "plan": plan,
        "reviews": reviews,
        "claim_scope": contract.claim_scope,
        "experiment_contract_path": contract_rel,
        "experiment_contract_sha256": contract_sha,
        "writer_model": manifest.writer_model,
        "critic_model": manifest.critic_model,
        "source_paper_path": "stage-17/paper_draft.md",
        "section_metadata": metadata,
        "attempts_text": attempts_text,
        "assessments_text": assessments_text,
        "unresolved_comments_text": unresolved_text,
        "completed": True,
        "validation_context_text": context_text,
    }
    try:
        expected_manifest = build_section_revision_manifest(**manifest_args)
        if manifest.sections != expected_manifest.sections:
            _raise(
                "sectional_manifest_sections_mismatch",
                "manifest section entries do not match the replayed merge",
                "stage-19/section_revision_manifest.json",
            )
        hash_fields = (
            "source_paper_sha256",
            "source_reviews_sha256",
            "ledger_sha256",
            "plan_sha256",
            "attempts_sha256",
            "assessments_sha256",
            "unresolved_comments_sha256",
            "validation_context_sha256",
            "merged_paper_sha256",
        )
        if any(
            getattr(manifest, field) != getattr(expected_manifest, field)
            for field in hash_fields
        ):
            _raise(
                "sectional_manifest_hash_mismatch",
                "one or more manifest-bound hashes do not recompute",
                "stage-19/section_revision_manifest.json",
            )
        validate_section_revision_manifest(
            manifest,
            **manifest_args,
        )
    except SectionalAuditError:
        raise
    except SectionalRevisionContractError as exc:
        _raise(
            "sectional_manifest_recompute_mismatch",
            f"section revision manifest does not recompute: {exc}",
            "stage-19/section_revision_manifest.json",
        )


def assigned_by_section_for_comment(plan: RevisionPlan, comment_id: str) -> tuple[str, ...]:
    for assignment in plan.assignments:
        if assignment.comment_id == comment_id and assignment.disposition == "assigned":
            return assignment.target_section_ids
    return ()


def _check_sealed_contract_anchor(run_dir: Path, contract_sha: str) -> None:
    path = run_dir / "stage-10" / "selected_candidate_manifest.json"
    if not path.is_file():
        _raise(
            "sectional_sealed_candidate_missing",
            "Stage 10 sealed candidate manifest is required",
            "stage-10/selected_candidate_manifest.json",
        )
    try:
        payload = _read_json(path)
        required = {
            "schema_version",
            "generated",
            "contract_path",
            "contract_sha256",
            "scaffold_sha256",
            "entry_point",
            "files",
        }
        allowed = required | {"scaffold_files", "plugin_files"}
        if set(payload) - allowed or required - set(payload):
            raise ValueError("sealed manifest has missing or unknown fields")
        if payload["schema_version"] != 1 or not isinstance(payload["files"], dict):
            raise ValueError("sealed manifest schema is invalid")
        if not payload["files"]:
            raise ValueError("sealed manifest files must be nonempty")
        for name, metadata in payload["files"].items():
            if (
                not isinstance(name, str)
                or not name
                or "/" in name
                or "\\" in name
                or not isinstance(metadata, dict)
                or set(metadata) != {"sha256"}
                or not _is_sha256(metadata["sha256"])
            ):
                raise ValueError("sealed manifest file metadata is invalid")
        for owner_key, owner in (
            ("scaffold_files", "scaffold"),
            ("plugin_files", "model"),
        ):
            owner_files = payload.get(owner_key)
            if owner_files is None:
                continue
            if not isinstance(owner_files, dict):
                raise ValueError(f"sealed manifest {owner_key} must be an object")
            for name, metadata in owner_files.items():
                if (
                    name not in payload["files"]
                    or not isinstance(metadata, dict)
                    or set(metadata) != {"sha256", "owner"}
                    or metadata["owner"] != owner
                    or metadata["sha256"] != payload["files"][name]["sha256"]
                ):
                    raise ValueError(f"sealed manifest {owner_key} is invalid")
        recorded = payload["contract_sha256"]
        if not _is_sha256(recorded) or not _is_sha256(payload["scaffold_sha256"]):
            raise ValueError("sealed manifest contract hash is invalid")
        if recorded != contract_sha:
            _raise(
                "sectional_contract_hash_mismatch",
                "Stage 10 and canonical Stage 9 contract hashes differ",
                "stage-10/selected_candidate_manifest.json",
            )
    except (OSError, ValueError) as exc:
        _raise(
            "sectional_sealed_candidate_invalid",
            f"Stage 10 sealed candidate manifest is invalid: {exc}",
            "stage-10/selected_candidate_manifest.json",
        )


def _verify_model_binding(
    run_manifest: Mapping[str, Any] | None,
    manifest: SectionRevisionManifest,
    attempts: tuple[SectionAttemptRecord, ...],
    assessments: tuple[ResolutionAssessmentRecord, ...],
) -> None:
    reviewer = run_manifest.get("reviewer") if isinstance(run_manifest, Mapping) else None
    if not isinstance(reviewer, Mapping):
        _raise(
            "sectional_run_manifest_binding_missing",
            "run_manifest reviewer binding is missing",
            "run_manifest.json",
        )
    writer = reviewer.get("sectional_writer_model")
    critic = reviewer.get("sectional_critic_model")
    root_writer = reviewer.get("writer_model")
    if not all(isinstance(value, str) and value.strip() for value in (writer, critic, root_writer)):
        _raise(
            "sectional_run_manifest_binding_missing",
            "run_manifest sectional model identities are missing",
            "run_manifest.json",
        )
    if (
        writer != manifest.writer_model
        or critic != manifest.critic_model
        or root_writer != writer
        or writer == critic
        or any(item.writer_model != writer for item in attempts)
        or any(item.critic_model != critic for item in assessments)
    ):
        _raise(
            "sectional_run_manifest_model_mismatch",
            "sectional model identities differ across the bundle",
            "run_manifest.json",
        )


def _verify_context_sources(run_dir: Path, payload: Mapping[str, Any]) -> None:
    for source in payload["sources"]:
        if source["kind"] == "config":
            continue
        rel = source["path"]
        if rel.split("/", 1)[0].startswith("stage-10"):
            _raise(
                "sectional_validation_source_disallowed",
                f"Stage 10 cannot ground sectional validation: {rel}",
                rel,
            )
        path = _run_path(run_dir, rel)
        if not path.is_file():
            _raise(
                "sectional_validation_source_missing",
                f"validation context source is missing: {rel}",
                rel,
            )
        if sha256_file(path) != source["sha256"]:
            _raise(
                "sectional_validation_source_hash_mismatch",
                f"validation context source hash differs: {rel}",
                rel,
            )


def _verify_default_deny_directory(
    run_dir: Path,
    directory: Path,
    expected_paths: set[str | None],
    code: str,
) -> None:
    expected = {path for path in expected_paths if path}
    actual: set[str] = set()
    for path in directory.rglob("*"):
        if (
            path.is_symlink()
            or not path.is_file()
            or path.parent != directory
            or path.name.endswith(".tmp")
        ):
            _raise(code, f"unmanifested or nested Stage 19 artifact: {path.name}")
        actual.add(_relative_path(path, run_dir))
    missing = expected - actual
    if missing:
        _raise(
            "sectional_attempt_artifact_missing",
            "Stage 19 attempt artifacts are missing: " + ", ".join(sorted(missing)),
        )
    if actual - expected:
        _raise(code, "Stage 19 artifact set differs from the attempt log")


def _read_json(path: Path) -> dict[str, Any]:
    return _read_json_text(path.read_text(encoding="utf-8"))


def _read_json_text(text: str) -> dict[str, Any]:
    value = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_text(path: Path, code: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        _raise(code, f"cannot read {path.name}: {exc}", path.name)


def _run_path(run_dir: Path, rel: str) -> Path:
    pure = PurePosixPath(rel)
    if not rel or pure.is_absolute() or ".." in pure.parts or "\\" in rel:
        _raise("sectional_revision_artifact_invalid", f"unsafe run path: {rel!r}")
    path = run_dir.joinpath(*pure.parts)
    try:
        path.resolve().relative_to(run_dir.resolve())
    except (OSError, ValueError):
        _raise("sectional_revision_artifact_invalid", f"path escapes run: {rel!r}")
    return path


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        _raise("sectional_revision_artifact_invalid", "artifact is outside the run")


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _raise(code: str, message: str, path: str = "") -> None:
    raise SectionalAuditError(code, message, path)
