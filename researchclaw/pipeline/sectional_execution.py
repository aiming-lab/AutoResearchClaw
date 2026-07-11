"""Feature-flagged deterministic execution shell for sectional Stage 19."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from researchclaw.config import PaperRevisionConfig
from researchclaw.experiment_runtime.contract import sha256_file
from researchclaw.literature.verify import parse_bibtex_entries
from researchclaw.pipeline.manuscript_sections import (
    ManuscriptDocument,
    ManuscriptSection,
    merge_manuscript,
    parse_manuscript,
)
from researchclaw.pipeline.sectional_revision import (
    ReviewComment,
    ReviewLedger,
    RevisionPlan,
    SectionalRevisionContractError,
    extract_review_ledger,
    make_attempt_id,
    validate_review_ledger,
    validate_revision_plan,
)
from researchclaw.pipeline.sectional_validation import (
    SectionManifestMetadata,
    SectionValidationContext,
    ValidatedSectionReplacement,
    build_section_revision_manifest,
    build_unresolved_comments_artifact,
    extract_citation_keys,
    merge_validated_sections,
    validate_section_candidate,
    validate_section_revision_manifest,
)


_OWNED_FILES = (
    "paper_revised.md",
    "revision_notes_internal.md",
    "revision_retry_failure.json",
    "revision_plan.json",
    "review_comment_ledger.json",
    "section_attempts.jsonl",
    "resolution_assessments.jsonl",
    "section_revision_manifest.json",
    "unresolved_comments.json",
    "consistency_audit.json",
    "validation_context.json",
)
_OWNED_DIRS = ("sections", "section_validation")


class SectionalExecutionError(RuntimeError):
    """Raised when the B2 execution shell cannot close its contracts."""


@dataclass(frozen=True)
class SectionProposal:
    section_id: str
    revised_body: str
    resolution_comment_ids: tuple[str, ...]


@dataclass(frozen=True)
class ResolutionAssessment:
    comment_id: str
    section_id: str
    attempt_id: str
    critic_model: str
    context_isolated: bool
    verdict: str
    reason: str


class SectionalRevisionProvider(Protocol):
    writer_model: str
    critic_model: str

    def build_plan(
        self,
        *,
        ledger: ReviewLedger,
        document: ManuscriptDocument,
    ) -> object: ...

    def propose(
        self,
        *,
        section: ManuscriptSection,
        comments: tuple[ReviewComment, ...],
        attempt: int,
        context: SectionValidationContext,
    ) -> SectionProposal: ...

    def assess(
        self,
        *,
        comment: ReviewComment,
        section: ManuscriptSection,
        original_body: str,
        revised_body: str,
        attempt_id: str,
        validator_codes: tuple[str, ...],
    ) -> ResolutionAssessment: ...


@dataclass(frozen=True)
class SectionalExecutionResult:
    completed: bool
    paper_text: str | None
    error: str | None
    artifacts: tuple[str, ...]


@dataclass(frozen=True)
class _ContextBundle:
    allowed_citation_keys: frozenset[str]
    grounded_numeric_values: tuple[float, ...]
    text: str


def execute_sectional_revision(
    *,
    stage_dir: Path,
    run_dir: Path,
    config: PaperRevisionConfig,
    claim_scope: str,
    provider: SectionalRevisionProvider | None,
) -> SectionalExecutionResult:
    """Execute B2 using an explicitly injected provider and deterministic gates."""

    clean_sectional_outputs(stage_dir)
    if provider is None:
        raise SectionalExecutionError(
            "sectional revision is enabled but no reviewed provider is configured"
        )
    writer_model = str(provider.writer_model or "").strip()
    critic_model = str(provider.critic_model or "").strip()
    if not writer_model or not critic_model or writer_model == critic_model:
        raise SectionalExecutionError(
            "sectional provider requires distinct nonempty writer and critic models"
        )
    if not config.critic_model.strip() or config.critic_model.strip() != critic_model:
        raise SectionalExecutionError(
            "sectional provider critic model does not match paper_revision.critic_model"
        )

    paper_path = run_dir / "stage-17" / "paper_draft.md"
    reviews_path = run_dir / "stage-18" / "reviews.md"
    if not paper_path.is_file() or not reviews_path.is_file():
        raise SectionalExecutionError(
            "sectional revision requires canonical stage-17 and stage-18 inputs"
        )
    paper = paper_path.read_text(encoding="utf-8")
    reviews = reviews_path.read_text(encoding="utf-8")
    document = parse_manuscript(paper, strict=True)
    ledger = extract_review_ledger(reviews, source_path="stage-18/reviews.md")
    plan = validate_revision_plan(
        provider.build_plan(ledger=ledger, document=document),
        ledger,
        document,
        reviews=reviews,
    )

    context_bundle = build_validation_context(
        run_dir=run_dir,
        document=document,
        config=config,
    )
    _write_text_atomic(stage_dir / "validation_context.json", context_bundle.text)
    _write_json_atomic(stage_dir / "review_comment_ledger.json", ledger.to_dict())
    _write_json_atomic(stage_dir / "revision_plan.json", plan.to_dict())

    comments_by_id = {comment.comment_id: comment for comment in ledger.comments}
    assigned_by_section: dict[str, list[ReviewComment]] = {}
    for assignment in plan.assignments:
        if assignment.disposition != "assigned":
            continue
        for section_id in assignment.target_section_ids:
            assigned_by_section.setdefault(section_id, []).append(
                comments_by_id[assignment.comment_id]
            )

    sections_dir = stage_dir / "sections"
    validation_dir = stage_dir / "section_validation"
    sections_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    assessments: list[dict[str, Any]] = []
    replacements: dict[str, ValidatedSectionReplacement] = {}
    metadata: dict[str, SectionManifestMetadata] = {}
    accepted_comment_sections: set[tuple[str, str]] = set()
    attempts_by_section: dict[str, list[str]] = {}

    section_lookup = {section.section_id: section for section in document.sections}
    for section_id, assigned_comments_list in assigned_by_section.items():
        section = section_lookup[section_id]
        assigned_comments = tuple(assigned_comments_list)
        required_ids = tuple(
            comment.comment_id for comment in assigned_comments if comment.required
        )
        accepted_replacement: ValidatedSectionReplacement | None = None
        for attempt in range(1, config.max_section_retries + 2):
            attempt_id = make_attempt_id(section_id, attempt)
            attempts_by_section.setdefault(section_id, []).append(attempt_id)
            context = SectionValidationContext(
                document=document,
                section_id=section_id,
                attempt=attempt,
                allowed_citation_keys=context_bundle.allowed_citation_keys,
                grounded_numeric_values=context_bundle.grounded_numeric_values,
                required_comment_ids=required_ids,
                resolution_comment_ids=(),
                min_length_ratio=config.min_length_ratio,
                max_length_ratio=config.max_length_ratio,
            )
            try:
                proposal = provider.propose(
                    section=section,
                    comments=assigned_comments,
                    attempt=attempt,
                    context=context,
                )
            except RuntimeError as exc:  # provider transport boundary
                attempts.append(
                    _transport_failure_attempt(
                        attempt_id=attempt_id,
                        section_id=section_id,
                        comment_ids=tuple(c.comment_id for c in assigned_comments),
                        writer_model=writer_model,
                        attempt=attempt,
                        source_section_sha256=section.original_sha256,
                        exc=exc,
                    )
                )
                continue
            _validate_proposal(proposal, section_id, assigned_comments)
            context = replace(
                context,
                resolution_comment_ids=proposal.resolution_comment_ids,
            )
            candidate_path = sections_dir / f"{section_id}.attempt-{attempt}.md"
            _write_text_atomic(candidate_path, proposal.revised_body)
            validation = validate_section_candidate(context, proposal.revised_body)
            validation_rel = (
                f"stage-19/section_validation/{section_id}.attempt-{attempt}.json"
            )
            validation_path = validation_dir / f"{section_id}.attempt-{attempt}.json"
            validation_text = _canonical_json_text(validation.to_dict())
            _write_text_atomic(validation_path, validation_text)
            failed_codes = tuple(
                check.code for check in validation.checks if check.status == "failed"
            )
            attempt_assessments: list[ResolutionAssessment] = []
            assessment_error: Exception | None = None
            if validation.accepted and proposal.revised_body != section.body:
                try:
                    for comment in assigned_comments:
                        assessment = provider.assess(
                            comment=comment,
                            section=section,
                            original_body=section.body,
                            revised_body=proposal.revised_body,
                            attempt_id=attempt_id,
                            validator_codes=failed_codes,
                        )
                        _validate_assessment(
                            assessment,
                            comment=comment,
                            section_id=section_id,
                            attempt_id=attempt_id,
                            expected_critic_model=critic_model,
                            writer_model=writer_model,
                        )
                        attempt_assessments.append(assessment)
                except RuntimeError as exc:  # isolated critic boundary
                    assessment_error = exc
                    attempt_assessments = []
            all_resolved = bool(attempt_assessments) and all(
                assessment.verdict == "resolved"
                for assessment in attempt_assessments
            )
            status = "accepted" if validation.accepted and all_resolved else "rejected"
            attempts.append(
                {
                    "schema_version": 1,
                    "attempt_id": attempt_id,
                    "section_id": section_id,
                    "source_section_sha256": section.original_sha256,
                    "comment_ids": [c.comment_id for c in assigned_comments],
                    "writer_model": writer_model,
                    "attempt": attempt,
                    "status": status,
                    "candidate_body_sha256": _sha256(proposal.revised_body),
                    "validation_report_path": validation_rel,
                    "validation_report_sha256": _sha256(validation_text),
                    "validator_codes": list(failed_codes),
                    "error_type": (
                        type(assessment_error).__name__ if assessment_error else None
                    ),
                    "error": (
                        None
                        if status == "accepted"
                        else str(assessment_error or "candidate not accepted")
                    ),
                    "timestamp": _utcnow(),
                }
            )
            for assessment in attempt_assessments:
                payload = _assessment_payload(assessment)
                assessments.append(payload)
                if status == "accepted" and assessment.verdict == "resolved":
                    accepted_comment_sections.add(
                        (assessment.comment_id, assessment.section_id)
                    )
            if status == "accepted":
                accepted_replacement = ValidatedSectionReplacement(
                    section_id=section_id,
                    body=proposal.revised_body,
                    validation=validation,
                    context=context,
                )
                replacements[section_id] = accepted_replacement
                metadata[section_id] = SectionManifestMetadata(
                    comment_ids=tuple(c.comment_id for c in assigned_comments),
                    attempt_ids=tuple(attempts_by_section[section_id]),
                    final_status="accepted",
                    validation_result=validation,
                    validation_context=context,
                )
                break
        if accepted_replacement is None:
            metadata[section_id] = SectionManifestMetadata(
                comment_ids=tuple(c.comment_id for c in assigned_comments),
                attempt_ids=tuple(attempts_by_section[section_id]),
                final_status="unresolved_original_preserved",
                validation_result=None,
                validation_context=None,
            )

    _verify_input_immutability(
        run_dir=run_dir,
        paper_path=paper_path,
        reviews_path=reviews_path,
        document=document,
        ledger=ledger,
        context_text=context_bundle.text,
        config=config,
    )
    final_ledger = _finalize_ledger(
        ledger=ledger,
        plan=plan,
        attempts_by_section=attempts_by_section,
        accepted_comment_sections=accepted_comment_sections,
    )
    validate_review_ledger(final_ledger, reviews=reviews, require_final=True)
    merge_result = merge_validated_sections(document, replacements)
    assessments_text = _jsonl_text(assessments)
    attempts_text = _jsonl_text(attempts)
    unresolved_text = _pretty_json_text(
        build_unresolved_comments_artifact(final_ledger)
    )
    _write_json_atomic(stage_dir / "review_comment_ledger.json", final_ledger.to_dict())
    _write_text_atomic(stage_dir / "section_attempts.jsonl", attempts_text)
    _write_text_atomic(stage_dir / "resolution_assessments.jsonl", assessments_text)
    _write_text_atomic(stage_dir / "unresolved_comments.json", unresolved_text)

    completed = True
    try:
        manifest = build_section_revision_manifest(
            document=document,
            merge_result=merge_result,
            ledger=final_ledger,
            plan=plan,
            reviews=reviews,
            claim_scope=claim_scope,
            source_paper_path="stage-17/paper_draft.md",
            section_metadata=metadata,
            assessments_text=assessments_text,
            unresolved_comments_text=unresolved_text,
            completed=True,
            validation_context_text=context_bundle.text,
        )
    except SectionalRevisionContractError as exc:
        if claim_scope == "pipeline_validation":
            raise
        completed = False
        manifest = build_section_revision_manifest(
            document=document,
            merge_result=merge_result,
            ledger=final_ledger,
            plan=plan,
            reviews=reviews,
            claim_scope=claim_scope,
            source_paper_path="stage-17/paper_draft.md",
            section_metadata=metadata,
            assessments_text=assessments_text,
            unresolved_comments_text=unresolved_text,
            completed=False,
            validation_context_text=context_bundle.text,
        )
        error = str(exc)
    else:
        error = None
    validate_section_revision_manifest(
        manifest,
        document=document,
        merge_result=merge_result,
        ledger=final_ledger,
        plan=plan,
        reviews=reviews,
        claim_scope=claim_scope,
        source_paper_path="stage-17/paper_draft.md",
        section_metadata=metadata,
        assessments_text=assessments_text,
        unresolved_comments_text=unresolved_text,
        completed=completed,
        validation_context_text=context_bundle.text,
    )
    _write_json_atomic(stage_dir / "section_revision_manifest.json", manifest.to_dict())
    artifacts = (
        "review_comment_ledger.json",
        "revision_plan.json",
        "validation_context.json",
        "section_attempts.jsonl",
        "resolution_assessments.jsonl",
        "unresolved_comments.json",
        "section_revision_manifest.json",
    )
    if not completed:
        return SectionalExecutionResult(False, None, error, artifacts)
    _write_text_atomic(stage_dir / "paper_revised.md", merge_result.merged_text)
    return SectionalExecutionResult(
        True,
        merge_result.merged_text,
        None,
        ("paper_revised.md",) + artifacts,
    )


def build_validation_context(
    *,
    run_dir: Path,
    document: ManuscriptDocument,
    config: PaperRevisionConfig,
) -> _ContextBundle:
    """Build and source-bind the B2 citation and metric validation context."""

    sources: list[dict[str, str]] = []
    original_citations = extract_citation_keys(merge_manuscript(document))
    bib_path = _latest_stage_artifact(run_dir, 4, "references.bib")
    citation_keys: set[str] = set()
    if bib_path is not None:
        bib_text = bib_path.read_text(encoding="utf-8")
        citation_keys.update(
            str(entry.get("key") or "").strip()
            for entry in parse_bibtex_entries(bib_text)
            if str(entry.get("key") or "").strip()
        )
        sources.append(_source_record(run_dir, bib_path, "citations"))
    missing_citations = sorted(original_citations - citation_keys)
    if missing_citations:
        raise SectionalExecutionError(
            "canonical references.bib is missing draft citation keys: "
            + ", ".join(missing_citations)
        )

    numeric_values: list[float] = []
    numeric_seen: set[float] = set()
    for path in _metric_source_paths(run_dir):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        before = len(numeric_values)
        for key in ("primary_metric", "metrics", "key_metrics", "metrics_summary"):
            if key in payload:
                _collect_numbers(payload[key], numeric_values, numeric_seen)
        per_seed = payload.get("per_seed")
        if isinstance(per_seed, list):
            for row in per_seed[:20]:
                if isinstance(row, dict) and isinstance(row.get("metrics"), dict):
                    _collect_numbers(row["metrics"], numeric_values, numeric_seen)
        if len(numeric_values) > before:
            sources.append(_source_record(run_dir, path, "metrics"))

    config_payload = {
        "max_section_retries": config.max_section_retries,
        "min_length_ratio": config.min_length_ratio,
        "max_length_ratio": config.max_length_ratio,
    }
    config_text = _canonical_json_text(config_payload)
    sources.append(
        {
            "kind": "config",
            "path": "config.paper_revision",
            "sha256": _sha256(config_text),
        }
    )
    payload = {
        "schema_version": 1,
        "source_paper_sha256": document.source_sha256,
        "allowed_citation_keys": sorted(citation_keys),
        "grounded_numeric_values": numeric_values,
        "max_section_retries": config.max_section_retries,
        "min_length_ratio": config.min_length_ratio,
        "max_length_ratio": config.max_length_ratio,
        "sources": sorted(sources, key=lambda item: (item["kind"], item["path"])),
    }
    text = _pretty_json_text(payload)
    return _ContextBundle(frozenset(citation_keys), tuple(numeric_values), text)


def _finalize_ledger(
    *,
    ledger: ReviewLedger,
    plan: RevisionPlan,
    attempts_by_section: dict[str, list[str]],
    accepted_comment_sections: set[tuple[str, str]],
) -> ReviewLedger:
    assignments = {assignment.comment_id: assignment for assignment in plan.assignments}
    comments: list[ReviewComment] = []
    for comment in ledger.comments:
        assignment = assignments[comment.comment_id]
        if assignment.disposition != "assigned":
            comments.append(
                replace(
                    comment,
                    final_status=assignment.disposition,
                    resolution_reason=assignment.reason,
                )
            )
            continue
        attempt_ids = tuple(
            attempt_id
            for section_id in assignment.target_section_ids
            for attempt_id in attempts_by_section.get(section_id, ())
        )
        resolved = all(
            (comment.comment_id, section_id) in accepted_comment_sections
            for section_id in assignment.target_section_ids
        )
        comments.append(
            replace(
                comment,
                working_status="assigned",
                target_section_ids=assignment.target_section_ids,
                final_status="resolved" if resolved else "unresolved",
                resolution_reason=(
                    "All assigned sections passed deterministic validation "
                    "and isolated assessment."
                    if resolved
                    else "One or more assigned sections did not close after bounded attempts."
                ),
                attempt_ids=attempt_ids,
            )
        )
    return replace(ledger, comments=tuple(comments))


def _verify_input_immutability(
    *,
    run_dir: Path,
    paper_path: Path,
    reviews_path: Path,
    document: ManuscriptDocument,
    ledger: ReviewLedger,
    context_text: str,
    config: PaperRevisionConfig,
) -> None:
    if _sha256(paper_path.read_text(encoding="utf-8")) != document.source_sha256:
        raise SectionalExecutionError("canonical Stage 17 paper changed during revision")
    if _sha256(reviews_path.read_text(encoding="utf-8")) != ledger.source_reviews_sha256:
        raise SectionalExecutionError("canonical Stage 18 reviews changed during revision")
    payload = json.loads(context_text)
    for source in payload["sources"]:
        kind = source["kind"]
        if kind == "config":
            config_payload = {
                "max_section_retries": config.max_section_retries,
                "min_length_ratio": config.min_length_ratio,
                "max_length_ratio": config.max_length_ratio,
            }
            actual = _sha256(_canonical_json_text(config_payload))
        else:
            path = run_dir / source["path"]
            if not path.is_file():
                raise SectionalExecutionError(
                    f"validation context source disappeared: {source['path']}"
                )
            actual = sha256_file(path)
        if actual != source["sha256"]:
            raise SectionalExecutionError(
                f"validation context source changed: {source['path']}"
            )


def _validate_proposal(
    proposal: SectionProposal,
    section_id: str,
    comments: tuple[ReviewComment, ...],
) -> None:
    if not isinstance(proposal, SectionProposal):
        raise SectionalExecutionError("provider returned an invalid proposal type")
    if proposal.section_id != section_id or not isinstance(proposal.revised_body, str):
        raise SectionalExecutionError("proposal section identity is invalid")
    known = {comment.comment_id for comment in comments}
    if len(set(proposal.resolution_comment_ids)) != len(proposal.resolution_comment_ids):
        raise SectionalExecutionError("proposal contains duplicate resolution IDs")
    if not set(proposal.resolution_comment_ids).issubset(known):
        raise SectionalExecutionError("proposal contains unknown resolution IDs")


def _validate_assessment(
    assessment: ResolutionAssessment,
    *,
    comment: ReviewComment,
    section_id: str,
    attempt_id: str,
    expected_critic_model: str,
    writer_model: str,
) -> None:
    if not isinstance(assessment, ResolutionAssessment):
        raise SectionalExecutionError("provider returned an invalid assessment type")
    if (
        assessment.comment_id != comment.comment_id
        or assessment.section_id != section_id
        or assessment.attempt_id != attempt_id
    ):
        raise SectionalExecutionError("assessment identity does not match its attempt")
    if (
        assessment.critic_model != expected_critic_model
        or assessment.critic_model == writer_model
        or not assessment.context_isolated
    ):
        raise SectionalExecutionError("assessment critic isolation is invalid")
    if assessment.verdict not in {"resolved", "unresolved"}:
        raise SectionalExecutionError("assessment verdict is invalid")
    if not assessment.reason.strip():
        raise SectionalExecutionError("assessment reason is required")


def _assessment_payload(assessment: ResolutionAssessment) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "assessment_id": f"ra-{assessment.comment_id}-{assessment.attempt_id}",
        "comment_id": assessment.comment_id,
        "section_id": assessment.section_id,
        "attempt_id": assessment.attempt_id,
        "critic_model": assessment.critic_model,
        "context_isolated": assessment.context_isolated,
        "verdict": assessment.verdict,
        "reason": assessment.reason,
        "timestamp": _utcnow(),
    }


def _transport_failure_attempt(
    *,
    attempt_id: str,
    section_id: str,
    comment_ids: tuple[str, ...],
    writer_model: str,
    attempt: int,
    source_section_sha256: str,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "section_id": section_id,
        "source_section_sha256": source_section_sha256,
        "comment_ids": list(comment_ids),
        "writer_model": writer_model,
        "attempt": attempt,
        "status": "transport_failed",
        "candidate_body_sha256": None,
        "validation_report_path": None,
        "validation_report_sha256": None,
        "validator_codes": [],
        "error_type": type(exc).__name__,
        "error": str(exc),
        "timestamp": _utcnow(),
    }


def clean_sectional_outputs(stage_dir: Path) -> None:
    """Remove only artifacts owned by the current Stage 19 attempt."""

    for name in _OWNED_FILES:
        (stage_dir / name).unlink(missing_ok=True)
        (stage_dir / f"{name}.tmp").unlink(missing_ok=True)
    for name in _OWNED_DIRS:
        path = stage_dir / name
        if path.exists():
            shutil.rmtree(path)


def _metric_source_paths(run_dir: Path) -> tuple[Path, ...]:
    paths: set[Path] = set()
    paths.update(path for path in run_dir.glob("stage-12*/runs/*.json") if path.is_file())
    paths.update(
        path
        for path in run_dir.glob("stage-14*/experiment_summary.json")
        if path.is_file()
    )
    best = run_dir / "experiment_summary_best.json"
    if best.is_file():
        paths.add(best)
    return tuple(sorted(paths, key=lambda path: path.relative_to(run_dir).as_posix()))


def _latest_stage_artifact(run_dir: Path, stage: int, name: str) -> Path | None:
    direct = run_dir / f"stage-{stage:02d}" / name
    if direct.is_file():
        return direct
    candidates = [
        path
        for path in run_dir.glob(f"stage-{stage:02d}_v*/{name}")
        if path.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: _stage_version(path.parent.name))


def _stage_version(name: str) -> int:
    try:
        return int(name.rsplit("_v", 1)[1])
    except (IndexError, ValueError):
        return -1


def _source_record(run_dir: Path, path: Path, kind: str) -> dict[str, str]:
    return {
        "kind": kind,
        "path": path.relative_to(run_dir).as_posix(),
        "sha256": sha256_file(path),
    }


def _collect_numbers(value: Any, output: list[float], seen: set[float]) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            _collect_numbers(nested, output, seen)
    elif isinstance(value, list):
        for nested in value[:100]:
            _collect_numbers(nested, output, seen)
    elif not isinstance(value, bool) and isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number) and number not in seen:
            seen.add(number)
            output.append(number)


def _write_json_atomic(path: Path, value: object) -> None:
    _write_text_atomic(path, _pretty_json_text(value))


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def _canonical_json_text(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _pretty_json_text(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n"


def _jsonl_text(rows: list[dict[str, Any]]) -> str:
    return "".join(_canonical_json_text(row) + "\n" for row in rows)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()
