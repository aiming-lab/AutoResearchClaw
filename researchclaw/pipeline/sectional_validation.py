"""Deterministic validation, merging, and manifests for sectional revision.

This B1 module does not call an LLM and is not wired into Stage 19. It accepts
one proposed section body, runs hard checks, and only lets accepted results
enter the lossless manuscript merger.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from markdown_it import MarkdownIt

from researchclaw.pipeline.manuscript_sections import (
    ManuscriptDocument,
    ManuscriptStructureError,
    merge_manuscript,
    parse_manuscript,
    split_commonmark_lines_keepends,
)
from researchclaw.pipeline.release_artifacts import numbers_match
from researchclaw.pipeline.sectional_revision import (
    ContractIssue,
    ReviewLedger,
    RevisionPlan,
    SectionalRevisionContractError,
    make_attempt_id,
    validate_review_ledger,
    validate_revision_plan,
)


SCHEMA_VERSION = 1
NUMERIC_REL_TOL = 1e-3
QUANTITATIVE_UNIT_LEXICON_VERSION = 1
QUANTITATIVE_UNIT_LEXICON_V1 = frozenset(
    {
        "%",
        "counter",
        "counters",
        "ghz",
        "hz",
        "khz",
        "mhz",
        "millisecond",
        "milliseconds",
        "ms",
        "microsecond",
        "microseconds",
        "percent",
        "percentage",
        "percentages",
        "s",
        "sample",
        "samples",
        "second",
        "seconds",
        "seed",
        "seeds",
        "trial",
        "trials",
        "us",
        "µs",
        "μs",
    }
)

_CHECK_CODES = (
    "empty_section_body",
    "new_heading_introduced",
    "html_heading_introduced",
    "markdown_structure_unbalanced",
    "post_merge_heading_mismatch",
    "section_order_mismatch",
    "unknown_citation_key",
    "required_citation_removed",
    "unknown_numeric_value",
    "required_reference_removed",
    "unknown_reference_introduced",
    "unparsed_quantitative_expression",
    "abnormal_section_shrink",
    "abnormal_section_growth",
    "unaddressed_required_comment",
)
_CHECK_STATUSES = frozenset({"passed", "failed"})
_SECTION_FINAL_STATUSES = frozenset(
    {"unchanged", "accepted", "unresolved_original_preserved"}
)
_CLAIM_SCOPES = frozenset({"pipeline_validation", "research_release", "exploratory"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HTML_HEADING_RE = re.compile(r"</?h[1-6](?:\s[^>]*)?>", re.IGNORECASE)
_LATEX_CITE_RE = re.compile(r"\\cite[a-zA-Z*]*(?:\[[^\]]*\])*\{([^}]+)\}")
_MARKDOWN_CITE_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*\d{4}[A-Za-z0-9_-]*$")
_REFERENCE_RE = re.compile(
    r"\b(Figures?|Figs?\.?|Tables?|Equations?|Eqs?\.?)\s*"
    r"(?:\(\s*)?"
    r"([A-Za-z]?\d+(?:\.\d+)*(?:\s*(?:,\s*(?:and\s+)?|(?:and|&)\s+)"
    r"[A-Za-z]?\d+(?:\.\d+)*)*)"
    r"(?:\s*\))?",
    re.IGNORECASE,
)
_REFERENCE_LABEL_RE = re.compile(r"[A-Za-z]?\d+(?:\.\d+)*")
_STRUCTURAL_NUMBER_RE = re.compile(
    r"\b(?:Section|Appendix)\s+[A-Za-z]?\d+(?:\.\d+)*",
    re.IGNORECASE,
)
_DECLARED_REFERENCE_RE = re.compile(
    r"(?im)^\s*(?:\*\*)?(Figure|Fig\.?|Table|Equation|Eq\.?)\s+"
    r"([A-Za-z]?\d+(?:\.\d+)*)\s*(?:\.(?:\*\*)?|:)",
)
_IMAGE_REFERENCE_RE = re.compile(
    r"!\[[^\]]*\b(Figure|Fig\.?|Table)\s+([A-Za-z]?\d+(?:\.\d+)*)[^\]]*\]",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(
    r"(?<![\w.])[-+]?(?:"
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"|(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?"
    r")\s*%?"
)
_AUTHOR_YEAR_RE = re.compile(
    r"\b[A-Z][A-Za-z'\u2019-]+"
    r"(?:\s+(?:(?:and|&)\s+[A-Z][A-Za-z'\u2019-]+|et\s+al\.))?"
    r"\s*\(\s*(?:19|20)\d{2}[a-z]?\s*\)",
)
_MATH_SPAN_RE = re.compile(
    r"\$\$.*?\$\$|\\\[.*?\\\]|"
    r"(?<![\\$])\$(?!\$).*?(?<![\\$])\$(?!\$)",
    re.DOTALL,
)
_PERCENT_WORD_RE = re.compile(r"\s*(percent|percentage|percentages)\b", re.IGNORECASE)
_NUMBER_WORD_TOKEN = (
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand"
)
_NUMBER_WORD_UNIT_RE = re.compile(
    rf"\b((?:{_NUMBER_WORD_TOKEN})(?:[ -](?:and[ -])?(?:{_NUMBER_WORD_TOKEN}))*)"
    rf"\s+({'|'.join(sorted(map(re.escape, QUANTITATIVE_UNIT_LEXICON_V1), key=len, reverse=True))})"
    r"\b",
    re.IGNORECASE,
)
_FRACTION_WORD_RE = re.compile(
    rf"\b((?:{_NUMBER_WORD_TOKEN}))\s+"
    r"(half|halves|third|thirds|quarter|quarters|fourth|fourths)\b",
    re.IGNORECASE,
)
_AMBIGUOUS_QUANTITY_RE = re.compile(
    rf"\b(few|several|many|dozen|dozens|couple)\s+"
    rf"({'|'.join(sorted(map(re.escape, QUANTITATIVE_UNIT_LEXICON_V1), key=len, reverse=True))})\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json_sha256(value: object) -> str:
    return _sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    )


@dataclass(frozen=True)
class SectionValidationCheck:
    code: str
    status: str
    details: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "status": self.status, "details": list(self.details)}

    @classmethod
    def from_dict(cls, value: object) -> SectionValidationCheck:
        data = _strict_object(
            value,
            expected={"code", "status", "details"},
            context="section validation check",
        )
        code = _required_str(data["code"], "code")
        status = _required_str(data["status"], "status")
        details = _string_tuple(data["details"], "details")
        if code not in _CHECK_CODES:
            _fail("validation_code_invalid", f"unknown validation code {code}")
        if status not in _CHECK_STATUSES:
            _fail("validation_status_invalid", f"invalid status for {code}")
        if status == "failed" and not details:
            _fail("validation_details_missing", f"failed check {code} needs details")
        if status == "passed" and details:
            _fail("validation_pass_has_details", f"passed check {code} has details")
        return cls(code=code, status=status, details=details)


@dataclass(frozen=True)
class SectionValidationResult:
    schema_version: int
    attempt_id: str
    section_id: str
    original_sha256: str
    candidate_sha256: str
    accepted: bool
    checks: tuple[SectionValidationCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "attempt_id": self.attempt_id,
            "section_id": self.section_id,
            "original_sha256": self.original_sha256,
            "candidate_sha256": self.candidate_sha256,
            "accepted": self.accepted,
            "checks": [check.to_dict() for check in self.checks],
        }

    @classmethod
    def from_dict(cls, value: object) -> SectionValidationResult:
        data = _strict_object(
            value,
            expected={
                "schema_version",
                "attempt_id",
                "section_id",
                "original_sha256",
                "candidate_sha256",
                "accepted",
                "checks",
            },
            context="section validation result",
        )
        checks_raw = data["checks"]
        if not isinstance(checks_raw, list):
            _fail("invalid_type", "checks must be a list")
        result = cls(
            schema_version=_required_int(data["schema_version"], "schema_version"),
            attempt_id=_required_str(data["attempt_id"], "attempt_id"),
            section_id=_required_str(data["section_id"], "section_id"),
            original_sha256=_required_hash(data["original_sha256"], "original_sha256"),
            candidate_sha256=_required_hash(data["candidate_sha256"], "candidate_sha256"),
            accepted=_required_bool(data["accepted"], "accepted"),
            checks=tuple(SectionValidationCheck.from_dict(item) for item in checks_raw),
        )
        _validate_result_shape(result)
        return result


@dataclass(frozen=True)
class SectionValidationContext:
    document: ManuscriptDocument
    section_id: str
    attempt: int
    allowed_citation_keys: frozenset[str]
    grounded_numeric_values: tuple[float, ...]
    required_comment_ids: tuple[str, ...] = ()
    resolution_comment_ids: tuple[str, ...] = ()
    min_length_ratio: float = 0.80
    max_length_ratio: float = 1.75


@dataclass(frozen=True)
class SectionAttemptRecord:
    schema_version: int
    attempt_id: str
    section_id: str
    source_section_sha256: str
    comment_ids: tuple[str, ...]
    resolution_comment_ids: tuple[str, ...]
    writer_model: str
    attempt: int
    status: str
    candidate_path: str | None
    candidate_body_sha256: str | None
    validation_report_path: str | None
    validation_report_sha256: str | None
    validator_codes: tuple[str, ...]
    error_type: str | None
    error: str | None
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "attempt_id": self.attempt_id,
            "section_id": self.section_id,
            "source_section_sha256": self.source_section_sha256,
            "comment_ids": list(self.comment_ids),
            "resolution_comment_ids": list(self.resolution_comment_ids),
            "writer_model": self.writer_model,
            "attempt": self.attempt,
            "status": self.status,
            "candidate_path": self.candidate_path,
            "candidate_body_sha256": self.candidate_body_sha256,
            "validation_report_path": self.validation_report_path,
            "validation_report_sha256": self.validation_report_sha256,
            "validator_codes": list(self.validator_codes),
            "error_type": self.error_type,
            "error": self.error,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, value: object) -> SectionAttemptRecord:
        data = _strict_object(
            value,
            expected={
                "schema_version",
                "attempt_id",
                "section_id",
                "source_section_sha256",
                "comment_ids",
                "resolution_comment_ids",
                "writer_model",
                "attempt",
                "status",
                "candidate_path",
                "candidate_body_sha256",
                "validation_report_path",
                "validation_report_sha256",
                "validator_codes",
                "error_type",
                "error",
                "timestamp",
            },
            context="section attempt",
        )
        record = cls(
            schema_version=_required_int(data["schema_version"], "schema_version"),
            attempt_id=_required_str(data["attempt_id"], "attempt_id"),
            section_id=_required_str(data["section_id"], "section_id"),
            source_section_sha256=_required_hash(
                data["source_section_sha256"], "source_section_sha256"
            ),
            comment_ids=_string_tuple(data["comment_ids"], "comment_ids"),
            resolution_comment_ids=_string_tuple(
                data["resolution_comment_ids"], "resolution_comment_ids"
            ),
            writer_model=_required_str(data["writer_model"], "writer_model"),
            attempt=_required_int(data["attempt"], "attempt"),
            status=_required_str(data["status"], "status"),
            candidate_path=_optional_str(data["candidate_path"], "candidate_path"),
            candidate_body_sha256=_optional_hash(
                data["candidate_body_sha256"], "candidate_body_sha256"
            ),
            validation_report_path=_optional_str(
                data["validation_report_path"], "validation_report_path"
            ),
            validation_report_sha256=_optional_hash(
                data["validation_report_sha256"], "validation_report_sha256"
            ),
            validator_codes=_string_tuple(data["validator_codes"], "validator_codes"),
            error_type=_optional_str(data["error_type"], "error_type"),
            error=_optional_str(data["error"], "error"),
            timestamp=_required_str(data["timestamp"], "timestamp"),
        )
        _validate_attempt_record(record)
        return record


@dataclass(frozen=True)
class ResolutionAssessmentRecord:
    schema_version: int
    assessment_id: str
    comment_id: str
    section_id: str
    attempt_id: str
    critic_model: str
    context_isolated: bool
    verdict: str
    reason: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "assessment_id": self.assessment_id,
            "comment_id": self.comment_id,
            "section_id": self.section_id,
            "attempt_id": self.attempt_id,
            "critic_model": self.critic_model,
            "context_isolated": self.context_isolated,
            "verdict": self.verdict,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, value: object) -> ResolutionAssessmentRecord:
        data = _strict_object(
            value,
            expected={
                "schema_version",
                "assessment_id",
                "comment_id",
                "section_id",
                "attempt_id",
                "critic_model",
                "context_isolated",
                "verdict",
                "reason",
                "timestamp",
            },
            context="resolution assessment",
        )
        record = cls(
            schema_version=_required_int(data["schema_version"], "schema_version"),
            assessment_id=_required_str(data["assessment_id"], "assessment_id"),
            comment_id=_required_str(data["comment_id"], "comment_id"),
            section_id=_required_str(data["section_id"], "section_id"),
            attempt_id=_required_str(data["attempt_id"], "attempt_id"),
            critic_model=_required_str(data["critic_model"], "critic_model"),
            context_isolated=_required_bool(
                data["context_isolated"], "context_isolated"
            ),
            verdict=_required_str(data["verdict"], "verdict"),
            reason=_required_str(data["reason"], "reason"),
            timestamp=_required_str(data["timestamp"], "timestamp"),
        )
        _validate_assessment_record(record)
        return record


def parse_section_attempts_jsonl(text: str) -> tuple[SectionAttemptRecord, ...]:
    records = tuple(
        SectionAttemptRecord.from_dict(value)
        for value in _parse_strict_jsonl(text, "section attempts")
    )
    ids = [record.attempt_id for record in records]
    if len(set(ids)) != len(ids):
        _fail("duplicate_attempt_id", "section attempt IDs must be unique")
    return records


def parse_resolution_assessments_jsonl(
    text: str,
) -> tuple[ResolutionAssessmentRecord, ...]:
    records = tuple(
        ResolutionAssessmentRecord.from_dict(value)
        for value in _parse_strict_jsonl(text, "resolution assessments")
    )
    ids = [record.assessment_id for record in records]
    if len(set(ids)) != len(ids):
        _fail("duplicate_assessment_id", "assessment IDs must be unique")
    return records


@dataclass(frozen=True)
class ValidatedSectionReplacement:
    section_id: str
    body: str
    validation: SectionValidationResult
    context: SectionValidationContext


@dataclass(frozen=True)
class MergeSectionResult:
    section_id: str
    original_sha256: str
    final_body_sha256: str
    changed: bool


@dataclass(frozen=True)
class MergeResult:
    source_paper_sha256: str
    merged_paper_sha256: str
    merged_text: str
    sections: tuple[MergeSectionResult, ...]


@dataclass(frozen=True)
class SectionManifestMetadata:
    comment_ids: tuple[str, ...]
    attempt_ids: tuple[str, ...]
    final_status: str
    validation_result: SectionValidationResult | None
    validation_context: SectionValidationContext | None


@dataclass(frozen=True)
class SectionManifestEntry:
    section_id: str
    original_sha256: str
    final_body_sha256: str
    changed: bool
    comment_ids: tuple[str, ...]
    attempt_ids: tuple[str, ...]
    validation_report_sha256: str | None
    final_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "original_sha256": self.original_sha256,
            "final_body_sha256": self.final_body_sha256,
            "changed": self.changed,
            "comment_ids": list(self.comment_ids),
            "attempt_ids": list(self.attempt_ids),
            "validation_report_sha256": self.validation_report_sha256,
            "final_status": self.final_status,
        }

    @classmethod
    def from_dict(cls, value: object) -> SectionManifestEntry:
        data = _strict_object(
            value,
            expected={
                "section_id",
                "original_sha256",
                "final_body_sha256",
                "changed",
                "comment_ids",
                "attempt_ids",
                "validation_report_sha256",
                "final_status",
            },
            context="manifest section entry",
        )
        entry = cls(
            section_id=_required_str(data["section_id"], "section_id"),
            original_sha256=_required_hash(data["original_sha256"], "original_sha256"),
            final_body_sha256=_required_hash(data["final_body_sha256"], "final_body_sha256"),
            changed=_required_bool(data["changed"], "changed"),
            comment_ids=_string_tuple(data["comment_ids"], "comment_ids"),
            attempt_ids=_string_tuple(data["attempt_ids"], "attempt_ids"),
            validation_report_sha256=_optional_hash(
                data["validation_report_sha256"], "validation_report_sha256"
            ),
            final_status=_required_str(data["final_status"], "final_status"),
        )
        if entry.final_status not in _SECTION_FINAL_STATUSES:
            _fail("section_final_status_invalid", f"invalid status for {entry.section_id}")
        if len(set(entry.comment_ids)) != len(entry.comment_ids):
            _fail("duplicate_comment_id", f"duplicate comments for {entry.section_id}")
        if len(set(entry.attempt_ids)) != len(entry.attempt_ids):
            _fail("duplicate_attempt_id", f"duplicate attempts for {entry.section_id}")
        return entry


@dataclass(frozen=True)
class SectionRevisionManifest:
    schema_version: int
    mode: str
    claim_scope: str
    experiment_contract_path: str
    experiment_contract_sha256: str
    writer_model: str
    critic_model: str
    source_paper_path: str
    source_paper_sha256: str
    source_reviews_path: str
    source_reviews_sha256: str
    ledger_sha256: str
    plan_sha256: str
    attempts_sha256: str
    assessments_sha256: str
    unresolved_comments_sha256: str
    validation_context_path: str
    validation_context_sha256: str
    sections: tuple[SectionManifestEntry, ...]
    comment_counts: tuple[tuple[str, int], ...]
    merged_paper_sha256: str
    completed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "claim_scope": self.claim_scope,
            "experiment_contract_path": self.experiment_contract_path,
            "experiment_contract_sha256": self.experiment_contract_sha256,
            "writer_model": self.writer_model,
            "critic_model": self.critic_model,
            "source_paper_path": self.source_paper_path,
            "source_paper_sha256": self.source_paper_sha256,
            "source_reviews_path": self.source_reviews_path,
            "source_reviews_sha256": self.source_reviews_sha256,
            "ledger_sha256": self.ledger_sha256,
            "plan_sha256": self.plan_sha256,
            "attempts_sha256": self.attempts_sha256,
            "assessments_sha256": self.assessments_sha256,
            "unresolved_comments_sha256": self.unresolved_comments_sha256,
            "validation_context_path": self.validation_context_path,
            "validation_context_sha256": self.validation_context_sha256,
            "sections": [entry.to_dict() for entry in self.sections],
            "comment_counts": dict(self.comment_counts),
            "merged_paper_sha256": self.merged_paper_sha256,
            "completed": self.completed,
        }

    @classmethod
    def from_dict(cls, value: object) -> SectionRevisionManifest:
        data = _strict_object(
            value,
            expected={
                "schema_version",
                "mode",
                "claim_scope",
                "experiment_contract_path",
                "experiment_contract_sha256",
                "writer_model",
                "critic_model",
                "source_paper_path",
                "source_paper_sha256",
                "source_reviews_path",
                "source_reviews_sha256",
                "ledger_sha256",
                "plan_sha256",
                "attempts_sha256",
                "assessments_sha256",
                "unresolved_comments_sha256",
                "validation_context_path",
                "validation_context_sha256",
                "sections",
                "comment_counts",
                "merged_paper_sha256",
                "completed",
            },
            context="section revision manifest",
        )
        sections_raw = data["sections"]
        if not isinstance(sections_raw, list):
            _fail("invalid_type", "manifest sections must be a list")
        counts_raw = _strict_object(
            data["comment_counts"],
            expected={"input", "resolved", "unresolved", "not_actionable_with_reason"},
            context="manifest comment_counts",
        )
        manifest = cls(
            schema_version=_required_int(data["schema_version"], "schema_version"),
            mode=_required_str(data["mode"], "mode"),
            claim_scope=_required_str(data["claim_scope"], "claim_scope"),
            experiment_contract_path=_required_str(
                data["experiment_contract_path"], "experiment_contract_path"
            ),
            experiment_contract_sha256=_required_hash(
                data["experiment_contract_sha256"], "experiment_contract_sha256"
            ),
            writer_model=_required_str(data["writer_model"], "writer_model"),
            critic_model=_required_str(data["critic_model"], "critic_model"),
            source_paper_path=_required_str(data["source_paper_path"], "source_paper_path"),
            source_paper_sha256=_required_hash(data["source_paper_sha256"], "source_paper_sha256"),
            source_reviews_path=_required_str(data["source_reviews_path"], "source_reviews_path"),
            source_reviews_sha256=_required_hash(
                data["source_reviews_sha256"], "source_reviews_sha256"
            ),
            ledger_sha256=_required_hash(data["ledger_sha256"], "ledger_sha256"),
            plan_sha256=_required_hash(data["plan_sha256"], "plan_sha256"),
            attempts_sha256=_required_hash(data["attempts_sha256"], "attempts_sha256"),
            assessments_sha256=_required_hash(data["assessments_sha256"], "assessments_sha256"),
            unresolved_comments_sha256=_required_hash(
                data["unresolved_comments_sha256"], "unresolved_comments_sha256"
            ),
            validation_context_path=_required_str(
                data["validation_context_path"], "validation_context_path"
            ),
            validation_context_sha256=_required_hash(
                data["validation_context_sha256"], "validation_context_sha256"
            ),
            sections=tuple(SectionManifestEntry.from_dict(item) for item in sections_raw),
            comment_counts=tuple(
                (key, _required_int(counts_raw[key], f"comment_counts.{key}"))
                for key in ("input", "resolved", "unresolved", "not_actionable_with_reason")
            ),
            merged_paper_sha256=_required_hash(data["merged_paper_sha256"], "merged_paper_sha256"),
            completed=_required_bool(data["completed"], "completed"),
        )
        if manifest.schema_version != SCHEMA_VERSION:
            _fail("schema_version", "manifest schema_version must be 1")
        if manifest.mode != "sectional":
            _fail("manifest_mode_invalid", "manifest mode must be sectional")
        if manifest.claim_scope not in _CLAIM_SCOPES:
            _fail("claim_scope_invalid", "manifest claim_scope is invalid")
        if manifest.writer_model == manifest.critic_model:
            _fail("manifest_model_identity_invalid", "writer and critic must differ")
        _validate_relative_path(
            manifest.experiment_contract_path, "experiment_contract_path"
        )
        if not re.fullmatch(
            r"stage-09(?:_v[1-9]\d*)?/experiment_contract\.yaml",
            manifest.experiment_contract_path,
        ):
            _fail(
                "experiment_contract_path_invalid",
                "manifest experiment contract path is not canonical",
            )
        _validate_relative_path(manifest.source_paper_path, "source_paper_path")
        _validate_relative_path(manifest.source_reviews_path, "source_reviews_path")
        _validate_relative_path(
            manifest.validation_context_path,
            "validation_context_path",
        )
        if manifest.validation_context_path != "stage-19/validation_context.json":
            _fail(
                "validation_context_path_invalid",
                "validation context path must be stage-19/validation_context.json",
            )
        if any(count < 0 for _, count in manifest.comment_counts):
            _fail("manifest_comment_count_invalid", "comment counts cannot be negative")
        counts = dict(manifest.comment_counts)
        terminal_total = (
            counts["resolved"]
            + counts["unresolved"]
            + counts["not_actionable_with_reason"]
        )
        if terminal_total > counts["input"]:
            _fail("manifest_comment_count_invalid", "terminal counts exceed input")
        if manifest.completed and terminal_total != counts["input"]:
            _fail("manifest_ledger_not_closed", "completed manifest is not closed")
        return manifest


def validate_section_candidate(
    context: SectionValidationContext,
    candidate_body: str,
) -> SectionValidationResult:
    """Run all B1 hard checks against one proposed section body."""

    if not isinstance(candidate_body, str):
        raise TypeError("candidate_body must be a string")
    section = _section_by_id(context.document, context.section_id)
    if context.document.structure_issues:
        _fail("manuscript_structure_ambiguous", "strict manuscript required")
    if (
        isinstance(context.attempt, bool)
        or not isinstance(context.attempt, int)
        or context.attempt <= 0
    ):
        _fail("attempt_invalid", "attempt must be positive")
    if (
        isinstance(context.min_length_ratio, bool)
        or not isinstance(context.min_length_ratio, (int, float))
        or not math.isfinite(float(context.min_length_ratio))
        or not (0.0 < context.min_length_ratio <= 1.0)
    ):
        _fail("length_ratio_invalid", "min_length_ratio must be in (0, 1]")
    if (
        isinstance(context.max_length_ratio, bool)
        or not isinstance(context.max_length_ratio, (int, float))
        or not math.isfinite(float(context.max_length_ratio))
        or context.max_length_ratio < 1.0
        or context.max_length_ratio < context.min_length_ratio
    ):
        _fail("length_ratio_invalid", "max_length_ratio is invalid")
    if len(set(context.required_comment_ids)) != len(context.required_comment_ids):
        _fail("duplicate_comment_id", "required_comment_ids contains duplicates")
    if len(set(context.resolution_comment_ids)) != len(context.resolution_comment_ids):
        _fail("duplicate_comment_id", "resolution_comment_ids contains duplicates")
    if any(
        not isinstance(comment_id, str) or not comment_id.strip()
        for comment_id in context.required_comment_ids
        + context.resolution_comment_ids
    ):
        _fail("comment_id_invalid", "comment IDs must be nonempty strings")
    if any(not isinstance(key, str) or not key.strip() for key in context.allowed_citation_keys):
        _fail("citation_key_invalid", "allowed citation keys must be nonempty strings")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in context.grounded_numeric_values
    ):
        _fail("grounded_numeric_value_invalid", "grounded values must be finite numbers")

    failures: dict[str, list[str]] = {code: [] for code in _CHECK_CODES}
    if not candidate_body.strip():
        failures["empty_section_body"].append("candidate body is empty")

    candidate_tokens = MarkdownIt("commonmark").parse(candidate_body)
    if any(token.type == "heading_open" for token in candidate_tokens):
        failures["new_heading_introduced"].append(
            "candidate body contains a CommonMark heading"
        )
    if _HTML_HEADING_RE.search(_strip_markdown_non_prose(candidate_body)):
        failures["html_heading_introduced"].append(
            "candidate body contains a raw HTML heading"
        )
    structure_errors = _markdown_balance_errors(candidate_body)
    failures["markdown_structure_unbalanced"].extend(structure_errors)

    merged_text: str | None = None
    reparsed: ManuscriptDocument | None = None
    try:
        merged_text = merge_manuscript(
            context.document, {context.section_id: candidate_body}
        )
        reparsed = parse_manuscript(merged_text, strict=True)
    except (ManuscriptStructureError, KeyError, TypeError) as exc:
        failures["post_merge_heading_mismatch"].append(str(exc))
        failures["section_order_mismatch"].append(
            "merged manuscript could not preserve strict section order"
        )
    if reparsed is not None:
        original_order = tuple(item.section_id for item in context.document.sections)
        reparsed_order = tuple(item.section_id for item in reparsed.sections)
        if reparsed_order != original_order:
            failures["section_order_mismatch"].append(
                "section ID order changed after merge"
            )
        original_headings = tuple(
            (item.section_id, item.level, item.path, item.title, item.heading_source)
            for item in context.document.sections
        )
        reparsed_headings = tuple(
            (item.section_id, item.level, item.path, item.title, item.heading_source)
            for item in reparsed.sections
        )
        if reparsed_headings != original_headings:
            failures["post_merge_heading_mismatch"].append(
                "heading metadata changed after merge"
            )

    original_citations = extract_citation_keys(section.body)
    candidate_citations = extract_citation_keys(candidate_body)
    unknown_citations = sorted(
        (candidate_citations - original_citations) - context.allowed_citation_keys
    )
    if unknown_citations:
        failures["unknown_citation_key"].append(
            "unknown citation keys: " + ", ".join(unknown_citations)
        )
    removed_citations = sorted(original_citations - candidate_citations)
    if removed_citations:
        failures["required_citation_removed"].append(
            "removed citation keys: " + ", ".join(removed_citations)
        )

    original_refs = extract_reference_mentions(section.body)
    candidate_refs = extract_reference_mentions(candidate_body)
    removed_refs = sorted(original_refs - candidate_refs)
    if removed_refs:
        failures["required_reference_removed"].append(
            "removed references: " + ", ".join(_format_reference(ref) for ref in removed_refs)
        )
    declared_refs = extract_declared_reference_targets(_document_source(context.document))
    unknown_refs = sorted((candidate_refs - original_refs) - declared_refs)
    if unknown_refs:
        failures["unknown_reference_introduced"].append(
            "references without declared targets: "
            + ", ".join(_format_reference(ref) for ref in unknown_refs)
        )

    original_values, _ = extract_quantitative_values(section.body)
    candidate_values, unparsed = extract_quantitative_values(candidate_body)
    if unparsed:
        failures["unparsed_quantitative_expression"].append(
            "unparsed quantities: " + ", ".join(sorted(unparsed))
        )
    unknown_values = [
        value
        for value in candidate_values
        if not _matches_any(value, original_values)
        and not _matches_any(value, context.grounded_numeric_values)
    ]
    if unknown_values:
        failures["unknown_numeric_value"].append(
            "ungrounded values: " + ", ".join(_format_number(v) for v in unknown_values)
        )

    original_words = len(_WORD_RE.findall(section.body))
    candidate_words = len(_WORD_RE.findall(candidate_body))
    if original_words:
        if candidate_words < original_words * context.min_length_ratio:
            failures["abnormal_section_shrink"].append(
                f"word count shrank from {original_words} to {candidate_words}"
            )
        if candidate_words > original_words * context.max_length_ratio:
            failures["abnormal_section_growth"].append(
                f"word count grew from {original_words} to {candidate_words}"
            )

    missing_resolutions = sorted(
        set(context.required_comment_ids) - set(context.resolution_comment_ids)
    )
    unknown_resolutions = sorted(
        set(context.resolution_comment_ids) - set(context.required_comment_ids)
    )
    if missing_resolutions:
        failures["unaddressed_required_comment"].append(
            "missing resolution records: " + ", ".join(missing_resolutions)
        )
    if unknown_resolutions:
        failures["unaddressed_required_comment"].append(
            "unknown resolution records: " + ", ".join(unknown_resolutions)
        )

    checks = tuple(
        SectionValidationCheck(
            code=code,
            status="failed" if failures[code] else "passed",
            details=tuple(failures[code]),
        )
        for code in _CHECK_CODES
    )
    result = SectionValidationResult(
        schema_version=SCHEMA_VERSION,
        attempt_id=make_attempt_id(context.section_id, context.attempt),
        section_id=context.section_id,
        original_sha256=section.original_sha256,
        candidate_sha256=_sha256(candidate_body),
        accepted=all(check.status == "passed" for check in checks),
        checks=checks,
    )
    _validate_result_shape(result)
    return result


def merge_validated_sections(
    document: ManuscriptDocument,
    replacements: Mapping[str, ValidatedSectionReplacement],
) -> MergeResult:
    """Merge only bodies whose validation result is accepted and hash-bound."""

    if document.structure_issues:
        _fail("manuscript_structure_ambiguous", "strict manuscript required")
    known = {section.section_id: section for section in document.sections}
    unknown = sorted(set(replacements) - set(known))
    if unknown:
        _fail("unknown_section_id", "unknown replacements: " + ", ".join(unknown))

    bodies: dict[str, str] = {}
    for section_id, replacement in replacements.items():
        section = known[section_id]
        if replacement.section_id != section_id:
            _fail("replacement_section_mismatch", f"replacement key mismatches {section_id}")
        if replacement.context.document != document:
            _fail("validation_context_mismatch", f"context document mismatches {section_id}")
        if replacement.context.section_id != section_id:
            _fail("validation_context_mismatch", f"context section mismatches {section_id}")
        result = SectionValidationResult.from_dict(replacement.validation.to_dict())
        if not result.accepted:
            _fail("unaccepted_section_revision", f"section {section_id} was not accepted")
        if result.section_id != section_id:
            _fail("validation_section_mismatch", f"validation mismatches {section_id}")
        if result.original_sha256 != section.original_sha256:
            _fail("validation_original_hash_mismatch", f"original hash mismatches {section_id}")
        if result.candidate_sha256 != _sha256(replacement.body):
            _fail("validation_candidate_hash_mismatch", f"candidate hash mismatches {section_id}")
        recomputed = validate_section_candidate(replacement.context, replacement.body)
        if recomputed != result:
            _fail(
                "validation_recompute_mismatch",
                f"stored validation does not match deterministic recomputation for {section_id}",
            )
        bodies[section_id] = replacement.body

    try:
        merged = merge_manuscript(document, bodies)
        reparsed = parse_manuscript(merged, strict=True)
    except (ManuscriptStructureError, KeyError, TypeError) as exc:
        raise SectionalRevisionContractError(
            (ContractIssue("merge_structure_mismatch", str(exc)),)
        ) from exc
    if _heading_identity(reparsed) != _heading_identity(document):
        _fail("merge_structure_mismatch", "merged heading identity changed")

    section_results: list[MergeSectionResult] = []
    for original, final in zip(document.sections, reparsed.sections, strict=True):
        expected_body = bodies.get(original.section_id, original.body)
        if final.body != expected_body:
            _fail("merge_body_mismatch", f"merged body mismatches {original.section_id}")
        section_results.append(
            MergeSectionResult(
                section_id=original.section_id,
                original_sha256=original.original_sha256,
                final_body_sha256=_sha256(final.body),
                changed=original.section_id in bodies and final.body != original.body,
            )
        )
    return MergeResult(
        source_paper_sha256=document.source_sha256,
        merged_paper_sha256=_sha256(merged),
        merged_text=merged,
        sections=tuple(section_results),
    )


def build_section_revision_manifest(
    *,
    document: ManuscriptDocument,
    merge_result: MergeResult,
    ledger: ReviewLedger,
    plan: RevisionPlan,
    reviews: str,
    claim_scope: str,
    experiment_contract_path: str,
    experiment_contract_sha256: str,
    writer_model: str,
    critic_model: str,
    source_paper_path: str,
    section_metadata: Mapping[str, SectionManifestMetadata],
    attempts_text: str,
    assessments_text: str,
    unresolved_comments_text: str,
    completed: bool,
    validation_context_text: str | None = None,
) -> SectionRevisionManifest:
    """Derive a manifest from authoritative inputs; never accept free-form counts."""

    manifest = _construct_manifest(
        document=document,
        merge_result=merge_result,
        ledger=ledger,
        plan=plan,
        reviews=reviews,
        claim_scope=claim_scope,
        experiment_contract_path=experiment_contract_path,
        experiment_contract_sha256=experiment_contract_sha256,
        writer_model=writer_model,
        critic_model=critic_model,
        source_paper_path=source_paper_path,
        section_metadata=section_metadata,
        attempts_text=attempts_text,
        assessments_text=assessments_text,
        unresolved_comments_text=unresolved_comments_text,
        completed=completed,
        validation_context_text=validation_context_text,
    )
    return SectionRevisionManifest.from_dict(manifest.to_dict())


def build_unresolved_comments_artifact(ledger: ReviewLedger) -> dict[str, Any]:
    """Derive the unresolved diagnostic payload from the authoritative ledger."""

    validate_review_ledger(ledger)
    return {
        "schema_version": SCHEMA_VERSION,
        "ledger_sha256": canonical_json_sha256(ledger.to_dict()),
        "comments": [
            {
                "comment_id": comment.comment_id,
                "final_status": comment.final_status,
                "reason": comment.resolution_reason,
            }
            for comment in ledger.comments
            if comment.final_status in {"unresolved", "not_actionable_with_reason"}
        ],
    }


def validate_section_revision_manifest(
    manifest: object,
    *,
    document: ManuscriptDocument,
    merge_result: MergeResult,
    ledger: ReviewLedger,
    plan: RevisionPlan,
    reviews: str,
    claim_scope: str,
    experiment_contract_path: str,
    experiment_contract_sha256: str,
    writer_model: str,
    critic_model: str,
    source_paper_path: str,
    section_metadata: Mapping[str, SectionManifestMetadata],
    attempts_text: str,
    assessments_text: str,
    unresolved_comments_text: str,
    completed: bool,
    validation_context_text: str | None = None,
) -> SectionRevisionManifest:
    """Recompute the complete manifest and reject any stale or edited field."""

    parsed = (
        SectionRevisionManifest.from_dict(manifest.to_dict())
        if isinstance(manifest, SectionRevisionManifest)
        else SectionRevisionManifest.from_dict(manifest)
    )
    expected = _construct_manifest(
        document=document,
        merge_result=merge_result,
        ledger=ledger,
        plan=plan,
        reviews=reviews,
        claim_scope=claim_scope,
        experiment_contract_path=experiment_contract_path,
        experiment_contract_sha256=experiment_contract_sha256,
        writer_model=writer_model,
        critic_model=critic_model,
        source_paper_path=source_paper_path,
        section_metadata=section_metadata,
        attempts_text=attempts_text,
        assessments_text=assessments_text,
        unresolved_comments_text=unresolved_comments_text,
        completed=completed,
        validation_context_text=validation_context_text,
    )
    if parsed != expected:
        _fail(
            "manifest_recompute_mismatch",
            "manifest does not match authoritative inputs",
        )
    return parsed


def _construct_manifest(
    *,
    document: ManuscriptDocument,
    merge_result: MergeResult,
    ledger: ReviewLedger,
    plan: RevisionPlan,
    reviews: str,
    claim_scope: str,
    experiment_contract_path: str,
    experiment_contract_sha256: str,
    writer_model: str,
    critic_model: str,
    source_paper_path: str,
    section_metadata: Mapping[str, SectionManifestMetadata],
    attempts_text: str,
    assessments_text: str,
    unresolved_comments_text: str,
    completed: bool,
    validation_context_text: str | None,
) -> SectionRevisionManifest:
    if claim_scope not in _CLAIM_SCOPES:
        _fail("claim_scope_invalid", f"invalid claim scope {claim_scope!r}")
    _validate_relative_path(experiment_contract_path, "experiment_contract_path")
    _required_hash(experiment_contract_sha256, "experiment_contract_sha256")
    writer_model = _required_str(writer_model, "writer_model")
    critic_model = _required_str(critic_model, "critic_model")
    if writer_model == critic_model:
        _fail("manifest_model_identity_invalid", "writer and critic must differ")
    _validate_relative_path(source_paper_path, "source_paper_path")
    if validation_context_text is None:
        validation_context_text = _default_validation_context_text(document)
    if not all(
        isinstance(value, str)
        for value in (
            assessments_text,
            attempts_text,
            unresolved_comments_text,
            validation_context_text,
        )
    ):
        _fail(
            "invalid_type",
            "attempt, assessment, unresolved, and validation context artifacts must be strings",
        )
    attempt_records = parse_section_attempts_jsonl(attempts_text)
    assessment_records = parse_resolution_assessments_jsonl(assessments_text)
    if any(record.writer_model != writer_model for record in attempt_records):
        _fail("manifest_writer_model_mismatch", "attempt writer model is inconsistent")
    if any(record.critic_model != critic_model for record in assessment_records):
        _fail("manifest_critic_model_mismatch", "assessment critic model is inconsistent")
    try:
        validation_context_payload = json.loads(validation_context_text)
    except json.JSONDecodeError as exc:
        raise SectionalRevisionContractError(
            (ContractIssue("validation_context_artifact_invalid", str(exc)),)
        ) from exc
    if not isinstance(validation_context_payload, dict):
        _fail(
            "validation_context_artifact_invalid",
            "validation context artifact must be an object",
        )
    _validate_validation_context_payload(
        validation_context_payload,
        document=document,
        section_metadata=section_metadata,
    )
    try:
        unresolved_payload = json.loads(unresolved_comments_text)
    except json.JSONDecodeError as exc:
        raise SectionalRevisionContractError(
            (ContractIssue("unresolved_artifact_invalid", str(exc)),)
        ) from exc
    if unresolved_payload != build_unresolved_comments_artifact(ledger):
        _fail(
            "unresolved_artifact_mismatch",
            "unresolved comments artifact does not match ledger",
        )
    if document.structure_issues:
        _fail("manuscript_structure_ambiguous", "strict manuscript required")
    merged_bodies = _validate_merge_result(document, merge_result)

    validate_revision_plan(plan, ledger, document, reviews=reviews)
    validate_review_ledger(ledger, reviews=reviews, require_final=completed)
    unknown_metadata = sorted(
        set(section_metadata) - {section.section_id for section in document.sections}
    )
    if unknown_metadata:
        _fail(
            "manifest_unknown_section",
            "metadata has unknown sections: " + ", ".join(unknown_metadata),
        )

    counts = {
        "input": len(ledger.comments),
        "resolved": sum(c.final_status == "resolved" for c in ledger.comments),
        "unresolved": sum(c.final_status == "unresolved" for c in ledger.comments),
        "not_actionable_with_reason": sum(
            c.final_status == "not_actionable_with_reason" for c in ledger.comments
        ),
    }
    terminal_total = (
        counts["resolved"]
        + counts["unresolved"]
        + counts["not_actionable_with_reason"]
    )
    if completed and terminal_total != counts["input"]:
        _fail("manifest_ledger_not_closed", "completed manifest requires ledger closure")
    if not completed and terminal_total > counts["input"]:
        _fail("manifest_comment_count_invalid", "terminal counts exceed input")
    if completed and claim_scope != "pipeline_validation":
        blocking = [
            comment.comment_id
            for comment in ledger.comments
            if comment.required and comment.final_status != "resolved"
        ]
        if blocking:
            _fail(
                "manifest_required_comment_unresolved",
                "strict claim scope has unresolved required comments: "
                + ", ".join(blocking),
            )

    comments_by_id = {comment.comment_id: comment for comment in ledger.comments}
    assignments_by_id = {assignment.comment_id: assignment for assignment in plan.assignments}
    entries: list[SectionManifestEntry] = []
    attempts_seen: set[str] = set()
    for merge_section in merge_result.sections:
        metadata = section_metadata.get(merge_section.section_id)
        if metadata is None:
            metadata = SectionManifestMetadata((), (), "unchanged", None, None)
        if metadata.final_status not in _SECTION_FINAL_STATUSES:
            _fail(
                "section_final_status_invalid",
                f"invalid status for {merge_section.section_id}",
            )
        if len(set(metadata.comment_ids)) != len(metadata.comment_ids):
            _fail("duplicate_comment_id", f"duplicate comments for {merge_section.section_id}")
        if len(set(metadata.attempt_ids)) != len(metadata.attempt_ids):
            _fail("duplicate_attempt_id", f"duplicate attempts for {merge_section.section_id}")
        if any(
            not _attempt_id_matches(attempt_id, merge_section.section_id)
            for attempt_id in metadata.attempt_ids
        ):
            _fail(
                "manifest_attempt_id_invalid",
                f"attempt ID does not bind to {merge_section.section_id}",
            )
        unknown_comments = sorted(set(metadata.comment_ids) - set(comments_by_id))
        if unknown_comments:
            _fail(
                "manifest_unknown_comment",
                "unknown comments: " + ", ".join(unknown_comments),
            )
        for comment_id in metadata.comment_ids:
            assignment = assignments_by_id[comment_id]
            if (
                assignment.disposition != "assigned"
                or merge_section.section_id not in assignment.target_section_ids
            ):
                _fail(
                    "manifest_comment_target_mismatch",
                    f"{comment_id} is not assigned to {merge_section.section_id}",
                )
        expected_comments = {
            comment_id
            for comment_id, assignment in assignments_by_id.items()
            if assignment.disposition == "assigned"
            and merge_section.section_id in assignment.target_section_ids
        }
        if completed and set(metadata.comment_ids) != expected_comments:
            _fail(
                "manifest_comment_link_mismatch",
                f"section comment links mismatch plan for {merge_section.section_id}",
            )
        required_comments = {
            comment_id
            for comment_id in metadata.comment_ids
            if comments_by_id[comment_id].required
        }
        duplicate_attempts = sorted(set(metadata.attempt_ids) & attempts_seen)
        if duplicate_attempts:
            _fail(
                "manifest_attempt_reused",
                "attempt IDs reused across sections: " + ", ".join(duplicate_attempts),
            )
        attempts_seen.update(metadata.attempt_ids)
        validation_hash = _validate_section_manifest_state(
            merge_section,
            merged_bodies[merge_section.section_id],
            document,
            metadata,
            required_comments,
        )
        entries.append(
            SectionManifestEntry(
                section_id=merge_section.section_id,
                original_sha256=merge_section.original_sha256,
                final_body_sha256=merge_section.final_body_sha256,
                changed=merge_section.changed,
                comment_ids=metadata.comment_ids,
                attempt_ids=metadata.attempt_ids,
                validation_report_sha256=validation_hash,
                final_status=metadata.final_status,
            )
        )

    entries_by_id = {entry.section_id: entry for entry in entries}
    for comment_id, assignment in assignments_by_id.items():
        comment = comments_by_id[comment_id]
        if assignment.disposition == "assigned":
            if completed:
                for target in assignment.target_section_ids:
                    if comment_id not in entries_by_id[target].comment_ids:
                        _fail(
                            "manifest_comment_link_missing",
                            f"{comment_id} missing from target section {target}",
                        )
                if comment.working_status != "assigned":
                    _fail("manifest_ledger_plan_mismatch", f"{comment_id} is not assigned")
                if comment.target_section_ids != assignment.target_section_ids:
                    _fail(
                        "manifest_ledger_plan_mismatch",
                        f"{comment_id} target sections mismatch",
                    )
                if comment.final_status not in {"resolved", "unresolved"}:
                    _fail(
                        "manifest_ledger_plan_mismatch",
                        f"{comment_id} has invalid assigned final status",
                    )
                if comment.final_status == "resolved" and any(
                    entries_by_id[target].final_status != "accepted"
                    for target in assignment.target_section_ids
                ):
                    _fail(
                        "manifest_resolved_section_not_accepted",
                        f"{comment_id} resolved while a target section is unresolved",
                    )
                linked_attempts = {
                    attempt_id
                    for target in assignment.target_section_ids
                    for attempt_id in entries_by_id[target].attempt_ids
                }
                if not set(comment.attempt_ids).issubset(linked_attempts):
                    _fail(
                        "manifest_attempt_link_mismatch",
                        f"{comment_id} ledger attempts are not linked",
                    )
                if comment.final_status == "resolved" and not comment.attempt_ids:
                    _fail(
                        "manifest_resolved_without_attempt",
                        f"{comment_id} resolved without attempt evidence",
                    )
        elif completed:
            if comment.working_status != "unassigned" or comment.target_section_ids:
                _fail(
                    "manifest_ledger_plan_mismatch",
                    f"{comment_id} terminal disposition has assignment state",
                )
            if comment.final_status != assignment.disposition:
                _fail(
                    "manifest_ledger_plan_mismatch",
                    f"{comment_id} final status mismatches plan disposition",
                )

    return SectionRevisionManifest(
        schema_version=SCHEMA_VERSION,
        mode="sectional",
        claim_scope=claim_scope,
        experiment_contract_path=experiment_contract_path,
        experiment_contract_sha256=experiment_contract_sha256,
        writer_model=writer_model,
        critic_model=critic_model,
        source_paper_path=source_paper_path,
        source_paper_sha256=document.source_sha256,
        source_reviews_path=ledger.source_reviews_path,
        source_reviews_sha256=ledger.source_reviews_sha256,
        ledger_sha256=canonical_json_sha256(ledger.to_dict()),
        plan_sha256=canonical_json_sha256(plan.to_dict()),
        attempts_sha256=_sha256(attempts_text),
        assessments_sha256=_sha256(assessments_text),
        unresolved_comments_sha256=_sha256(unresolved_comments_text),
        validation_context_path="stage-19/validation_context.json",
        validation_context_sha256=_sha256(validation_context_text),
        sections=tuple(entries),
        comment_counts=tuple(
            (key, counts[key])
            for key in ("input", "resolved", "unresolved", "not_actionable_with_reason")
        ),
        merged_paper_sha256=merge_result.merged_paper_sha256,
        completed=completed,
    )


def _default_validation_context_text(document: ManuscriptDocument) -> str:
    config_payload = {
        "max_section_retries": 1,
        "min_length_ratio": 0.80,
        "max_length_ratio": 1.75,
    }
    payload = {
        "schema_version": 1,
        "source_paper_sha256": document.source_sha256,
        "allowed_citation_keys": [],
        "grounded_numeric_values": [],
        **config_payload,
        "sources": [
            {
                "kind": "config",
                "path": "config.paper_revision",
                "sha256": canonical_json_sha256(config_payload),
            }
        ],
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n"


def _validate_validation_context_payload(
    value: Mapping[str, Any],
    *,
    document: ManuscriptDocument,
    section_metadata: Mapping[str, SectionManifestMetadata],
) -> None:
    data = _strict_object(
        value,
        expected={
            "schema_version",
            "source_paper_sha256",
            "allowed_citation_keys",
            "grounded_numeric_values",
            "max_section_retries",
            "min_length_ratio",
            "max_length_ratio",
            "sources",
        },
        context="validation context artifact",
    )
    if _required_int(data["schema_version"], "schema_version") != SCHEMA_VERSION:
        _fail("schema_version", "validation context schema_version must be 1")
    if _required_hash(
        data["source_paper_sha256"], "source_paper_sha256"
    ) != document.source_sha256:
        _fail(
            "validation_context_source_mismatch",
            "validation context paper hash mismatches",
        )
    citation_keys = _string_tuple(
        data["allowed_citation_keys"], "allowed_citation_keys"
    )
    if len(set(citation_keys)) != len(citation_keys):
        _fail("validation_context_duplicate", "duplicate citation keys")
    values_raw = data["grounded_numeric_values"]
    if not isinstance(values_raw, list):
        _fail("invalid_type", "grounded_numeric_values must be a list")
    numeric_values: list[float] = []
    for value_raw in values_raw:
        if (
            isinstance(value_raw, bool)
            or not isinstance(value_raw, (int, float))
            or not math.isfinite(float(value_raw))
        ):
            _fail(
                "grounded_numeric_value_invalid",
                "grounded numeric values must be finite numbers",
            )
        numeric_values.append(float(value_raw))
    if len(set(numeric_values)) != len(numeric_values):
        _fail("validation_context_duplicate", "duplicate grounded numeric values")
    retries = _required_int(data["max_section_retries"], "max_section_retries")
    if not 0 <= retries <= 3:
        _fail("validation_context_config_invalid", "max_section_retries is unsafe")
    min_ratio = _required_number(data["min_length_ratio"], "min_length_ratio")
    max_ratio = _required_number(data["max_length_ratio"], "max_length_ratio")
    if not 0.5 <= min_ratio <= 1.0 or not 1.0 <= max_ratio <= 3.0:
        _fail("validation_context_config_invalid", "length ratios are unsafe")

    sources_raw = data["sources"]
    if not isinstance(sources_raw, list):
        _fail("invalid_type", "validation context sources must be a list")
    sources: list[tuple[str, str, str]] = []
    for index, source_raw in enumerate(sources_raw):
        source = _strict_object(
            source_raw,
            expected={"kind", "path", "sha256"},
            context=f"validation context source {index}",
        )
        kind = _required_str(source["kind"], "source.kind")
        path = _required_str(source["path"], "source.path")
        digest = _required_hash(source["sha256"], "source.sha256")
        if kind not in {"citations", "metrics", "config"}:
            _fail("validation_context_source_invalid", f"invalid source kind {kind}")
        if kind == "config":
            if path != "config.paper_revision":
                _fail(
                    "validation_context_source_invalid",
                    "config source path must be config.paper_revision",
                )
        else:
            _validate_relative_path(path, "source.path")
            if path.split("/", 1)[0].startswith("stage-10"):
                _fail(
                    "validation_context_stage10_source",
                    "Stage 10 artifacts cannot ground sectional revision",
                )
        sources.append((kind, path, digest))
    if len(set(sources)) != len(sources):
        _fail("validation_context_duplicate", "duplicate context sources")
    config_payload = {
        "max_section_retries": retries,
        "min_length_ratio": min_ratio,
        "max_length_ratio": max_ratio,
    }
    expected_config_source = (
        "config",
        "config.paper_revision",
        canonical_json_sha256(config_payload),
    )
    if sources.count(expected_config_source) != 1:
        _fail(
            "validation_context_config_source_mismatch",
            "validation context config source hash mismatches",
        )
    if citation_keys and not any(kind == "citations" for kind, _, _ in sources):
        _fail("validation_context_source_missing", "citation source is missing")
    if numeric_values and not any(kind == "metrics" for kind, _, _ in sources):
        _fail("validation_context_source_missing", "metric source is missing")
    for metadata in section_metadata.values():
        context = metadata.validation_context
        if context is None:
            continue
        if context.allowed_citation_keys != frozenset(citation_keys):
            _fail(
                "validation_context_citation_mismatch",
                "section citation whitelist mismatches context artifact",
            )
        if context.grounded_numeric_values != tuple(numeric_values):
            _fail(
                "validation_context_numeric_mismatch",
                "section numeric whitelist mismatches context artifact",
            )
        if (
            context.min_length_ratio != min_ratio
            or context.max_length_ratio != max_ratio
        ):
            _fail(
                "validation_context_ratio_mismatch",
                "section length ratios mismatch context artifact",
            )


def _validate_section_manifest_state(
    merge_section: MergeSectionResult,
    final_body: str,
    document: ManuscriptDocument,
    metadata: SectionManifestMetadata,
    required_comment_ids: set[str],
) -> str | None:
    validation = metadata.validation_result
    context = metadata.validation_context
    validation_hash: str | None = None
    if (validation is None) is not (context is None):
        _fail(
            "manifest_validation_context_missing",
            f"validation/context pair incomplete for {merge_section.section_id}",
        )
    if validation is not None:
        validation = SectionValidationResult.from_dict(validation.to_dict())
        validation_hash = canonical_json_sha256(validation.to_dict())
        if context is None:
            _fail(
                "manifest_validation_context_missing",
                f"validation context missing for {merge_section.section_id}",
            )
        if context.document != document or context.section_id != merge_section.section_id:
            _fail(
                "manifest_validation_context_mismatch",
                f"validation context mismatches {merge_section.section_id}",
            )
        if set(context.required_comment_ids) != required_comment_ids:
            _fail(
                "manifest_validation_comment_mismatch",
                f"validation required comments mismatch {merge_section.section_id}",
            )
        recomputed = validate_section_candidate(context, final_body)
        if recomputed != validation:
            _fail(
                "manifest_validation_recompute_mismatch",
                f"validation does not recompute for {merge_section.section_id}",
            )
        if validation.section_id != merge_section.section_id:
            _fail(
                "manifest_validation_section_mismatch",
                f"validation section mismatches {merge_section.section_id}",
            )
        if validation.original_sha256 != merge_section.original_sha256:
            _fail(
                "manifest_validation_original_mismatch",
                f"validation original hash mismatches {merge_section.section_id}",
            )
        if validation.attempt_id not in metadata.attempt_ids:
            _fail(
                "manifest_validation_attempt_mismatch",
                f"validation attempt is not linked for {merge_section.section_id}",
            )
    if metadata.final_status == "unchanged":
        if merge_section.changed or metadata.comment_ids or metadata.attempt_ids:
            _fail(
                "manifest_unchanged_state_invalid",
                f"unchanged section {merge_section.section_id} has revision state",
            )
        if validation is not None:
            _fail(
                "manifest_unchanged_state_invalid",
                f"unchanged section {merge_section.section_id} has validation hash",
            )
    elif metadata.final_status == "accepted":
        if not merge_section.changed:
            _fail(
                "manifest_accepted_state_invalid",
                f"accepted section {merge_section.section_id} is byte-identical",
            )
        if not metadata.comment_ids or not metadata.attempt_ids:
            _fail(
                "manifest_accepted_state_invalid",
                f"accepted section {merge_section.section_id} lacks links",
            )
        if validation is None or not validation.accepted:
            _fail(
                "manifest_accepted_state_invalid",
                f"accepted section {merge_section.section_id} lacks accepted validation",
            )
        if validation.candidate_sha256 != merge_section.final_body_sha256:
            _fail(
                "manifest_validation_candidate_mismatch",
                f"validation candidate mismatches {merge_section.section_id}",
            )
    else:
        if merge_section.changed:
            _fail(
                "manifest_unresolved_state_invalid",
                f"unresolved section {merge_section.section_id} changed",
            )
        if not metadata.comment_ids or not metadata.attempt_ids:
            _fail(
                "manifest_unresolved_state_invalid",
                f"unresolved section {merge_section.section_id} lacks links",
            )
        if validation is not None and validation.accepted:
            _fail(
                "manifest_unresolved_state_invalid",
                f"unresolved section {merge_section.section_id} has accepted validation",
            )
    return validation_hash


def _validate_merge_result(
    document: ManuscriptDocument,
    merge_result: MergeResult,
) -> dict[str, str]:
    if merge_result.source_paper_sha256 != document.source_sha256:
        _fail("merge_source_hash_mismatch", "merge result source hash mismatches")
    if merge_result.merged_paper_sha256 != _sha256(merge_result.merged_text):
        _fail("merge_output_hash_mismatch", "merge result output hash mismatches")
    try:
        reparsed = parse_manuscript(merge_result.merged_text, strict=True)
    except ManuscriptStructureError as exc:
        raise SectionalRevisionContractError(
            (ContractIssue("merge_structure_mismatch", str(exc)),)
        ) from exc
    if _heading_identity(reparsed) != _heading_identity(document):
        _fail("merge_structure_mismatch", "merge result heading identity mismatches")
    if len(merge_result.sections) != len(document.sections):
        _fail("merge_section_order_mismatch", "merge result section count mismatches")
    bodies: dict[str, str] = {}
    for original, final, record in zip(
        document.sections,
        reparsed.sections,
        merge_result.sections,
        strict=True,
    ):
        if record.section_id != original.section_id:
            _fail("merge_section_order_mismatch", "merge result section order mismatches")
        if record.original_sha256 != original.original_sha256:
            _fail(
                "merge_original_hash_mismatch",
                f"merge original hash mismatches {original.section_id}",
            )
        if record.final_body_sha256 != _sha256(final.body):
            _fail(
                "merge_final_body_hash_mismatch",
                f"merge body hash mismatches {original.section_id}",
            )
        if record.changed is not (final.body != original.body):
            _fail(
                "merge_changed_flag_mismatch",
                f"merge changed flag mismatches {original.section_id}",
            )
        bodies[original.section_id] = final.body
    return bodies


def extract_citation_keys(text: str) -> frozenset[str]:
    stripped = _strip_markdown_non_prose(text)
    keys: set[str] = set()
    for body in _LATEX_CITE_RE.findall(stripped):
        keys.update(key.strip() for key in body.split(",") if key.strip())
    for bracket in re.findall(r"\[([^\[\]]{4,300})\]", stripped):
        parts = [part.strip() for part in re.split(r"[,;]", bracket)]
        if parts and all(_MARKDOWN_CITE_KEY_RE.fullmatch(part) for part in parts if part):
            keys.update(part for part in parts if part)
    return frozenset(keys)


def extract_reference_mentions(text: str) -> frozenset[tuple[str, str]]:
    prose = _strip_markdown_non_prose(text)
    references: set[tuple[str, str]] = set()
    for match in _REFERENCE_RE.finditer(prose):
        references.update(
            _normalize_reference(match.group(1), label)
            for label in _REFERENCE_LABEL_RE.findall(match.group(2))
        )
    return frozenset(references)


def extract_declared_reference_targets(text: str) -> frozenset[tuple[str, str]]:
    text = _strip_markdown_non_prose(text)
    targets = {
        _normalize_reference(match.group(1), match.group(2))
        for match in _DECLARED_REFERENCE_RE.finditer(text)
    }
    targets.update(
        _normalize_reference(match.group(1), match.group(2))
        for match in _IMAGE_REFERENCE_RE.finditer(text)
    )
    return frozenset(targets)


def extract_quantitative_values(text: str) -> tuple[tuple[float, ...], tuple[str, ...]]:
    """Return normalized numeric values and unparsed quantitative phrases."""

    cleaned = _strip_quantitative_non_metrics(text)
    values = list(_extract_math_metric_literals(text))
    unparsed = [match.group(0) for match in _AMBIGUOUS_QUANTITY_RE.finditer(cleaned)]
    occupied: list[tuple[int, int]] = []

    for match in _FRACTION_WORD_RE.finditer(cleaned):
        numerator = _parse_number_words(match.group(1))
        denominators = {
            "half": 2,
            "halves": 2,
            "third": 3,
            "thirds": 3,
            "quarter": 4,
            "quarters": 4,
            "fourth": 4,
            "fourths": 4,
        }
        if numerator is None:
            unparsed.append(match.group(0))
        else:
            values.append(numerator / denominators[match.group(2).casefold()])
        occupied.append(match.span())

    for match in _NUMBER_WORD_UNIT_RE.finditer(cleaned):
        if _span_overlaps(match.span(), occupied):
            continue
        parsed = _parse_number_words(match.group(1))
        if parsed is None:
            unparsed.append(match.group(0))
        else:
            unit = match.group(2).casefold()
            percent_units = {"%", "percent", "percentage", "percentages"}
            values.append(parsed / 100.0 if unit in percent_units else parsed)
        occupied.append(match.span())

    for match in _NUMBER_RE.finditer(cleaned):
        if _span_overlaps(match.span(), occupied):
            continue
        raw = match.group(0).strip()
        numeric = raw.rstrip("%").strip().replace(",", "")
        try:
            value = float(numeric)
        except ValueError:
            unparsed.append(raw)
            continue
        percent_word = _PERCENT_WORD_RE.match(cleaned, match.end())
        if raw.endswith("%") or percent_word is not None:
            value /= 100.0
        values.append(value)
    return tuple(values), tuple(unparsed)


def _validate_result_shape(result: SectionValidationResult) -> None:
    issues: list[ContractIssue] = []
    if result.schema_version != SCHEMA_VERSION:
        issues.append(ContractIssue("schema_version", "schema_version must be 1"))
    if not _attempt_id_matches(result.attempt_id, result.section_id):
        issues.append(
            ContractIssue(
                "attempt_id_invalid",
                "attempt_id must contain the complete section_id and positive attempt",
            )
        )
    codes = tuple(check.code for check in result.checks)
    if codes != _CHECK_CODES:
        issues.append(
            ContractIssue(
                "validation_checks_incomplete",
                "validation checks must contain every B1 code in canonical order",
            )
        )
    expected = all(check.status == "passed" for check in result.checks)
    if result.accepted is not expected:
        issues.append(
            ContractIssue(
                "validation_acceptance_mismatch",
                "accepted must equal all mandatory checks passing",
            )
        )
    if issues:
        raise SectionalRevisionContractError(issues)


def _attempt_id_matches(attempt_id: str, section_id: str) -> bool:
    return bool(
        re.fullmatch(
            re.escape(f"sec-{section_id}-a") + r"[1-9]\d*",
            attempt_id,
        )
    )


def _markdown_balance_errors(text: str) -> list[str]:
    errors: list[str] = []
    open_fence: tuple[str, int] | None = None
    non_fence_lines: list[str] = []
    for line in split_commonmark_lines_keepends(text):
        match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
        if not match:
            if open_fence is None:
                non_fence_lines.append(line)
            continue
        marker = match.group(1)[0]
        length = len(match.group(1))
        tail = match.group(2)
        if open_fence is None:
            open_fence = (marker, length)
        elif marker == open_fence[0] and length >= open_fence[1] and not tail.strip():
            open_fence = None
    if open_fence is not None:
        errors.append("unclosed fenced code block")
    prose = _strip_inline_code("".join(non_fence_lines))
    if prose.count("<!--") != prose.count("-->"):
        errors.append("unbalanced HTML comment")
    math_prose = re.sub(r"<!--.*?-->", "", prose, flags=re.DOTALL)
    if len(re.findall(r"(?<!\\)\$\$", math_prose)) % 2:
        errors.append("unbalanced $$ display math delimiter")
    if math_prose.count(r"\[") != math_prose.count(r"\]"):
        errors.append("unbalanced \\[ display math delimiter")
    return errors


def _strip_markdown_non_prose(text: str) -> str:
    text = re.sub(r"```.*?```|~~~.*?~~~", "", text, flags=re.DOTALL)
    text = _strip_inline_code(text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    text = re.sub(r"\\\[.*?\\\]", "", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\\)\$.*?(?<!\\)\$", "", text)
    return text


def _strip_inline_code(text: str) -> str:
    return re.sub(r"(`+).*?\1", "", text, flags=re.DOTALL)


def _strip_quantitative_non_metrics(text: str) -> str:
    cleaned = _strip_markdown_non_prose(text)
    cleaned = _LATEX_CITE_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\[[^\[\]]{4,300}\]", " ", cleaned)
    cleaned = _AUTHOR_YEAR_RE.sub(" ", cleaned)
    cleaned = _REFERENCE_RE.sub(" ", cleaned)
    cleaned = _STRUCTURAL_NUMBER_RE.sub(" ", cleaned)
    return cleaned


def _extract_math_metric_literals(text: str) -> tuple[float, ...]:
    math_source = re.sub(r"```.*?```|~~~.*?~~~", "", text, flags=re.DOTALL)
    math_source = _strip_inline_code(math_source)
    math_source = re.sub(r"<!--.*?-->", "", math_source, flags=re.DOTALL)
    values: list[float] = []
    for math_match in _MATH_SPAN_RE.finditer(math_source):
        for number_match in _NUMBER_RE.finditer(math_match.group(0)):
            raw = number_match.group(0).strip()
            numeric = raw.rstrip("%").strip().replace(",", "")
            if "." not in numeric and "e" not in numeric.casefold() and not raw.endswith("%"):
                continue
            try:
                value = float(numeric)
            except ValueError:
                continue
            values.append(value / 100.0 if raw.endswith("%") else value)
    return tuple(values)


def _parse_number_words(text: str) -> float | None:
    ones = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
        "eighteen": 18, "nineteen": 19,
    }
    tens = {
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    }
    total = 0
    current = 0
    tokens = [token for token in re.split(r"[ -]+", text.casefold()) if token != "and"]
    if not tokens:
        return None
    for token in tokens:
        if token in ones:
            current += ones[token]
        elif token in tens:
            current += tens[token]
        elif token == "hundred":
            current = max(current, 1) * 100
        elif token == "thousand":
            total += max(current, 1) * 1000
            current = 0
        else:
            return None
    return float(total + current)


def _section_by_id(document: ManuscriptDocument, section_id: str):
    if document.structure_issues:
        _fail("manuscript_structure_ambiguous", "strict manuscript required")
    matches = [section for section in document.sections if section.section_id == section_id]
    if len(matches) != 1:
        _fail("unknown_section_id", f"section {section_id!r} is not unique")
    return matches[0]


def _document_source(document: ManuscriptDocument) -> str:
    return merge_manuscript(document)


def _heading_identity(document: ManuscriptDocument) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (section.section_id, section.level, section.path, section.title, section.heading_source)
        for section in document.sections
    )


def _normalize_reference(kind: str, label: str) -> tuple[str, str]:
    kind_lower = kind.casefold().rstrip(".")
    normalized_kind = {
        "fig": "figure",
        "figs": "figure",
        "figure": "figure",
        "figures": "figure",
        "table": "table",
        "tables": "table",
        "eq": "equation",
        "eqs": "equation",
        "equation": "equation",
        "equations": "equation",
    }[kind_lower]
    return normalized_kind, label.casefold()


def _format_reference(reference: tuple[str, str]) -> str:
    return f"{reference[0]} {reference[1]}"


def _matches_any(value: float, candidates: Sequence[float]) -> bool:
    return any(numbers_match(value, candidate, rel_tol=NUMERIC_REL_TOL) for candidate in candidates)


def _format_number(value: float) -> str:
    return format(value, ".12g")


def _span_overlaps(span: tuple[int, int], occupied: Sequence[tuple[int, int]]) -> bool:
    return any(span[0] < end and start < span[1] for start, end in occupied)


def _validate_attempt_record(record: SectionAttemptRecord) -> None:
    if record.schema_version != SCHEMA_VERSION:
        _fail("schema_version", "section attempt schema_version must be 1")
    if not 1 <= record.attempt <= 4 or record.attempt_id != make_attempt_id(
        record.section_id, record.attempt
    ):
        _fail(
            "attempt_id_invalid",
            "attempt identity must match section and bounded ordinal 1..4",
        )
    if not record.comment_ids:
        _fail("attempt_state_invalid", "section attempt requires assigned comments")
    if len(set(record.comment_ids)) != len(record.comment_ids):
        _fail("duplicate_comment_id", "attempt comment IDs must be unique")
    if len(set(record.resolution_comment_ids)) != len(record.resolution_comment_ids):
        _fail("duplicate_comment_id", "attempt resolution IDs must be unique")
    if not set(record.resolution_comment_ids).issubset(record.comment_ids):
        _fail("unknown_comment_id", "attempt resolution IDs must be assigned comments")
    if len(set(record.validator_codes)) != len(record.validator_codes):
        _fail("duplicate_validator_code", "attempt validator codes must be unique")
    if not set(record.validator_codes).issubset(_CHECK_CODES):
        _fail("validator_code_invalid", "attempt contains an unknown validator code")
    if record.status not in {"accepted", "rejected", "transport_failed"}:
        _fail("attempt_status_invalid", "attempt status is invalid")

    if record.status == "transport_failed":
        if any(
            value is not None
            for value in (
                record.candidate_path,
                record.candidate_body_sha256,
                record.validation_report_path,
                record.validation_report_sha256,
            )
        ):
            _fail("attempt_state_invalid", "transport failure cannot own artifacts")
        if record.resolution_comment_ids or record.validator_codes:
            _fail("attempt_state_invalid", "transport failure cannot report validation")
        if record.error_type is None or record.error is None:
            _fail("attempt_state_invalid", "transport failure requires error details")
        return

    expected_candidate = (
        f"stage-19/sections/{record.section_id}.attempt-{record.attempt}.md"
    )
    expected_validation = (
        "stage-19/section_validation/"
        f"{record.section_id}.attempt-{record.attempt}.json"
    )
    if record.candidate_path != expected_candidate:
        _fail("attempt_path_invalid", "candidate path is not canonical")
    if record.validation_report_path != expected_validation:
        _fail("attempt_path_invalid", "validation report path is not canonical")
    if record.candidate_body_sha256 is None or record.validation_report_sha256 is None:
        _fail("attempt_state_invalid", "non-transport attempt requires artifact hashes")
    _validate_relative_path(record.candidate_path, "candidate_path")
    _validate_relative_path(record.validation_report_path, "validation_report_path")
    if record.status == "accepted":
        if record.validator_codes or record.error_type is not None or record.error is not None:
            _fail("attempt_state_invalid", "accepted attempt cannot contain failures")
    elif record.error is None:
        _fail("attempt_state_invalid", "rejected attempt requires an error")


def _validate_assessment_record(record: ResolutionAssessmentRecord) -> None:
    if record.schema_version != SCHEMA_VERSION:
        _fail("schema_version", "assessment schema_version must be 1")
    if record.assessment_id != f"ra-{record.comment_id}-{record.attempt_id}":
        _fail("assessment_id_invalid", "assessment identity is not canonical")
    if not _attempt_id_matches(record.attempt_id, record.section_id):
        _fail("attempt_id_invalid", "assessment attempt does not match section")
    if not record.context_isolated:
        _fail("critic_isolation_invalid", "assessment context must be isolated")
    if record.verdict not in {"resolved", "unresolved"}:
        _fail("assessment_verdict_invalid", "assessment verdict is invalid")


def _parse_strict_jsonl(text: str, context: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(text, str):
        _fail("invalid_type", f"{context} must be text")
    if not text:
        return ()
    records: list[dict[str, Any]] = []
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            _fail("jsonl_blank_line", f"{context} line {line_no} is blank")
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_json_keys)
        except (json.JSONDecodeError, ValueError) as exc:
            _fail("jsonl_invalid", f"{context} line {line_no} is invalid: {exc}")
        if not isinstance(value, dict):
            _fail("invalid_type", f"{context} line {line_no} must be an object")
        records.append(value)
    return tuple(records)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_object(value: object, *, expected: set[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail("invalid_type", f"{context} must be an object")
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        _fail("missing_fields", f"{context} missing fields: {', '.join(missing)}")
    if unknown:
        _fail("unknown_fields", f"{context} has unknown fields: {', '.join(unknown)}")
    return value


def _required_str(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("invalid_type", f"{field} must be a nonempty string")
    return value


def _required_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("invalid_type", f"{field} must be an integer")
    return value


def _required_number(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        _fail("invalid_type", f"{field} must be a finite number")
    return float(value)


def _required_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        _fail("invalid_type", f"{field} must be a boolean")
    return value


def _required_hash(value: object, field: str) -> str:
    text = _required_str(value, field)
    if not _SHA256_RE.fullmatch(text):
        _fail("hash_invalid", f"{field} must be lowercase SHA-256")
    return text


def _optional_hash(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _required_hash(value, field)


def _optional_str(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _required_str(value, field)


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        _fail("invalid_type", f"{field} must be a list")
    return tuple(_required_str(item, field) for item in value)


def _validate_relative_path(value: str, field: str) -> None:
    value = _required_str(value, field)
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or "\\" in value
        or str(path) != value
    ):
        _fail("artifact_path_invalid", f"{field} must be a safe relative path")


def _fail(code: str, message: str) -> None:
    raise SectionalRevisionContractError((ContractIssue(code, message),))
