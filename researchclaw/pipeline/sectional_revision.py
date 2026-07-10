"""Deterministic contracts for sectional manuscript revision.

This B0 module owns review extraction, ledger validation, and revision-plan
closure. It intentionally has no LLM, executor, or release-check integration.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping

from markdown_it import MarkdownIt

from researchclaw.pipeline.manuscript_sections import (
    ManuscriptDocument,
    split_commonmark_lines_keepends,
)


SCHEMA_VERSION = 1
EXTRACTOR_VERSION = 1
PLANNER_VERSION = 1
SECTION_MODEL_VERSION = 1

_COMMENT_CATEGORIES = frozenset({"actionable_revision", "general_comment"})
_WORKING_STATUSES = frozenset({"unassigned", "assigned"})
_FINAL_STATUSES = frozenset(
    {"resolved", "unresolved", "not_actionable_with_reason"}
)
_PLAN_DISPOSITIONS = frozenset(
    {"assigned", "unresolved", "not_actionable_with_reason"}
)
_CONTEXT_SUBSECTIONS = frozenset({"strengths", "weaknesses"})
_ACTIONABLE_SUBSECTION = "actionable revisions"
_GENERAL_SUBSECTION = "general comments"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TRAILING_QUALIFIER_RE = re.compile(r"\s*\([^()]*\)\s*$")
_THEMATIC_BREAK_RE = re.compile(
    r"^(?:(?:\*\s*){3,}|(?:-\s*){3,}|(?:_\s*){3,})$"
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ContractIssue:
    """One fail-closed contract or review-structure violation."""

    code: str
    message: str
    line: int | None = None


class SectionalRevisionContractError(ValueError):
    """Raised when a sectional revision artifact violates its contract."""

    def __init__(self, issues: Iterable[ContractIssue]):
        self.issues = tuple(issues)
        detail = "; ".join(issue.message for issue in self.issues)
        super().__init__(detail or "invalid sectional revision contract")


@dataclass(frozen=True)
class ReviewComment:
    comment_id: str
    reviewer: str
    category: str
    exact_text: str
    source_line_start: int
    source_line_end: int
    source_text_sha256: str
    required: bool
    required_source: str
    working_status: str
    target_section_ids: tuple[str, ...]
    final_status: str | None
    resolution_reason: str | None
    attempt_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "comment_id": self.comment_id,
            "reviewer": self.reviewer,
            "category": self.category,
            "exact_text": self.exact_text,
            "source_line_start": self.source_line_start,
            "source_line_end": self.source_line_end,
            "source_text_sha256": self.source_text_sha256,
            "required": self.required,
            "required_source": self.required_source,
            "working_status": self.working_status,
            "target_section_ids": list(self.target_section_ids),
            "final_status": self.final_status,
            "resolution_reason": self.resolution_reason,
            "attempt_ids": list(self.attempt_ids),
        }

    @classmethod
    def from_dict(cls, value: object) -> ReviewComment:
        data = _strict_object(
            value,
            expected={
                "comment_id",
                "reviewer",
                "category",
                "exact_text",
                "source_line_start",
                "source_line_end",
                "source_text_sha256",
                "required",
                "required_source",
                "working_status",
                "target_section_ids",
                "final_status",
                "resolution_reason",
                "attempt_ids",
            },
            context="review comment",
        )
        return cls(
            comment_id=_required_str(data["comment_id"], "comment_id"),
            reviewer=_required_str(data["reviewer"], "reviewer"),
            category=_required_str(data["category"], "category"),
            exact_text=_required_str(data["exact_text"], "exact_text"),
            source_line_start=_required_int(
                data["source_line_start"], "source_line_start"
            ),
            source_line_end=_required_int(data["source_line_end"], "source_line_end"),
            source_text_sha256=_required_str(
                data["source_text_sha256"], "source_text_sha256"
            ),
            required=_required_bool(data["required"], "required"),
            required_source=_required_str(
                data["required_source"], "required_source"
            ),
            working_status=_required_str(
                data["working_status"], "working_status"
            ),
            target_section_ids=_string_tuple(
                data["target_section_ids"], "target_section_ids"
            ),
            final_status=_optional_str(data["final_status"], "final_status"),
            resolution_reason=_optional_str(
                data["resolution_reason"], "resolution_reason"
            ),
            attempt_ids=_string_tuple(data["attempt_ids"], "attempt_ids"),
        )


@dataclass(frozen=True)
class ReviewLedger:
    schema_version: int
    extractor_version: int
    source_reviews_path: str
    source_reviews_sha256: str
    comments: tuple[ReviewComment, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "extractor_version": self.extractor_version,
            "source_reviews_path": self.source_reviews_path,
            "source_reviews_sha256": self.source_reviews_sha256,
            "comments": [comment.to_dict() for comment in self.comments],
        }

    @classmethod
    def from_dict(cls, value: object) -> ReviewLedger:
        data = _strict_object(
            value,
            expected={
                "schema_version",
                "extractor_version",
                "source_reviews_path",
                "source_reviews_sha256",
                "comments",
            },
            context="review ledger",
        )
        comments_raw = data["comments"]
        if not isinstance(comments_raw, list):
            _raise("invalid_type", "comments must be a list")
        ledger = cls(
            schema_version=_required_int(data["schema_version"], "schema_version"),
            extractor_version=_required_int(
                data["extractor_version"], "extractor_version"
            ),
            source_reviews_path=_required_str(
                data["source_reviews_path"], "source_reviews_path"
            ),
            source_reviews_sha256=_required_str(
                data["source_reviews_sha256"], "source_reviews_sha256"
            ),
            comments=tuple(ReviewComment.from_dict(item) for item in comments_raw),
        )
        validate_review_ledger(ledger)
        return ledger


@dataclass(frozen=True)
class RevisionAssignment:
    comment_id: str
    target_section_ids: tuple[str, ...]
    disposition: str
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "comment_id": self.comment_id,
            "target_section_ids": list(self.target_section_ids),
            "disposition": self.disposition,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: object) -> RevisionAssignment:
        data = _strict_object(
            value,
            expected={"comment_id", "target_section_ids", "disposition", "reason"},
            context="revision assignment",
        )
        return cls(
            comment_id=_required_str(data["comment_id"], "comment_id"),
            target_section_ids=_string_tuple(
                data["target_section_ids"], "target_section_ids"
            ),
            disposition=_required_str(data["disposition"], "disposition"),
            reason=_optional_str(data["reason"], "reason"),
        )


@dataclass(frozen=True)
class RevisionPlan:
    schema_version: int
    planner_version: int
    source_paper_sha256: str
    source_reviews_sha256: str
    section_model_version: int
    assignments: tuple[RevisionAssignment, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "planner_version": self.planner_version,
            "source_paper_sha256": self.source_paper_sha256,
            "source_reviews_sha256": self.source_reviews_sha256,
            "section_model_version": self.section_model_version,
            "assignments": [assignment.to_dict() for assignment in self.assignments],
        }

    @classmethod
    def from_dict(cls, value: object) -> RevisionPlan:
        data = _strict_object(
            value,
            expected={
                "schema_version",
                "planner_version",
                "source_paper_sha256",
                "source_reviews_sha256",
                "section_model_version",
                "assignments",
            },
            context="revision plan",
        )
        assignments_raw = data["assignments"]
        if not isinstance(assignments_raw, list):
            _raise("invalid_type", "assignments must be a list")
        return cls(
            schema_version=_required_int(data["schema_version"], "schema_version"),
            planner_version=_required_int(data["planner_version"], "planner_version"),
            source_paper_sha256=_required_str(
                data["source_paper_sha256"], "source_paper_sha256"
            ),
            source_reviews_sha256=_required_str(
                data["source_reviews_sha256"], "source_reviews_sha256"
            ),
            section_model_version=_required_int(
                data["section_model_version"], "section_model_version"
            ),
            assignments=tuple(
                RevisionAssignment.from_dict(item) for item in assignments_raw
            ),
        )


@dataclass(frozen=True)
class _HeadingSpan:
    title: str
    level: int
    start_line: int
    heading_end_line: int
    body_end_line: int


def extract_review_ledger(reviews: str, *, source_path: str) -> ReviewLedger:
    """Extract a complete, fail-closed ledger from the Stage 18 review format."""

    if not isinstance(reviews, str):
        raise TypeError("reviews must be a string")
    _validate_relative_artifact_path(source_path, "source_path")

    lines = split_commonmark_lines_keepends(reviews)
    tokens = MarkdownIt("commonmark").parse(reviews)
    headings = _review_headings(tokens, len(lines))
    heading_lines = {
        line_no
        for heading in headings
        for line_no in range(heading.start_line, heading.heading_end_line)
    }
    list_ranges = _outer_list_item_ranges(tokens)
    comments_raw: list[tuple[str, str, str, int, int]] = []
    issues: list[ContractIssue] = []
    accepted_item_lines: set[int] = set()

    for heading in headings:
        normalized = _normalize_subsection_name(heading.title)
        recognized = (
            heading.level == 2
            and normalized.startswith("reviewer ")
        ) or (
            heading.level == 3
            and normalized
            in (_CONTEXT_SUBSECTIONS | {_ACTIONABLE_SUBSECTION, _GENERAL_SUBSECTION})
        )
        if not recognized and _range_has_meaningful_content(
            lines, heading.heading_end_line, heading.body_end_line
        ):
            issues.append(
                ContractIssue(
                    code="unknown_review_subsection",
                    message=f"unknown content-bearing subsection {heading.title!r}",
                    line=heading.start_line + 1,
                )
            )

    for heading in headings:
        normalized = _normalize_subsection_name(heading.title)
        if heading.level != 3 or normalized not in {
            _ACTIONABLE_SUBSECTION,
            _GENERAL_SUBSECTION,
        }:
            continue
        reviewer = "all"
        category = "general_comment"
        if normalized == _ACTIONABLE_SUBSECTION:
            reviewer_heading = _parent_reviewer_heading(headings, heading)
            if reviewer_heading is None:
                issues.append(
                    ContractIssue(
                        code="reviewer_heading_missing",
                        message="Actionable Revisions has no Reviewer parent",
                        line=heading.start_line + 1,
                    )
                )
                continue
            reviewer = reviewer_heading.title
            category = "actionable_revision"

        item_ranges = [
            item
            for item in list_ranges
            if item[0] >= heading.heading_end_line
            and item[1] <= heading.body_end_line
        ]
        if not item_ranges and _range_has_meaningful_content(
            lines, heading.heading_end_line, heading.body_end_line
        ):
            issues.append(
                ContractIssue(
                    code="unparsed_review_content",
                    message=f"{heading.title!r} contains no parseable list items",
                    line=heading.start_line + 1,
                )
            )
        for start, end in item_ranges:
            exact_text = "".join(lines[start:end])
            comments_raw.append((reviewer, category, exact_text, start + 1, end))
            accepted_item_lines.update(range(start, end))

    issue_keys: set[tuple[str, int | None]] = {
        (issue.code, issue.line) for issue in issues
    }
    for line_no, line in enumerate(lines):
        if line_no in heading_lines or line_no in accepted_item_lines:
            continue
        if not _is_meaningful_review_line(line):
            continue
        owner = _deepest_heading_owner(headings, line_no)
        if owner is None or owner.level == 1:
            _append_unique_issue(
                issues,
                issue_keys,
                ContractIssue(
                    code="review_content_outside_subsection",
                    message="review content appears outside a recognized subsection",
                    line=line_no + 1,
                ),
            )
            continue
        normalized = _normalize_subsection_name(owner.title)
        if owner.level == 2:
            _append_unique_issue(
                issues,
                issue_keys,
                ContractIssue(
                    code="review_content_outside_subsection",
                    message=f"content under {owner.title!r} is not in a subsection",
                    line=line_no + 1,
                ),
            )
        elif owner.level != 3 or normalized not in (
            _CONTEXT_SUBSECTIONS
            | {_ACTIONABLE_SUBSECTION, _GENERAL_SUBSECTION}
        ):
            _append_unique_issue(
                issues,
                issue_keys,
                ContractIssue(
                    code="unknown_review_subsection",
                    message=f"unknown content-bearing subsection {owner.title!r}",
                    line=owner.start_line + 1,
                ),
            )
        elif normalized in {_ACTIONABLE_SUBSECTION, _GENERAL_SUBSECTION}:
            _append_unique_issue(
                issues,
                issue_keys,
                ContractIssue(
                    code="unparsed_review_content",
                    message=f"content in {owner.title!r} is outside a list item",
                    line=line_no + 1,
                ),
            )

    if reviews.strip() and not comments_raw:
        _append_unique_issue(
            issues,
            issue_keys,
            ContractIssue(
                code="review_comments_empty",
                message="nonempty reviews produced zero review comments",
            ),
        )
    if issues:
        raise SectionalRevisionContractError(issues)

    comments: list[ReviewComment] = []
    for ordinal, (reviewer, category, exact_text, line_start, line_end) in enumerate(
        comments_raw, start=1
    ):
        required = category == "actionable_revision"
        required_source = (
            "policy-v1:actionable_revision"
            if required
            else "policy-v1:general_comment"
        )
        comments.append(
            ReviewComment(
                comment_id=_comment_id(
                    ordinal,
                    reviewer=reviewer,
                    category=category,
                    exact_text=exact_text,
                    line_start=line_start,
                    line_end=line_end,
                ),
                reviewer=reviewer,
                category=category,
                exact_text=exact_text,
                source_line_start=line_start,
                source_line_end=line_end,
                source_text_sha256=_sha256(exact_text),
                required=required,
                required_source=required_source,
                working_status="unassigned",
                target_section_ids=(),
                final_status=None,
                resolution_reason=None,
                attempt_ids=(),
            )
        )

    ledger = ReviewLedger(
        schema_version=SCHEMA_VERSION,
        extractor_version=EXTRACTOR_VERSION,
        source_reviews_path=source_path,
        source_reviews_sha256=_sha256(reviews),
        comments=tuple(comments),
    )
    validate_review_ledger(ledger)
    return ledger


def validate_review_ledger(
    ledger: ReviewLedger,
    *,
    reviews: str | None = None,
    source_path: str | None = None,
    require_final: bool = False,
) -> ReviewLedger:
    """Validate ledger schema invariants and optionally bind it to source text."""

    issues: list[ContractIssue] = []
    if ledger.schema_version != SCHEMA_VERSION:
        issues.append(ContractIssue("schema_version", "schema_version must be 1"))
    if ledger.extractor_version != EXTRACTOR_VERSION:
        issues.append(
            ContractIssue("extractor_version", "extractor_version must be 1")
        )
    try:
        _validate_relative_artifact_path(
            ledger.source_reviews_path, "source_reviews_path"
        )
    except SectionalRevisionContractError as exc:
        issues.extend(exc.issues)
    if not _SHA256_RE.fullmatch(ledger.source_reviews_sha256):
        issues.append(
            ContractIssue(
                "source_reviews_hash_invalid",
                "source_reviews_sha256 must be a lowercase SHA-256",
            )
        )

    seen_ids: set[str] = set()
    for ordinal, comment in enumerate(ledger.comments, start=1):
        prefix = f"comment {ordinal}"
        if comment.comment_id in seen_ids:
            issues.append(
                ContractIssue("duplicate_comment_id", f"duplicate {comment.comment_id}")
            )
        seen_ids.add(comment.comment_id)
        expected_id = _comment_id(
            ordinal,
            reviewer=comment.reviewer,
            category=comment.category,
            exact_text=comment.exact_text,
            line_start=comment.source_line_start,
            line_end=comment.source_line_end,
        )
        if comment.comment_id != expected_id:
            issues.append(
                ContractIssue(
                    "comment_id_mismatch", f"{prefix} has a non-canonical comment_id"
                )
            )
        if comment.category not in _COMMENT_CATEGORIES:
            issues.append(
                ContractIssue("comment_category_invalid", f"{prefix} category is invalid")
            )
        if not comment.reviewer.strip() or not comment.exact_text:
            issues.append(
                ContractIssue("comment_text_invalid", f"{prefix} text/reviewer is empty")
            )
        if (
            comment.source_line_start <= 0
            or comment.source_line_end < comment.source_line_start
        ):
            issues.append(
                ContractIssue("comment_span_invalid", f"{prefix} source span is invalid")
            )
        if comment.source_text_sha256 != _sha256(comment.exact_text):
            issues.append(
                ContractIssue("comment_hash_mismatch", f"{prefix} text hash mismatches")
            )
        expected_required = comment.category == "actionable_revision"
        expected_required_source = (
            "policy-v1:actionable_revision"
            if expected_required
            else "policy-v1:general_comment"
        )
        if (
            comment.required is not expected_required
            or comment.required_source != expected_required_source
        ):
            issues.append(
                ContractIssue(
                    "required_policy_mismatch", f"{prefix} required policy mismatches"
                )
            )
        if comment.working_status not in _WORKING_STATUSES:
            issues.append(
                ContractIssue("working_status_invalid", f"{prefix} status is invalid")
            )
        if len(set(comment.target_section_ids)) != len(comment.target_section_ids):
            issues.append(
                ContractIssue("duplicate_section_id", f"{prefix} has duplicate targets")
            )
        if comment.working_status == "unassigned" and comment.target_section_ids:
            issues.append(
                ContractIssue("unassigned_has_targets", f"{prefix} has targets")
            )
        if comment.working_status == "assigned" and not comment.target_section_ids:
            issues.append(
                ContractIssue("assigned_without_targets", f"{prefix} has no targets")
            )
        if comment.final_status is not None and comment.final_status not in _FINAL_STATUSES:
            issues.append(
                ContractIssue("final_status_invalid", f"{prefix} final status is invalid")
            )
        if comment.final_status is None and comment.resolution_reason is not None:
            issues.append(
                ContractIssue("premature_resolution_reason", f"{prefix} has a reason")
            )
        if comment.final_status is not None and not (comment.resolution_reason or "").strip():
            issues.append(
                ContractIssue("resolution_reason_missing", f"{prefix} reason is missing")
            )
        if len(set(comment.attempt_ids)) != len(comment.attempt_ids):
            issues.append(
                ContractIssue("duplicate_attempt_id", f"{prefix} has duplicate attempts")
            )
        if require_final and comment.final_status is None:
            issues.append(
                ContractIssue("ledger_not_closed", f"{prefix} has no final status")
            )

    if reviews is not None:
        if _sha256(reviews) != ledger.source_reviews_sha256:
            issues.append(
                ContractIssue(
                    "source_reviews_hash_mismatch", "ledger does not match reviews text"
                )
            )
        lines = split_commonmark_lines_keepends(reviews)
        for ordinal, comment in enumerate(ledger.comments, start=1):
            if comment.source_line_end > len(lines):
                issues.append(
                    ContractIssue(
                        "comment_span_outside_source",
                        f"comment {ordinal} span is outside reviews text",
                    )
                )
                continue
            actual = "".join(
                lines[comment.source_line_start - 1 : comment.source_line_end]
            )
            if actual != comment.exact_text:
                issues.append(
                    ContractIssue(
                        "comment_source_mismatch",
                        f"comment {ordinal} does not match its source span",
                    )
                )
        try:
            fresh = extract_review_ledger(
                reviews, source_path=ledger.source_reviews_path
            )
        except SectionalRevisionContractError as exc:
            issues.extend(exc.issues)
        else:
            immutable = tuple(_comment_source_identity(item) for item in ledger.comments)
            expected = tuple(_comment_source_identity(item) for item in fresh.comments)
            if immutable != expected:
                issues.append(
                    ContractIssue(
                        "ledger_source_closure_mismatch",
                        "ledger comments do not exactly match deterministic extraction",
                    )
                )
        if reviews.strip() and not ledger.comments:
            issues.append(
                ContractIssue(
                    "review_comments_empty",
                    "nonempty reviews cannot have an empty ledger",
                )
            )
    if source_path is not None and ledger.source_reviews_path != source_path:
        issues.append(
            ContractIssue(
                "source_reviews_path_mismatch", "ledger source path mismatches"
            )
        )
    if issues:
        raise SectionalRevisionContractError(issues)
    return ledger


def validate_revision_plan(
    plan: object,
    ledger: ReviewLedger,
    document: ManuscriptDocument,
    *,
    reviews: str,
) -> RevisionPlan:
    """Validate strict plan shape and complete comment/section closure."""

    parsed = (
        RevisionPlan.from_dict(plan.to_dict())
        if isinstance(plan, RevisionPlan)
        else RevisionPlan.from_dict(plan)
    )
    issues: list[ContractIssue] = []
    try:
        validate_review_ledger(ledger, reviews=reviews)
    except SectionalRevisionContractError as exc:
        issues.extend(exc.issues)
    if document.structure_issues:
        issues.append(
            ContractIssue(
                "manuscript_structure_ambiguous",
                "revision plan requires a strict manuscript document",
            )
        )
    if parsed.schema_version != SCHEMA_VERSION:
        issues.append(ContractIssue("schema_version", "schema_version must be 1"))
    if parsed.planner_version != PLANNER_VERSION:
        issues.append(ContractIssue("planner_version", "planner_version must be 1"))
    if parsed.section_model_version != SECTION_MODEL_VERSION:
        issues.append(
            ContractIssue("section_model_version", "section_model_version must be 1")
        )
    for field_name, value in (
        ("source_paper_sha256", parsed.source_paper_sha256),
        ("source_reviews_sha256", parsed.source_reviews_sha256),
    ):
        if not _SHA256_RE.fullmatch(value):
            issues.append(
                ContractIssue("source_hash_invalid", f"{field_name} is not SHA-256")
            )
    if parsed.source_paper_sha256 != document.source_sha256:
        issues.append(
            ContractIssue("source_paper_hash_mismatch", "plan paper hash mismatches")
        )
    if parsed.source_reviews_sha256 != ledger.source_reviews_sha256:
        issues.append(
            ContractIssue("source_reviews_hash_mismatch", "plan review hash mismatches")
        )

    ledger_ids = {comment.comment_id for comment in ledger.comments}
    known_sections = {section.section_id for section in document.sections}
    if len(known_sections) != len(document.sections):
        issues.append(
            ContractIssue(
                "duplicate_document_section_id",
                "manuscript contains duplicate section IDs",
            )
        )
    seen_comments: set[str] = set()
    for assignment in parsed.assignments:
        if assignment.comment_id in seen_comments:
            issues.append(
                ContractIssue(
                    "duplicate_plan_comment",
                    f"duplicate assignment for {assignment.comment_id}",
                )
            )
        seen_comments.add(assignment.comment_id)
        if assignment.comment_id not in ledger_ids:
            issues.append(
                ContractIssue(
                    "unknown_plan_comment",
                    f"unknown comment {assignment.comment_id}",
                )
            )
        if assignment.disposition not in _PLAN_DISPOSITIONS:
            issues.append(
                ContractIssue(
                    "plan_disposition_invalid",
                    f"invalid disposition for {assignment.comment_id}",
                )
            )
            continue
        targets = assignment.target_section_ids
        if len(set(targets)) != len(targets):
            issues.append(
                ContractIssue(
                    "duplicate_plan_target",
                    f"duplicate section target for {assignment.comment_id}",
                )
            )
        if assignment.disposition == "assigned":
            if not targets:
                issues.append(
                    ContractIssue(
                        "assigned_without_targets",
                        f"assigned comment {assignment.comment_id} has no target",
                    )
                )
            if assignment.reason is not None:
                issues.append(
                    ContractIssue(
                        "assigned_has_reason",
                        f"assigned comment {assignment.comment_id} must have null reason",
                    )
                )
            for target in targets:
                if target == "global":
                    issues.append(
                        ContractIssue(
                            "global_target_not_executable",
                            f"comment {assignment.comment_id} still targets global",
                        )
                    )
                elif target not in known_sections:
                    issues.append(
                        ContractIssue(
                            "unknown_section_id",
                            f"unknown section {target}",
                        )
                    )
        else:
            if targets:
                issues.append(
                    ContractIssue(
                        "terminal_disposition_has_targets",
                        f"{assignment.comment_id} has non-assigned targets",
                    )
                )
            if not (assignment.reason or "").strip():
                issues.append(
                    ContractIssue(
                        "plan_reason_missing",
                        f"{assignment.comment_id} requires a reason",
                    )
                )

    missing = sorted(ledger_ids - seen_comments)
    if missing:
        issues.append(
            ContractIssue(
                "plan_comments_missing",
                "plan omits comments: " + ", ".join(missing),
            )
        )
    if issues:
        raise SectionalRevisionContractError(issues)
    return parsed


def make_attempt_id(section_id: str, attempt: int) -> str:
    """Build an unambiguous attempt ID from the complete section ID."""

    section_id = _required_str(section_id, "section_id")
    attempt = _required_int(attempt, "attempt")
    if attempt <= 0:
        _raise("attempt_invalid", "attempt must be positive")
    return f"sec-{section_id}-a{attempt}"


def _review_headings(tokens: list[Any], line_count: int) -> tuple[_HeadingSpan, ...]:
    provisional: list[tuple[str, int, int, int]] = []
    for index, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        if token.map is None or len(token.map) != 2:
            _raise("heading_map_missing", "review heading has no source map")
        try:
            level = int(token.tag.removeprefix("h"))
        except ValueError:
            _raise("heading_level_invalid", f"invalid heading tag {token.tag!r}")
        if index + 1 >= len(tokens) or tokens[index + 1].type != "inline":
            _raise("heading_inline_missing", "review heading has no inline token")
        title = str(tokens[index + 1].content or "").strip()
        provisional.append((title, level, int(token.map[0]), int(token.map[1])))

    headings: list[_HeadingSpan] = []
    for index, (title, level, start, heading_end) in enumerate(provisional):
        body_end = line_count
        for _, next_level, next_start, _ in provisional[index + 1 :]:
            if next_level <= level:
                body_end = next_start
                break
        headings.append(
            _HeadingSpan(
                title=title,
                level=level,
                start_line=start,
                heading_end_line=heading_end,
                body_end_line=body_end,
            )
        )
    return tuple(headings)


def _outer_list_item_ranges(tokens: list[Any]) -> tuple[tuple[int, int], ...]:
    candidates = [
        (int(token.map[0]), int(token.map[1]))
        for token in tokens
        if token.type == "list_item_open"
        and token.map is not None
        and len(token.map) == 2
    ]
    selected: list[tuple[int, int]] = []
    for candidate in candidates:
        if any(
            parent_start <= candidate[0] and candidate[1] <= parent_end
            for parent_start, parent_end in selected
        ):
            continue
        selected.append(candidate)
    return tuple(selected)


def _parent_reviewer_heading(
    headings: tuple[_HeadingSpan, ...], child: _HeadingSpan
) -> _HeadingSpan | None:
    parents = [
        heading
        for heading in headings
        if heading.level == 2
        and heading.start_line < child.start_line < heading.body_end_line
        and _normalize_subsection_name(heading.title).startswith("reviewer ")
    ]
    return parents[-1] if parents else None


def _deepest_heading_owner(
    headings: tuple[_HeadingSpan, ...], line_no: int
) -> _HeadingSpan | None:
    owners = [
        heading
        for heading in headings
        if heading.heading_end_line <= line_no < heading.body_end_line
    ]
    if not owners:
        return None
    return max(owners, key=lambda heading: (heading.level, heading.start_line))


def _normalize_subsection_name(title: str) -> str:
    value = unicodedata.normalize("NFKC", title)
    value = " ".join(value.split())
    value = _TRAILING_QUALIFIER_RE.sub("", value)
    return value.casefold().strip()


def _range_has_meaningful_content(
    lines: list[str], start: int, end: int
) -> bool:
    return any(_is_meaningful_review_line(line) for line in lines[start:end])


def _is_meaningful_review_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped and not _THEMATIC_BREAK_RE.fullmatch(stripped))


def _append_unique_issue(
    issues: list[ContractIssue],
    keys: set[tuple[str, int | None]],
    issue: ContractIssue,
) -> None:
    key = (issue.code, issue.line)
    if key not in keys:
        issues.append(issue)
        keys.add(key)


def _comment_id(
    ordinal: int,
    *,
    reviewer: str,
    category: str,
    exact_text: str,
    line_start: int,
    line_end: int,
) -> str:
    material = "\0".join(
        (reviewer, category, exact_text, str(line_start), str(line_end))
    )
    return f"rc-{ordinal:03d}-{_sha256(material)[:12]}"


def _comment_source_identity(comment: ReviewComment) -> tuple[Any, ...]:
    return (
        comment.comment_id,
        comment.reviewer,
        comment.category,
        comment.exact_text,
        comment.source_line_start,
        comment.source_line_end,
        comment.source_text_sha256,
        comment.required,
        comment.required_source,
    )


def _strict_object(
    value: object, *, expected: set[str], context: str
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _raise("invalid_type", f"{context} must be an object")
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        _raise("missing_fields", f"{context} missing fields: {', '.join(missing)}")
    if unknown:
        _raise("unknown_fields", f"{context} has unknown fields: {', '.join(unknown)}")
    return value


def _required_str(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _raise("invalid_type", f"{field} must be a nonempty string")
    return value


def _optional_str(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        _raise("invalid_type", f"{field} must be null or a nonempty string")
    return value


def _required_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _raise("invalid_type", f"{field} must be an integer")
    return value


def _required_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        _raise("invalid_type", f"{field} must be a boolean")
    return value


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        _raise("invalid_type", f"{field} must be a list")
    return tuple(_required_str(item, field) for item in value)


def _validate_relative_artifact_path(value: str, field: str) -> None:
    value = _required_str(value, field)
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or "\\" in value
        or str(path) != value
    ):
        _raise("artifact_path_invalid", f"{field} must be a safe relative path")


def _raise(code: str, message: str) -> None:
    raise SectionalRevisionContractError((ContractIssue(code, message),))
