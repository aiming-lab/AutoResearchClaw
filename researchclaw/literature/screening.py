"""Strict Stage 5 screening response and report contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


SCREENING_SCHEMA_VERSION = 1
SCREENING_POLICY_VERSION = 2
SCREEN_BATCH_SIZE = 8
MAX_SCREEN_REASON_CHARS = 160
MAX_SCREEN_CANDIDATES = 150
MIN_RELEVANCE_SCORE = 0.5


class ScreeningContractError(ValueError):
    """Raised when a screening response or report violates its contract."""


@dataclass(frozen=True)
class ScreeningDecision:
    source_identity: str
    decision: str
    relevance_score: float
    quality_score: float
    reason: str

    @property
    def keep(self) -> bool:
        return self.decision == "keep"


def parse_screening_response(
    text: str,
    *,
    expected_batch_id: str,
    expected_source_ids: Iterable[str],
    minimum_relevance_score: float = MIN_RELEVANCE_SCORE,
    minimum_quality_score: float = 0.0,
) -> tuple[ScreeningDecision, ...]:
    """Parse one complete batch response with exact candidate-ID closure."""
    payload = _parse_json_object(text, "screening response")
    _require_exact_keys(
        payload,
        {"schema_version", "batch_id", "decisions"},
        "screening response",
    )
    if payload["schema_version"] != SCREENING_SCHEMA_VERSION:
        raise ScreeningContractError("unsupported screening response schema_version")
    if payload["batch_id"] != expected_batch_id:
        raise ScreeningContractError("screening response batch_id mismatch")
    raw_decisions = payload["decisions"]
    if not isinstance(raw_decisions, list):
        raise ScreeningContractError("screening response decisions must be a list")

    decisions: list[ScreeningDecision] = []
    seen: set[str] = set()
    for raw in raw_decisions:
        if not isinstance(raw, dict):
            raise ScreeningContractError("screening decision must be an object")
        _require_exact_keys(
            raw,
            {
                "source_identity",
                "decision",
                "relevance_score",
                "quality_score",
                "reason",
            },
            "screening decision",
        )
        source_identity = _required_string(raw, "source_identity")
        if source_identity in seen:
            raise ScreeningContractError(
                f"duplicate screening decision for {source_identity}"
            )
        seen.add(source_identity)
        decision = raw["decision"]
        if decision not in {"keep", "reject"}:
            raise ScreeningContractError("decision must be keep or reject")
        relevance_score = _score(raw, "relevance_score")
        quality_score = _score(raw, "quality_score")
        qualifies = (
            relevance_score >= minimum_relevance_score
            and quality_score >= minimum_quality_score
        )
        if (decision == "keep") != qualifies:
            raise ScreeningContractError(
                "screening decision contradicts configured score thresholds"
            )
        reason = _required_string(raw, "reason")
        if len(reason) > MAX_SCREEN_REASON_CHARS:
            raise ScreeningContractError(
                f"reason exceeds {MAX_SCREEN_REASON_CHARS} Unicode code points"
            )
        decisions.append(
            ScreeningDecision(
                source_identity=source_identity,
                decision=decision,
                relevance_score=relevance_score,
                quality_score=quality_score,
                reason=reason,
            )
        )

    expected = tuple(expected_source_ids)
    if len(expected) != len(set(expected)):
        raise ScreeningContractError("expected source identities are not unique")
    expected_set = set(expected)
    if seen != expected_set:
        missing = sorted(expected_set - seen)
        extra = sorted(seen - expected_set)
        raise ScreeningContractError(
            f"screening candidate-ID closure mismatch: missing={missing}, extra={extra}"
        )
    decision_map = {decision.source_identity: decision for decision in decisions}
    return tuple(decision_map[source_id] for source_id in expected)


def parse_screening_candidates(text: str) -> tuple[dict[str, Any], ...]:
    """Load Stage 4 candidates without silently dropping malformed records."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    lines = text.split("\n")
    for line_number, line in enumerate(lines, start=1):
        if not line:
            if line_number == len(lines):
                continue
            raise ScreeningContractError(
                f"blank candidate JSONL line {line_number}"
            )
        payload = _parse_json_object(line, f"candidate JSONL line {line_number}")
        source_identity = _required_string(payload, "source_identity")
        if source_identity in seen:
            raise ScreeningContractError(
                f"duplicate candidate source_identity: {source_identity}"
            )
        seen.add(source_identity)
        if payload.get("cite_key_version") != 2:
            raise ScreeningContractError("candidate cite_key_version must equal 2")
        _required_string(payload, "cite_key")
        _required_string(payload, "title")
        _required_string(payload, "paper_id")
        _required_string(payload, "source")
        for field in ("abstract", "venue", "doi", "arxiv_id", "url"):
            if not isinstance(payload.get(field), str):
                raise ScreeningContractError(
                    f"candidate {field} must be a string"
                )
        for field in ("year", "citation_count"):
            value = payload.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ScreeningContractError(
                    f"candidate {field} must be a nonnegative integer"
                )
        authors = payload.get("authors")
        if not isinstance(authors, list):
            raise ScreeningContractError("candidate authors must be a list")
        for author in authors:
            if not isinstance(author, dict):
                raise ScreeningContractError("candidate author must be an object")
            _required_string(author, "name")
        rows.append(payload)
    if not rows:
        raise ScreeningContractError("candidate JSONL must not be empty")
    return tuple(rows)


def build_screening_report(
    *,
    candidates_sha256: str,
    registry_sha256: str,
    references_sha256: str,
    screening_output_path: str,
    screening_output_sha256: str,
    minimum_quality_score: float,
    claim_scope: str,
    candidate_ids: Iterable[str],
    prefilter_rejected_ids: Iterable[str],
    screened_ids: Iterable[str],
    selected_ids: Iterable[str],
    semantic_duplicate_ids: Iterable[str],
    unscreened_ids: Iterable[str],
    batch_count: int,
    failed_batches: Iterable[Mapping[str, str]],
    degraded: bool,
    degradation_codes: Iterable[str],
) -> dict[str, Any]:
    """Build and self-validate a Stage 5 screening report."""
    candidate_ids_tuple = tuple(candidate_ids)
    prefilter_ids = list(prefilter_rejected_ids)
    screened_ids_list = list(screened_ids)
    selected_ids_list = list(selected_ids)
    duplicate_ids = list(semantic_duplicate_ids)
    unscreened_ids_list = list(unscreened_ids)
    degradation_codes_list = list(degradation_codes)
    report = {
        "schema_version": SCREENING_SCHEMA_VERSION,
        "screening_policy_version": SCREENING_POLICY_VERSION,
        "candidates_path": "stage-04/candidates.jsonl",
        "candidates_sha256": candidates_sha256,
        "registry_path": "stage-04/cite_key_registry.json",
        "registry_sha256": registry_sha256,
        "references_path": "stage-04/references.bib",
        "references_sha256": references_sha256,
        "screening_output_path": screening_output_path,
        "screening_output_sha256": screening_output_sha256,
        "candidate_count": len(candidate_ids_tuple),
        "batch_size": SCREEN_BATCH_SIZE,
        "max_screen_candidates": MAX_SCREEN_CANDIDATES,
        "minimum_relevance_score": MIN_RELEVANCE_SCORE,
        "minimum_quality_score": minimum_quality_score,
        "claim_scope": claim_scope,
        "prefilter_rejected_candidate_ids": prefilter_ids,
        "screened_candidate_ids": screened_ids_list,
        "selected_candidate_ids": selected_ids_list,
        "semantic_duplicate_candidate_ids": duplicate_ids,
        "unscreened_candidate_ids": unscreened_ids_list,
        "batch_count": batch_count,
        "failed_batches": [dict(item) for item in failed_batches],
        "screening_complete": not unscreened_ids_list,
        "degraded": degraded,
        "degradation_codes": degradation_codes_list,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return parse_screening_report(
        text,
        candidates_text_sha256=candidates_sha256,
        registry_text_sha256=registry_sha256,
        references_text_sha256=references_sha256,
        expected_screening_output_path=screening_output_path,
        screening_output_text_sha256=screening_output_sha256,
        expected_minimum_quality_score=minimum_quality_score,
        expected_claim_scope=claim_scope,
        expected_candidate_ids=candidate_ids_tuple,
        expected_selected_ids=selected_ids_list,
    )


def parse_screening_report(
    text: str,
    *,
    candidates_text_sha256: str,
    registry_text_sha256: str,
    references_text_sha256: str,
    expected_screening_output_path: str,
    screening_output_text_sha256: str,
    expected_minimum_quality_score: float,
    expected_claim_scope: str,
    expected_candidate_ids: Iterable[str],
    expected_selected_ids: Iterable[str],
) -> dict[str, Any]:
    """Strictly parse and validate a complete Stage 5 screening report."""
    payload = _parse_json_object(text, "screening report")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "screening_policy_version",
            "candidates_path",
            "candidates_sha256",
            "registry_path",
            "registry_sha256",
            "references_path",
            "references_sha256",
            "screening_output_path",
            "screening_output_sha256",
            "candidate_count",
            "batch_size",
            "max_screen_candidates",
            "minimum_relevance_score",
            "minimum_quality_score",
            "claim_scope",
            "prefilter_rejected_candidate_ids",
            "screened_candidate_ids",
            "selected_candidate_ids",
            "semantic_duplicate_candidate_ids",
            "unscreened_candidate_ids",
            "batch_count",
            "failed_batches",
            "screening_complete",
            "degraded",
            "degradation_codes",
        },
        "screening report",
    )
    if payload["schema_version"] != SCREENING_SCHEMA_VERSION:
        raise ScreeningContractError("unsupported screening report schema_version")
    if payload["screening_policy_version"] != SCREENING_POLICY_VERSION:
        raise ScreeningContractError("unsupported screening_policy_version")
    if payload["candidates_path"] != "stage-04/candidates.jsonl":
        raise ScreeningContractError("noncanonical screening candidates_path")
    if payload["registry_path"] != "stage-04/cite_key_registry.json":
        raise ScreeningContractError("noncanonical screening registry_path")
    if payload["references_path"] != "stage-04/references.bib":
        raise ScreeningContractError("noncanonical screening references_path")
    if payload["screening_output_path"] != expected_screening_output_path or payload[
        "screening_output_path"
    ] not in {
        "stage-05/shortlist.jsonl",
        "stage-05/screening_partial.jsonl",
    }:
        raise ScreeningContractError("noncanonical screening_output_path")
    if payload["candidates_sha256"] != candidates_text_sha256:
        raise ScreeningContractError("screening candidates_sha256 mismatch")
    if payload["registry_sha256"] != registry_text_sha256:
        raise ScreeningContractError("screening registry_sha256 mismatch")
    if payload["references_sha256"] != references_text_sha256:
        raise ScreeningContractError("screening references_sha256 mismatch")
    if payload["screening_output_sha256"] != screening_output_text_sha256:
        raise ScreeningContractError("screening output sha256 mismatch")
    for field in (
        "candidates_sha256",
        "registry_sha256",
        "references_sha256",
        "screening_output_sha256",
    ):
        if not isinstance(payload[field], str) or not re.fullmatch(
            r"[0-9a-f]{64}", payload[field]
        ):
            raise ScreeningContractError(f"invalid {field}")

    candidate_ids = tuple(expected_candidate_ids)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ScreeningContractError("expected candidate identities are not unique")
    if payload["candidate_count"] != len(candidate_ids):
        raise ScreeningContractError("screening candidate_count mismatch")
    if payload["batch_size"] != SCREEN_BATCH_SIZE:
        raise ScreeningContractError("screening batch_size mismatch")
    if payload["max_screen_candidates"] != MAX_SCREEN_CANDIDATES:
        raise ScreeningContractError("screening max_screen_candidates mismatch")
    if _score(payload, "minimum_relevance_score") != MIN_RELEVANCE_SCORE:
        raise ScreeningContractError("screening minimum_relevance_score mismatch")
    expected_quality = _score(
        {"minimum_quality_score": expected_minimum_quality_score},
        "minimum_quality_score",
    )
    if _score(payload, "minimum_quality_score") != expected_quality:
        raise ScreeningContractError("screening minimum_quality_score mismatch")
    if payload["claim_scope"] != expected_claim_scope or payload[
        "claim_scope"
    ] not in {"pipeline_validation", "exploratory", "research_release"}:
        raise ScreeningContractError("screening claim_scope mismatch")
    if not isinstance(payload["batch_count"], int) or isinstance(
        payload["batch_count"], bool
    ) or payload["batch_count"] < 0:
        raise ScreeningContractError("invalid screening batch_count")
    if not isinstance(payload["screening_complete"], bool):
        raise ScreeningContractError("screening_complete must be boolean")
    if not isinstance(payload["degraded"], bool):
        raise ScreeningContractError("degraded must be boolean")

    list_fields = (
        "prefilter_rejected_candidate_ids",
        "screened_candidate_ids",
        "selected_candidate_ids",
        "semantic_duplicate_candidate_ids",
        "unscreened_candidate_ids",
        "degradation_codes",
    )
    parsed_lists = {
        field: _unique_string_list(payload[field], field) for field in list_fields
    }
    failed_batch_ids: set[str] = set()
    if not isinstance(payload["failed_batches"], list):
        raise ScreeningContractError("failed_batches must be a list")
    for item in payload["failed_batches"]:
        if not isinstance(item, dict):
            raise ScreeningContractError("failed batch must be an object")
        _require_exact_keys(item, {"batch_id", "error"}, "failed batch")
        batch_id = _required_string(item, "batch_id")
        _required_string(item, "error")
        match = re.fullmatch(r"screen-batch-(\d{3})", batch_id)
        if match is None or not 1 <= int(match.group(1)) <= payload["batch_count"]:
            raise ScreeningContractError("invalid failed batch_id")
        if batch_id in failed_batch_ids:
            raise ScreeningContractError("duplicate failed batch_id")
        failed_batch_ids.add(batch_id)

    all_ids = set(candidate_ids)
    prefilter_ids = set(parsed_lists["prefilter_rejected_candidate_ids"])
    screened_ids = set(parsed_lists["screened_candidate_ids"])
    selected_ids = set(parsed_lists["selected_candidate_ids"])
    duplicate_ids = set(parsed_lists["semantic_duplicate_candidate_ids"])
    unscreened_ids = set(parsed_lists["unscreened_candidate_ids"])
    expected_selected = tuple(expected_selected_ids)
    if len(expected_selected) != len(set(expected_selected)):
        raise ScreeningContractError("expected selected identities are not unique")
    if parsed_lists["selected_candidate_ids"] != list(expected_selected):
        raise ScreeningContractError("screening selected_candidate_ids mismatch")
    if (prefilter_ids | screened_ids | unscreened_ids) != all_ids:
        raise ScreeningContractError("screening report does not cover all candidates")
    if (prefilter_ids & screened_ids) or (prefilter_ids & unscreened_ids) or (
        screened_ids & unscreened_ids
    ):
        raise ScreeningContractError("screening candidate partitions overlap")
    if not selected_ids <= screened_ids:
        raise ScreeningContractError("selected candidates were not screened")
    if not duplicate_ids <= screened_ids or duplicate_ids & selected_ids:
        raise ScreeningContractError("invalid semantic duplicate partition")
    admitted_count = len(screened_ids | unscreened_ids)
    if admitted_count > MAX_SCREEN_CANDIDATES:
        raise ScreeningContractError("screening admitted candidate limit exceeded")
    expected_batch_count = (
        (admitted_count + SCREEN_BATCH_SIZE - 1) // SCREEN_BATCH_SIZE
        if admitted_count
        else 0
    )
    if payload["batch_count"] != expected_batch_count:
        raise ScreeningContractError("screening batch_count mismatch")
    if payload["screening_complete"] != (not unscreened_ids):
        raise ScreeningContractError("screening_complete contradicts unscreened IDs")
    if bool(failed_batch_ids) != bool(unscreened_ids):
        raise ScreeningContractError("failed batches contradict unscreened IDs")
    expected_degradation_codes = (
        ["screen_batch_failed", "screening_incomplete"]
        if failed_batch_ids
        else []
    )
    if parsed_lists["degradation_codes"] != expected_degradation_codes:
        raise ScreeningContractError("screening degradation_codes mismatch")
    if payload["degraded"] != bool(expected_degradation_codes):
        raise ScreeningContractError("degraded contradicts degradation_codes")
    return payload


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_quality_threshold(value: object) -> float:
    """Convert the configured 0-10 threshold to the Stage 5 0-1 scale."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScreeningContractError("quality threshold must be numeric")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 10.0:
        raise ScreeningContractError("quality threshold must be between 0 and 10")
    return threshold / 10.0


def _parse_json_object(text: str, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ScreeningContractError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ScreeningContractError(f"{label} root must be an object")
    return payload


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ScreeningContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(payload)
    if actual != expected:
        raise ScreeningContractError(
            f"{label} fields mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _required_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ScreeningContractError(f"{field} must be a nonempty string")
    return value


def _score(payload: Mapping[str, Any], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScreeningContractError(f"{field} must be numeric")
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise ScreeningContractError(f"{field} must be between 0 and 1")
    return score


def _unique_string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ScreeningContractError(f"{field} must be a list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ScreeningContractError(f"{field} contains invalid string")
        result.append(item)
    if len(result) != len(set(result)):
        raise ScreeningContractError(f"{field} contains duplicates")
    return result
