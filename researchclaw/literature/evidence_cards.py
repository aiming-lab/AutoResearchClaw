"""Strict Stage 6 evidence-card contracts and deterministic rendering."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from researchclaw.literature.citation_identity import (
    CitationIdentityError,
    parse_cite_key_registry,
    validate_registry_artifacts,
)
from researchclaw.literature.screening import (
    ScreeningContractError,
    normalize_quality_threshold,
    parse_screening_candidates,
    parse_screening_report,
    sha256_text,
)


CARD_SCHEMA_VERSION = 1
CARDS_MANIFEST_SCHEMA_VERSION = 1
CARD_RENDERER_VERSION = 1
EXTRACTION_BATCH_SIZE = 4
MIN_EXCERPT_CHARS = 25
SUMMARY_FIELDS = (
    "problem",
    "method",
    "data",
    "metrics",
    "findings",
    "limitations",
)
SHORTLIST_SCREENING_FIELDS = {
    "screening_policy_version",
    "screen_rank_score",
    "keyword_overlap",
    "relevance_score",
    "quality_score",
    "keep_reason",
}


class EvidenceCardContractError(ValueError):
    """Raised when Stage 6 evidence provenance violates its contract."""


@dataclass(frozen=True)
class CardProposal:
    source_identity: str
    summary_text: dict[str, str]
    excerpt_texts: tuple[str, ...]


@dataclass(frozen=True)
class ValidatedCardInputs:
    candidates_text: str
    shortlist_text: str
    screening_report_text: str
    candidates: tuple[dict[str, Any], ...]
    shortlist: tuple[dict[str, Any], ...]


def load_validated_card_inputs(run_dir: Path, config: Any) -> ValidatedCardInputs:
    """Load Stage 4/5 canonical inputs and replay their strict bindings."""
    paths = {
        "candidates": run_dir / "stage-04" / "candidates.jsonl",
        "registry": run_dir / "stage-04" / "cite_key_registry.json",
        "bibliography": run_dir / "stage-04" / "references.bib",
        "shortlist": run_dir / "stage-05" / "shortlist.jsonl",
        "screening_report": run_dir / "stage-05" / "screening_report.json",
    }
    texts: dict[str, str] = {}
    try:
        for label, path in paths.items():
            if path.is_symlink() or not path.is_file():
                raise EvidenceCardContractError(
                    f"canonical {label} artifact is missing or not a file"
                )
            texts[label] = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise EvidenceCardContractError(f"cannot read canonical card input: {exc}") from exc

    try:
        registry = parse_cite_key_registry(texts["registry"])
        validate_registry_artifacts(
            registry, texts["candidates"], texts["bibliography"]
        )
        candidates = parse_screening_candidates(texts["candidates"])
        shortlist = parse_screening_candidates(texts["shortlist"])
    except (CitationIdentityError, ScreeningContractError) as exc:
        raise EvidenceCardContractError(f"invalid Stage 4/5 citation input: {exc}") from exc

    candidate_map = {str(row["source_identity"]): row for row in candidates}
    selected_ids: list[str] = []
    for row in shortlist:
        source_identity = str(row["source_identity"])
        if source_identity in selected_ids:
            raise EvidenceCardContractError("duplicate shortlist source_identity")
        selected_ids.append(source_identity)
        candidate = candidate_map.get(source_identity)
        if candidate is None:
            raise EvidenceCardContractError("shortlist identity is absent from candidates")
        if set(row) != set(candidate) | SHORTLIST_SCREENING_FIELDS:
            raise EvidenceCardContractError("shortlist screening fields mismatch")
        for key, value in candidate.items():
            if row.get(key) != value:
                raise EvidenceCardContractError(
                    f"shortlist mutated canonical candidate field: {key}"
                )
        if row.get("screening_policy_version") != 1:
            raise EvidenceCardContractError("invalid shortlist screening policy")
        for field in ("screen_rank_score", "relevance_score", "quality_score"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EvidenceCardContractError(f"invalid shortlist {field}")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise EvidenceCardContractError(f"shortlist {field} is out of range")
            if field != "screen_rank_score" and float(value) > 1.0:
                raise EvidenceCardContractError(f"shortlist {field} is out of range")
        overlap = row.get("keyword_overlap")
        if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 1:
            raise EvidenceCardContractError("invalid shortlist keyword_overlap")
        _required_string(row, "keep_reason")

    try:
        report = parse_screening_report(
            texts["screening_report"],
            candidates_text_sha256=sha256_text(texts["candidates"]),
            registry_text_sha256=sha256_text(texts["registry"]),
            references_text_sha256=sha256_text(texts["bibliography"]),
            expected_screening_output_path="stage-05/shortlist.jsonl",
            screening_output_text_sha256=sha256_text(texts["shortlist"]),
            expected_minimum_quality_score=normalize_quality_threshold(
                config.research.quality_threshold
            ),
            expected_claim_scope=config.experiment.claim_scope,
            expected_candidate_ids=[str(row["source_identity"]) for row in candidates],
            expected_selected_ids=selected_ids,
        )
    except ScreeningContractError as exc:
        raise EvidenceCardContractError(f"invalid Stage 5 screening report: {exc}") from exc

    if config.experiment.claim_scope != "pipeline_validation" and (
        not report["screening_complete"]
        or report["degraded"]
        or report["failed_batches"]
    ):
        raise EvidenceCardContractError(
            "strict claim scope requires complete non-degraded screening"
        )
    return ValidatedCardInputs(
        candidates_text=texts["candidates"],
        shortlist_text=texts["shortlist"],
        screening_report_text=texts["screening_report"],
        candidates=candidates,
        shortlist=shortlist,
    )


def parse_card_batch_response(
    text: str,
    *,
    expected_batch_id: str,
    expected_source_ids: Iterable[str],
) -> tuple[CardProposal, ...]:
    """Parse one model response with exact source-identity closure."""
    payload = _parse_object(text, "card batch response")
    _exact_keys(payload, {"schema_version", "batch_id", "cards"}, "card batch")
    if payload["schema_version"] != CARD_SCHEMA_VERSION:
        raise EvidenceCardContractError("unsupported card response schema_version")
    if payload["batch_id"] != expected_batch_id:
        raise EvidenceCardContractError("card response batch_id mismatch")
    if not isinstance(payload["cards"], list):
        raise EvidenceCardContractError("card response cards must be a list")

    proposals: dict[str, CardProposal] = {}
    for raw in payload["cards"]:
        if not isinstance(raw, dict):
            raise EvidenceCardContractError("card proposal must be an object")
        _exact_keys(
            raw,
            {"source_identity", "summary_text", "evidence_excerpt_texts"},
            "card proposal",
        )
        source_identity = _required_string(raw, "source_identity")
        if source_identity in proposals:
            raise EvidenceCardContractError("duplicate card proposal identity")
        summary = raw["summary_text"]
        if not isinstance(summary, dict):
            raise EvidenceCardContractError("summary_text must be an object")
        _exact_keys(summary, set(SUMMARY_FIELDS), "summary_text")
        summary_text = {
            field: _required_string(summary, field) for field in SUMMARY_FIELDS
        }
        excerpts = raw["evidence_excerpt_texts"]
        if not isinstance(excerpts, list) or not 1 <= len(excerpts) <= 4:
            raise EvidenceCardContractError(
                "evidence_excerpt_texts must contain one to four excerpts"
            )
        excerpt_texts = tuple(
            _required_list_string(item, "evidence excerpt") for item in excerpts
        )
        if any(len(excerpt) < MIN_EXCERPT_CHARS for excerpt in excerpt_texts):
            raise EvidenceCardContractError(
                f"evidence excerpts must contain at least {MIN_EXCERPT_CHARS} characters"
            )
        if len(excerpt_texts) != len(set(excerpt_texts)):
            raise EvidenceCardContractError("duplicate evidence excerpt text")
        proposals[source_identity] = CardProposal(
            source_identity=source_identity,
            summary_text=summary_text,
            excerpt_texts=excerpt_texts,
        )

    expected = tuple(expected_source_ids)
    if len(expected) != len(set(expected)):
        raise EvidenceCardContractError("expected card identities are not unique")
    if set(proposals) != set(expected):
        raise EvidenceCardContractError(
            "card response source-identity closure mismatch"
        )
    return tuple(proposals[source_identity] for source_identity in expected)


def build_evidence_card(
    *,
    card_id: str,
    candidate: Mapping[str, Any],
    candidates_sha256: str,
    proposal: CardProposal | None,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    """Build one card; only exact retained-abstract slices become evidence."""
    source_identity = _required_string(candidate, "source_identity")
    summary = {field: "" for field in SUMMARY_FIELDS}
    status = "failed"
    fallback_reason = failure_reason or "card_extraction_failed"
    evidence: list[dict[str, Any]] = []
    if proposal is not None:
        if proposal.source_identity != source_identity:
            raise EvidenceCardContractError("proposal identity mismatch")
        summary = dict(proposal.summary_text)
        abstract = candidate.get("abstract")
        if not isinstance(abstract, str) or not abstract:
            status = "fallback"
            fallback_reason = "source_abstract_empty"
        else:
            located: list[dict[str, Any]] = []
            for excerpt_text in proposal.excerpt_texts:
                start = abstract.find(excerpt_text)
                if start < 0:
                    located = []
                    fallback_reason = "excerpt_not_in_retained_abstract"
                    break
                end = start + len(excerpt_text)
                excerpt_hash = sha256_text(excerpt_text)
                excerpt_id = "ev-" + hashlib.sha256(
                    f"{source_identity}\n{start}\n{end}\n{excerpt_hash}".encode("utf-8")
                ).hexdigest()[:16]
                located.append(
                    {
                        "excerpt_id": excerpt_id,
                        "source_type": "abstract",
                        "source_artifact_path": "stage-04/candidates.jsonl",
                        "source_artifact_sha256": candidates_sha256,
                        "source_record_id": source_identity,
                        "json_pointer": "/abstract",
                        "char_start": start,
                        "char_end": end,
                        "excerpt_text": excerpt_text,
                        "excerpt_sha256": excerpt_hash,
                    }
                )
            if located:
                status = "success"
                fallback_reason = None
                evidence = located
            else:
                status = "fallback"
    card = {
        "schema_version": CARD_SCHEMA_VERSION,
        "card_id": card_id,
        "source_identity": source_identity,
        "cite_key": _required_string(candidate, "cite_key"),
        "title": _required_string(candidate, "title"),
        "extraction_status": status,
        "fallback_reason": fallback_reason,
        "summary_text": summary,
        "evidence_excerpts": evidence,
    }
    return parse_evidence_card(
        json.dumps(card, ensure_ascii=False),
        candidate=candidate,
        candidates_sha256=candidates_sha256,
    )


def parse_evidence_card(
    text: str,
    *,
    candidate: Mapping[str, Any],
    candidates_sha256: str,
) -> dict[str, Any]:
    """Strictly validate one authoritative JSON card."""
    card = _parse_object(text, "evidence card")
    _exact_keys(
        card,
        {
            "schema_version",
            "card_id",
            "source_identity",
            "cite_key",
            "title",
            "extraction_status",
            "fallback_reason",
            "summary_text",
            "evidence_excerpts",
        },
        "evidence card",
    )
    if card["schema_version"] != CARD_SCHEMA_VERSION:
        raise EvidenceCardContractError("unsupported evidence-card schema_version")
    card_id = _required_string(card, "card_id")
    if re.fullmatch(r"card-\d{3,}", card_id) is None:
        raise EvidenceCardContractError("invalid card_id")
    for field in ("source_identity", "cite_key", "title"):
        if card[field] != candidate.get(field):
            raise EvidenceCardContractError(f"card {field} mismatch")
    status = card["extraction_status"]
    if status not in {"success", "fallback", "failed"}:
        raise EvidenceCardContractError("invalid extraction_status")
    reason = card["fallback_reason"]
    if status == "success":
        if reason is not None:
            raise EvidenceCardContractError("successful card has fallback_reason")
    elif not isinstance(reason, str) or not reason.strip():
        raise EvidenceCardContractError("non-success card requires fallback_reason")
    summary = card["summary_text"]
    if not isinstance(summary, dict):
        raise EvidenceCardContractError("card summary_text must be an object")
    _exact_keys(summary, set(SUMMARY_FIELDS), "card summary_text")
    for field in SUMMARY_FIELDS:
        value = summary[field]
        if not isinstance(value, str):
            raise EvidenceCardContractError("card summary values must be strings")
        if status == "success" and not value.strip():
            raise EvidenceCardContractError("successful card summary is incomplete")

    excerpts = card["evidence_excerpts"]
    if not isinstance(excerpts, list):
        raise EvidenceCardContractError("evidence_excerpts must be a list")
    if status == "success" and not excerpts:
        raise EvidenceCardContractError("successful card requires evidence")
    if status != "success" and excerpts:
        raise EvidenceCardContractError("non-success card cannot carry evidence")
    abstract = candidate.get("abstract")
    if not isinstance(abstract, str):
        raise EvidenceCardContractError("candidate abstract must be a string")
    seen_excerpt_ids: set[str] = set()
    for excerpt in excerpts:
        if not isinstance(excerpt, dict):
            raise EvidenceCardContractError("evidence excerpt must be an object")
        _exact_keys(
            excerpt,
            {
                "excerpt_id",
                "source_type",
                "source_artifact_path",
                "source_artifact_sha256",
                "source_record_id",
                "json_pointer",
                "char_start",
                "char_end",
                "excerpt_text",
                "excerpt_sha256",
            },
            "evidence excerpt",
        )
        excerpt_id = _required_string(excerpt, "excerpt_id")
        if excerpt_id in seen_excerpt_ids:
            raise EvidenceCardContractError("duplicate excerpt_id")
        seen_excerpt_ids.add(excerpt_id)
        if excerpt["source_type"] != "abstract":
            raise EvidenceCardContractError("unsupported excerpt source_type")
        if excerpt["source_artifact_path"] != "stage-04/candidates.jsonl":
            raise EvidenceCardContractError("noncanonical excerpt source path")
        if excerpt["source_artifact_sha256"] != candidates_sha256:
            raise EvidenceCardContractError("excerpt source artifact hash mismatch")
        if excerpt["source_record_id"] != candidate.get("source_identity"):
            raise EvidenceCardContractError("excerpt source_record_id mismatch")
        if excerpt["json_pointer"] != "/abstract":
            raise EvidenceCardContractError("unsupported excerpt json_pointer")
        start = _required_nonnegative_int(excerpt, "char_start")
        end = _required_nonnegative_int(excerpt, "char_end")
        if not start < end <= len(abstract):
            raise EvidenceCardContractError("invalid excerpt span")
        excerpt_text = _required_string(excerpt, "excerpt_text")
        if len(excerpt_text) < MIN_EXCERPT_CHARS:
            raise EvidenceCardContractError(
                f"evidence excerpt is shorter than {MIN_EXCERPT_CHARS} characters"
            )
        if abstract[start:end] != excerpt_text:
            raise EvidenceCardContractError("excerpt text does not match source slice")
        excerpt_hash = sha256_text(excerpt_text)
        if excerpt["excerpt_sha256"] != excerpt_hash:
            raise EvidenceCardContractError("excerpt_sha256 mismatch")
        expected_id = "ev-" + hashlib.sha256(
            f"{candidate['source_identity']}\n{start}\n{end}\n{excerpt_hash}".encode(
                "utf-8"
            )
        ).hexdigest()[:16]
        if excerpt_id != expected_id:
            raise EvidenceCardContractError("excerpt_id mismatch")
    return card


def render_evidence_card_markdown(card: Mapping[str, Any]) -> str:
    """Render the human view exclusively from validated JSON card data."""
    lines = [f"# {card['title']}", "", f"- Cite key: `{card['cite_key']}`"]
    lines.append(f"- Source identity: `{card['source_identity']}`")
    lines.append(f"- Extraction status: `{card['extraction_status']}`")
    if card.get("fallback_reason"):
        lines.append(f"- Fallback reason: `{card['fallback_reason']}`")
    lines.append("")
    summary = card["summary_text"]
    for field in SUMMARY_FIELDS:
        lines.extend([f"## {field.title()}", str(summary[field]), ""])
    lines.append("## Evidence Excerpts")
    excerpts = card["evidence_excerpts"]
    if not excerpts:
        lines.extend(["No eligible evidence excerpt.", ""])
    else:
        for excerpt in excerpts:
            lines.append(
                f"- `{excerpt['excerpt_id']}` chars "
                f"{excerpt['char_start']}:{excerpt['char_end']}"
            )
            lines.extend([f"> {excerpt['excerpt_text']}", ""])
    return "\n".join(lines).rstrip() + "\n"


def build_cards_manifest(
    *,
    shortlist_sha256: str,
    screening_report_sha256: str,
    cards: Iterable[tuple[Mapping[str, Any], str, str]],
) -> dict[str, Any]:
    """Build the Stage 6 manifest from canonical serialized card views."""
    entries: list[dict[str, Any]] = []
    for card, json_text, markdown_text in cards:
        card_id = str(card["card_id"])
        entries.append(
            {
                "card_id": card_id,
                "source_identity": card["source_identity"],
                "cite_key": card["cite_key"],
                "json_path": f"stage-06/cards/{card_id}.json",
                "json_sha256": sha256_text(json_text),
                "markdown_path": f"stage-06/cards/{card_id}.md",
                "markdown_sha256": sha256_text(markdown_text),
            }
        )
    manifest = {
        "schema_version": CARDS_MANIFEST_SCHEMA_VERSION,
        "shortlist_path": "stage-05/shortlist.jsonl",
        "shortlist_sha256": shortlist_sha256,
        "screening_report_path": "stage-05/screening_report.json",
        "screening_report_sha256": screening_report_sha256,
        "renderer_version": CARD_RENDERER_VERSION,
        "cards": entries,
    }
    return parse_cards_manifest(json.dumps(manifest, ensure_ascii=False))


def parse_cards_manifest(text: str) -> dict[str, Any]:
    manifest = _parse_object(text, "cards manifest")
    _exact_keys(
        manifest,
        {
            "schema_version",
            "shortlist_path",
            "shortlist_sha256",
            "screening_report_path",
            "screening_report_sha256",
            "renderer_version",
            "cards",
        },
        "cards manifest",
    )
    if manifest["schema_version"] != CARDS_MANIFEST_SCHEMA_VERSION:
        raise EvidenceCardContractError("unsupported cards manifest schema")
    if manifest["shortlist_path"] != "stage-05/shortlist.jsonl":
        raise EvidenceCardContractError("noncanonical manifest shortlist_path")
    if manifest["screening_report_path"] != "stage-05/screening_report.json":
        raise EvidenceCardContractError("noncanonical manifest report path")
    if manifest["renderer_version"] != CARD_RENDERER_VERSION:
        raise EvidenceCardContractError("unsupported card renderer_version")
    for field in ("shortlist_sha256", "screening_report_sha256"):
        _sha256_field(manifest, field)
    if not isinstance(manifest["cards"], list) or not manifest["cards"]:
        raise EvidenceCardContractError("cards manifest must not be empty")
    card_ids: set[str] = set()
    identities: set[str] = set()
    cite_keys: set[str] = set()
    for entry in manifest["cards"]:
        if not isinstance(entry, dict):
            raise EvidenceCardContractError("manifest card entry must be an object")
        _exact_keys(
            entry,
            {
                "card_id",
                "source_identity",
                "cite_key",
                "json_path",
                "json_sha256",
                "markdown_path",
                "markdown_sha256",
            },
            "manifest card entry",
        )
        card_id = _required_string(entry, "card_id")
        identity = _required_string(entry, "source_identity")
        cite_key = _required_string(entry, "cite_key")
        if card_id in card_ids or identity in identities or cite_key in cite_keys:
            raise EvidenceCardContractError("duplicate cards manifest identity")
        card_ids.add(card_id)
        identities.add(identity)
        cite_keys.add(cite_key)
        if entry["json_path"] != f"stage-06/cards/{card_id}.json":
            raise EvidenceCardContractError("noncanonical card JSON path")
        if entry["markdown_path"] != f"stage-06/cards/{card_id}.md":
            raise EvidenceCardContractError("noncanonical card Markdown path")
        _sha256_field(entry, "json_sha256")
        _sha256_field(entry, "markdown_sha256")
    return manifest


def validate_cards_artifacts(
    *,
    stage_dir: Path,
    manifest: Mapping[str, Any],
    shortlist_text: str,
    screening_report_text: str,
    candidates_sha256: str,
    shortlist: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Recompute every manifest/card/view binding from disk, default-deny."""
    parsed_manifest = parse_cards_manifest(
        json.dumps(manifest, ensure_ascii=False)
    )
    if parsed_manifest["shortlist_sha256"] != sha256_text(shortlist_text):
        raise EvidenceCardContractError("cards manifest shortlist hash mismatch")
    if parsed_manifest["screening_report_sha256"] != sha256_text(
        screening_report_text
    ):
        raise EvidenceCardContractError("cards manifest report hash mismatch")
    shortlist_rows = tuple(shortlist)
    candidate_map = {
        str(candidate["source_identity"]): candidate for candidate in shortlist_rows
    }
    expected_identities = [str(row["source_identity"]) for row in shortlist_rows]
    actual_identities = [
        str(entry["source_identity"]) for entry in parsed_manifest["cards"]
    ]
    if actual_identities != expected_identities:
        raise EvidenceCardContractError("cards manifest shortlist-order mismatch")
    expected_card_ids = [f"card-{index:03d}" for index in range(1, len(shortlist_rows) + 1)]
    if [entry["card_id"] for entry in parsed_manifest["cards"]] != expected_card_ids:
        raise EvidenceCardContractError("cards manifest card-id sequence mismatch")
    cards_dir = stage_dir / "cards"
    if cards_dir.is_symlink() or not cards_dir.is_dir():
        raise EvidenceCardContractError("cards directory is missing or unsafe")
    actual_files = set()
    for path in cards_dir.iterdir():
        if path.is_symlink() or not path.is_file():
            raise EvidenceCardContractError("cards directory must be flat files only")
        actual_files.add(path.name)
    expected_files = {
        Path(entry[key]).name
        for entry in parsed_manifest["cards"]
        for key in ("json_path", "markdown_path")
    }
    if actual_files != expected_files:
        raise EvidenceCardContractError("cards directory manifest closure mismatch")

    parsed_cards: list[dict[str, Any]] = []
    for entry in parsed_manifest["cards"]:
        json_path = cards_dir / Path(entry["json_path"]).name
        markdown_path = cards_dir / Path(entry["markdown_path"]).name
        json_text = json_path.read_text(encoding="utf-8")
        markdown_text = markdown_path.read_text(encoding="utf-8")
        if sha256_text(json_text) != entry["json_sha256"]:
            raise EvidenceCardContractError("card JSON hash mismatch")
        if sha256_text(markdown_text) != entry["markdown_sha256"]:
            raise EvidenceCardContractError("card Markdown hash mismatch")
        candidate = candidate_map.get(str(entry["source_identity"]))
        if candidate is None:
            raise EvidenceCardContractError("manifest card identity is not shortlisted")
        card = parse_evidence_card(
            json_text,
            candidate=candidate,
            candidates_sha256=candidates_sha256,
        )
        if card["card_id"] != entry["card_id"] or card["cite_key"] != entry["cite_key"]:
            raise EvidenceCardContractError("manifest card metadata mismatch")
        if render_evidence_card_markdown(card) != markdown_text:
            raise EvidenceCardContractError("Markdown card is not deterministic")
        parsed_cards.append(card)
    return tuple(parsed_cards)


def load_validated_cards(run_dir: Path, config: Any) -> tuple[dict[str, Any], ...]:
    """Replay Stage 4-6 bindings before any downstream card consumption."""
    inputs = load_validated_card_inputs(run_dir, config)
    manifest_path = run_dir / "stage-06" / "cards_manifest.json"
    try:
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise EvidenceCardContractError("canonical cards manifest is missing")
        manifest_text = manifest_path.read_text(encoding="utf-8")
        manifest = parse_cards_manifest(manifest_text)
        return validate_cards_artifacts(
            stage_dir=run_dir / "stage-06",
            manifest=manifest,
            shortlist_text=inputs.shortlist_text,
            screening_report_text=inputs.screening_report_text,
            candidates_sha256=sha256_text(inputs.candidates_text),
            shortlist=inputs.shortlist,
        )
    except (OSError, UnicodeDecodeError) as exc:
        raise EvidenceCardContractError(
            f"cannot read canonical Stage 6 card artifacts: {exc}"
        ) from exc


def canonical_json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _parse_object(text: str, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise EvidenceCardContractError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvidenceCardContractError(f"{label} root must be an object")
    return payload


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceCardContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _exact_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise EvidenceCardContractError(
            f"{label} fields mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _required_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise EvidenceCardContractError(f"{field} must be a nonempty string")
    return value


def _required_list_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceCardContractError(f"{label} must be a nonempty string")
    return value


def _required_nonnegative_int(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceCardContractError(f"{field} must be a nonnegative integer")
    return value


def _sha256_field(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise EvidenceCardContractError(f"invalid {field}")
    return value
