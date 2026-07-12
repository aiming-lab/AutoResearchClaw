"""Deterministic citation identity and Stage 4 registry construction."""

from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


CITE_KEY_VERSION = 2
CITE_KEY_REGISTRY_SCHEMA_VERSION = 1

_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "into",
        "over",
        "upon",
        "about",
        "through",
        "using",
        "based",
        "towards",
        "toward",
        "between",
        "under",
        "more",
        "than",
        "when",
        "what",
        "which",
        "where",
        "does",
        "have",
        "been",
        "some",
        "each",
        "also",
        "much",
        "very",
        "learning",
    }
)

_ARXIV_ID_RE = re.compile(
    r"(?i)^(?:arxiv:\s*)?"
    r"((?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7}))"
    r"(?:v\d+)?$"
)
_ARXIV_URL_RE = re.compile(
    r"(?i)^https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/"
    r"((?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7}))"
    r"(?:v\d+)?(?:\.pdf)?(?:[?#].*)?$"
)
_TAG_RE = re.compile(r"<[^>]*>")
_ARXIV_CATEGORY_RE = re.compile(
    r"^(?:cs|math|stat|eess|physics|q-bio|q-fin|astro-ph|cond-mat|"
    r"gr-qc|hep-ex|hep-lat|hep-ph|hep-th|nlin|nucl-ex|nucl-th|"
    r"quant-ph)\.[A-Z]{2}$"
)


class CitationIdentityError(ValueError):
    """Raised when a candidate collection cannot be sealed consistently."""


@dataclass(frozen=True)
class SealedCitationCollection:
    """Exact serialized Stage 4 artifacts projected from one registry."""

    candidates: tuple[dict[str, Any], ...]
    candidates_jsonl: str
    bibliography: str
    registry: dict[str, Any]


def parse_cite_key_registry(text: str) -> dict[str, Any]:
    """Parse a registry with duplicate-key and exact-schema rejection."""
    try:
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CitationIdentityError(f"invalid cite-key registry JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CitationIdentityError("cite-key registry root must be an object")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "cite_key_version",
            "candidates_path",
            "candidates_sha256",
            "references_path",
            "references_sha256",
            "entries",
        },
        "cite-key registry",
    )
    if payload["schema_version"] != CITE_KEY_REGISTRY_SCHEMA_VERSION:
        raise CitationIdentityError("unsupported cite-key registry schema_version")
    if payload["cite_key_version"] != CITE_KEY_VERSION:
        raise CitationIdentityError("unsupported cite_key_version")
    if payload["candidates_path"] != "stage-04/candidates.jsonl":
        raise CitationIdentityError("noncanonical candidates_path")
    if payload["references_path"] != "stage-04/references.bib":
        raise CitationIdentityError("noncanonical references_path")
    for field in ("candidates_sha256", "references_sha256"):
        if not isinstance(payload[field], str) or not re.fullmatch(
            r"[0-9a-f]{64}", payload[field]
        ):
            raise CitationIdentityError(f"invalid {field}")
    if not isinstance(payload["entries"], list) or not payload["entries"]:
        raise CitationIdentityError("registry entries must be a nonempty list")

    identities: set[str] = set()
    cite_keys: set[str] = set()
    for raw_entry in payload["entries"]:
        if not isinstance(raw_entry, dict):
            raise CitationIdentityError("registry entry must be an object")
        _require_exact_keys(
            raw_entry,
            {"source_identity", "cite_key", "base_key", "collision_suffix"},
            "cite-key registry entry",
        )
        identity = _required_string(raw_entry, "source_identity")
        cite_key = _required_string(raw_entry, "cite_key")
        base_key = _required_string(raw_entry, "base_key")
        suffix = raw_entry["collision_suffix"]
        if suffix is not None and (
            not isinstance(suffix, str) or not re.fullmatch(r"[0-9a-f]{8}", suffix)
        ):
            raise CitationIdentityError("invalid collision_suffix")
        if cite_key != f"{base_key}{suffix or ''}":
            raise CitationIdentityError("cite_key does not match base_key and suffix")
        if identity in identities:
            raise CitationIdentityError("duplicate source_identity in registry")
        if cite_key in cite_keys:
            raise CitationIdentityError("duplicate cite_key in registry")
        identities.add(identity)
        cite_keys.add(cite_key)
    return payload


def validate_registry_artifacts(
    registry: Mapping[str, Any],
    candidates_jsonl: str,
    bibliography: str,
) -> None:
    """Recompute registry bindings against exact candidate and BibTeX text."""
    if registry.get("candidates_sha256") != _sha256_text(candidates_jsonl):
        raise CitationIdentityError("registry candidates_sha256 mismatch")
    if registry.get("references_sha256") != _sha256_text(bibliography):
        raise CitationIdentityError("registry references_sha256 mismatch")

    rows = _parse_candidates_jsonl(candidates_jsonl)
    row_map: dict[str, str] = {}
    for row in rows:
        identity = _required_string(row, "source_identity")
        cite_key = _required_string(row, "cite_key")
        if row.get("cite_key_version") != CITE_KEY_VERSION:
            raise CitationIdentityError("candidate cite_key_version mismatch")
        if identity in row_map:
            raise CitationIdentityError("duplicate source_identity in candidates")
        row_map[identity] = cite_key

    entries = registry.get("entries")
    if not isinstance(entries, list):
        raise CitationIdentityError("registry entries must be a list")
    registry_map = {
        _required_string(entry, "source_identity"): _required_string(entry, "cite_key")
        for entry in entries
        if isinstance(entry, Mapping)
    }
    if row_map != registry_map:
        raise CitationIdentityError("candidate identities/keys do not match registry")

    bib_keys = re.findall(r"(?m)^@\w+\s*\{\s*([^,\s]+)\s*,", bibliography)
    if len(bib_keys) != len(set(bib_keys)):
        raise CitationIdentityError("duplicate cite key in bibliography")
    if set(bib_keys) != set(registry_map.values()):
        raise CitationIdentityError("bibliography keys do not match registry")


def normalize_doi(value: object) -> str:
    """Return a stable lowercase DOI without URL or ``doi:`` prefixes."""
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    text = re.sub(r"\s+", "", text).rstrip(".,;)")
    return text if text.startswith("10.") and "/" in text else ""


def normalize_arxiv_id(value: object) -> str:
    """Return a stable arXiv work identity without a version suffix."""
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    match = _ARXIV_ID_RE.fullmatch(text)
    return match.group(1).lower() if match else ""


def normalize_arxiv_url(value: object) -> str:
    """Extract an arXiv identity only from a canonical arxiv.org URL."""
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    match = _ARXIV_URL_RE.fullmatch(text)
    return match.group(1).lower() if match else ""


def clean_title_text(value: object) -> str:
    """Normalize title text and remove HTML/XML markup before tokenization."""
    text = unicodedata.normalize("NFKC", html.unescape(str(value or "")))
    return re.sub(r"\s+", " ", _TAG_RE.sub(" ", text)).strip()


def source_identity_for_candidate(candidate: Mapping[str, Any]) -> str:
    """Derive a versioned source identity from canonical candidate metadata."""
    doi = normalize_doi(candidate.get("doi"))
    if doi:
        return f"doi:{doi}"

    arxiv_id = normalize_arxiv_id(candidate.get("arxiv_id"))
    if not arxiv_id:
        arxiv_id = normalize_arxiv_url(candidate.get("url"))
    if arxiv_id:
        return f"arxiv:{arxiv_id}"

    provider_id = str(candidate.get("paper_id") or candidate.get("id") or "").strip()
    if provider_id:
        source = _ascii_token(candidate.get("source")) or "unknown"
        normalized_id = unicodedata.normalize("NFKC", provider_id).strip().lower()
        return f"provider:{source}:{normalized_id}"

    title = _normalize_metadata_text(candidate.get("title"))
    author = _normalize_metadata_text(_first_author_name(candidate))
    year = _safe_year(candidate.get("year"))
    if not title:
        raise CitationIdentityError("candidate lacks DOI, arXiv ID, provider ID, and title")
    material = f"{title}\n{author}\n{year}".encode("utf-8")
    return f"metadata:{hashlib.sha256(material).hexdigest()[:24]}"


def base_cite_key_for_candidate(candidate: Mapping[str, Any]) -> str:
    """Build the human-readable citation-key base for one candidate."""
    author = _first_author_name(candidate)
    surname = _ascii_token(author.split()[-1] if author.split() else "") or "anon"
    year = str(_safe_year(candidate.get("year")) or "0000")

    normalized_title = clean_title_text(candidate.get("title"))
    ascii_title = unicodedata.normalize("NFKD", normalized_title)
    ascii_title = ascii_title.encode("ascii", "ignore").decode("ascii")
    keyword = ""
    for token in re.findall(r"[A-Za-z0-9]+", ascii_title):
        lowered = token.lower()
        if len(lowered) > 3 and lowered not in _STOPWORDS:
            keyword = lowered
            break
    return f"{surname}{year}{keyword or 'paper'}"


def seal_citation_collection(
    candidates: Iterable[Mapping[str, Any]],
) -> SealedCitationCollection:
    """Seal candidates, bibliography, and registry from one complete collection."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in candidates:
        candidate = dict(raw)
        if candidate.get("is_placeholder"):
            raise CitationIdentityError("placeholder candidates cannot enter cite-key registry")
        identity = source_identity_for_candidate(candidate)
        candidate["source_identity"] = identity
        grouped.setdefault(identity, []).append(candidate)

    if not grouped:
        raise CitationIdentityError("cannot seal an empty citation collection")

    deduplicated: dict[str, dict[str, Any]] = {}
    for identity, rows in grouped.items():
        titles = {
            _normalize_metadata_text(row.get("title"))
            for row in rows
            if _normalize_metadata_text(row.get("title"))
        }
        if len(titles) > 1:
            raise CitationIdentityError(
                f"conflicting titles for source identity {identity}"
            )
        deduplicated[identity] = max(rows, key=_candidate_preference_key)
    base_groups: dict[str, list[str]] = {}
    for identity, candidate in deduplicated.items():
        base = base_cite_key_for_candidate(candidate)
        base_groups.setdefault(base, []).append(identity)

    sealed_rows: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    for identity in sorted(deduplicated):
        candidate = dict(deduplicated[identity])
        base = base_cite_key_for_candidate(candidate)
        collision_suffix: str | None = None
        if len(base_groups[base]) > 1:
            collision_suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8]
        cite_key = f"{base}{collision_suffix or ''}"
        candidate["source_identity"] = identity
        candidate["cite_key_version"] = CITE_KEY_VERSION
        candidate["cite_key"] = cite_key
        sealed_rows.append(candidate)
        entries.append(
            {
                "source_identity": identity,
                "cite_key": cite_key,
                "base_key": base,
                "collision_suffix": collision_suffix,
            }
        )

    cite_keys = [entry["cite_key"] for entry in entries]
    if len(cite_keys) != len(set(cite_keys)):
        raise CitationIdentityError("cite-key collision remained after suffixing")

    candidates_jsonl = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in sealed_rows
    )
    bibliography = "\n\n".join(
        render_candidate_bibtex(row) for row in sealed_rows
    ) + "\n"
    registry = {
        "schema_version": CITE_KEY_REGISTRY_SCHEMA_VERSION,
        "cite_key_version": CITE_KEY_VERSION,
        "candidates_path": "stage-04/candidates.jsonl",
        "candidates_sha256": _sha256_text(candidates_jsonl),
        "references_path": "stage-04/references.bib",
        "references_sha256": _sha256_text(bibliography),
        "entries": entries,
    }
    registry_text = json.dumps(
        registry, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    parsed_registry = parse_cite_key_registry(registry_text)
    validate_registry_artifacts(parsed_registry, candidates_jsonl, bibliography)
    return SealedCitationCollection(
        candidates=tuple(sealed_rows),
        candidates_jsonl=candidates_jsonl,
        bibliography=bibliography,
        registry=registry,
    )


def render_candidate_bibtex(candidate: Mapping[str, Any]) -> str:
    """Render one registry-keyed candidate as a deterministic BibTeX entry."""
    cite_key = str(candidate.get("cite_key") or "").strip()
    if not cite_key:
        raise CitationIdentityError("candidate is missing sealed cite_key")

    title = _sanitize_bibtex_value(candidate.get("title") or "Untitled")
    author_text = _sanitize_bibtex_value(
        " and ".join(_author_names(candidate)) or "Unknown"
    )
    year = _safe_year(candidate.get("year")) or "Unknown"
    venue = _sanitize_bibtex_value(candidate.get("venue"))
    arxiv_id = normalize_arxiv_id(candidate.get("arxiv_id"))
    doi = normalize_doi(candidate.get("doi"))
    url = _sanitize_bibtex_value(candidate.get("url"))

    venue_lower = venue.lower()
    is_arxiv_category = bool(_ARXIV_CATEGORY_RE.match(venue))
    is_proceedings = bool(
        venue
        and not is_arxiv_category
        and any(
            marker in venue_lower
            for marker in (
                "conference",
                "proc",
                "workshop",
                "neurips",
                "icml",
                "iclr",
                "aaai",
                "cvpr",
                "acl",
                "emnlp",
                "naacl",
                "eccv",
                "iccv",
                "sigir",
                "kdd",
                "www",
                "ijcai",
            )
        )
    )
    if is_proceedings:
        entry_type = "inproceedings"
        venue_field = f"  booktitle = {{{venue}}},"
    elif arxiv_id and (not venue or is_arxiv_category):
        entry_type = "article"
        venue_field = f"  journal = {{arXiv preprint arXiv:{arxiv_id}}},"
    else:
        entry_type = "article"
        venue_field = f"  journal = {{{venue}}}," if venue else ""

    lines = [f"@{entry_type}{{{cite_key},"]
    lines.append(f"  title = {{{title}}},")
    lines.append(f"  author = {{{author_text}}},")
    lines.append(f"  year = {{{year}}},")
    if venue_field:
        lines.append(venue_field)
    if doi:
        lines.append(f"  doi = {{{doi}}},")
    if arxiv_id:
        lines.append(f"  eprint = {{{arxiv_id}}},")
        lines.append("  archiveprefix = {arXiv},")
    if url:
        lines.append(f"  url = {{{url}}},")
    lines.append("}")
    return "\n".join(lines)


def _candidate_preference_key(candidate: Mapping[str, Any]) -> tuple[int, int, str]:
    completeness = sum(
        bool(candidate.get(field))
        for field in ("title", "authors", "year", "abstract", "venue", "doi", "arxiv_id", "url")
    )
    try:
        citations = int(candidate.get("citation_count") or 0)
    except (TypeError, ValueError):
        citations = 0
    stable = {
        key: value
        for key, value in candidate.items()
        if key not in {"collected_at", "cite_key", "cite_key_version", "source_identity"}
    }
    stable_json = json.dumps(
        stable, ensure_ascii=False, sort_keys=True, default=str
    )
    return completeness, citations, stable_json


def _author_names(candidate: Mapping[str, Any]) -> list[str]:
    raw = candidate.get("authors")
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(" and ") if part.strip()]
    if not isinstance(raw, (list, tuple)):
        return []
    names: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            names.append(item.strip())
        elif isinstance(item, Mapping):
            name = str(item.get("name") or "").strip()
            if name:
                names.append(name)
    return names


def _first_author_name(candidate: Mapping[str, Any]) -> str:
    names = _author_names(candidate)
    return names[0] if names else ""


def _safe_year(value: object) -> int:
    try:
        year = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return year if 0 <= year <= 9999 else 0


def _normalize_metadata_text(value: object) -> str:
    text = clean_title_text(value).casefold()
    return " ".join(re.findall(r"\w+", text, flags=re.UNICODE))


def _ascii_token(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    return re.sub(r"[^a-zA-Z0-9]", "", text.encode("ascii", "ignore").decode("ascii")).lower()


def _sanitize_bibtex_value(value: object) -> str:
    """Keep candidate metadata from changing BibTeX entry structure."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = re.sub(r"[\r\n]+", " ", text)
    return re.sub(r"[{}]", "", text).strip()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_candidates_jsonl(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.split("\n"), start=1):
        if not line:
            if line_number == len(text.split("\n")):
                continue
            raise CitationIdentityError(f"blank candidates JSONL line {line_number}")
        try:
            row = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as exc:
            raise CitationIdentityError(
                f"invalid candidates JSONL line {line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise CitationIdentityError(
                f"candidates JSONL line {line_number} must be an object"
            )
        rows.append(row)
    if not rows:
        raise CitationIdentityError("candidates JSONL must not be empty")
    return rows


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CitationIdentityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise CitationIdentityError(
            f"{label} fields mismatch: missing={missing}, extra={extra}"
        )


def _required_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise CitationIdentityError(f"{field} must be a nonempty string")
    return value
