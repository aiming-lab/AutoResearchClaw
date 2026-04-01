"""Citation verification engine — detect hallucinated references.

Verifies each BibTeX entry against real academic APIs using a three-layer
strategy:

  L1: **arXiv ID lookup** — direct ``id_list`` query to arXiv API
  L2: **DOI resolution** — HTTP GET to CrossRef ``/works/{doi}``
  L3: **Title search** — search Semantic Scholar + arXiv by title

Classifications:

  - ``VERIFIED``:      API confirms existence + title similarity ≥ 0.80
  - ``SUSPICIOUS``:    Found a paper but metadata diverges (0.50 ≤ sim < 0.80)
  - ``HALLUCINATED``:  Not found via any API or sim < 0.50
  - ``SKIPPED``:       Entry cannot be verified (no title, or all APIs unreachable)

All network I/O uses stdlib ``urllib`` — zero extra pip dependencies.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from researchclaw.literature.models import Author, Paper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public enums & data classes
# ---------------------------------------------------------------------------


class VerifyStatus(str, Enum):
    """Verification outcome for a single citation."""

    VERIFIED = "verified"
    SUSPICIOUS = "suspicious"
    HALLUCINATED = "hallucinated"
    SKIPPED = "skipped"


@dataclass
class CitationResult:
    """Verification result for one BibTeX entry."""

    cite_key: str
    title: str
    status: VerifyStatus
    confidence: float  # 0.0–1.0
    method: str  # "arxiv_id" | "doi" | "title_search" | "skipped"
    details: str = ""
    matched_paper: Paper | None = None
    relevance_score: float | None = None  # 0.0–1.0, set by LLM relevance check

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "cite_key": self.cite_key,
            "title": self.title,
            "status": self.status.value,
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "details": self.details,
        }
        if self.relevance_score is not None:
            d["relevance_score"] = round(self.relevance_score, 2)
        if self.matched_paper:
            d["matched_paper"] = {
                "title": self.matched_paper.title,
                "authors": [a.name for a in self.matched_paper.authors],
                "year": self.matched_paper.year,
                "source": self.matched_paper.source,
            }
        return d


@dataclass
class VerificationReport:
    """Aggregate report for all citations in a paper."""

    total: int = 0
    verified: int = 0
    suspicious: int = 0
    hallucinated: int = 0
    skipped: int = 0
    results: list[CitationResult] = field(default_factory=list)

    @property
    def integrity_score(self) -> float:
        """Fraction of verifiable citations that are verified (0.0–1.0)."""
        verifiable = self.total - self.skipped
        if verifiable <= 0:
            return 1.0
        return round(self.verified / verifiable, 3)

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": {
                "total": self.total,
                "verified": self.verified,
                "suspicious": self.suspicious,
                "hallucinated": self.hallucinated,
                "skipped": self.skipped,
                "integrity_score": self.integrity_score,
            },
            "results": [r.to_dict() for r in self.results],
        }


# ---------------------------------------------------------------------------
# BibTeX parsing
# ---------------------------------------------------------------------------

_ENTRY_RE = re.compile(
    r"@(\w+)\s*\{\s*([^,\s]+)\s*,\s*(.*?)\s*\}(?=\s*(?:@|\Z))",
    re.DOTALL,
)

_FIELD_RE = re.compile(
    r"(\w+)\s*=\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}",
    re.DOTALL,
)


def parse_bibtex_entries(bib_text: str) -> list[dict[str, str]]:
    """Parse BibTeX text into a list of field dicts.

    Each dict contains at least ``key`` and ``type``, plus any parsed fields
    (``title``, ``author``, ``year``, ``doi``, ``eprint``, ``url``, …).
    """
    entries: list[dict[str, str]] = []
    for m in _ENTRY_RE.finditer(bib_text):
        entry: dict[str, str] = {
            "type": m.group(1).lower(),
            "key": m.group(2).strip(),
        }
        body = m.group(3)
        for fm in _FIELD_RE.finditer(body):
            entry[fm.group(1).lower()] = fm.group(2).strip()
        entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Title similarity
# ---------------------------------------------------------------------------


def title_similarity(a: str, b: str) -> float:
    """Word-overlap Jaccard-ish similarity between two titles.

    Returns 0.0–1.0.  Uses max(len) as denominator so short titles don't
    inflate the score.
    """

    def _words(t: str) -> set[str]:
        return set(re.sub(r"[^a-z0-9\s]", "", t.lower()).split()) - {""}

    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


# ---------------------------------------------------------------------------
# L1: arXiv ID verification
# ---------------------------------------------------------------------------

_ARXIV_API = "https://export.arxiv.org/api/query"
_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
_ARXIV_TIMEOUT = 20


def verify_by_arxiv_id(arxiv_id: str, expected_title: str) -> CitationResult | None:
    """Look up a paper by arXiv ID and compare titles.

    Returns *None* on network failure so that the caller can fall through
    to the next verification layer.
    """
    # arXiv ID lookup uses id_list, not search_query
    params = urllib.parse.urlencode({"id_list": arxiv_id, "max_results": "1"})
    url = f"{_ARXIV_API}?{params}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ResearchClaw/0.1"})
        with urllib.request.urlopen(req, timeout=_ARXIV_TIMEOUT) as resp:
            data = resp.read().decode("utf-8")
    except Exception as exc:
        logger.debug("arXiv ID verification failed for %s: %s", arxiv_id, exc)
        return None

    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return None

    entries = root.findall("atom:entry", _ARXIV_NS)
    if not entries:
        return CitationResult(
            cite_key="",
            title=expected_title,
            status=VerifyStatus.HALLUCINATED,
            confidence=0.9,
            method="arxiv_id",
            details=f"arXiv ID {arxiv_id} not found in arXiv",
        )

    # arXiv returns an "error" entry when ID is invalid
    entry = entries[0]
    found_title_el = entry.find("atom:title", _ARXIV_NS)
    found_title = (
        (found_title_el.text or "").strip() if found_title_el is not None else ""
    )
    found_title = re.sub(r"\s+", " ", found_title)

    # Check for arXiv error responses (they return entry with id but title "Error")
    entry_id = entry.findtext("atom:id", "", _ARXIV_NS)
    if "api/errors" in entry_id or not found_title or found_title.lower() == "error":
        return CitationResult(
            cite_key="",
            title=expected_title,
            status=VerifyStatus.HALLUCINATED,
            confidence=0.9,
            method="arxiv_id",
            details=f"arXiv ID {arxiv_id} returned error or empty response",
        )

    sim = title_similarity(expected_title, found_title)
    if sim >= 0.80:
        return CitationResult(
            cite_key="",
            title=expected_title,
            status=VerifyStatus.VERIFIED,
            confidence=sim,
            method="arxiv_id",
            details=f"Confirmed via arXiv: '{found_title}'",
        )
    elif sim >= 0.50:
        return CitationResult(
            cite_key="",
            title=expected_title,
            status=VerifyStatus.SUSPICIOUS,
            confidence=sim,
            method="arxiv_id",
            details=f"arXiv ID exists but title differs (sim={sim:.2f}): '{found_title}'",
        )
    else:
        return CitationResult(
            cite_key="",
            title=expected_title,
            status=VerifyStatus.SUSPICIOUS,
            confidence=sim,
            method="arxiv_id",
            details=f"arXiv ID exists but title mismatch (sim={sim:.2f}): '{found_title}'",
        )


# ---------------------------------------------------------------------------
# L2: DOI verification via CrossRef
# ---------------------------------------------------------------------------

_CROSSREF_API = "https://api.crossref.org/works"
_CROSSREF_TIMEOUT = 20
_DATACITE_API = "https://api.datacite.org/dois"
_DATACITE_TIMEOUT = 15


def _verify_doi_datacite(doi: str, expected_title: str) -> CitationResult | None:
    """Fallback DOI verification via DataCite API.

    arXiv DOIs (10.48550/arXiv.*) are registered with DataCite, not CrossRef.
    Returns *None* on network failure.
    """
    encoded_doi = urllib.parse.quote(doi, safe="")
    url = f"{_DATACITE_API}/{encoded_doi}"

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "ResearchClaw/0.1",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=_DATACITE_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return CitationResult(
                cite_key="",
                title=expected_title,
                status=VerifyStatus.HALLUCINATED,
                confidence=0.9,
                method="doi",
                details=f"DOI {doi} not found via CrossRef or DataCite",
            )
        logger.debug("DataCite HTTP error for DOI %s: %s", doi, exc)
        return None
    except Exception as exc:
        logger.debug("DataCite verification failed for %s: %s", doi, exc)
        return None

    # Extract title from DataCite response
    attrs = body.get("data", {}).get("attributes", {})
    dc_titles = attrs.get("titles", [])
    found_title = dc_titles[0].get("title", "") if dc_titles else ""

    if not found_title:
        # DOI exists in DataCite but no title — still counts as verified
        return CitationResult(
            cite_key="",
            title=expected_title,
            status=VerifyStatus.VERIFIED,
            confidence=0.85,
            method="doi",
            details=f"DOI {doi} resolves via DataCite (no title comparison)",
        )

    sim = title_similarity(expected_title, found_title)
    if sim >= 0.80:
        return CitationResult(
            cite_key="",
            title=expected_title,
            status=VerifyStatus.VERIFIED,
            confidence=sim,
            method="doi",
            details=f"Confirmed via DataCite: '{found_title}'",
        )
    elif sim >= 0.50:
        return CitationResult(
            cite_key="",
            title=expected_title,
            status=VerifyStatus.SUSPICIOUS,
            confidence=sim,
            method="doi",
            details=f"DataCite DOI resolves but title differs (sim={sim:.2f}): '{found_title}'",
        )
    else:
        return CitationResult(
            cite_key="",
            title=expected_title,
            status=VerifyStatus.SUSPICIOUS,
            confidence=sim,
            method="doi",
            details=f"DataCite DOI resolves but title mismatch (sim={sim:.2f}): '{found_title}'",
        )


def verify_by_doi(doi: str, expected_title: str) -> CitationResult | None:
    """Verify a DOI via CrossRef API, with DataCite fallback for arXiv DOIs.

    Returns *None* on network failure.
    """
    encoded_doi = urllib.parse.quote(doi, safe="")
    url = f"{_CROSSREF_API}/{encoded_doi}"

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "ResearchClaw/0.1 (mailto:researchclaw@example.com)",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=_CROSSREF_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            # CrossRef 404 — try DataCite for arXiv/DataCite DOIs
            if doi.startswith("10.48550/") or doi.startswith("10.5281/"):
                dc_result = _verify_doi_datacite(doi, expected_title)
                if dc_result is not None:
                    return dc_result
            return CitationResult(
                cite_key="",
                title=expected_title,
                status=VerifyStatus.HALLUCINATED,
                confidence=0.9,
                method="doi",
                details=f"DOI {doi} not found (HTTP 404)",
            )
        logger.debug("CrossRef HTTP error for DOI %s: %s", doi, exc)
        return None
    except Exception as exc:
        logger.debug("DOI verification failed for %s: %s", doi, exc)
        return None

    # Extract title from CrossRef response
    message = body.get("message", {})
    titles = message.get("title", [])
    found_title = titles[0] if titles else ""

    if not found_title:
        # DOI exists but no title in response — still counts as verified
        return CitationResult(
            cite_key="",
            title=expected_title,
            status=VerifyStatus.VERIFIED,
            confidence=0.85,
            method="doi",
            details=f"DOI {doi} resolves via CrossRef (no title comparison)",
        )

    sim = title_similarity(expected_title, found_title)
    if sim >= 0.80:
        return CitationResult(
            cite_key="",
            title=expected_title,
            status=VerifyStatus.VERIFIED,
            confidence=sim,
            method="doi",
            details=f"Confirmed via CrossRef: '{found_title}'",
        )
    elif sim >= 0.50:
        return CitationResult(
            cite_key="",
            title=expected_title,
            status=VerifyStatus.SUSPICIOUS,
            confidence=sim,
            method="doi",
            details=f"DOI resolves but title differs (sim={sim:.2f}): '{found_title}'",
        )
    else:
        # DOI exists but title is very different — the DOI may be real
        # but the BibTeX entry may have wrong metadata
        return CitationResult(
            cite_key="",
            title=expected_title,
            status=VerifyStatus.SUSPICIOUS,
            confidence=sim,
            method="doi",
            details=f"DOI resolves but title mismatch (sim={sim:.2f}): '{found_title}'",
        )


# ---------------------------------------------------------------------------
# L3-alt: OpenAlex title search (primary L3 source — higher rate limits)
# ---------------------------------------------------------------------------

_OPENALEX_API = "https://api.openalex.org/works"
_OPENALEX_TIMEOUT = 15
_OPENALEX_EMAIL = "researchclaw@users.noreply.github.com"


def verify_by_openalex(title: str) -> CitationResult | None:
    """Verify a paper via OpenAlex API (10K+ calls/day vs S2's ~1 req/s).

    Returns *None* only on network failure (allows fallthrough to S2).
    """
    params = urllib.parse.urlencode({
        "filter": "title.search:" + title.replace(",", " ").replace(":", " "),
        "per_page": "5",
        "mailto": _OPENALEX_EMAIL,
    })
    url = f"{_OPENALEX_API}?{params}"

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": f"ResearchClaw/0.1 (mailto:{_OPENALEX_EMAIL})",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=_OPENALEX_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.debug("OpenAlex search failed for %r: %s", title, exc)
        return None

    results = body.get("results", [])
    if not results:
        return CitationResult(
            cite_key="",
            title=title,
            status=VerifyStatus.HALLUCINATED,
            confidence=0.7,
            method="openalex",
            details="No results found via OpenAlex",
        )

    best_sim = 0.0
    best_result = None
    for r in results:
        found_title = r.get("title", "")
        if found_title:
            sim = title_similarity(title, found_title)
            if sim > best_sim:
                best_sim = sim
                best_result = r

    if best_sim >= 0.80:
        return CitationResult(
            cite_key="",
            title=title,
            status=VerifyStatus.VERIFIED,
            confidence=best_sim,
            method="openalex",
            details=f"Confirmed via OpenAlex: '{best_result.get('title', '')}'",
        )
    elif best_sim >= 0.50:
        return CitationResult(
            cite_key="",
            title=title,
            status=VerifyStatus.SUSPICIOUS,
            confidence=best_sim,
            method="openalex",
            details=f"Partial match via OpenAlex (sim={best_sim:.2f}): '{best_result.get('title', '')}'",
        )
    else:
        return CitationResult(
            cite_key="",
            title=title,
            status=VerifyStatus.HALLUCINATED,
            confidence=0.7,
            method="openalex",
            details="No close match found via OpenAlex",
        )


# ---------------------------------------------------------------------------
# Verification result cache (avoids re-verifying known papers)
# ---------------------------------------------------------------------------

import hashlib
from pathlib import Path

_CACHE_DIR = Path.home() / ".cache" / "researchclaw" / "citation_verify"


def _cache_key(title: str) -> str:
    return hashlib.sha256(title.lower().strip().encode()).hexdigest()[:16]


def _read_cache(title: str) -> CitationResult | None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / f"{_cache_key(title)}.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return CitationResult(
                cite_key=data.get("cite_key", ""),
                title=data.get("title", title),
                status=VerifyStatus(data["status"]),
                confidence=data["confidence"],
                method=data["method"],
                details=data.get("details", ""),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return None
    return None


def _write_cache(title: str, result: CitationResult) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / f"{_cache_key(title)}.json"
    cache_file.write_text(
        json.dumps(result.to_dict(), indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# L3: Title search via Semantic Scholar + arXiv
# ---------------------------------------------------------------------------


def verify_by_title_search(
    title: str,
    *,
    s2_api_key: str = "",
) -> CitationResult | None:
    """Search for a paper by title and verify its existence.

    Uses the unified ``search_papers`` function from our literature module.
    Returns *None* only on total network failure.
    """
    from researchclaw.literature.search import search_papers

    try:
        results = search_papers(
            title,
            limit=5,
            s2_api_key=s2_api_key,
            deduplicate=True,
        )
    except Exception as exc:
        logger.debug("Title search failed for %r: %s", title, exc)
        return None

    if not results:
        return CitationResult(
            cite_key="",
            title=title,
            status=VerifyStatus.HALLUCINATED,
            confidence=0.7,
            method="title_search",
            details="No results found via Semantic Scholar + arXiv",
        )

    # Find best title match
    best_sim = 0.0
    best_paper: Paper | None = None
    for paper in results:
        sim = title_similarity(title, paper.title)
        if sim > best_sim:
            best_sim = sim
            best_paper = paper

    if best_sim >= 0.80:
        return CitationResult(
            cite_key="",
            title=title,
            status=VerifyStatus.VERIFIED,
            confidence=best_sim,
            method="title_search",
            details=f"Found via search: '{best_paper.title}'" if best_paper else "",
            matched_paper=best_paper,
        )
    elif best_sim >= 0.50:
        return CitationResult(
            cite_key="",
            title=title,
            status=VerifyStatus.SUSPICIOUS,
            confidence=best_sim,
            method="title_search",
            details=(
                f"Partial match (sim={best_sim:.2f}): '{best_paper.title}'"
                if best_paper
                else ""
            ),
            matched_paper=best_paper,
        )
    else:
        return CitationResult(
            cite_key="",
            title=title,
            status=VerifyStatus.HALLUCINATED,
            confidence=1.0 - best_sim,
            method="title_search",
            details=(
                f"Best match too weak (sim={best_sim:.2f}): '{best_paper.title}'"
                if best_paper
                else "No match found"
            ),
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def verify_citations(
    bib_text: str,
    *,
    s2_api_key: str = "",
    inter_verify_delay: float = 1.5,
) -> VerificationReport:
    """Verify all BibTeX entries against real academic APIs.

    Three-layer verification:

    1. If entry has ``eprint`` (arXiv ID) → arXiv API lookup
    2. If entry has ``doi`` → CrossRef API lookup
    3. Otherwise → title search via Semantic Scholar + arXiv

    Parameters
    ----------
    bib_text:
        Raw BibTeX string.
    s2_api_key:
        Optional Semantic Scholar API key for L3 title search.
    inter_verify_delay:
        Seconds to wait between API calls (rate limiting).
    """
    entries = parse_bibtex_entries(bib_text)
    report = VerificationReport(total=len(entries))

    # Adaptive delays: OpenAlex/CrossRef can be queried much faster than arXiv
    _DELAY_ARXIV = inter_verify_delay       # arXiv: conservative (1.5s default)
    _DELAY_CROSSREF = 0.3                   # CrossRef: 50 req/s polite pool
    _DELAY_OPENALEX = 0.2                   # OpenAlex: 10K/day
    api_call_count = 0

    # BUG-22: Global timeout — stop verifying after 5 minutes total
    _verify_start = time.monotonic()
    _VERIFY_TIMEOUT_SEC = 300  # 5 minutes

    for i, entry in enumerate(entries):
        # BUG-22: Check global timeout — mark remaining as SKIPPED
        if time.monotonic() - _verify_start > _VERIFY_TIMEOUT_SEC:
            logger.warning(
                "Verification timeout (%.0fs). Marking remaining %d/%d "
                "citations as SKIPPED.",
                _VERIFY_TIMEOUT_SEC, len(entries) - i, len(entries),
            )
            for remaining_entry in entries[i:]:
                _rkey = remaining_entry.get("key", f"unknown_{i}")
                _rtitle = remaining_entry.get("title", "")
                report.results.append(CitationResult(
                    cite_key=_rkey,
                    title=_rtitle,
                    status=VerifyStatus.SKIPPED,
                    confidence=0.0,
                    method="skipped",
                    details="Verification timeout exceeded",
                ))
                report.skipped += 1
            break

        key = entry.get("key", f"unknown_{i}")
        title = entry.get("title", "")
        arxiv_id = entry.get("eprint", "")
        doi = entry.get("doi", "")

        # Skip entries with no title
        if not title:
            result = CitationResult(
                cite_key=key,
                title="",
                status=VerifyStatus.SKIPPED,
                confidence=0.0,
                method="skipped",
                details="No title in BibTeX entry",
            )
            report.skipped += 1
            report.results.append(result)
            continue

        # Check cache first
        cached = _read_cache(title)
        if cached is not None:
            cached.cite_key = key
            report.results.append(cached)
            if cached.status == VerifyStatus.VERIFIED:
                report.verified += 1
            elif cached.status == VerifyStatus.SUSPICIOUS:
                report.suspicious += 1
            elif cached.status == VerifyStatus.HALLUCINATED:
                report.hallucinated += 1
            else:
                report.skipped += 1
            logger.debug("[cache] verify HIT [%s] %r → %s", key, title[:50], cached.status.value)
            continue

        result: CitationResult | None = None

        # Verification order optimized to minimize arXiv API pressure:
        #   DOI → CrossRef (fast, high limit)
        #   > OpenAlex title search (10K/day)
        #   > arXiv ID lookup (only if others fail, 1/3s)
        #   > S2 title search (last resort)

        # L2 first: DOI verification via CrossRef (fast, generous limits)
        if result is None and doi:
            if api_call_count > 0:
                time.sleep(_DELAY_CROSSREF)
            result = verify_by_doi(doi, title)
            api_call_count += 1
            if result is not None:
                logger.info(
                    "L2 DOI [%s] %s → %s (%.2f)",
                    key,
                    doi,
                    result.status.value,
                    result.confidence,
                )

        # L3a: OpenAlex title search (high rate limits, good coverage)
        if result is None:
            if api_call_count > 0:
                time.sleep(_DELAY_OPENALEX)
            result = verify_by_openalex(title)
            api_call_count += 1
            if result is not None:
                logger.info(
                    "L3a OpenAlex [%s] %r → %s (%.2f)",
                    key,
                    title[:50],
                    result.status.value,
                    result.confidence,
                )

        # L1: arXiv ID — only if DOI and OpenAlex both failed
        if result is None and arxiv_id:
            if api_call_count > 0:
                time.sleep(_DELAY_ARXIV)
            result = verify_by_arxiv_id(arxiv_id, title)
            api_call_count += 1
            if result is not None:
                logger.info(
                    "L1 arXiv ID [%s] %s → %s (%.2f)",
                    key,
                    arxiv_id,
                    result.status.value,
                    result.confidence,
                )

        # L3b: S2 title search — last resort fallback
        if result is None:
            result = verify_by_title_search(title, s2_api_key=s2_api_key)
            api_call_count += 1
            if result is not None:
                logger.info(
                    "L3b S2 [%s] %r → %s (%.2f)",
                    key,
                    title[:50],
                    result.status.value,
                    result.confidence,
                )

        # Fallback: all layers failed (network issues)
        if result is None:
            result = CitationResult(
                cite_key=key,
                title=title,
                status=VerifyStatus.SKIPPED,
                confidence=0.0,
                method="skipped",
                details="All verification methods failed (network error?)",
            )


        result = CitationResult(
            cite_key=key,
            title=result.title,
            status=result.status,
            confidence=result.confidence,
            method=result.method,
            details=result.details,
            matched_paper=result.matched_paper,
        )

        # Cache the result (skip SKIPPED — network failures shouldn't be cached)
        if result.status != VerifyStatus.SKIPPED:
            _write_cache(title, result)

        if result.status == VerifyStatus.VERIFIED:
            report.verified += 1
        elif result.status == VerifyStatus.SUSPICIOUS:
            report.suspicious += 1
        elif result.status == VerifyStatus.HALLUCINATED:
            report.hallucinated += 1
        else:
            report.skipped += 1

        report.results.append(result)

    return report


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------


def filter_verified_bibtex(
    bib_text: str,
    report: VerificationReport,
    *,
    include_suspicious: bool = True,
) -> str:
    """Return a cleaned BibTeX string with only verified entries.

    Parameters
    ----------
    bib_text:
        Original BibTeX string.
    report:
        Verification report from ``verify_citations()``.
    include_suspicious:
        If True, keep SUSPICIOUS entries.  If False, only keep VERIFIED.
    """
    # Build set of keys to keep
    keep_keys: set[str] = set()
    for r in report.results:
        if r.status == VerifyStatus.VERIFIED:
            keep_keys.add(r.cite_key)
        elif r.status == VerifyStatus.SUSPICIOUS and include_suspicious:
            keep_keys.add(r.cite_key)
        elif r.status == VerifyStatus.SKIPPED:
            keep_keys.add(r.cite_key)  # keep unverifiable entries

    # Rebuild BibTeX keeping only entries whose keys are in keep_keys
    kept: list[str] = []
    for m in _ENTRY_RE.finditer(bib_text):
        key = m.group(2).strip()
        if key in keep_keys:
            kept.append(m.group(0))

    return "\n\n".join(kept) + "\n" if kept else ""


def annotate_paper_hallucinations(
    paper_text: str,
    report: VerificationReport,
) -> str:
    """Remove hallucinated citations from paper text.

    - HALLUCINATED citations: removed from text (recorded in verification report)
    - SUSPICIOUS/VERIFIED/SKIPPED: left as-is

    Works with both ``\\cite{key}`` (LaTeX) and ``[key]`` (Markdown) formats.
    """
    hallucinated_keys: set[str] = set()
    for r in report.results:
        if r.status == VerifyStatus.HALLUCINATED:
            hallucinated_keys.add(r.cite_key)

    if not hallucinated_keys:
        return paper_text

    result = paper_text

    # Handle \cite{key1, key2} format — remove only hallucinated keys
    def _replace_latex(m: re.Match[str]) -> str:
        keys = [k.strip() for k in m.group(1).split(",")]
        kept = [k for k in keys if k not in hallucinated_keys]
        if not kept:
            return ""  # All keys hallucinated — remove entire cite
        return "\\cite{" + ", ".join(kept) + "}"

    result = re.sub(r"\\cite\{([^}]+)\}", _replace_latex, result)

    # Handle [key1, key2] and [key1; key2] format (Markdown multi-key)
    _CITE_KEY_PAT = r"[a-zA-Z]+\d{4}[a-zA-Z]*"

    def _replace_markdown_multi(m: re.Match[str]) -> str:
        keys = [k.strip() for k in re.split(r"[,;]\s*", m.group(1))]
        kept = [k for k in keys if k not in hallucinated_keys]
        if not kept:
            return ""
        return "[" + ", ".join(kept) + "]"

    result = re.sub(
        rf"\[({_CITE_KEY_PAT}(?:\s*[,;]\s*{_CITE_KEY_PAT})*)\]",
        _replace_markdown_multi,
        result,
    )

    # Clean up artifacts: double spaces, empty parenthetical citations, orphan punctuation
    result = re.sub(r" {2,}", " ", result)
    result = re.sub(r"\(\s*\)", "", result)
    result = re.sub(r"\[\s*\]", "", result)

    return result
