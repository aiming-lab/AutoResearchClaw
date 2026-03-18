"""OpenAlex API client.

Uses stdlib ``urllib`` + ``json`` — zero extra dependencies.

Public API
----------
- ``search_openalex(query, limit, year_min)`` → ``list[Paper]``

Rate limits (with polite pool email):
  - List/filter: 10,000/day
  - Full-text search: 1,000/day

OpenAlex provides generous rate limits and indexes arXiv, PubMed,
CrossRef, and many other sources — making it an excellent primary
search backend that reduces pressure on arXiv and Semantic Scholar.
"""

from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from researchclaw.literature.models import Author, Paper

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.openalex.org/works"
_POLITE_EMAIL = "researchclaw@users.noreply.github.com"
_MAX_PER_REQUEST = 50
_MAX_RETRIES = 3
_MAX_WAIT_SEC = 60
_TIMEOUT_SEC = 20
_RATE_LIMIT_SEC = 0.2  # OpenAlex is generous; 200ms is more than enough


# Last request timestamp for rate limiting
_last_request_time: float = 0.0
_rate_lock = threading.Lock()


def search_openalex(
    query: str,
    *,
    limit: int = 20,
    year_min: int = 0,
    email: str = _POLITE_EMAIL,
) -> list[Paper]:
    """Search OpenAlex for papers matching *query*.

    Parameters
    ----------
    query:
        Free-text search query.
    limit:
        Maximum number of results (capped at 50).
    year_min:
        If >0, restrict to papers published in this year or later.
    email:
        Polite pool email for higher rate limits.

    Returns
    -------
    list[Paper]
        Parsed papers.  Empty list on network failure.
    """
    global _last_request_time  # noqa: PLW0603

    # Rate limiting (locked to serialize concurrent callers)
    with _rate_lock:
        now = time.monotonic()
        elapsed = now - _last_request_time
        if elapsed < _RATE_LIMIT_SEC:
            time.sleep(_RATE_LIMIT_SEC - elapsed)
        _last_request_time = time.monotonic()

    limit = min(limit, _MAX_PER_REQUEST)

    # Build filter string
    filters = []
    if year_min > 0:
        filters.append(f"from_publication_date:{year_min}-01-01")

    params: dict[str, str] = {
        "search": query,
        "per_page": str(limit),
        "mailto": email,
        "select": (
            "id,title,authorships,publication_year,primary_location,"
            "cited_by_count,doi,ids,abstract_inverted_index,type"
        ),
    }
    if filters:
        params["filter"] = ",".join(filters)

    url = f"{_BASE_URL}?{urllib.parse.urlencode(params)}"
    data = _request_with_retry(url, email)
    if data is None:
        return []

    results = data.get("results", [])
    if not isinstance(results, list):
        return []

    papers: list[Paper] = []
    for item in results:
        try:
            papers.append(_parse_openalex_work(item))
        except Exception:  # noqa: BLE001
            logger.debug("Failed to parse OpenAlex work: %s", item.get("id", "?"))
    return papers


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _request_with_retry(
    url: str,
    email: str,
) -> dict[str, Any] | None:
    """GET *url* with exponential back-off retries."""
    for attempt in range(_MAX_RETRIES):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": f"ResearchClaw/1.0 (mailto:{email})",
                },
            )
            with urllib.request.urlopen(req, timeout=_TIMEOUT_SEC) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                if retry_after:
                    try:
                        wait = float(retry_after)
                    except (ValueError, TypeError):
                        wait = 2 ** (attempt + 1)
                else:
                    wait = 2 ** (attempt + 1)
                # BUG-22: If Retry-After is absurdly long (>300s), skip immediately
                if wait > 300:
                    logger.warning(
                        "[rate-limit] OpenAlex Retry-After=%s (>300s). "
                        "Skipping request instead of waiting.",
                        retry_after,
                    )
                    return None
                wait = min(wait, _MAX_WAIT_SEC)
                jitter = random.uniform(0, wait * 0.2)
                logger.warning(
                    "[rate-limit] OpenAlex 429 (Retry-After: %s). "
                    "Waiting %.1fs (attempt %d/%d)...",
                    retry_after or "none",
                    wait + jitter,
                    attempt + 1,
                    _MAX_RETRIES,
                )
                time.sleep(wait + jitter)
                continue

            if exc.code in (500, 502, 503, 504):
                wait = 2 ** attempt
                jitter = random.uniform(0, wait * 0.2)
                logger.warning(
                    "OpenAlex HTTP %d. Retry %d/%d in %.0fs...",
                    exc.code,
                    attempt + 1,
                    _MAX_RETRIES,
                    wait + jitter,
                )
                time.sleep(wait + jitter)
                continue

            logger.warning("OpenAlex HTTP %d for %s", exc.code, url)
            return None

        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            wait = min(2**attempt, _MAX_WAIT_SEC)
            jitter = random.uniform(0, wait * 0.2)
            logger.warning(
                "OpenAlex request failed (%s). Retry %d/%d in %ds...",
                exc,
                attempt + 1,
                _MAX_RETRIES,
                wait,
            )
            time.sleep(wait + jitter)

    logger.error("OpenAlex request exhausted retries for: %s", url)
    return None


def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """Reconstruct abstract from OpenAlex inverted index format."""
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    # Build word -> position mapping
    words: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            words.append((pos, word))
    words.sort(key=lambda x: x[0])
    return " ".join(w for _, w in words)


def _parse_openalex_work(item: dict[str, Any]) -> Paper:
    """Convert a single OpenAlex work JSON to a ``Paper``."""
    # Title
    title = str(item.get("title") or "").strip()
    title = re.sub(r"\s+", " ", title)

    # Authors
    authorships = item.get("authorships") or []
    authors = tuple(
        Author(
            name=str(a.get("author", {}).get("display_name", "Unknown")),
            affiliation=str(
                (a.get("institutions") or [{}])[0].get("display_name", "")
                if a.get("institutions")
                else ""
            ),
        )
        for a in authorships
        if isinstance(a, dict)
    )

    # Year
    year = int(item.get("publication_year") or 0)

    # Abstract (inverted index format)
    abstract = _reconstruct_abstract(item.get("abstract_inverted_index"))

    # Venue from primary_location
    primary_loc = item.get("primary_location") or {}
    source_info = primary_loc.get("source") or {}
    venue = str(source_info.get("display_name") or "").strip()
    # BUG-33: arXiv category codes (e.g. cs.LG, stat.ML) are not proper venue names
    if venue and re.match(r"^[a-z]{2,}\.[A-Z]{2}$", venue):
        venue = ""

    # Citation count
    citation_count = int(item.get("cited_by_count") or 0)

    # DOI
    raw_doi = str(item.get("doi") or "").strip()
    doi = raw_doi.replace("https://doi.org/", "").replace("http://doi.org/", "")

    # IDs
    ids = item.get("ids") or {}
    openalex_id = str(ids.get("openalex") or item.get("id") or "").strip()

    # arXiv ID from ids or DOI
    arxiv_id = ""
    raw_arxiv = str(ids.get("arxiv") or "").strip()
    if raw_arxiv:
        # Extract numeric ID from URLs like https://arxiv.org/abs/2301.00001
        m = re.search(r"(\d{4}\.\d{4,5})", raw_arxiv)
        if m:
            arxiv_id = m.group(1)

    # URL
    url = ""
    if arxiv_id:
        url = f"https://arxiv.org/abs/{arxiv_id}"
    elif doi:
        url = f"https://doi.org/{doi}"
    elif openalex_id:
        url = openalex_id

    # Paper ID
    paper_id = f"oalex-{openalex_id.split('/')[-1]}" if openalex_id else f"oalex-{title[:20]}"

    return Paper(
        paper_id=paper_id,
        title=title,
        authors=authors,
        year=year,
        abstract=abstract,
        venue=venue,
        citation_count=citation_count,
        doi=doi,
        arxiv_id=arxiv_id,
        url=url,
        source="openalex",
    )
