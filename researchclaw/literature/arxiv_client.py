"""arXiv API client.

Uses stdlib ``urllib`` + ``xml.etree`` — zero extra dependencies.

Public API
----------
- ``search_arxiv(query, limit)`` → ``list[Paper]``

Rate limit: arXiv requests 3-second gaps between calls.  Retries up to
3 times with exponential back-off on transient failures.

Circuit breaker has three states:
  CLOSED    → normal operation
  OPEN      → skip all requests, auto-recover after cooldown
  HALF_OPEN → try one probe request, success→CLOSED, fail→OPEN (doubled cooldown)
"""

from __future__ import annotations

import logging
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

from researchclaw.literature.models import Author, Paper

logger = logging.getLogger(__name__)

_BASE_URL = "https://export.arxiv.org/api/query"
_MAX_RESULTS = 50
_RATE_LIMIT_SEC = 3.1  # arXiv asks for ≥3 s between requests
_RATE_LIMIT_ELEVATED = 5.0  # raised after 429 to be extra conservative
_MAX_RETRIES = 3
_MAX_WAIT_SEC = 60
_TIMEOUT_SEC = 30

# ---------------------------------------------------------------------------
# Three-state circuit breaker  (mirrors S2 breaker in semantic_scholar.py)
# ---------------------------------------------------------------------------

_CB_THRESHOLD = 3           # consecutive 429s to trip
_CB_INITIAL_COOLDOWN = 180  # seconds before first HALF_OPEN probe (3 min)
_CB_MAX_COOLDOWN = 600      # cap cooldown at 10 minutes

_CB_CLOSED = "closed"
_CB_OPEN = "open"
_CB_HALF_OPEN = "half_open"

_cb_state: str = _CB_CLOSED
_cb_consecutive_429s: int = 0
_cb_cooldown_sec: float = _CB_INITIAL_COOLDOWN
_cb_open_since: float = 0.0
_cb_trip_count: int = 0
_rate_elevated: bool = False  # temporarily use slower rate after 429
_cb_lock = threading.Lock()


def _reset_circuit_breaker() -> None:
    """Reset circuit breaker state (for tests)."""
    global _cb_state, _cb_consecutive_429s, _cb_cooldown_sec  # noqa: PLW0603
    global _cb_open_since, _cb_trip_count, _rate_elevated  # noqa: PLW0603
    with _cb_lock:
        _cb_state = _CB_CLOSED
        _cb_consecutive_429s = 0
        _cb_cooldown_sec = _CB_INITIAL_COOLDOWN
        _cb_open_since = 0.0
        _cb_trip_count = 0
        _rate_elevated = False


def _cb_should_allow() -> bool:
    """Check if circuit breaker allows a request."""
    global _cb_state  # noqa: PLW0603
    with _cb_lock:
        if _cb_state == _CB_CLOSED:
            return True
        if _cb_state == _CB_OPEN:
            elapsed = time.monotonic() - _cb_open_since
            if elapsed >= _cb_cooldown_sec:
                _cb_state = _CB_HALF_OPEN
                logger.info(
                    "arXiv circuit breaker → HALF_OPEN after %.0fs cooldown. "
                    "Trying one probe request...",
                    elapsed,
                )
                return True
            logger.debug(
                "arXiv circuit breaker OPEN — %.0fs remaining",
                _cb_cooldown_sec - elapsed,
            )
            return False
        # HALF_OPEN: allow the probe
        return True


def _cb_on_success() -> None:
    """Record a successful request."""
    global _cb_state, _cb_consecutive_429s, _cb_cooldown_sec  # noqa: PLW0603
    global _rate_elevated  # noqa: PLW0603
    with _cb_lock:
        _cb_consecutive_429s = 0
        if _cb_state != _CB_CLOSED:
            logger.info("arXiv circuit breaker → CLOSED (request succeeded)")
            _cb_state = _CB_CLOSED
            _cb_cooldown_sec = _CB_INITIAL_COOLDOWN
        _rate_elevated = False  # restore normal rate on success


def _cb_on_429() -> bool:
    """Record a 429 response. Returns True if breaker is now OPEN."""
    global _cb_state, _cb_consecutive_429s, _cb_cooldown_sec  # noqa: PLW0603
    global _cb_open_since, _cb_trip_count, _rate_elevated  # noqa: PLW0603
    with _cb_lock:
        _cb_consecutive_429s += 1
        _rate_elevated = True  # slow down future requests

        if _cb_state == _CB_HALF_OPEN:
            _cb_cooldown_sec = min(_cb_cooldown_sec * 2, _CB_MAX_COOLDOWN)
            _cb_state = _CB_OPEN
            _cb_open_since = time.monotonic()
            _cb_trip_count += 1
            logger.warning(
                "arXiv circuit breaker → OPEN (probe failed). "
                "Next cooldown: %.0fs (trip #%d)",
                _cb_cooldown_sec,
                _cb_trip_count,
            )
            return True

        if _cb_consecutive_429s >= _CB_THRESHOLD:
            _cb_state = _CB_OPEN
            _cb_open_since = time.monotonic()
            _cb_trip_count += 1
            logger.warning(
                "arXiv circuit breaker TRIPPED after %d consecutive 429s. "
                "Cooldown: %.0fs (trip #%d). Other sources still active.",
                _cb_consecutive_429s,
                _cb_cooldown_sec,
                _cb_trip_count,
            )
            return True
        return False


# Last request timestamp for rate limiting
_last_request_time: float = 0.0
_rate_lock = threading.Lock()

# Atom XML namespaces
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def search_arxiv(
    query: str,
    *,
    limit: int = 20,
) -> list[Paper]:
    """Search arXiv for papers matching *query*.

    Parameters
    ----------
    query:
        Free-text search query (mapped to ``all:`` arXiv field).
    limit:
        Maximum number of results (capped at 50).

    Returns
    -------
    list[Paper]
        Parsed papers.  Empty list on network failure.
    """
    global _last_request_time  # noqa: PLW0603

    # Rate limiting: locked to serialize concurrent callers
    with _rate_lock:
        now = time.monotonic()
        with _cb_lock:
            rate_elevated = _rate_elevated
        rate = _RATE_LIMIT_ELEVATED if rate_elevated else _RATE_LIMIT_SEC
        elapsed_since_last = now - _last_request_time
        if elapsed_since_last < rate:
            time.sleep(rate - elapsed_since_last)
        _last_request_time = time.monotonic()

    limit = min(limit, _MAX_RESULTS)
    params = {
        "search_query": f"all:{query}",
        "start": "0",
        "max_results": str(limit),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{_BASE_URL}?{urllib.parse.urlencode(params)}"
    xml_text = _fetch_with_retry(url)
    if xml_text is None:
        return []

    return _parse_atom_feed(xml_text)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _fetch_with_retry(url: str) -> str | None:
    """GET *url* returning raw text, with retries.

    Handles HTTP 429 explicitly:
      - Parses ``Retry-After`` header when present
      - Notifies circuit breaker on 429 responses
      - Falls back to exponential back-off if no header
    """
    if not _cb_should_allow():
        logger.info("[rate-limit] arXiv circuit breaker OPEN — skipping request")
        return None

    for attempt in range(_MAX_RETRIES):
        try:
            req = urllib.request.Request(
                url, headers={"Accept": "application/atom+xml"}
            )
            with urllib.request.urlopen(req, timeout=_TIMEOUT_SEC) as resp:
                body = resp.read().decode("utf-8")
                _cb_on_success()
                return body
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                # Parse Retry-After header if present
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                if retry_after:
                    try:
                        wait = float(retry_after)
                    except (ValueError, TypeError):
                        wait = _RATE_LIMIT_ELEVATED * (2**attempt)
                else:
                    wait = _RATE_LIMIT_ELEVATED * (2**attempt)
                wait = min(wait, _MAX_WAIT_SEC)
                jitter = random.uniform(0, wait * 0.2)

                if _cb_on_429():
                    logger.warning(
                        "[rate-limit] arXiv 429 — circuit breaker OPEN. "
                        "All arXiv requests paused for %.0fs.",
                        _cb_cooldown_sec,
                    )
                    return None

                logger.warning(
                    "[rate-limit] arXiv 429 (Retry-After: %s). "
                    "Waiting %.1fs (attempt %d/%d)...",
                    retry_after or "none",
                    wait + jitter,
                    attempt + 1,
                    _MAX_RETRIES,
                )
                time.sleep(wait + jitter)
                continue

            if exc.code == 503:
                # Service unavailable — transient, retry
                wait = _RATE_LIMIT_SEC * (2**attempt)
                jitter = random.uniform(0, wait * 0.2)
                logger.warning(
                    "arXiv 503 (service unavailable). "
                    "Retry %d/%d in %.0fs...",
                    attempt + 1,
                    _MAX_RETRIES,
                    wait + jitter,
                )
                time.sleep(wait + jitter)
                continue

            # Other HTTP errors — not retryable
            logger.warning("arXiv HTTP %d for %s", exc.code, url)
            return None

        except (urllib.error.URLError, OSError) as exc:
            wait = min(_RATE_LIMIT_SEC * (2**attempt), _MAX_WAIT_SEC)
            jitter = random.uniform(0, wait * 0.2)
            logger.warning(
                "arXiv request failed (%s). Retry %d/%d in %.0fs…",
                exc,
                attempt + 1,
                _MAX_RETRIES,
                wait,
            )
            time.sleep(wait + jitter)

    logger.error("arXiv request exhausted retries for: %s", url)
    return None


def _parse_atom_feed(xml_text: str) -> list[Paper]:
    """Parse arXiv Atom XML feed into ``Paper`` objects."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        logger.error("Failed to parse arXiv Atom XML")
        return []

    papers: list[Paper] = []
    for entry in root.findall("atom:entry", _NS):
        try:
            papers.append(_parse_entry(entry))
        except Exception:  # noqa: BLE001
            logger.debug("Failed to parse arXiv entry")
    return papers


def _text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return (element.text or "").strip()


def _parse_entry(entry: ET.Element) -> Paper:
    """Convert a single Atom <entry> to a ``Paper``."""
    title = re.sub(r"\s+", " ", _text(entry.find("atom:title", _NS)))
    abstract = re.sub(r"\s+", " ", _text(entry.find("atom:summary", _NS)))

    # Authors
    authors = tuple(
        Author(name=_text(a.find("atom:name", _NS)))
        for a in entry.findall("atom:author", _NS)
    )

    # arXiv ID from the <id> element (e.g. "http://arxiv.org/abs/2301.00001v2")
    raw_id = _text(entry.find("atom:id", _NS))
    arxiv_id = ""
    m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?$", raw_id)
    if m:
        arxiv_id = m.group(1)

    # Year from <published>
    published = _text(entry.find("atom:published", _NS))
    year = 0
    if published:
        ym = re.match(r"(\d{4})", published)
        if ym:
            year = int(ym.group(1))

    # DOI (may be absent)
    doi_el = entry.find("arxiv:doi", _NS)
    doi = _text(doi_el) if doi_el is not None else ""

    # Primary category
    primary = entry.find("arxiv:primary_category", _NS)
    venue = ""
    if primary is not None:
        venue = primary.get("term", "")

    # URL — prefer abs link
    url = ""
    for link in entry.findall("atom:link", _NS):
        if link.get("type") == "text/html":
            url = link.get("href", "")
            break
    if not url:
        url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else raw_id

    return Paper(
        paper_id=f"arxiv-{arxiv_id}" if arxiv_id else f"arxiv-{raw_id}",
        title=title,
        authors=authors,
        year=year,
        abstract=abstract,
        venue=venue,
        citation_count=0,  # arXiv doesn't provide citation counts
        doi=doi,
        arxiv_id=arxiv_id,
        url=url,
        source="arxiv",
    )
