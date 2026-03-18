"""Semantic Scholar API client.

Uses only stdlib ``urllib`` — zero extra dependencies.

Public API
----------
- ``search_semantic_scholar(query, limit, year_min)`` → ``list[Paper]``

Rate limit: 1 req/s (free, no API key).  Retries up to 3 times with
exponential back-off on transient failures.

Circuit breaker has three states:
  CLOSED → normal operation
  OPEN   → skip all requests, auto-recover after cooldown
  HALF_OPEN → try one probe request, success→CLOSED, fail→OPEN (doubled cooldown)
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from researchclaw.literature.models import Author, Paper

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "paperId,title,abstract,year,venue,citationCount,authors,externalIds,url"
_MAX_PER_REQUEST = 100
_RATE_LIMIT_SEC = 1.5  # conservative spacing between requests
_MAX_RETRIES = 3
_MAX_WAIT_SEC = 60
_TIMEOUT_SEC = 30

# ---------------------------------------------------------------------------
# Three-state circuit breaker
# ---------------------------------------------------------------------------

_CB_THRESHOLD = 3           # consecutive 429s to trip
_CB_INITIAL_COOLDOWN = 120  # seconds before first HALF_OPEN probe
_CB_MAX_COOLDOWN = 600      # cap cooldown at 10 minutes

# States
_CB_CLOSED = "closed"
_CB_OPEN = "open"
_CB_HALF_OPEN = "half_open"

_cb_state: str = _CB_CLOSED
_cb_consecutive_429s: int = 0
_cb_cooldown_sec: float = _CB_INITIAL_COOLDOWN
_cb_open_since: float = 0.0  # monotonic timestamp when breaker opened
_cb_trip_count: int = 0      # total number of trips in this process
_cb_lock = threading.Lock()


def _reset_circuit_breaker() -> None:
    """Reset circuit breaker state (for tests)."""
    global _cb_state, _cb_consecutive_429s, _cb_cooldown_sec  # noqa: PLW0603
    global _cb_open_since, _cb_trip_count  # noqa: PLW0603
    with _cb_lock:
        _cb_state = _CB_CLOSED
        _cb_consecutive_429s = 0
        _cb_cooldown_sec = _CB_INITIAL_COOLDOWN
        _cb_open_since = 0.0
        _cb_trip_count = 0


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
                    "S2 circuit breaker → HALF_OPEN after %.0fs cooldown. "
                    "Trying one probe request...",
                    elapsed,
                )
                return True
            return False
        # HALF_OPEN: allow the probe
        return True


def _cb_on_success() -> None:
    """Record a successful request."""
    global _cb_state, _cb_consecutive_429s, _cb_cooldown_sec  # noqa: PLW0603
    with _cb_lock:
        _cb_consecutive_429s = 0
        if _cb_state != _CB_CLOSED:
            logger.info("S2 circuit breaker → CLOSED (request succeeded)")
            _cb_state = _CB_CLOSED
            _cb_cooldown_sec = _CB_INITIAL_COOLDOWN  # reset cooldown


def _cb_on_429() -> bool:
    """Record a 429 response. Returns True if breaker is now OPEN."""
    global _cb_state, _cb_consecutive_429s, _cb_cooldown_sec  # noqa: PLW0603
    global _cb_open_since, _cb_trip_count  # noqa: PLW0603
    with _cb_lock:
        _cb_consecutive_429s += 1

        if _cb_state == _CB_HALF_OPEN:
            # Probe failed — back to OPEN with doubled cooldown
            _cb_cooldown_sec = min(_cb_cooldown_sec * 2, _CB_MAX_COOLDOWN)
            _cb_state = _CB_OPEN
            _cb_open_since = time.monotonic()
            _cb_trip_count += 1
            logger.warning(
                "S2 circuit breaker → OPEN (probe failed). "
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
                "S2 circuit breaker TRIPPED after %d consecutive 429s. "
                "Cooldown: %.0fs (trip #%d). arXiv still active.",
                _cb_consecutive_429s,
                _cb_cooldown_sec,
                _cb_trip_count,
            )
            return True
        return False


# Last request timestamp for rate limiting
_last_request_time: float = 0.0
_rate_lock = threading.Lock()


def search_semantic_scholar(
    query: str,
    *,
    limit: int = 20,
    year_min: int = 0,
    api_key: str = "",
) -> list[Paper]:
    """Search Semantic Scholar for papers matching *query*.

    Parameters
    ----------
    query:
        Free-text search query.
    limit:
        Maximum number of results (capped at 100 per API constraint).
    year_min:
        If >0, restrict to papers published in this year or later.
    api_key:
        Optional S2 API key (raises rate limit to 10 req/s).

    Returns
    -------
    list[Paper]
        Parsed papers.  Empty list on network failure.
    """
    global _last_request_time  # noqa: PLW0603

    # Rate limiting: locked to serialize concurrent callers
    with _rate_lock:
        now = time.monotonic()
        rate_limit = 0.3 if api_key else _RATE_LIMIT_SEC
        elapsed_since_last = now - _last_request_time
        if elapsed_since_last < rate_limit:
            time.sleep(rate_limit - elapsed_since_last)
        _last_request_time = time.monotonic()

    limit = min(limit, _MAX_PER_REQUEST)
    params: dict[str, str] = {
        "query": query,
        "limit": str(limit),
        "fields": _FIELDS,
    }
    if year_min > 0:
        params["year"] = f"{year_min}-"

    url = f"{_BASE_URL}?{urllib.parse.urlencode(params)}"

    headers: dict[str, str] = {"Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    data = _request_with_retry(url, headers)
    if data is None:
        return []

    raw_papers = data.get("data", [])
    if not isinstance(raw_papers, list):
        return []

    papers: list[Paper] = []
    for item in raw_papers:
        try:
            papers.append(_parse_s2_paper(item))
        except Exception:  # noqa: BLE001
            logger.debug("Failed to parse S2 paper entry: %s", item)
    return papers


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _request_with_retry(
    url: str,
    headers: dict[str, str],
) -> dict[str, Any] | None:
    """GET *url* with exponential back-off retries."""
    if not _cb_should_allow():
        return None

    for attempt in range(_MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=_TIMEOUT_SEC) as resp:
                body = resp.read().decode("utf-8")
                _cb_on_success()
                return json.loads(body)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                if _cb_on_429():
                    return None  # breaker tripped
                delay = min(2 ** (attempt + 1), _MAX_WAIT_SEC)
                jitter = random.uniform(0, delay * 0.3)
                wait = delay + jitter
                logger.warning(
                    "S2 rate-limited (429). Waiting %.1fs (attempt %d/%d)...",
                    wait,
                    attempt + 1,
                    _MAX_RETRIES,
                )
                time.sleep(wait)
                continue
            logger.warning("S2 HTTP %d for %s", exc.code, url)
            return None
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            wait = min(2**attempt, _MAX_WAIT_SEC)
            jitter = random.uniform(0, wait * 0.2)
            logger.warning(
                "S2 request failed (%s). Retry %d/%d in %ds \u2026",
                exc,
                attempt + 1,
                _MAX_RETRIES,
                wait,
            )
            time.sleep(wait + jitter)
    logger.error("S2 request exhausted retries for: %s", url)
    return None


_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
_BATCH_MAX = 500  # S2 batch endpoint max


def batch_fetch_papers(
    paper_ids: list[str],
    *,
    api_key: str = "",
    fields: str = _FIELDS,
) -> list[Paper]:
    """Batch fetch paper details via POST /graph/v1/paper/batch.

    Accepts S2 paper IDs, arXiv IDs (prefixed ``ARXIV:``), or DOIs.
    Returns parsed papers; silently skips papers that fail to resolve.
    """
    if not paper_ids:
        return []

    if not _cb_should_allow():
        return []

    global _last_request_time  # noqa: PLW0603
    rate = 0.3 if api_key else _RATE_LIMIT_SEC
    with _rate_lock:
        now = time.monotonic()
        elapsed = now - _last_request_time
        if elapsed < rate:
            time.sleep(rate - elapsed)
        _last_request_time = time.monotonic()

    all_papers: list[Paper] = []

    # Process in chunks of _BATCH_MAX
    for i in range(0, len(paper_ids), _BATCH_MAX):
        chunk = paper_ids[i : i + _BATCH_MAX]
        url = f"{_BATCH_URL}?fields={fields}"

        headers: dict[str, str] = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if api_key:
            headers["x-api-key"] = api_key

        body = json.dumps({"ids": chunk}).encode("utf-8")

        result = _post_with_retry(url, headers, body)
        with _rate_lock:
            _last_request_time = time.monotonic()
        if result is None:
            continue

        for item in result:
            if item is None:
                continue  # unresolved ID
            try:
                all_papers.append(_parse_s2_paper(item))
            except Exception:  # noqa: BLE001
                logger.debug("Failed to parse batch S2 paper entry")

        # Delay between chunks (sleep outside lock to avoid contention)
        if i + _BATCH_MAX < len(paper_ids):
            time.sleep(rate)
            with _rate_lock:
                _last_request_time = time.monotonic()

    return all_papers


def _post_with_retry(
    url: str,
    headers: dict[str, str],
    body: bytes,
) -> list[dict[str, Any]] | None:
    """POST *url* with exponential back-off retries."""
    if not _cb_should_allow():
        return None

    for attempt in range(_MAX_RETRIES):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=_TIMEOUT_SEC) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                _cb_on_success()
                return data if isinstance(data, list) else None
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                if _cb_on_429():
                    return None
                delay = min(2 ** (attempt + 1), _MAX_WAIT_SEC)
                jitter = random.uniform(0, delay * 0.3)
                logger.warning(
                    "S2 batch rate-limited (429). Waiting %.1fs (attempt %d/%d)...",
                    delay + jitter,
                    attempt + 1,
                    _MAX_RETRIES,
                )
                time.sleep(delay + jitter)
                continue
            logger.warning("S2 batch HTTP %d", exc.code)
            return None
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            wait = min(2**attempt, _MAX_WAIT_SEC)
            jitter = random.uniform(0, wait * 0.2)
            logger.warning(
                "S2 batch request failed (%s). Retry %d/%d in %ds…",
                exc,
                attempt + 1,
                _MAX_RETRIES,
                wait,
            )
            time.sleep(wait + jitter)

    logger.error("S2 batch request exhausted retries")
    return None


def _parse_s2_paper(item: dict[str, Any]) -> Paper:
    """Convert a single Semantic Scholar JSON entry to a ``Paper``."""
    ext_ids = item.get("externalIds") or {}
    authors_raw = item.get("authors") or []
    authors = tuple(
        Author(name=a.get("name", "Unknown"))
        for a in authors_raw
        if isinstance(a, dict)
    )
    return Paper(
        paper_id=f"s2-{item.get('paperId', '')}",
        title=str(item.get("title", "")).strip(),
        authors=authors,
        year=int(item.get("year") or 0),
        abstract=str(item.get("abstract") or "").strip(),
        venue=str(item.get("venue") or "").strip(),
        citation_count=int(item.get("citationCount") or 0),
        doi=str(ext_ids.get("DOI") or "").strip(),
        arxiv_id=str(ext_ids.get("ArXiv") or "").strip(),
        url=str(item.get("url") or "").strip(),
        source="semantic_scholar",
    )
