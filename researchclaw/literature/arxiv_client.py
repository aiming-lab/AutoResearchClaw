"""arXiv API client powered by the ``arxiv`` library.

The ``arxiv`` pip package (2.4+) provides robust arXiv search with
built-in rate limiting, retries, pagination, and PDF download support.

Public API
----------
- ``search_arxiv(query, limit, sort_by, year_min)`` → ``list[Paper]``
- ``download_pdf(arxiv_id, dirpath)`` → ``Path | None``
- ``get_paper_by_id(arxiv_id)`` → ``Paper | None``

Circuit breaker is preserved for extra resilience beyond the library's
built-in retry logic.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

try:
    import arxiv  # pip install arxiv
except ImportError:
    arxiv = None  # type: ignore[assignment]

from researchclaw.literature.models import Author, Paper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Circuit breaker (kept for extra safety on top of arxiv library retries)
# ---------------------------------------------------------------------------

_CB_THRESHOLD = 3
_CB_INITIAL_COOLDOWN = 180
_CB_MAX_COOLDOWN = 600

_CB_CLOSED = "closed"
_CB_OPEN = "open"
_CB_HALF_OPEN = "half_open"

_cb_state: str = _CB_CLOSED
_cb_consecutive_429s: int = 0
_cb_cooldown_sec: float = _CB_INITIAL_COOLDOWN
_cb_open_since: float = 0.0
_cb_trip_count: int = 0
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
    global _cb_state  # noqa: PLW0603
    with _cb_lock:
        if _cb_state == _CB_CLOSED:
            return True
        if _cb_state == _CB_OPEN:
            elapsed = time.monotonic() - _cb_open_since
            if elapsed >= _cb_cooldown_sec:
                _cb_state = _CB_HALF_OPEN
                logger.info("arXiv circuit breaker → HALF_OPEN (%.0fs cooldown elapsed)", elapsed)
                return True
            return False
        return True  # HALF_OPEN: allow probe


def _cb_on_success() -> None:
    global _cb_state, _cb_consecutive_429s, _cb_cooldown_sec  # noqa: PLW0603
    with _cb_lock:
        _cb_consecutive_429s = 0
        if _cb_state != _CB_CLOSED:
            logger.info("arXiv circuit breaker → CLOSED (request succeeded)")
            _cb_state = _CB_CLOSED
            _cb_cooldown_sec = _CB_INITIAL_COOLDOWN


def _cb_on_failure() -> bool:
    global _cb_state, _cb_consecutive_429s, _cb_cooldown_sec  # noqa: PLW0603
    global _cb_open_since, _cb_trip_count  # noqa: PLW0603
    with _cb_lock:
        _cb_consecutive_429s += 1
        if _cb_state == _CB_HALF_OPEN or _cb_consecutive_429s >= _CB_THRESHOLD:
            if _cb_state == _CB_HALF_OPEN:
                _cb_cooldown_sec = min(_cb_cooldown_sec * 2, _CB_MAX_COOLDOWN)
            _cb_state = _CB_OPEN
            _cb_open_since = time.monotonic()
            _cb_trip_count += 1
            logger.warning(
                "arXiv circuit breaker TRIPPED (trip #%d, cooldown %.0fs)",
                _cb_trip_count, _cb_cooldown_sec,
            )
            return True
        return False


# ---------------------------------------------------------------------------
# Shared arxiv.Client instance (reuses connection, respects rate limits)
# ---------------------------------------------------------------------------

_client: arxiv.Client | None = None


def _get_client() -> arxiv.Client:
    """Get or create the shared arxiv Client."""
    global _client  # noqa: PLW0603
    if _client is None:
        _client = arxiv.Client(
            page_size=100,       # fetch up to 100 per API call
            delay_seconds=3.1,   # arXiv requires ≥3s between requests
            num_retries=3,       # built-in retry on failure
        )
    return _client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_arxiv(
    query: str,
    *,
    limit: int = 50,
    sort_by: str = "relevance",
    year_min: int = 0,
) -> list[Paper]:
    """Search arXiv for papers matching *query*.

    Parameters
    ----------
    query:
        Free-text search query. Supports arXiv field syntax
        (e.g., ``ti:transformer``, ``au:vaswani``, ``cat:cs.LG``).
    limit:
        Maximum number of results (up to 300).
    sort_by:
        Sort criterion: "relevance", "submitted_date", or "last_updated".
    year_min:
        If > 0, only return papers published in this year or later.

    Returns
    -------
    list[Paper]
        Parsed papers. Empty list on failure.
    """
    if arxiv is None:
        logger.warning("arxiv library not installed — skipping arXiv search")
        return []
    if not _cb_should_allow():
        logger.info("[rate-limit] arXiv circuit breaker OPEN — skipping")
        return []

    limit = min(limit, 300)

    sort_map = {
        "relevance": arxiv.SortCriterion.Relevance,
        "submitted_date": arxiv.SortCriterion.SubmittedDate,
        "last_updated": arxiv.SortCriterion.LastUpdatedDate,
    }
    criterion = sort_map.get(sort_by, arxiv.SortCriterion.Relevance)

    search = arxiv.Search(
        query=query,
        max_results=limit,
        sort_by=criterion,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers: list[Paper] = []
    try:
        client = _get_client()
        for result in client.results(search):
            paper = _convert_result(result)
            if year_min > 0 and paper.year < year_min:
                continue
            papers.append(paper)
        _cb_on_success()
        logger.info("arXiv: found %d papers for %r", len(papers), query)
    except arxiv.HTTPError as exc:
        logger.warning("arXiv HTTP error: %s", exc)
        _cb_on_failure()
    except arxiv.UnexpectedEmptyPageError:
        logger.warning("arXiv returned unexpected empty page for %r", query)
        _cb_on_failure()
    except Exception as exc:  # noqa: BLE001
        logger.warning("arXiv search failed: %s", exc)
        _cb_on_failure()

    return papers


def get_paper_by_id(arxiv_id: str) -> Paper | None:
    """Fetch a single paper by arXiv ID (e.g., '2301.00001')."""
    if arxiv is None:
        logger.warning("arxiv library not installed — cannot look up %s", arxiv_id)
        return None
    try:
        search = arxiv.Search(id_list=[arxiv_id])
        client = _get_client()
        for result in client.results(search):
            return _convert_result(result)
    except Exception as exc:  # noqa: BLE001
        logger.warning("arXiv ID lookup failed for %s: %s", arxiv_id, exc)
    return None


def download_pdf(
    arxiv_id: str,
    dirpath: str | Path = ".",
    filename: str = "",
) -> Path | None:
    """Download PDF for a given arXiv ID.

    Parameters
    ----------
    arxiv_id:
        arXiv paper ID (e.g., '2301.00001').
    dirpath:
        Directory to save the PDF.
    filename:
        Custom filename. If empty, uses ``{arxiv_id}.pdf``.

    Returns
    -------
    Path | None
        Path to downloaded PDF, or None on failure.
    """
    if arxiv is None:
        logger.warning("arxiv library not installed — cannot download PDF")
        return None
    try:
        search = arxiv.Search(id_list=[arxiv_id])
        client = _get_client()
        for result in client.results(search):
            dirpath = Path(dirpath)
            dirpath.mkdir(parents=True, exist_ok=True)
            fname = filename or f"{arxiv_id.replace('/', '_')}.pdf"
            result.download_pdf(dirpath=str(dirpath), filename=fname)
            pdf_path = dirpath / fname
            logger.info("Downloaded arXiv PDF: %s → %s", arxiv_id, pdf_path)
            return pdf_path
    except Exception as exc:  # noqa: BLE001
        logger.warning("PDF download failed for %s: %s", arxiv_id, exc)
    return None


def search_arxiv_advanced(
    *,
    title: str = "",
    author: str = "",
    abstract: str = "",
    category: str = "",
    limit: int = 50,
    year_min: int = 0,
) -> list[Paper]:
    """Advanced arXiv search using field-specific queries.

    Example: search_arxiv_advanced(title="transformer", category="cs.LG")
    """
    parts = []
    if title:
        parts.append(f"ti:{title}")
    if author:
        parts.append(f"au:{author}")
    if abstract:
        parts.append(f"abs:{abstract}")
    if category:
        parts.append(f"cat:{category}")

    if not parts:
        return []

    query = " AND ".join(parts)
    return search_arxiv(query, limit=limit, year_min=year_min)


# ---------------------------------------------------------------------------
# Internal: convert arxiv.Result → Paper
# ---------------------------------------------------------------------------


def _convert_result(result: arxiv.Result) -> Paper:
    """Convert an ``arxiv.Result`` to our ``Paper`` dataclass."""
    # Extract arXiv ID from entry_id URL
    arxiv_id = ""
    if result.entry_id:
        m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?$", result.entry_id)
        if m:
            arxiv_id = m.group(1)

    # Authors
    authors = tuple(Author(name=a.name) for a in result.authors)

    # Year from published date
    year = result.published.year if result.published else 0

    # DOI
    doi = result.doi or ""

    # Primary category as venue
    venue = result.primary_category or ""

    # Prefer HTML abstract URL
    url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else result.entry_id

    return Paper(
        paper_id=f"arxiv-{arxiv_id}" if arxiv_id else f"arxiv-{result.entry_id}",
        title=result.title or "",
        authors=authors,
        year=year,
        abstract=result.summary or "",
        venue=venue,
        citation_count=0,  # arXiv doesn't provide citation counts
        doi=doi,
        arxiv_id=arxiv_id,
        url=url,
        source="arxiv",
    )
