"""Unified literature search with deduplication.

Combines results from OpenAlex, Semantic Scholar, arXiv, and a suite of
open free-access sources. Deduplicates by DOI -> arXiv ID -> fuzzy title
match, and returns a merged list sorted by citation count (descending).

All sources are open, require no API key, and need no registration.

Sources
-------
- openalex, semantic_scholar, arxiv  (original three)
- crossref, europepmc, hal, datacite, scielo, inspirehep, dblp, jstage

Public API
----------
- ``search_papers(query, limit, sources, year_min, deduplicate)``
  -> ``list[Paper]``
- ``search_papers_parallel(query, limit, sources, year_min)``
  -> ``list[Paper]``  (3-5x faster for 8+ sources)
- ``search_papers_multi_query_parallel(queries, limit, sources, year_min)``
  -> ``list[Paper]``  (fan-out across queries and sources)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
import importlib
import logging
import re
import time
import urllib.error
from typing import cast

from researchclaw.literature.arxiv_client import search_arxiv
from researchclaw.literature.models import Author, Paper
from researchclaw.literature.openalex_client import search_openalex
from researchclaw.literature.query_adapter import adapt_query
from researchclaw.literature.semantic_scholar import search_semantic_scholar

# Open sources, lazy-imported so missing optional deps don't break search.py.
_NEW_SOURCE_IMPORTS: dict[str, tuple[str, str]] = {
    "crossref":    ("researchclaw.literature.crossref_client",   "search_crossref"),
    "europepmc":   ("researchclaw.literature.europepmc_client",  "search_europepmc"),
    "hal":         ("researchclaw.literature.hal_client",        "search_hal"),
    "datacite":    ("researchclaw.literature.datacite_client",   "search_datacite"),
    "scielo":      ("researchclaw.literature.scielo_client",     "search_scielo"),
    "inspirehep":  ("researchclaw.literature.inspirehep_client", "search_inspirehep"),
    "dblp":        ("researchclaw.literature.dblp_client",       "search_dblp"),
    "jstage":      ("researchclaw.literature.jstage_client",     "search_jstage"),
}

logger = logging.getLogger(__name__)

# Original three sources (kept as a named constant for tests and explicit opt-out).
_ORIGINAL_SOURCES: tuple[str, ...] = (
    "openalex", "semantic_scholar", "arxiv",
)
# All open sources: original 3 + 8 Tier 1 additions (no key, no signup).
_EXTENDED_SOURCES: tuple[str, ...] = _ORIGINAL_SOURCES + (
    "crossref", "europepmc", "hal", "datacite",
    "scielo", "inspirehep", "dblp", "jstage",
)
# Default: all Tier 0 + Tier 1 sources. Each client handles its own errors
# gracefully (returns [] on failure), so including all sources by default is
# safe and maximizes literature coverage.
_DEFAULT_SOURCES: tuple[str, ...] = _EXTENDED_SOURCES


CacheGet = Callable[[str, str, int], list[dict[str, object]] | None]
CachePut = Callable[[str, str, int, list[dict[str, object]]], None]


def _cache_api() -> tuple[CacheGet, CachePut]:
    cache_mod = importlib.import_module("researchclaw.literature.cache")
    return cast(CacheGet, cache_mod.get_cached), cast(CachePut, cache_mod.put_cache)


def _papers_to_dicts(papers: list[Paper]) -> list[dict[str, object]]:
    """Convert papers to serializable dicts for caching."""
    return [asdict(p) for p in papers]


def _as_int(value: object, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _dicts_to_papers(dicts: list[dict[str, object]]) -> list[Paper]:
    """Reconstruct Paper objects from cached dicts."""
    papers: list[Paper] = []
    for d in dicts:
        try:
            authors_raw = d.get("authors", ())
            if not isinstance(authors_raw, list):
                authors_raw = []
            authors = tuple(
                Author(
                    name=str(cast(dict[str, object], a).get("name", "")),
                    affiliation=str(cast(dict[str, object], a).get("affiliation", "")),
                )
                for a in authors_raw
                if isinstance(a, dict)
            )
            paper_id = cast(str, d["paper_id"])
            title = cast(str, d["title"])
            papers.append(
                Paper(
                    paper_id=paper_id,
                    title=title,
                    authors=authors,
                    year=_as_int(d.get("year", 0), 0),
                    abstract=str(d.get("abstract", "")),
                    venue=str(d.get("venue", "")),
                    citation_count=_as_int(d.get("citation_count", 0), 0),
                    doi=str(d.get("doi", "")),
                    arxiv_id=str(d.get("arxiv_id", "")),
                    url=str(d.get("url", "")),
                    source=str(d.get("source", "")),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return papers


def search_papers(
    query: str,
    *,
    limit: int = 20,
    sources: Sequence[str] = _DEFAULT_SOURCES,
    year_min: int = 0,
    deduplicate: bool = True,
    s2_api_key: str = "",
) -> list[Paper]:
    """Search multiple academic sources and return deduplicated results.

    Parameters
    ----------
    query:
        Free-text search query.
    limit:
        Maximum results *per source*.
    sources:
        Which backends to query.  Default: both S2 and arXiv.
    year_min:
        If >0, pass to backends that support year filtering.
    deduplicate:
        Whether to remove duplicates across sources.
    s2_api_key:
        Optional Semantic Scholar API key.

    Returns
    -------
    list[Paper]
        Merged results, sorted by citation_count descending.
    """
    all_papers: list[Paper] = []
    cache_get: CacheGet
    cache_put: CachePut
    cache_get, cache_put = _cache_api()

    source_stats: dict[str, int] = {}  # track per-source counts
    cache_hits = 0

    for src in sources:
        src_lower = src.lower().replace("-", "_").replace(" ", "_")
        # Adapt query for this source's syntax only -- year filtering is the
        # client's responsibility via its native params (avoids double-filtering).
        adapted_query = adapt_query(query, src_lower, 0)
        cache_source = (
            "semantic_scholar" if src_lower in ("semantic_scholar", "s2") else src_lower
        )
        # Cache key incorporates year_min so different year ranges don't collide.
        cache_key = f"{query}@{year_min}" if year_min else query
        try:
            if src_lower == "openalex":
                papers = search_openalex(
                    adapted_query,
                    limit=limit,
                    year_min=year_min,
                )
                all_papers.extend(papers)
                cache_put(cache_key, "openalex", limit, _papers_to_dicts(papers))
                source_stats["openalex"] = len(papers)
                logger.info(
                    "OpenAlex returned %d papers for %r", len(papers), adapted_query
                )
                time.sleep(0.5)

            elif src_lower in ("semantic_scholar", "s2"):
                papers = search_semantic_scholar(
                    adapted_query,
                    limit=limit,
                    year_min=year_min,
                    api_key=s2_api_key,
                )
                all_papers.extend(papers)
                cache_put(cache_key, "semantic_scholar", limit, _papers_to_dicts(papers))
                source_stats["semantic_scholar"] = len(papers)
                logger.info(
                    "Semantic Scholar returned %d papers for %r", len(papers), adapted_query
                )
                # Rate-limit gap before next source
                time.sleep(1.0)

            elif src_lower == "arxiv":
                papers = search_arxiv(adapted_query, limit=limit, year_min=year_min)
                all_papers.extend(papers)
                cache_put(cache_key, "arxiv", limit, _papers_to_dicts(papers))
                source_stats["arxiv"] = len(papers)
                logger.info("arXiv returned %d papers for %r", len(papers), adapted_query)

            elif src_lower in _NEW_SOURCE_IMPORTS:
                mod_path, func_name = _NEW_SOURCE_IMPORTS[src_lower]
                mod = importlib.import_module(mod_path)
                search_fn = getattr(mod, func_name)
                kwargs: dict[str, object] = {"limit": limit}
                if year_min:
                    kwargs["year_min"] = year_min
                papers = search_fn(adapted_query, **kwargs)
                all_papers.extend(papers)
                cache_put(cache_key, src_lower, limit, _papers_to_dicts(papers))
                source_stats[src_lower] = len(papers)
                logger.info(
                    "%s returned %d papers for %r", src_lower, len(papers), adapted_query
                )
                time.sleep(0.5)

            else:
                logger.warning("Unknown literature source: %s (skipped)", src)
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            ImportError,
            ModuleNotFoundError,
            AttributeError,
            urllib.error.HTTPError,
            urllib.error.URLError,
        ):
            logger.warning(
                "[rate-limit] Source %s failed for %r — trying cache", src, query
            )
            cached = cache_get(cache_key, cache_source, limit)
            if cached:
                papers = _dicts_to_papers(cached)
                all_papers.extend(papers)
                cache_hits += len(papers)
                logger.info(
                    "[cache] HIT: %d papers for %s/%r", len(papers), src, query
                )
            else:
                logger.warning(
                    "No cache available for %s/%r — skipping", src, query
                )

    # Summary log
    total = len(all_papers)
    parts = [f"{src}: {n}" for src, n in source_stats.items()]
    if cache_hits:
        parts.append(f"cache: {cache_hits}")
    logger.info(
        "[literature] Found %d papers (%s) for %r",
        total,
        ", ".join(parts) if parts else "none",
        query,
    )

    if deduplicate:
        all_papers = _deduplicate(all_papers)

    # Sort by citation count descending, then year descending
    all_papers.sort(key=lambda p: (p.citation_count, p.year), reverse=True)

    return all_papers


def search_papers_multi_query(
    queries: list[str],
    *,
    limit_per_query: int = 20,
    sources: Sequence[str] = _DEFAULT_SOURCES,
    year_min: int = 0,
    s2_api_key: str = "",
    inter_query_delay: float = 1.5,
) -> list[Paper]:
    """Run multiple queries and return deduplicated union.

    Adds a delay between queries to respect rate limits.
    """
    all_papers: list[Paper] = []

    for i, q in enumerate(queries):
        if i > 0:
            time.sleep(inter_query_delay)
        results = search_papers(
            q,
            limit=limit_per_query,
            sources=sources,
            year_min=year_min,
            s2_api_key=s2_api_key,
            deduplicate=False,  # we dedup globally below
        )
        all_papers.extend(results)
        logger.info("Query %d/%d %r → %d papers", i + 1, len(queries), q, len(results))

    deduped = _deduplicate(all_papers)
    deduped.sort(key=lambda p: (p.citation_count, p.year), reverse=True)
    return deduped


def search_papers_parallel(
    query: str,
    *,
    limit: int = 20,
    sources: Sequence[str] = _DEFAULT_SOURCES,
    year_min: int = 0,
    deduplicate: bool = True,
    s2_api_key: str = "",
    max_workers: int = 6,
) -> list[Paper]:
    """Search multiple sources in parallel using ThreadPoolExecutor.

    Sources with independent rate limits are queried concurrently.
    Each source still respects its own internal rate limiting.
    Falls back to serial search_papers() on any threading error.

    Typical speedup: 3-5x for 11 sources (from ~25s serial to ~7s parallel).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _search_one_source(src: str) -> tuple[str, list[Paper]]:
        """Search a single source. Returns (source_name, papers)."""
        try:
            # Use the serial search with a single source
            return src, search_papers(
                query,
                limit=limit,
                sources=(src,),
                year_min=year_min,
                deduplicate=False,
                s2_api_key=s2_api_key,
            )
        except Exception:  # noqa: BLE001
            logger.warning("Parallel search failed for %s", src)
            return src, []

    all_papers: list[Paper] = []
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_search_one_source, src): src
                for src in sources
            }
            for future in as_completed(futures):
                src_name, papers = future.result()
                all_papers.extend(papers)
                if papers:
                    logger.info(
                        "[parallel] %s returned %d papers", src_name, len(papers)
                    )
    except Exception:  # noqa: BLE001
        logger.warning("Parallel search failed, falling back to serial")
        return search_papers(
            query,
            limit=limit,
            sources=sources,
            year_min=year_min,
            deduplicate=deduplicate,
            s2_api_key=s2_api_key,
        )

    if deduplicate:
        all_papers = _deduplicate(all_papers)
    all_papers.sort(key=lambda p: (p.citation_count, p.year), reverse=True)
    return all_papers


def search_papers_multi_query_parallel(
    queries: list[str],
    *,
    limit_per_query: int = 20,
    sources: Sequence[str] = _DEFAULT_SOURCES,
    year_min: int = 0,
    s2_api_key: str = "",
    max_workers: int = 6,
) -> list[Paper]:
    """Run multiple queries with parallel source fetching per query.

    Each query fans out to all sources in parallel, then queries
    run sequentially (to avoid overwhelming APIs with N*M concurrent).
    """
    all_papers: list[Paper] = []

    for i, q in enumerate(queries):
        results = search_papers_parallel(
            q,
            limit=limit_per_query,
            sources=sources,
            year_min=year_min,
            deduplicate=False,
            s2_api_key=s2_api_key,
            max_workers=max_workers,
        )
        all_papers.extend(results)
        logger.info(
            "Query %d/%d %r -> %d papers (parallel)", i + 1, len(queries), q, len(results)
        )

    deduped = _deduplicate(all_papers)
    deduped.sort(key=lambda p: (p.citation_count, p.year), reverse=True)
    return deduped


# ------------------------------------------------------------------
# Deduplication
# ------------------------------------------------------------------


def _normalise_title(title: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    t = title.lower()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _deduplicate(papers: list[Paper]) -> list[Paper]:
    """Remove duplicates.  Priority: DOI > arXiv ID > fuzzy title.

    When a duplicate is found, the entry with higher citation_count wins
    (i.e. Semantic Scholar data is preferred over arXiv-only data).
    """
    seen_doi: dict[str, int] = {}
    seen_arxiv: dict[str, int] = {}
    seen_title: dict[str, int] = {}
    result: list[Paper] = []

    def _update_indices(p: Paper, idx: int) -> None:
        """Register all identifiers of *p* in the lookup dicts at *idx*."""
        if p.doi:
            seen_doi[p.doi.lower().strip()] = idx
        if p.arxiv_id:
            seen_arxiv[p.arxiv_id.strip()] = idx
        norm = _normalise_title(p.title)
        if norm:
            seen_title[norm] = idx

    for paper in papers:
        is_dup = False

        # Check DOI
        if paper.doi:
            doi_key = paper.doi.lower().strip()
            if doi_key in seen_doi:
                idx = seen_doi[doi_key]
                if paper.citation_count > result[idx].citation_count:
                    result[idx] = paper
                    _update_indices(paper, idx)
                is_dup = True

        # Check arXiv ID
        if not is_dup and paper.arxiv_id:
            ax_key = paper.arxiv_id.strip()
            if ax_key in seen_arxiv:
                idx = seen_arxiv[ax_key]
                if paper.citation_count > result[idx].citation_count:
                    result[idx] = paper
                    _update_indices(paper, idx)
                is_dup = True

        # Check fuzzy title
        if not is_dup:
            norm = _normalise_title(paper.title)
            if norm and norm in seen_title:
                idx = seen_title[norm]
                if paper.citation_count > result[idx].citation_count:
                    result[idx] = paper
                    _update_indices(paper, idx)
                is_dup = True

        if is_dup:
            continue

        # Not a duplicate — store indices and append
        new_idx = len(result)
        _update_indices(paper, new_idx)
        result.append(paper)

    return result


def papers_to_bibtex(papers: Sequence[Paper]) -> str:
    """Generate a combined BibTeX file from a list of papers."""
    entries = [p.to_bibtex() for p in papers]
    return "\n\n".join(entries) + "\n"
