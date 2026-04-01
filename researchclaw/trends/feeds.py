"""ArXiv / Semantic Scholar / OpenAlex feed management."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

logger = logging.getLogger(__name__)


class FeedManager:
    """Manage paper feeds from multiple sources."""

    SUPPORTED_SOURCES = ("arxiv", "semantic_scholar", "openalex")

    def __init__(
        self,
        sources: tuple[str, ...] = ("arxiv", "semantic_scholar"),
        s2_api_key: str = "",
    ):
        self.sources = tuple(
            s for s in sources if s in self.SUPPORTED_SOURCES
        )
        self.s2_api_key = s2_api_key

    def fetch_recent_papers(
        self,
        domains: list[str],
        max_papers: int = 20,
        since_date: date | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch recent papers from configured sources.

        Returns a list of paper dicts with: title, authors, abstract,
        url, source, published_date, domains.
        """
        all_papers: list[dict[str, Any]] = []
        target_date = since_date or date.today()

        for source in self.sources:
            try:
                if source == "arxiv":
                    papers = self._fetch_arxiv(domains, max_papers, target_date)
                elif source == "semantic_scholar":
                    papers = self._fetch_s2(domains, max_papers, target_date)
                elif source == "openalex":
                    papers = self._fetch_openalex(domains, max_papers, target_date)
                else:
                    continue
                all_papers.extend(papers)
            except Exception as exc:
                logger.warning("Feed fetch failed for %s: %s", source, exc)

        # Deduplicate by title similarity
        seen_titles: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for paper in all_papers:
            norm_title = paper.get("title", "").lower().strip()
            if norm_title and norm_title not in seen_titles:
                seen_titles.add(norm_title)
                deduped.append(paper)

        return deduped[:max_papers]

    def _fetch_arxiv(
        self,
        domains: list[str],
        max_papers: int,
        since_date: date,
    ) -> list[dict[str, Any]]:
        """Fetch papers from arXiv API."""
        try:
            from researchclaw.literature.arxiv_client import search_arxiv
        except ImportError:
            logger.debug("arxiv_client not available")
            return []

        query = " OR ".join(domains) if domains else "machine learning"
        try:
            results = search_arxiv(query, limit=max_papers)
            return [
                {
                    "title": r.get("title", ""),
                    "authors": r.get("authors", []),
                    "abstract": r.get("abstract", ""),
                    "url": r.get("url", ""),
                    "source": "arxiv",
                    "published_date": r.get("published", since_date.isoformat()),
                    "arxiv_id": r.get("arxiv_id", ""),
                }
                for r in results
            ]
        except Exception as exc:
            logger.warning("ArXiv fetch failed: %s", exc)
            return []

    def _fetch_s2(
        self,
        domains: list[str],
        max_papers: int,
        since_date: date,
    ) -> list[dict[str, Any]]:
        """Fetch papers from Semantic Scholar API."""
        try:
            from researchclaw.literature.semantic_scholar import search_s2
        except ImportError:
            logger.debug("semantic_scholar client not available")
            return []

        query = " ".join(domains) if domains else "machine learning"
        try:
            results = search_s2(
                query,
                limit=max_papers,
                year_min=since_date.year,
                api_key=self.s2_api_key,
            )
            return [
                {
                    "title": r.get("title", ""),
                    "authors": [
                        a.get("name", "") for a in r.get("authors", [])
                    ],
                    "abstract": r.get("abstract", ""),
                    "url": r.get("url", ""),
                    "source": "semantic_scholar",
                    "published_date": str(r.get("year", since_date.year)),
                    "citation_count": r.get("citationCount", 0),
                }
                for r in results
            ]
        except Exception as exc:
            logger.warning("S2 fetch failed: %s", exc)
            return []

    def _fetch_openalex(
        self,
        domains: list[str],
        max_papers: int,
        since_date: date,
    ) -> list[dict[str, Any]]:
        """Fetch papers from OpenAlex API."""
        try:
            from researchclaw.literature.openalex_client import search_openalex
        except ImportError:
            logger.debug("openalex_client not available")
            return []

        query = " ".join(domains) if domains else "machine learning"
        try:
            results = search_openalex(query, limit=max_papers)
            return [
                {
                    "title": r.get("title", ""),
                    "authors": r.get("authors", []),
                    "abstract": r.get("abstract", ""),
                    "url": r.get("url", ""),
                    "source": "openalex",
                    "published_date": r.get("publication_date", ""),
                    "citation_count": r.get("cited_by_count", 0),
                }
                for r in results
            ]
        except Exception as exc:
            logger.warning("OpenAlex fetch failed: %s", exc)
            return []
