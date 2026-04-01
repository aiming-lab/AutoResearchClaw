"""Google Scholar search powered by the ``scholarly`` library.

scholarly is installed as a dependency and provides direct access to
Google Scholar search, citation graph traversal, and author lookup.

Usage::

    client = GoogleScholarClient()
    papers = client.search("attention is all you need", limit=5)
    citing = client.get_citations(papers[0].scholar_id, limit=10)
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

try:
    from scholarly import scholarly, ProxyGenerator
    HAS_SCHOLARLY = True
except ImportError:
    scholarly = None  # type: ignore[assignment]
    ProxyGenerator = None  # type: ignore[assignment,misc]
    HAS_SCHOLARLY = False

logger = logging.getLogger(__name__)


@dataclass
class ScholarPaper:
    """A paper result from Google Scholar."""

    title: str
    authors: list[str] = field(default_factory=list)
    year: int = 0
    abstract: str = ""
    citation_count: int = 0
    url: str = ""
    scholar_id: str = ""
    venue: str = ""
    source: str = "google_scholar"

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "abstract": self.abstract,
            "citation_count": self.citation_count,
            "url": self.url,
            "scholar_id": self.scholar_id,
            "venue": self.venue,
            "source": self.source,
        }

    def to_literature_paper(self) -> Any:
        """Convert to researchclaw.literature.models.Paper."""
        from researchclaw.literature.models import Author, Paper
        authors_tuple = tuple(Author(name=a) for a in self.authors)
        return Paper(
            paper_id=self.scholar_id or f"gs-{hashlib.sha256(self.title.encode()).hexdigest()[:8]}",
            title=self.title,
            authors=authors_tuple,
            year=self.year,
            abstract=self.abstract,
            venue=self.venue,
            citation_count=self.citation_count,
            url=self.url,
            source="google_scholar",
        )


class GoogleScholarClient:
    """Google Scholar search client using the ``scholarly`` library.

    Parameters
    ----------
    inter_request_delay:
        Seconds between requests to avoid rate limiting.
    use_proxy:
        Whether to set up a free proxy to reduce blocking risk.
    """

    def __init__(
        self,
        *,
        inter_request_delay: float = 2.0,
        use_proxy: bool = False,
    ) -> None:
        if not HAS_SCHOLARLY:
            raise ImportError(
                "scholarly is required for Google Scholar search. "
                "Install: pip install 'researchclaw[web]'"
            )
        self.delay = inter_request_delay
        self._last_request_time: float = 0.0

        if use_proxy:
            try:
                pg = ProxyGenerator()
                pg.FreeProxies()
                scholarly.use_proxy(pg)
                logger.info("Google Scholar: proxy enabled")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to set up proxy: %s", exc)

    @property
    def available(self) -> bool:
        """Always True — scholarly is installed as a dependency."""
        return True

    def search(self, query: str, *, limit: int = 10) -> list[ScholarPaper]:
        """Search Google Scholar for papers matching query."""
        self._rate_limit()
        results: list[ScholarPaper] = []
        try:
            search_gen = scholarly.search_pubs(query)
            for i, pub in enumerate(search_gen):
                if i >= limit:
                    break
                results.append(self._parse_pub(pub))
                if i < limit - 1:
                    self._rate_limit()

            logger.info("Google Scholar: found %d papers for %r", len(results), query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Google Scholar search failed: %s", exc)

        return results

    def get_citations(self, scholar_id: str, *, limit: int = 20) -> list[ScholarPaper]:
        """Get papers that cite the given paper (citation graph traversal)."""
        self._rate_limit()
        results: list[ScholarPaper] = []
        try:
            pub = scholarly.search_single_pub(scholar_id)
            if pub:
                citations = scholarly.citedby(pub)
                for i, cit in enumerate(citations):
                    if i >= limit:
                        break
                    results.append(self._parse_pub(cit))
                    if i < limit - 1:
                        self._rate_limit()

            logger.info("Google Scholar: found %d citations for %s", len(results), scholar_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Citation retrieval failed for %s: %s", scholar_id, exc)

        return results

    def search_author(self, name: str) -> list[dict[str, Any]]:
        """Search for an author on Google Scholar."""
        self._rate_limit()
        try:
            results = []
            for author in scholarly.search_author(name):
                results.append({
                    "name": author.get("name", ""),
                    "affiliation": author.get("affiliation", ""),
                    "scholar_id": author.get("scholar_id", ""),
                    "citedby": author.get("citedby", 0),
                    "interests": author.get("interests", []),
                })
                if len(results) >= 5:
                    break
            return results
        except Exception as exc:  # noqa: BLE001
            logger.warning("Author search failed for %s: %s", name, exc)
            return []

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request_time = time.monotonic()

    @staticmethod
    def _parse_pub(pub: Any) -> ScholarPaper:
        """Parse a scholarly publication object into ScholarPaper."""
        bib = pub.get("bib", {}) if isinstance(pub, dict) else getattr(pub, "bib", {})
        info = pub if isinstance(pub, dict) else pub.__dict__ if hasattr(pub, "__dict__") else {}

        authors = bib.get("author", [])
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(" and ")]

        year = 0
        year_raw = bib.get("pub_year", bib.get("year", 0))
        try:
            year = int(year_raw)
        except (ValueError, TypeError):
            pass

        cites_id = info.get("cites_id", [])
        scholar_id = info.get("author_pub_id", "") or (
            cites_id[0] if isinstance(cites_id, list) and cites_id else ""
        )

        return ScholarPaper(
            title=bib.get("title", ""),
            authors=authors,
            year=year,
            abstract=bib.get("abstract", ""),
            citation_count=info.get("num_citations", 0),
            url=info.get("pub_url", info.get("eprint_url", "")),
            scholar_id=scholar_id,
            venue=bib.get("venue", bib.get("journal", "")),
        )
