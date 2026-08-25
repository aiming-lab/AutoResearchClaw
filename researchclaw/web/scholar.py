"""Google Scholar search powered by the ``scholarly`` library.

scholarly is installed as a dependency and provides direct access to
Google Scholar search, citation graph traversal, and author lookup.

When a Serply API key is configured, paper search goes through Serply's
Scholar endpoint (a keyed REST API, https://serply.io/docs) first and
only falls back to scholarly scraping if that call fails.

Usage::

    client = GoogleScholarClient()
    papers = client.search("attention is all you need", limit=5)
    citing = client.get_citations(papers[0].scholar_id, limit=10)

    client = GoogleScholarClient(serply_api_key="...")
    papers = client.search("attention is all you need", limit=5)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    from scholarly import scholarly, ProxyGenerator
    HAS_SCHOLARLY = True
except ImportError:
    scholarly = None  # type: ignore[assignment]
    ProxyGenerator = None  # type: ignore[assignment,misc]
    HAS_SCHOLARLY = False

logger = logging.getLogger(__name__)

SERPLY_SCHOLAR_URL = "https://api.serply.io/v1/scholar/"
# Serply sits behind Cloudflare and rejects requests without a User-Agent.
SERPLY_USER_AGENT = "researchclaw (+https://github.com/aiming-lab/AutoResearchClaw)"
_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
_TRAILING_YEAR_PATTERN = re.compile(r",?\s*\b((?:19|20)\d{2})\s*$")


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
    serply_api_key:
        Serply API key. Falls back to ``SERPLY_API_KEY`` env var. When
        set, ``search()`` uses Serply's Scholar endpoint before scholarly.
    """

    def __init__(
        self,
        *,
        inter_request_delay: float = 2.0,
        use_proxy: bool = False,
        serply_api_key: str = "",
    ) -> None:
        self.serply_api_key = serply_api_key or os.environ.get("SERPLY_API_KEY", "")
        if not HAS_SCHOLARLY and not self.serply_api_key:
            raise ImportError(
                "scholarly is required for Google Scholar search. "
                "Install: pip install 'researchclaw[web]' "
                "(or set SERPLY_API_KEY to use Serply's Scholar API instead)"
            )
        self.delay = inter_request_delay
        self._last_request_time: float = 0.0

        if use_proxy and HAS_SCHOLARLY:
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
        if self.serply_api_key:
            try:
                return self._search_serply(query, limit=limit)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Serply Scholar search failed: %s", exc)
                if not HAS_SCHOLARLY:
                    return []

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
    # Serply Scholar backend (keyed REST API, stdlib only)
    # ------------------------------------------------------------------

    def _search_serply(self, query: str, *, limit: int = 10) -> list[ScholarPaper]:
        """Search via Serply's Scholar endpoint."""
        params = {"q": query, "num": max(1, min(limit, 100))}
        req = Request(f"{SERPLY_SCHOLAR_URL}?{urlencode(params)}", headers={
            "X-Api-Key": self.serply_api_key,
            "Accept": "application/json",
            "User-Agent": SERPLY_USER_AGENT,
        })
        resp = urlopen(req, timeout=15)  # noqa: S310
        payload = json.loads(resp.read().decode("utf-8"))

        results: list[ScholarPaper] = []
        for article in payload.get("articles", [])[:limit]:
            paper = self._parse_serply_article(article)
            if paper.title:
                results.append(paper)

        logger.info("Serply Scholar: found %d papers for %r", len(results), query)
        return results

    @staticmethod
    def _parse_serply_article(article: dict[str, Any]) -> ScholarPaper:
        """Parse one Serply ``articles[]`` entry into a ScholarPaper.

        Serply returns ``author.authors[]`` plus a ``description`` line of
        the form ``"A Author, B Author - Venue, 2021"``; venue and year are
        recovered from that line.
        """
        author_info = article.get("author") or {}
        authors = [
            str(a.get("name", "")).strip()
            for a in author_info.get("authors", [])
            if isinstance(a, dict) and a.get("name")
        ]
        description = str(article.get("description") or author_info.get("names") or "")

        venue = ""
        year = 0
        meta = description.rsplit(" - ", 1)[1] if " - " in description else ""
        if meta:
            trailing_year = _TRAILING_YEAR_PATTERN.search(meta)
            if trailing_year:
                year = int(trailing_year.group(1))
                venue = meta[: trailing_year.start()].strip(" ,")
            else:
                venue = meta.strip(" ,")
                any_year = _YEAR_PATTERN.search(meta)
                if any_year:
                    year = int(any_year.group(0))
        if not authors and " - " in description:
            authors = [a.strip() for a in description.rsplit(" - ", 1)[0].split(",") if a.strip()]

        doc = article.get("doc") or {}
        citations = (article.get("extras") or {}).get("citations") or {}
        try:
            citation_count = int(citations.get("count", 0))
        except (TypeError, ValueError):
            citation_count = 0

        return ScholarPaper(
            title=str(article.get("title", "")).strip(),
            authors=authors,
            year=year,
            abstract=str(article.get("snippet", "")),
            citation_count=citation_count,
            url=str(article.get("link") or doc.get("link") or ""),
            scholar_id=str(article.get("id", "")),
            venue=venue,
        )

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
