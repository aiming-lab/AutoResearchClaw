"""Literature trend analysis — analyze trends from search results."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class LiteratureTrendAnalyzer:
    """Analyze trends from literature search results."""

    def __init__(self, search_client: Any = None):
        self.client = search_client

    def get_daily_papers(
        self,
        domains: list[str],
        max_papers: int = 20,
    ) -> list[dict[str, Any]]:
        """Get today's most relevant papers via literature search."""
        if self.client is None:
            return []

        try:
            from researchclaw.literature.search import search_papers

            query = " OR ".join(domains) if domains else "machine learning"
            papers = search_papers(query, limit=max_papers)
            return [
                {
                    "title": p.title,
                    "authors": [a.name for a in p.authors],
                    "abstract": p.abstract or "",
                    "url": p.url or "",
                    "year": p.year,
                    "citation_count": p.citation_count,
                    "source": p.source,
                }
                for p in papers
            ]
        except Exception as exc:
            logger.warning("Literature trend fetch failed: %s", exc)
            return []

    def analyze_keyword_trends(
        self,
        domains: list[str],
        window_days: int = 30,
    ) -> dict[str, Any]:
        """Analyze keyword frequency trends."""
        papers = self.get_daily_papers(domains)
        if not papers:
            return {"keywords": [], "total_papers": 0}

        from researchclaw.trends.trend_analyzer import TrendAnalyzer

        analyzer = TrendAnalyzer()
        analysis = analyzer.analyze(papers, window_days)
        return {
            "keywords": analysis.get("rising_keywords", []),
            "total_papers": len(papers),
            "methods": analysis.get("method_trends", []),
        }

    def find_emerging_topics(
        self,
        domains: list[str],
    ) -> list[dict[str, Any]]:
        """Discover emerging research directions."""
        papers = self.get_daily_papers(domains, max_papers=50)
        if not papers:
            return []

        from researchclaw.trends.trend_analyzer import TrendAnalyzer

        analyzer = TrendAnalyzer()
        analysis = analyzer.analyze(papers)
        keywords = analysis.get("rising_keywords", [])

        # Emerging topics = high-frequency bigrams
        emerging = [
            {
                "topic": kw["keyword"],
                "frequency": kw["count"],
                "type": kw.get("type", "unigram"),
            }
            for kw in keywords
            if kw.get("type") == "bigram" and kw["count"] >= 3
        ]
        return emerging[:10]
