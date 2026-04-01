"""Unified Web Search Agent.

Orchestrates all web capabilities (Tavily, Google Scholar, Crawl4AI,
PDF extraction) into a single search-and-extract pipeline.

Usage::

    agent = WebSearchAgent()
    result = agent.search_and_extract(
        topic="knowledge distillation for vision transformers",
        search_queries=["knowledge distillation survey", "ViT compression"],
    )
    # result.papers — Google Scholar papers
    # result.web_results — Tavily/DDG web search results
    # result.crawled_pages — full-text from crawled URLs
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from researchclaw.web.crawler import CrawlResult, WebCrawler
from researchclaw.web.pdf_extractor import PDFContent, PDFExtractor
from researchclaw.web.scholar import GoogleScholarClient, ScholarPaper
from researchclaw.web.search import SearchResult, WebSearchClient, WebSearchResponse

logger = logging.getLogger(__name__)


@dataclass
class WebSearchAgentResult:
    """Combined result from all web search sources."""

    topic: str
    web_results: list[SearchResult] = field(default_factory=list)
    scholar_papers: list[ScholarPaper] = field(default_factory=list)
    crawled_pages: list[CrawlResult] = field(default_factory=list)
    pdf_extractions: list[PDFContent] = field(default_factory=list)
    search_answer: str = ""  # Tavily AI answer if available
    elapsed_seconds: float = 0.0

    @property
    def total_results(self) -> int:
        return (
            len(self.web_results)
            + len(self.scholar_papers)
            + len(self.crawled_pages)
            + len(self.pdf_extractions)
        )

    def to_context_string(self, *, max_length: int = 30_000) -> str:
        """Convert all results to a single context string for LLM injection.

        The output is structured Markdown suitable for prompt injection.
        """
        parts: list[str] = []

        # Tavily AI answer
        if self.search_answer:
            parts.append("## AI Search Summary")
            parts.append(self.search_answer)
            parts.append("")

        # Web search results
        if self.web_results:
            parts.append("## Web Search Results")
            for i, r in enumerate(self.web_results[:15], 1):
                parts.append(f"### [{i}] {r.title}")
                parts.append(f"URL: {r.url}")
                if r.snippet:
                    parts.append(r.snippet)
                parts.append("")

        # Google Scholar papers
        if self.scholar_papers:
            parts.append("## Google Scholar Papers")
            for i, p in enumerate(self.scholar_papers[:10], 1):
                authors = ", ".join(p.authors[:3])
                if len(p.authors) > 3:
                    authors += " et al."
                parts.append(
                    f"- **{p.title}** ({authors}, {p.year}) "
                    f"[{p.citation_count} citations]"
                )
                if p.abstract:
                    parts.append(f"  {p.abstract[:200]}...")
            parts.append("")

        # Crawled page content
        if self.crawled_pages:
            parts.append("## Crawled Page Content")
            for cr in self.crawled_pages:
                if cr.has_content:
                    parts.append(f"### {cr.title or cr.url}")
                    parts.append(cr.markdown[:3000])
                    parts.append("")

        # PDF extractions
        if self.pdf_extractions:
            parts.append("## PDF Full-Text Extractions")
            for pdf in self.pdf_extractions:
                if pdf.has_content:
                    label = pdf.title or pdf.path
                    parts.append(f"### {label}")
                    if pdf.abstract:
                        parts.append(f"**Abstract:** {pdf.abstract}")
                    parts.append(pdf.text[:3000])
                    parts.append("")

        result = "\n".join(parts)
        if len(result) > max_length:
            result = result[:max_length] + "\n\n[... truncated]"
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "topic": self.topic,
            "web_results_count": len(self.web_results),
            "scholar_papers_count": len(self.scholar_papers),
            "crawled_pages_count": len(self.crawled_pages),
            "pdf_extractions_count": len(self.pdf_extractions),
            "has_search_answer": bool(self.search_answer),
            "elapsed_seconds": self.elapsed_seconds,
            "web_results": [r.to_dict() for r in self.web_results[:20]],
            "scholar_papers": [p.to_dict() for p in self.scholar_papers[:20]],
        }


class WebSearchAgent:
    """Orchestrates all web search and content extraction capabilities.

    Parameters
    ----------
    tavily_api_key:
        Tavily API key (optional, falls back to env var or DuckDuckGo).
    enable_scholar:
        Whether to include Google Scholar search.
    enable_crawling:
        Whether to crawl top URLs for full content.
    enable_pdf:
        Whether to extract PDF content.
    max_web_results:
        Maximum web search results.
    max_scholar_results:
        Maximum Google Scholar results.
    max_crawl_urls:
        Maximum URLs to crawl for full content.
    """

    def __init__(
        self,
        *,
        tavily_api_key: str = "",
        enable_scholar: bool = True,
        enable_crawling: bool = True,
        enable_pdf: bool = True,
        max_web_results: int = 10,
        max_scholar_results: int = 10,
        max_crawl_urls: int = 5,
    ) -> None:
        self.web_client = WebSearchClient(api_key=tavily_api_key)
        try:
            self.scholar_client = GoogleScholarClient()
        except ImportError:
            self.scholar_client = None  # type: ignore[assignment]
        self.crawler = WebCrawler()
        self.pdf_extractor = PDFExtractor()
        self.enable_scholar = enable_scholar
        self.enable_crawling = enable_crawling
        self.enable_pdf = enable_pdf
        self.max_web_results = max_web_results
        self.max_scholar_results = max_scholar_results
        self.max_crawl_urls = max_crawl_urls

    def search_and_extract(
        self,
        topic: str,
        *,
        search_queries: list[str] | None = None,
        crawl_urls: list[str] | None = None,
        pdf_urls: list[str] | None = None,
    ) -> WebSearchAgentResult:
        """Run the full search + extraction pipeline.

        Parameters
        ----------
        topic:
            Research topic string.
        search_queries:
            Custom search queries. If None, auto-generates from topic.
        crawl_urls:
            Specific URLs to crawl. If None, crawls top search result URLs.
        pdf_urls:
            Specific PDF URLs to extract. If None, extracts PDFs from search.
        """
        t0 = time.monotonic()
        result = WebSearchAgentResult(topic=topic)

        # 1. Generate search queries if not provided
        if search_queries is None:
            search_queries = self._generate_queries(topic)

        # 2. Web search (Tavily / DuckDuckGo)
        self._run_web_search(result, search_queries)

        # 3. Google Scholar search
        if self.enable_scholar and self.scholar_client and self.scholar_client.available:
            self._run_scholar_search(result, topic)

        # 4. Crawl top URLs for full content
        if self.enable_crawling:
            urls_to_crawl = crawl_urls or self._select_urls_to_crawl(result)
            if urls_to_crawl:
                self._run_crawling(result, urls_to_crawl)

        # 5. Extract PDFs
        if self.enable_pdf:
            pdf_targets = pdf_urls or self._find_pdf_urls(result)
            if pdf_targets:
                self._run_pdf_extraction(result, pdf_targets)

        result.elapsed_seconds = time.monotonic() - t0
        logger.info(
            "[WebSearchAgent] Done: %d web, %d scholar, %d crawled, %d PDFs (%.1fs)",
            len(result.web_results),
            len(result.scholar_papers),
            len(result.crawled_pages),
            len(result.pdf_extractions),
            result.elapsed_seconds,
        )
        return result

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _run_web_search(
        self, result: WebSearchAgentResult, queries: list[str]
    ) -> None:
        """Run web search across all queries."""
        try:
            responses = self.web_client.search_multi(
                queries, max_results=self.max_web_results
            )
            for resp in responses:
                result.web_results.extend(resp.results)
                if resp.answer and not result.search_answer:
                    result.search_answer = resp.answer
        except Exception as exc:  # noqa: BLE001
            logger.warning("Web search failed: %s", exc)

    def _run_scholar_search(
        self, result: WebSearchAgentResult, topic: str
    ) -> None:
        """Run Google Scholar search."""
        try:
            papers = self.scholar_client.search(
                topic, limit=self.max_scholar_results
            )
            result.scholar_papers.extend(papers)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Scholar search failed: %s", exc)

    def _run_crawling(
        self, result: WebSearchAgentResult, urls: list[str]
    ) -> None:
        """Crawl URLs for full content."""
        try:
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

            if loop and loop.is_running():
                # We're inside an async context — use sync fallback
                for url in urls[: self.max_crawl_urls]:
                    cr = self.crawler.crawl_sync(url)
                    if cr.has_content:
                        result.crawled_pages.append(cr)
            else:
                crawl_results = asyncio.run(
                    self.crawler.crawl_many(urls[: self.max_crawl_urls])
                )
                result.crawled_pages.extend(
                    cr for cr in crawl_results if cr.has_content
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Crawling failed: %s", exc)

    def _run_pdf_extraction(
        self, result: WebSearchAgentResult, urls: list[str]
    ) -> None:
        """Extract text from PDF URLs."""
        for url in urls[:5]:
            try:
                pdf = self.pdf_extractor.extract_from_url(url)
                if pdf.has_content:
                    result.pdf_extractions.append(pdf)
            except Exception as exc:  # noqa: BLE001
                logger.warning("PDF extraction failed for %s: %s", url, exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_queries(topic: str) -> list[str]:
        """Generate search queries from a topic string."""
        queries = [
            topic,
            f"{topic} survey",
            f"{topic} benchmark state of the art",
        ]
        return queries

    def _select_urls_to_crawl(self, result: WebSearchAgentResult) -> list[str]:
        """Select top URLs from search results for crawling."""
        urls = []
        seen = set()
        for r in result.web_results:
            if r.url and r.url not in seen:
                # Skip PDF URLs (handled separately) and common non-content sites
                if r.url.endswith(".pdf"):
                    continue
                seen.add(r.url)
                urls.append(r.url)
                if len(urls) >= self.max_crawl_urls:
                    break
        return urls

    @staticmethod
    def _find_pdf_urls(result: WebSearchAgentResult) -> list[str]:
        """Find PDF URLs from search results."""
        pdf_urls = []
        seen = set()
        for r in result.web_results:
            if r.url and r.url.endswith(".pdf") and r.url not in seen:
                seen.add(r.url)
                pdf_urls.append(r.url)
                if len(pdf_urls) >= 3:
                    break
        return pdf_urls
