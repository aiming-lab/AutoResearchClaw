"""Integration tests for researchclaw.web — WebSearchAgent end-to-end."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from researchclaw.web.agent import WebSearchAgent, WebSearchAgentResult
from researchclaw.web.crawler import CrawlResult
from researchclaw.web.search import SearchResult, WebSearchResponse
from researchclaw.web.scholar import ScholarPaper


# ---------------------------------------------------------------------------
# WebSearchAgentResult
# ---------------------------------------------------------------------------


class TestWebSearchAgentResult:
    def test_total_results(self):
        r = WebSearchAgentResult(
            topic="test",
            web_results=[SearchResult(title="A", url="u1")],
            scholar_papers=[ScholarPaper(title="B")],
        )
        assert r.total_results == 2

    def test_to_context_string_empty(self):
        r = WebSearchAgentResult(topic="test")
        ctx = r.to_context_string()
        assert isinstance(ctx, str)

    def test_to_context_string_with_results(self):
        r = WebSearchAgentResult(
            topic="knowledge distillation",
            web_results=[
                SearchResult(
                    title="KD Survey",
                    url="https://example.com/kd",
                    snippet="A comprehensive survey on KD",
                    source="tavily",
                ),
            ],
            scholar_papers=[
                ScholarPaper(
                    title="Distilling Knowledge",
                    authors=["Hinton", "Vinyals", "Dean"],
                    year=2015,
                    citation_count=5000,
                    abstract="We propose a technique for model compression.",
                ),
            ],
            search_answer="KD is a model compression technique.",
        )
        ctx = r.to_context_string()
        assert "AI Search Summary" in ctx
        assert "KD Survey" in ctx
        assert "Distilling Knowledge" in ctx
        assert "Hinton" in ctx

    def test_to_context_string_truncation(self):
        r = WebSearchAgentResult(
            topic="test",
            web_results=[
                SearchResult(title=f"R{i}", url=f"u{i}", snippet="x" * 1000)
                for i in range(50)
            ],
        )
        ctx = r.to_context_string(max_length=5000)
        assert len(ctx) <= 5100

    def test_to_dict(self):
        r = WebSearchAgentResult(
            topic="test",
            web_results=[SearchResult(title="A", url="u1")],
        )
        d = r.to_dict()
        assert d["topic"] == "test"
        assert d["web_results_count"] == 1

    def test_to_context_with_crawled_pages(self):
        r = WebSearchAgentResult(
            topic="test",
            crawled_pages=[
                CrawlResult(
                    url="https://blog.example.com",
                    markdown="# Great Blog Post\n\nContent " * 50,
                    title="Great Blog Post",
                    success=True,
                ),
            ],
        )
        ctx = r.to_context_string()
        assert "Crawled Page Content" in ctx
        assert "Great Blog Post" in ctx


# ---------------------------------------------------------------------------
# WebSearchAgent — orchestration
# ---------------------------------------------------------------------------


class TestWebSearchAgent:
    def test_generate_queries(self):
        queries = WebSearchAgent._generate_queries("knowledge distillation")
        assert len(queries) == 3
        assert "knowledge distillation" in queries
        assert any("survey" in q for q in queries)
        assert any("benchmark" in q for q in queries)

    def test_select_urls_to_crawl(self):
        agent = WebSearchAgent(max_crawl_urls=3)
        result = WebSearchAgentResult(
            topic="test",
            web_results=[
                SearchResult(title=f"R{i}", url=f"https://ex.com/{i}")
                for i in range(10)
            ],
        )
        urls = agent._select_urls_to_crawl(result)
        assert len(urls) <= 3
        assert all(url.startswith("https://") for url in urls)

    def test_select_urls_skips_pdf(self):
        agent = WebSearchAgent(max_crawl_urls=5)
        result = WebSearchAgentResult(
            topic="test",
            web_results=[
                SearchResult(title="Paper", url="https://ex.com/paper.pdf"),
                SearchResult(title="Blog", url="https://ex.com/blog"),
            ],
        )
        urls = agent._select_urls_to_crawl(result)
        assert "https://ex.com/paper.pdf" not in urls
        assert "https://ex.com/blog" in urls

    def test_find_pdf_urls(self):
        result = WebSearchAgentResult(
            topic="test",
            web_results=[
                SearchResult(title="P1", url="https://ex.com/a.pdf"),
                SearchResult(title="P2", url="https://ex.com/b.html"),
                SearchResult(title="P3", url="https://ex.com/c.pdf"),
            ],
        )
        pdfs = WebSearchAgent._find_pdf_urls(result)
        assert len(pdfs) == 2
        assert all(u.endswith(".pdf") for u in pdfs)

    @patch("researchclaw.web.search.urlopen")
    @patch("researchclaw.web.scholar.scholarly")
    def test_search_and_extract_minimal(self, mock_scholarly, mock_urlopen):
        """End-to-end test with mocked HTTP — DuckDuckGo + mocked Scholar."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"""
        <a class="result__a" href="https://arxiv.org/abs/1234">Paper About KD</a>
        <a class="result__snippet">A study on knowledge distillation</a>
        """
        mock_urlopen.return_value = mock_resp

        # Mock scholarly to return empty (avoid network calls)
        mock_scholarly.search_pubs.return_value = iter([])

        agent = WebSearchAgent(
            enable_scholar=True,
            enable_crawling=False,
            enable_pdf=False,
        )
        result = agent.search_and_extract("knowledge distillation")
        assert result.topic == "knowledge distillation"
        assert result.elapsed_seconds > 0

    @patch("researchclaw.web.search.urlopen")
    @patch("researchclaw.web.scholar.scholarly")
    @patch("researchclaw.web.crawler.urlopen")
    def test_search_and_extract_with_crawling(self, mock_crawl_urlopen, mock_scholarly, mock_search_urlopen):
        """Test with crawling enabled."""
        mock_search_resp = MagicMock()
        mock_search_resp.read.return_value = b"""
        <a class="result__a" href="https://blog.example.com/kd">KD Tutorial</a>
        <a class="result__snippet">A tutorial</a>
        """
        mock_search_urlopen.return_value = mock_search_resp

        mock_crawl_resp = MagicMock()
        mock_crawl_resp.read.return_value = (
            b"<html><title>KD Tutorial</title><body><p>"
            + b"Tutorial content about knowledge distillation. " * 20
            + b"</p></body></html>"
        )
        mock_crawl_resp.headers = {"Content-Type": "text/html"}
        mock_crawl_urlopen.return_value = mock_crawl_resp

        mock_scholarly.search_pubs.return_value = iter([])

        agent = WebSearchAgent(
            enable_scholar=False,
            enable_crawling=True,
            enable_pdf=False,
            max_crawl_urls=2,
        )
        result = agent.search_and_extract("knowledge distillation")
        assert result.elapsed_seconds > 0


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestWebSearchConfig:
    def test_default_config(self):
        from researchclaw.config import WebSearchConfig
        cfg = WebSearchConfig()
        assert cfg.enabled is True
        assert cfg.max_web_results == 10
        assert cfg.enable_scholar is True

    def test_config_in_rcconfig(self):
        from researchclaw.config import RCConfig
        import dataclasses
        field_names = [f.name for f in dataclasses.fields(RCConfig)]
        assert "web_search" in field_names
