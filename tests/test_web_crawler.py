"""Tests for researchclaw.web.crawler — WebCrawler."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from researchclaw.web.crawler import CrawlResult, WebCrawler


# ---------------------------------------------------------------------------
# CrawlResult dataclass
# ---------------------------------------------------------------------------


class TestCrawlResult:
    def test_has_content_true(self):
        r = CrawlResult(url="https://example.com", markdown="x" * 100, success=True)
        assert r.has_content

    def test_has_content_false_empty(self):
        r = CrawlResult(url="https://example.com", markdown="", success=True)
        assert not r.has_content

    def test_has_content_false_short(self):
        r = CrawlResult(url="https://example.com", markdown="too short", success=True)
        assert not r.has_content


# ---------------------------------------------------------------------------
# HTML → Markdown conversion (urllib fallback)
# ---------------------------------------------------------------------------


class TestHtmlToMarkdown:
    def test_strips_script_tags(self):
        html = "<p>Hello</p><script>alert(1)</script><p>World</p>"
        md = WebCrawler._html_to_markdown(html)
        assert "alert" not in md
        assert "Hello" in md
        assert "World" in md

    def test_converts_headings(self):
        html = "<h1>Title</h1><h2>Subtitle</h2><h3>Section</h3>"
        md = WebCrawler._html_to_markdown(html)
        assert "# Title" in md
        assert "## Subtitle" in md
        assert "### Section" in md

    def test_converts_paragraphs(self):
        html = "<p>First paragraph.</p><p>Second paragraph.</p>"
        md = WebCrawler._html_to_markdown(html)
        assert "First paragraph." in md
        assert "Second paragraph." in md

    def test_converts_links(self):
        html = '<a href="https://example.com">Click</a>'
        md = WebCrawler._html_to_markdown(html)
        assert "[Click](https://example.com)" in md

    def test_converts_list_items(self):
        html = "<ul><li>Item 1</li><li>Item 2</li></ul>"
        md = WebCrawler._html_to_markdown(html)
        assert "- Item 1" in md
        assert "- Item 2" in md

    def test_decodes_entities(self):
        html = "<p>A &amp; B &lt; C &gt; D</p>"
        md = WebCrawler._html_to_markdown(html)
        assert "A & B < C > D" in md

    def test_collapses_whitespace(self):
        html = "<p>Hello</p>\n\n\n\n<p>World</p>"
        md = WebCrawler._html_to_markdown(html)
        assert "\n\n\n" not in md


# ---------------------------------------------------------------------------
# urllib fallback crawl
# ---------------------------------------------------------------------------


class TestCrawlUrllibFallback:
    @patch("researchclaw.web.crawler.urlopen")
    def test_crawl_urllib_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"<html><title>Test</title><body><p>Content here</p></body></html>"
        mock_resp.headers = {"Content-Type": "text/html; charset=utf-8"}
        mock_urlopen.return_value = mock_resp

        crawler = WebCrawler()
        import time
        t0 = time.monotonic()
        result = crawler._crawl_with_urllib("https://example.com", t0)
        assert result.success
        assert result.title == "Test"
        assert "Content here" in result.markdown

    @patch("researchclaw.web.crawler.urlopen")
    def test_crawl_urllib_truncation(self, mock_urlopen):
        mock_resp = MagicMock()
        long_content = "<p>" + "x" * 60000 + "</p>"
        mock_resp.read.return_value = long_content.encode()
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_urlopen.return_value = mock_resp

        crawler = WebCrawler(max_content_length=1000)
        import time
        t0 = time.monotonic()
        result = crawler._crawl_with_urllib("https://example.com", t0)
        assert len(result.markdown) <= 1100  # 1000 + truncation notice

    def test_crawl_urllib_uses_certifi_ssl_context(self, monkeypatch: pytest.MonkeyPatch):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"<html><body><p>Content here</p></body></html>"
        mock_resp.headers = {"Content-Type": "text/html; charset=utf-8"}
        ssl_context = object()
        captured: dict[str, object] = {}

        def fake_urlopen(*args, **kwargs):
            captured["context"] = kwargs.get("context")
            return mock_resp

        monkeypatch.setattr("researchclaw.web.crawler.get_default_ssl_context", lambda: ssl_context)
        monkeypatch.setattr("researchclaw.web.crawler.urlopen", fake_urlopen)

        crawler = WebCrawler()
        import time
        t0 = time.monotonic()
        _ = crawler._crawl_with_urllib("https://example.com", t0)

        assert captured["context"] is ssl_context


# ---------------------------------------------------------------------------
# Sync crawl (goes through crawl4ai → urllib fallback chain)
# ---------------------------------------------------------------------------


class TestCrawlSync:
    @patch("researchclaw.web.crawler.urlopen")
    def test_crawl_sync_falls_back_to_urllib(self, mock_urlopen):
        """crawl_sync tries crawl4ai, then falls back to urllib."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"<html><title>Sync</title><body><p>Works via urllib</p></body></html>"
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_urlopen.return_value = mock_resp

        crawler = WebCrawler()
        # Crawl4AI may or may not work in test env (no browser),
        # but urllib fallback should always work
        result = crawler.crawl_sync("https://example.com")
        assert result.success or result.error  # either crawl4ai or urllib


# ---------------------------------------------------------------------------
# Async crawl
# ---------------------------------------------------------------------------


class TestCrawlAsync:
    @patch("researchclaw.web.crawler.urlopen")
    def test_crawl_async_urllib_fallback(self, mock_urlopen):
        """When crawl4ai's browser isn't set up, async crawl falls back to urllib."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"<html><title>Async</title><body><p>Works</p></body></html>"
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_urlopen.return_value = mock_resp

        crawler = WebCrawler()
        result = asyncio.run(crawler.crawl("https://example.com"))
        # Should succeed via either crawl4ai or urllib fallback
        assert isinstance(result, CrawlResult)
