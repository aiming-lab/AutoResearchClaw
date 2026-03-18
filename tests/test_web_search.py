"""Tests for researchclaw.web.search — WebSearchClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from researchclaw.web.search import SearchResult, WebSearchClient, WebSearchResponse


# ---------------------------------------------------------------------------
# SearchResult dataclass
# ---------------------------------------------------------------------------


class TestSearchResult:
    def test_to_dict(self):
        r = SearchResult(
            title="Test", url="https://example.com", snippet="A snippet", source="tavily"
        )
        d = r.to_dict()
        assert d["title"] == "Test"
        assert d["url"] == "https://example.com"
        assert d["source"] == "tavily"


# ---------------------------------------------------------------------------
# WebSearchResponse dataclass
# ---------------------------------------------------------------------------


class TestWebSearchResponse:
    def test_has_results_true(self):
        r = WebSearchResponse(
            query="test", results=[SearchResult(title="A", url="u")],
        )
        assert r.has_results

    def test_has_results_false(self):
        r = WebSearchResponse(query="test")
        assert not r.has_results


# ---------------------------------------------------------------------------
# DuckDuckGo HTML parsing
# ---------------------------------------------------------------------------


class TestDDGParsing:
    def test_parse_ddg_html_basic(self):
        html = """
        <div class="result">
            <a class="result__a" href="https://example.com/1">Title One</a>
            <a class="result__snippet">Snippet one here</a>
        </div>
        <div class="result">
            <a class="result__a" href="https://example.com/2">Title Two</a>
            <a class="result__snippet">Snippet two here</a>
        </div>
        """
        results = WebSearchClient._parse_ddg_html(html, limit=10)
        assert len(results) == 2
        assert results[0].title == "Title One"
        assert results[0].url == "https://example.com/1"
        assert results[0].snippet == "Snippet one here"

    def test_parse_ddg_html_skips_ddg_links(self):
        html = """
        <a class="result__a" href="https://duckduckgo.com/internal">DDG Link</a>
        <a class="result__a" href="https://example.com/real">Real</a>
        """
        results = WebSearchClient._parse_ddg_html(html, limit=10)
        assert len(results) == 1
        assert results[0].url == "https://example.com/real"

    def test_parse_ddg_html_respects_limit(self):
        html = ""
        for i in range(20):
            html += f'<a class="result__a" href="https://ex.com/{i}">T{i}</a>\n'
        results = WebSearchClient._parse_ddg_html(html, limit=5)
        assert len(results) == 5


# ---------------------------------------------------------------------------
# WebSearchClient.search
# ---------------------------------------------------------------------------


class TestWebSearchClient:
    @patch("researchclaw.web.search.urlopen")
    def test_search_ddg_fallback_no_api_key(self, mock_urlopen):
        """When no API key is set, uses DuckDuckGo fallback."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"""
        <a class="result__a" href="https://paper.com">Paper Title</a>
        <a class="result__snippet">About the paper</a>
        """
        mock_urlopen.return_value = mock_resp

        client = WebSearchClient(api_key="")  # No API key
        response = client.search("test query")
        assert response.source == "duckduckgo"

    @patch("researchclaw.web.search.urlopen")
    def test_search_ddg_error_graceful(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Network error")

        client = WebSearchClient(api_key="")
        response = client.search("test query")
        assert response.source == "duckduckgo"
        assert len(response.results) == 0

    def test_search_tavily_with_mock(self):
        """Test Tavily search with mocked SDK."""
        mock_client_instance = MagicMock()
        mock_client_instance.search.return_value = {
            "results": [
                {
                    "title": "Tavily Result",
                    "url": "https://tavily.com/r1",
                    "content": "Content from Tavily",
                    "score": 0.95,
                }
            ],
            "answer": "AI summary answer",
        }

        mock_tavily_module = MagicMock()
        mock_tavily_module.TavilyClient.return_value = mock_client_instance

        with patch.dict("sys.modules", {"tavily": mock_tavily_module}):
            client = WebSearchClient(api_key="test-key")
            import time
            response = client._search_tavily("test query", 10, None, None, time.monotonic())
            assert response.source == "tavily"
            assert len(response.results) == 1
            assert response.results[0].title == "Tavily Result"
            assert response.answer == "AI summary answer"

    @patch("researchclaw.web.search.urlopen")
    def test_search_multi_deduplication(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"""
        <a class="result__a" href="https://ex.com/same">Same Result</a>
        """
        mock_urlopen.return_value = mock_resp

        client = WebSearchClient(api_key="")
        responses = client.search_multi(["query1", "query2"], inter_query_delay=0.0)
        assert len(responses) == 2
        # Second query should have same URL deduped
        if responses[0].results:
            assert all(
                r.url != responses[0].results[0].url
                for r in responses[1].results
            )
