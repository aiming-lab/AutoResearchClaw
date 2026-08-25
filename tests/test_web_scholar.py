"""Tests for researchclaw.web.scholar — GoogleScholarClient."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from researchclaw.web.scholar import GoogleScholarClient, ScholarPaper


# ---------------------------------------------------------------------------
# ScholarPaper dataclass
# ---------------------------------------------------------------------------


class TestScholarPaper:
    def test_to_dict(self):
        p = ScholarPaper(
            title="Attention Is All You Need",
            authors=["Vaswani", "Shazeer"],
            year=2017,
            citation_count=50000,
        )
        d = p.to_dict()
        assert d["title"] == "Attention Is All You Need"
        assert d["year"] == 2017
        assert d["source"] == "google_scholar"

    def test_to_literature_paper(self):
        p = ScholarPaper(
            title="Test Paper",
            authors=["Author One", "Author Two"],
            year=2024,
            abstract="An abstract.",
            citation_count=100,
            url="https://example.com",
        )
        lit = p.to_literature_paper()
        assert lit.title == "Test Paper"
        assert lit.source == "google_scholar"
        assert len(lit.authors) == 2
        assert lit.authors[0].name == "Author One"


# ---------------------------------------------------------------------------
# GoogleScholarClient
# ---------------------------------------------------------------------------


class TestGoogleScholarClient:
    @patch("researchclaw.web.scholar.HAS_SCHOLARLY", True)
    def test_available_always_true(self):
        """scholarly is now an installed dependency, always available."""
        client = GoogleScholarClient()
        assert client.available

    def test_parse_pub_full(self):
        """Test _parse_pub with a complete publication dict."""
        pub = {
            "bib": {
                "title": "Deep Learning",
                "author": ["LeCun", "Bengio", "Hinton"],
                "pub_year": "2015",
                "abstract": "Deep learning review.",
                "venue": "Nature",
            },
            "num_citations": 30000,
            "pub_url": "https://nature.com/dl",
            "cites_id": ["abc123"],
        }
        paper = GoogleScholarClient._parse_pub(pub)
        assert paper.title == "Deep Learning"
        assert paper.year == 2015
        assert paper.citation_count == 30000
        assert "LeCun" in paper.authors
        assert paper.venue == "Nature"

    def test_parse_pub_string_authors(self):
        pub = {
            "bib": {
                "title": "Paper",
                "author": "Smith and Jones",
                "pub_year": "2023",
            },
            "num_citations": 10,
            "pub_url": "https://example.com",
        }
        paper = GoogleScholarClient._parse_pub(pub)
        assert paper.title == "Paper"
        assert "Smith" in paper.authors
        assert "Jones" in paper.authors

    def test_parse_pub_missing_fields(self):
        pub = {"bib": {}, "num_citations": 0}
        paper = GoogleScholarClient._parse_pub(pub)
        assert paper.title == ""
        assert paper.year == 0
        assert paper.authors == []

    @patch("researchclaw.web.scholar.HAS_SCHOLARLY", True)
    def test_rate_limiting(self):
        client = GoogleScholarClient(inter_request_delay=0.01)
        t0 = time.monotonic()
        client._rate_limit()
        client._rate_limit()
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.01

    @patch("researchclaw.web.scholar.HAS_SCHOLARLY", True)
    @patch("researchclaw.web.scholar.scholarly")
    def test_search_with_mocked_scholarly(self, mock_scholarly):
        """Test search using mocked scholarly library."""
        mock_pub = {
            "bib": {
                "title": "Test Paper",
                "author": ["Author A"],
                "pub_year": "2024",
            },
            "num_citations": 5,
            "pub_url": "https://example.com",
        }
        mock_scholarly.search_pubs.return_value = iter([mock_pub])

        client = GoogleScholarClient(inter_request_delay=0.0)
        results = client.search("test query", limit=5)
        assert len(results) == 1
        assert results[0].title == "Test Paper"

    @patch("researchclaw.web.scholar.HAS_SCHOLARLY", True)
    @patch("researchclaw.web.scholar.scholarly")
    def test_search_error_graceful(self, mock_scholarly):
        """Search should return empty list on error, not raise."""
        mock_scholarly.search_pubs.side_effect = Exception("Rate limited")

        client = GoogleScholarClient(inter_request_delay=0.0)
        results = client.search("test query")
        assert results == []


# ---------------------------------------------------------------------------
# Serply Scholar backend
# ---------------------------------------------------------------------------


_SERPLY_ARTICLE = {
    "title": "Knowledge Distillation: A Survey",
    "link": "https://doi.org/10.1007/s11263-021-01453-z",
    "id": "W3034368386",
    "author": {
        "names": "Jianping Gou, Baosheng Yu - International Journal of Computer Vision, 2021",
        "authors": [
            {"name": "Jianping Gou", "link": "https://openalex.org/A1"},
            {"name": "Baosheng Yu", "link": "https://openalex.org/A2"},
        ],
    },
    "description": "Jianping Gou, Baosheng Yu - International Journal of Computer Vision, 2021",
    "doc": {"link": "https://arxiv.org/pdf/2006.05525", "type": "PDF"},
    "extras": {"citations": {"count": 3698, "link": "https://example.org/cites"}},
}


def _serply_scholar_response(payload: dict) -> MagicMock:
    import json

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    return mock_resp


class TestSerplyScholarBackend:
    def test_parse_serply_article_full(self):
        paper = GoogleScholarClient._parse_serply_article(_SERPLY_ARTICLE)
        assert paper.title == "Knowledge Distillation: A Survey"
        assert paper.authors == ["Jianping Gou", "Baosheng Yu"]
        assert paper.year == 2021
        assert paper.venue == "International Journal of Computer Vision"
        assert paper.citation_count == 3698
        assert paper.url == "https://doi.org/10.1007/s11263-021-01453-z"
        assert paper.scholar_id == "W3034368386"
        assert paper.source == "google_scholar"

    def test_parse_serply_article_minimal(self):
        paper = GoogleScholarClient._parse_serply_article({
            "title": "Untitled Preprint",
            "description": "A Author, B Author - 2019",
            "doc": {"link": "https://example.org/paper.pdf"},
        })
        assert paper.authors == ["A Author", "B Author"]
        assert paper.year == 2019
        assert paper.venue == ""
        assert paper.citation_count == 0
        assert paper.url == "https://example.org/paper.pdf"

    def test_parse_serply_article_venue_starting_with_year(self):
        paper = GoogleScholarClient._parse_serply_article({
            "title": "Decoupled Knowledge Distillation",
            "description": (
                "Borui Zhao, Quan Cui - 2022 IEEE/CVF Conference on Computer Vision "
                "and Pattern Recognition (CVPR), 2022"
            ),
        })
        assert paper.year == 2022
        assert paper.venue == (
            "2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)"
        )
        assert paper.authors == ["Borui Zhao", "Quan Cui"]

    @patch("researchclaw.web.scholar.HAS_SCHOLARLY", False)
    def test_constructor_without_scholarly_requires_serply_key(self, monkeypatch):
        monkeypatch.delenv("SERPLY_API_KEY", raising=False)
        with pytest.raises(ImportError):
            GoogleScholarClient()
        client = GoogleScholarClient(serply_api_key="serply-key")
        assert client.available

    @patch("researchclaw.web.scholar.HAS_SCHOLARLY", False)
    @patch("researchclaw.web.scholar.urlopen")
    def test_search_uses_serply(self, mock_urlopen):
        mock_urlopen.return_value = _serply_scholar_response({
            "articles": [_SERPLY_ARTICLE, {"title": "", "link": "https://dropped"}],
        })
        client = GoogleScholarClient(serply_api_key="serply-key")
        papers = client.search("knowledge distillation", limit=5)

        assert len(papers) == 1
        assert papers[0].title == "Knowledge Distillation: A Survey"
        request = mock_urlopen.call_args.args[0]
        assert request.full_url.startswith("https://api.serply.io/v1/scholar/?")
        assert "num=5" in request.full_url
        assert request.get_header("X-api-key") == "serply-key"

    @patch("researchclaw.web.scholar.HAS_SCHOLARLY", False)
    @patch("researchclaw.web.scholar.urlopen")
    def test_search_serply_failure_without_scholarly_returns_empty(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("HTTP 403")
        client = GoogleScholarClient(serply_api_key="serply-key")
        assert client.search("anything") == []

    @patch("researchclaw.web.scholar.HAS_SCHOLARLY", True)
    @patch("researchclaw.web.scholar.urlopen")
    def test_search_serply_failure_falls_back_to_scholarly(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("HTTP 403")
        mock_scholarly = MagicMock()
        mock_scholarly.search_pubs.return_value = iter([
            {"bib": {"title": "Fallback Paper", "author": ["X"], "pub_year": "2020"}},
        ])
        with patch("researchclaw.web.scholar.scholarly", mock_scholarly):
            client = GoogleScholarClient(serply_api_key="serply-key", inter_request_delay=0.0)
            papers = client.search("anything", limit=1)

        assert [p.title for p in papers] == ["Fallback Paper"]
        mock_scholarly.search_pubs.assert_called_once_with("anything")
