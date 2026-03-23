"""Tests for researchclaw.trends — Research Trend Tracker (Agent D1).

25+ tests covering feeds, trend_analyzer, opportunity_finder,
daily_digest, auto_topic, and literature/trends.
"""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from researchclaw.trends.feeds import FeedManager
from researchclaw.trends.trend_analyzer import TrendAnalyzer, _STOPWORDS
from researchclaw.trends.opportunity_finder import OpportunityFinder
from researchclaw.trends.daily_digest import DailyDigest
from researchclaw.trends.auto_topic import AutoTopicGenerator
from researchclaw.literature.trends import LiteratureTrendAnalyzer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_papers(n: int = 10) -> list[dict[str, Any]]:
    """Generate synthetic papers for testing."""
    papers = []
    for i in range(n):
        papers.append({
            "title": f"Transformer attention mechanism for graph neural networks part {i}",
            "authors": [
                {"name": "Alice Smith"},
                {"name": "Bob Jones"},
            ] if i % 2 == 0 else ["Alice Smith", "Charlie Brown"],
            "abstract": (
                "We propose a transformer-based attention approach for "
                "graph neural networks using contrastive learning on ImageNet "
                "and CIFAR datasets. Our diffusion model achieves SOTA results."
            ),
            "url": f"https://arxiv.org/abs/2026.{i:05d}",
            "source": "arxiv" if i % 2 == 0 else "semantic_scholar",
            "published_date": "2026-03-01",
        })
    return papers


class MockLLM:
    async def chat_async(self, prompt: str) -> str:
        return (
            "TOPIC: Graph transformers for drug discovery | "
            "WHY: Rising trend | FEASIBILITY: high\n"
            "TOPIC: Diffusion models for 3D generation | "
            "WHY: New paradigm | FEASIBILITY: medium\n"
        )


class FailingLLM:
    async def chat_async(self, prompt: str) -> str:
        raise RuntimeError("API error")


# ===================================================================
# FeedManager tests
# ===================================================================


class TestFeedManager:
    def test_init_filters_supported_sources(self):
        fm = FeedManager(sources=("arxiv", "invalid_source", "semantic_scholar"))
        assert fm.sources == ("arxiv", "semantic_scholar")

    def test_supported_sources(self):
        assert "arxiv" in FeedManager.SUPPORTED_SOURCES
        assert "semantic_scholar" in FeedManager.SUPPORTED_SOURCES
        assert "openalex" in FeedManager.SUPPORTED_SOURCES

    def test_fetch_deduplicates_by_title(self):
        fm = FeedManager(sources=("arxiv",))
        # Mock _fetch_arxiv to return duplicates
        papers = [
            {"title": "Same Title", "source": "arxiv"},
            {"title": "Same Title", "source": "arxiv"},
            {"title": "Different Title", "source": "arxiv"},
        ]
        with patch.object(fm, "_fetch_arxiv", return_value=papers):
            result = fm.fetch_recent_papers(["ml"], max_papers=10)
        assert len(result) == 2

    def test_fetch_respects_max_papers(self):
        fm = FeedManager(sources=("arxiv",))
        papers = [{"title": f"Paper {i}", "source": "arxiv"} for i in range(20)]
        with patch.object(fm, "_fetch_arxiv", return_value=papers):
            result = fm.fetch_recent_papers(["ml"], max_papers=5)
        assert len(result) == 5

    def test_fetch_handles_source_failure(self):
        fm = FeedManager(sources=("arxiv",))
        with patch.object(fm, "_fetch_arxiv", side_effect=RuntimeError("fail")):
            result = fm.fetch_recent_papers(["ml"])
        assert result == []

    def test_fetch_empty_title_excluded(self):
        fm = FeedManager(sources=("arxiv",))
        papers = [
            {"title": "", "source": "arxiv"},
            {"title": "  ", "source": "arxiv"},
            {"title": "Valid Paper", "source": "arxiv"},
        ]
        with patch.object(fm, "_fetch_arxiv", return_value=papers):
            result = fm.fetch_recent_papers(["ml"])
        assert len(result) == 1


# ===================================================================
# TrendAnalyzer tests
# ===================================================================


class TestTrendAnalyzer:
    def test_analyze_empty(self):
        analyzer = TrendAnalyzer()
        result = analyzer.analyze([])
        assert result["paper_count"] == 0
        assert result["rising_keywords"] == []

    def test_analyze_extracts_keywords(self):
        analyzer = TrendAnalyzer()
        papers = _make_papers(10)
        result = analyzer.analyze(papers)
        assert result["paper_count"] == 10
        assert len(result["rising_keywords"]) > 0

    def test_keywords_exclude_stopwords(self):
        analyzer = TrendAnalyzer()
        papers = _make_papers(10)
        result = analyzer.analyze(papers)
        for kw in result["rising_keywords"]:
            for word in kw["keyword"].split():
                assert word not in _STOPWORDS

    def test_extract_authors_dict_format(self):
        analyzer = TrendAnalyzer()
        papers = [
            {"authors": [{"name": "Alice"}, {"name": "Bob"}]} for _ in range(5)
        ]
        authors = analyzer._extract_authors(papers)
        assert any(a["author"] == "Alice" for a in authors)

    def test_extract_authors_string_format(self):
        analyzer = TrendAnalyzer()
        papers = [{"authors": ["Alice", "Bob"]} for _ in range(5)]
        authors = analyzer._extract_authors(papers)
        assert any(a["author"] == "Alice" for a in authors)

    def test_extract_datasets(self):
        analyzer = TrendAnalyzer()
        papers = [
            {"title": "Training on ImageNet and CIFAR", "abstract": ""},
            {"title": "MNIST results", "abstract": "evaluated on GLUE benchmark"},
        ]
        datasets = analyzer._extract_datasets(papers)
        ds_names = {d["dataset"] for d in datasets}
        assert "ImageNet" in ds_names
        assert "CIFAR" in ds_names

    def test_extract_methods(self):
        analyzer = TrendAnalyzer()
        papers = [
            {"title": "Transformer attention", "abstract": "using diffusion models"},
            {"title": "GAN for images", "abstract": "contrastive learning approach"},
        ]
        methods = analyzer._extract_methods(papers)
        method_names = {m["method"] for m in methods}
        assert "transformer" in method_names or "attention" in method_names

    def test_tokenize(self):
        tokens = TrendAnalyzer._tokenize("Hello World! It's a test-case.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "it's" in tokens
        assert "test-case" in tokens

    def test_source_distribution(self):
        papers = [
            {"source": "arxiv"},
            {"source": "arxiv"},
            {"source": "semantic_scholar"},
        ]
        dist = TrendAnalyzer._source_distribution(papers)
        assert dist["arxiv"] == 2
        assert dist["semantic_scholar"] == 1

    def test_generate_trend_report(self):
        analyzer = TrendAnalyzer()
        analysis = analyzer.analyze(_make_papers(10))
        report = analyzer.generate_trend_report(analysis)
        assert "Research Trend Analysis" in report
        assert "10 papers" in report

    def test_min_keyword_length(self):
        analyzer = TrendAnalyzer(min_keyword_length=5)
        papers = [{"title": "AI is a big deal", "abstract": ""}] * 5
        keywords = analyzer._extract_keywords(papers)
        # Short words like "deal" (4 chars) should be excluded by min_keyword_length=5
        # but "big" is only 3 chars so excluded too
        for kw in keywords:
            for word in kw["keyword"].split():
                assert len(word) >= 5 or word in _STOPWORDS


# ===================================================================
# OpportunityFinder tests
# ===================================================================


class TestOpportunityFinder:
    def test_heuristic_no_llm(self):
        finder = OpportunityFinder()
        trend_analysis = {
            "rising_keywords": [
                {"keyword": "graph neural", "count": 10},
                {"keyword": "attention", "count": 8},
                {"keyword": "diffusion", "count": 6},
            ],
            "method_trends": [
                {"method": "transformer", "mention_count": 12},
                {"method": "contrastive learning", "mention_count": 7},
            ],
        }
        result = asyncio.run(finder.find_opportunities(trend_analysis, ["ml"]))
        assert len(result) > 0
        assert all("topic" in opp for opp in result)
        assert all(opp["source"] == "heuristic" for opp in result)

    def test_heuristic_empty_trends(self):
        finder = OpportunityFinder()
        result = asyncio.run(finder.find_opportunities(
            {"rising_keywords": [], "method_trends": []}, ["ml"]
        ))
        assert result == []

    def test_llm_path(self):
        finder = OpportunityFinder(llm_client=MockLLM())
        trend_analysis = {
            "rising_keywords": [{"keyword": "graph", "count": 10}],
            "method_trends": [{"method": "transformer", "mention_count": 5}],
        }
        result = asyncio.run(finder.find_opportunities(trend_analysis, ["ml"]))
        assert len(result) >= 1
        assert result[0]["source"] == "llm"

    def test_llm_fallback_on_failure(self):
        finder = OpportunityFinder(llm_client=FailingLLM())
        trend_analysis = {
            "rising_keywords": [{"keyword": "test", "count": 5}],
            "method_trends": [{"method": "GAN", "mention_count": 3}],
        }
        result = asyncio.run(finder.find_opportunities(trend_analysis, ["ml"]))
        assert all(opp["source"] == "heuristic" for opp in result)

    def test_parse_opportunities(self):
        response = (
            "TOPIC: Adaptive transformers | WHY: Trending | FEASIBILITY: high\n"
            "TOPIC: Diffusion for audio | WHY: New area | FEASIBILITY: medium\n"
            "Some noise line\n"
        )
        result = OpportunityFinder._parse_opportunities(response)
        assert len(result) == 2
        assert result[0]["topic"] == "Adaptive transformers"
        assert result[0]["feasibility"] == "high"

    def test_heuristic_max_five(self):
        finder = OpportunityFinder()
        trend_analysis = {
            "rising_keywords": [
                {"keyword": f"kw{i}", "count": 10} for i in range(10)
            ],
            "method_trends": [
                {"method": f"method{i}", "mention_count": 5} for i in range(10)
            ],
        }
        result = asyncio.run(finder.find_opportunities(trend_analysis, ["ml"]))
        assert len(result) <= 5


# ===================================================================
# DailyDigest tests
# ===================================================================


class TestDailyDigest:
    def test_generate_basic_no_papers(self):
        fm = FeedManager(sources=())
        digest = DailyDigest(fm)
        result = asyncio.run(digest.generate(["ml"]))
        assert "No new papers found" in result

    def test_generate_basic_with_papers(self):
        fm = FeedManager(sources=("arxiv",))
        papers = _make_papers(3)
        with patch.object(fm, "fetch_recent_papers", return_value=papers):
            digest = DailyDigest(fm)
            result = asyncio.run(digest.generate(["ml"]))
        assert "Daily Paper Digest" in result
        assert "Papers found: 3" in result

    def test_generate_basic_truncates_abstract(self):
        fm = FeedManager(sources=("arxiv",))
        papers = [{"title": "Test", "abstract": "x" * 500, "authors": [], "url": ""}]
        with patch.object(fm, "fetch_recent_papers", return_value=papers):
            digest = DailyDigest(fm)
            result = asyncio.run(digest.generate(["ml"]))
        assert "..." in result

    def test_parse_summary_valid(self):
        response = "SUMMARY: Great paper on attention | RELEVANCE: 4"
        summary, relevance = DailyDigest._parse_summary(response)
        assert summary == "Great paper on attention"
        assert relevance == 4

    def test_parse_summary_no_format(self):
        response = "Just a plain text response."
        summary, relevance = DailyDigest._parse_summary(response)
        assert summary == response
        assert relevance == 3  # default

    def test_parse_summary_clamped(self):
        response = "SUMMARY: x | RELEVANCE: 99"
        _, relevance = DailyDigest._parse_summary(response)
        assert relevance == 5

    def test_generate_and_save(self, tmp_path: Path):
        fm = FeedManager(sources=("arxiv",))
        papers = _make_papers(2)
        with patch.object(fm, "fetch_recent_papers", return_value=papers):
            digest = DailyDigest(fm)
            result_path = asyncio.run(digest.generate_and_save(tmp_path, ["ml"]))
        assert result_path.exists()
        assert result_path.read_text(encoding="utf-8").startswith("## Daily Paper Digest")

    def test_author_formatting_dict(self):
        fm = FeedManager(sources=("arxiv",))
        papers = [{
            "title": "T",
            "abstract": "",
            "url": "",
            "authors": [{"name": "A"}, {"name": "B"}, {"name": "C"}, {"name": "D"}],
        }]
        with patch.object(fm, "fetch_recent_papers", return_value=papers):
            digest = DailyDigest(fm)
            result = asyncio.run(digest.generate(["ml"]))
        assert "et al." in result


# ===================================================================
# AutoTopicGenerator tests
# ===================================================================


class TestAutoTopicGenerator:
    def test_generate_candidates(self):
        analyzer = TrendAnalyzer()
        finder = OpportunityFinder()
        gen = AutoTopicGenerator(analyzer, finder)
        papers = _make_papers(10)
        candidates = asyncio.run(gen.generate_candidates(["ml"], papers, count=3))
        assert len(candidates) <= 3
        if candidates:
            assert "topic" in candidates[0]
            assert "overall_score" in candidates[0]

    def test_generate_candidates_empty(self):
        analyzer = TrendAnalyzer()
        finder = OpportunityFinder()
        gen = AutoTopicGenerator(analyzer, finder)
        candidates = asyncio.run(gen.generate_candidates(["ml"], [], count=3))
        # With empty papers, heuristic has no keywords/methods → no opportunities
        assert isinstance(candidates, list)

    def test_auto_select_default_fallback(self):
        analyzer = TrendAnalyzer()
        finder = OpportunityFinder()
        gen = AutoTopicGenerator(analyzer, finder)
        result = asyncio.run(gen.auto_select(["ml"], []))
        assert "topic" in result
        assert result["source"] == "default"

    def test_score_candidate_feasibility(self):
        opp_high = {"topic": "unique topic xyz", "feasibility": "high"}
        opp_low = {"topic": "unique topic xyz", "feasibility": "low"}
        trend = {"rising_keywords": [], "paper_count": 50}
        score_h = AutoTopicGenerator._score_candidate(opp_high, trend)
        score_l = AutoTopicGenerator._score_candidate(opp_low, trend)
        assert score_h["feasibility"] == 0.9
        assert score_l["feasibility"] == 0.3
        assert score_h["overall"] > score_l["overall"]

    def test_score_candidate_novelty_decay(self):
        opp = {"topic": "graph neural", "feasibility": "medium"}
        trend = {
            "rising_keywords": [
                {"keyword": "graph neural", "count": 10},
                {"keyword": "neural network", "count": 8},
            ],
            "paper_count": 50,
        }
        score = AutoTopicGenerator._score_candidate(opp, trend)
        assert score["novelty"] < 1.0  # overlap penalizes novelty

    def test_score_candidate_weights(self):
        """Overall = 0.4*novelty + 0.3*feasibility + 0.3*impact."""
        opp = {"topic": "totally unique xyz", "feasibility": "high"}
        trend = {"rising_keywords": [], "paper_count": 50}
        score = AutoTopicGenerator._score_candidate(opp, trend)
        expected = round(0.4 * score["novelty"] + 0.3 * score["feasibility"] + 0.3 * score["impact"], 3)
        assert score["overall"] == expected

    def test_format_candidates_empty(self):
        analyzer = TrendAnalyzer()
        finder = OpportunityFinder()
        gen = AutoTopicGenerator(analyzer, finder)
        assert "No candidate" in gen.format_candidates([])

    def test_format_candidates_with_data(self):
        analyzer = TrendAnalyzer()
        finder = OpportunityFinder()
        gen = AutoTopicGenerator(analyzer, finder)
        candidates = [
            {
                "topic": "Test topic",
                "overall_score": 0.75,
                "novelty_score": 0.8,
                "feasibility_score": 0.7,
                "impact_score": 0.6,
                "rationale": "Good idea",
            }
        ]
        output = gen.format_candidates(candidates)
        assert "Test topic" in output
        assert "0.75" in output


# ===================================================================
# LiteratureTrendAnalyzer tests
# ===================================================================


class TestLiteratureTrendAnalyzer:
    def test_no_client_returns_empty(self):
        lta = LiteratureTrendAnalyzer()
        assert lta.get_daily_papers(["ml"]) == []

    def test_analyze_keyword_trends_no_client(self):
        lta = LiteratureTrendAnalyzer()
        result = lta.analyze_keyword_trends(["ml"])
        assert result["total_papers"] == 0

    def test_find_emerging_topics_no_client(self):
        lta = LiteratureTrendAnalyzer()
        assert lta.find_emerging_topics(["ml"]) == []

    def test_find_emerging_topics_filters_bigrams(self):
        """Only bigrams with count >= 3 are considered emerging."""
        lta = LiteratureTrendAnalyzer(search_client="fake")
        papers = _make_papers(20)
        with patch.object(lta, "get_daily_papers", return_value=papers):
            topics = lta.find_emerging_topics(["ml"])
        for t in topics:
            assert t["type"] == "bigram"
            assert t["frequency"] >= 3
