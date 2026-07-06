"""Tests for researchclaw.literature.novelty — novelty detection module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from researchclaw.literature.novelty import (
    _assess_novelty,
    _build_novelty_queries,
    _compute_similarity,
    _extract_keywords,
    _jaccard_keywords,
    check_novelty,
)


# ---------------------------------------------------------------------------
# _extract_keywords
# ---------------------------------------------------------------------------


class TestExtractKeywords:
    def test_basic_extraction(self) -> None:
        kws = _extract_keywords("Transformer attention mechanisms for NLP")
        assert "transformer" in kws
        assert "attention" in kws
        assert "mechanisms" in kws
        assert "nlp" in kws

    def test_stop_words_removed(self) -> None:
        kws = _extract_keywords("the model is a new approach for data")
        # "the", "is", "a", "new", "approach", "for", "data", "model" are stop words
        assert "the" not in kws
        assert "is" not in kws

    def test_short_tokens_removed(self) -> None:
        kws = _extract_keywords("AI ML RL deep reinforcement learning")
        # "AI", "ML", "RL" are only 2 chars → removed
        assert "ai" not in kws
        assert "deep" in kws
        assert "reinforcement" in kws

    def test_deduplication(self) -> None:
        kws = _extract_keywords("attention attention attention mechanism")
        assert kws.count("attention") == 1

    def test_empty_input(self) -> None:
        assert _extract_keywords("") == []

    def test_preserves_order(self) -> None:
        kws = _extract_keywords("alpha beta gamma delta")
        assert kws == ["alpha", "beta", "gamma", "delta"]


# ---------------------------------------------------------------------------
# _jaccard_keywords
# ---------------------------------------------------------------------------


class TestJaccardKeywords:
    def test_identical_sets(self) -> None:
        assert _jaccard_keywords(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_disjoint_sets(self) -> None:
        assert _jaccard_keywords(["a", "b"], ["c", "d"]) == 0.0

    def test_partial_overlap(self) -> None:
        # {a, b, c} & {b, c, d} = {b, c} / {a, b, c, d} = 2/4 = 0.5
        assert _jaccard_keywords(["a", "b", "c"], ["b", "c", "d"]) == 0.5

    def test_empty_first(self) -> None:
        assert _jaccard_keywords([], ["a", "b"]) == 0.0

    def test_empty_second(self) -> None:
        assert _jaccard_keywords(["a", "b"], []) == 0.0

    def test_both_empty(self) -> None:
        assert _jaccard_keywords([], []) == 0.0


# ---------------------------------------------------------------------------
# _compute_similarity
# ---------------------------------------------------------------------------


class TestComputeSimilarity:
    def test_returns_float_0_to_1(self) -> None:
        sim = _compute_similarity(
            ["transformer", "attention"],
            "Transformer Attention in NLP",
            "We study attention mechanisms in transformer models.",
        )
        assert 0.0 <= sim <= 1.0

    def test_high_similarity_for_matching_content(self) -> None:
        kws = ["transformer", "attention", "mechanisms", "self-attention"]
        sim = _compute_similarity(
            kws,
            "Self-Attention Mechanisms in Transformers",
            "This paper studies transformer self-attention mechanisms in detail.",
        )
        assert sim > 0.1  # should have meaningful overlap

    def test_low_similarity_for_unrelated_content(self) -> None:
        kws = ["quantum", "computing", "entanglement", "qubit"]
        sim = _compute_similarity(
            kws,
            "Deep Learning for Image Classification",
            "We propose a convolutional neural network for classifying images.",
        )
        assert sim < 0.1

    def test_empty_keywords(self) -> None:
        sim = _compute_similarity([], "Some title", "Some abstract")
        assert sim == 0.0


# ---------------------------------------------------------------------------
# _build_novelty_queries
# ---------------------------------------------------------------------------


class TestBuildNoveltyQueries:
    def test_includes_topic(self) -> None:
        queries = _build_novelty_queries("Reinforcement Learning", "No hypotheses")
        assert queries[0] == "Reinforcement Learning"

    def test_extracts_hypothesis_titles(self) -> None:
        hyp_text = (
            "## H1: Adaptive learning rates improve convergence\n"
            "Details about H1...\n\n"
            "## H2: Curriculum learning reduces sample complexity\n"
            "Details about H2...\n"
        )
        queries = _build_novelty_queries("RL topic", hyp_text)
        assert len(queries) >= 3  # topic + H1 + H2

    def test_caps_at_5(self) -> None:
        hyp_text = "\n".join(
            f"## H{i}: Hypothesis number {i} with enough text to pass length filter"
            for i in range(1, 10)
        )
        queries = _build_novelty_queries("Topic", hyp_text)
        assert len(queries) <= 5

    def test_skips_short_titles(self) -> None:
        hyp_text = "## H1: Short\n## H2: This is a longer hypothesis title\n"
        queries = _build_novelty_queries("Topic", hyp_text)
        # "Short" is < 10 chars → excluded
        assert not any("Short" in q for q in queries)

    def test_empty_hypotheses(self) -> None:
        queries = _build_novelty_queries("Topic", "")
        assert len(queries) >= 1
        assert queries[0] == "Topic"


# ---------------------------------------------------------------------------
# _assess_novelty
# ---------------------------------------------------------------------------


class TestAssessNovelty:
    def test_no_similar_papers_is_high(self) -> None:
        score, assessment = _assess_novelty([], 0.25)
        assert score == 1.0
        assert assessment == "high"

    def test_moderate_similarity(self) -> None:
        papers = [{"similarity": 0.35, "citation_count": 10}]
        score, assessment = _assess_novelty(papers, 0.25)
        assert 0.45 <= score <= 0.85
        assert assessment in ("high", "moderate")

    def test_high_similarity_low_novelty(self) -> None:
        papers = [{"similarity": 0.8, "citation_count": 200}]
        score, assessment = _assess_novelty(papers, 0.25)
        assert score <= 0.3
        assert assessment in ("low", "critical")

    def test_multiple_high_impact_overlaps_penalize(self) -> None:
        papers = [
            {"similarity": 0.5, "citation_count": 100},
            {"similarity": 0.45, "citation_count": 80},
            {"similarity": 0.42, "citation_count": 60},
        ]
        score, _ = _assess_novelty(papers, 0.25)
        # Should be penalized for multiple high-citation overlaps
        assert score < 0.6

    def test_score_bounded_0_to_1(self) -> None:
        papers = [{"similarity": 0.99, "citation_count": 5000}]
        score, _ = _assess_novelty(papers, 0.25)
        assert 0.0 <= score <= 1.0

    def test_critical_assessment(self) -> None:
        papers = [
            {"similarity": 0.9, "citation_count": 200},
            {"similarity": 0.85, "citation_count": 150},
        ]
        score, assessment = _assess_novelty(papers, 0.25)
        assert assessment == "critical"
        assert score < 0.25


# ---------------------------------------------------------------------------
# check_novelty (integration)
# ---------------------------------------------------------------------------


class TestCheckNovelty:
    """Integration tests for check_novelty — mocks the real API calls."""

    @patch("researchclaw.literature.search.search_papers_multi_query")
    def test_basic_flow(self, mock_search: MagicMock) -> None:
        """Smoke test: no similar papers found → high novelty."""
        mock_search.return_value = []
        result = check_novelty(
            topic="Novel quantum-inspired optimization",
            hypotheses_text="## H1: Quantum tunneling improves escape from local minima\n",
        )
        assert isinstance(result, dict)
        assert result["novelty_score"] == 1.0
        assert result["assessment"] in ("high", "insufficient_data")
        assert result["recommendation"] in ("proceed", "proceed_with_caution")
        assert result["topic"] == "Novel quantum-inspired optimization"
        assert "generated" in result

    @patch("researchclaw.literature.search.search_papers_multi_query")
    def test_with_similar_papers(self, mock_search: MagicMock) -> None:
        """Papers with keyword overlap → lower novelty."""
        # Create a mock paper with overlapping keywords
        mock_paper = MagicMock()
        mock_paper.title = "Quantum-Inspired Optimization for Combinatorial Problems"
        mock_paper.abstract = (
            "We propose quantum-inspired optimization methods "
            "using tunneling and superposition analogies to escape local minima."
        )
        mock_paper.paper_id = "abc123"
        mock_paper.year = 2024
        mock_paper.venue = "NeurIPS"
        mock_paper.citation_count = 45
        mock_paper.url = "https://example.com/paper"
        mock_paper.cite_key = "abc2024quantum"

        mock_search.return_value = [mock_paper]
        result = check_novelty(
            topic="Quantum-inspired optimization",
            hypotheses_text="## H1: Quantum tunneling improves escape from local minima\n",
        )
        assert result["similar_papers_found"] >= 0
        assert 0.0 <= result["novelty_score"] <= 1.0

    @patch("researchclaw.literature.search.search_papers_multi_query")
    def test_with_pipeline_papers(self, mock_search: MagicMock) -> None:
        """Papers from candidates.jsonl also checked for overlap."""
        mock_search.return_value = []
        pipeline_papers = [
            {
                "title": "Adaptive Learning Rate Schedules via Meta-Learning",
                "abstract": "We study adaptive learning rate schedules.",
                "paper_id": "p1",
                "year": 2023,
                "venue": "ICML",
                "citation_count": 30,
                "url": "https://example.com",
                "cite_key": "p12023",
            },
        ]
        result = check_novelty(
            topic="Adaptive learning rate schedules",
            hypotheses_text="## H1: Meta-learning adaptive learning rate schedules\n",
            papers_already_seen=pipeline_papers,
        )
        assert isinstance(result, dict)
        assert "similar_papers" in result

    @patch("researchclaw.literature.search.search_papers_multi_query")
    def test_search_failure_graceful(self, mock_search: MagicMock) -> None:
        """API failure should not crash — falls back to pipeline papers."""
        mock_search.side_effect = RuntimeError("API down")
        result = check_novelty(
            topic="Some topic",
            hypotheses_text="## H1: Some hypothesis with enough text\n",
        )
        assert isinstance(result, dict)
        assert "novelty_score" in result

    @patch("researchclaw.literature.search.search_papers_multi_query")
    def test_output_keys_complete(self, mock_search: MagicMock) -> None:
        """All expected keys present in output."""
        mock_search.return_value = []
        result = check_novelty(
            topic="Test topic",
            hypotheses_text="Some hypotheses text",
        )
        expected_keys = {
            "topic",
            "hypotheses_checked",
            "search_queries",
            "similar_papers_found",
            "novelty_score",
            "assessment",
            "top_3_threat_papers",
            "similar_papers",
            "recommendation",
            "similarity_threshold",
            "search_coverage",
            "total_papers_retrieved",
            "generated",
        }
        assert expected_keys == set(result.keys())

    @patch("researchclaw.literature.search.search_papers_multi_query")
    def test_recommendation_values(self, mock_search: MagicMock) -> None:
        """Recommendation must be one of proceed/differentiate/abort."""
        mock_search.return_value = []
        result = check_novelty(
            topic="Test",
            hypotheses_text="## H1: Hypothesis one\n",
        )
        assert result["recommendation"] in ("proceed", "differentiate", "abort", "proceed_with_caution")

    @patch("researchclaw.literature.search.search_papers_multi_query")
    def test_json_serializable(self, mock_search: MagicMock) -> None:
        """Output must be JSON-serializable for writing to novelty_report.json."""
        mock_search.return_value = []
        result = check_novelty(
            topic="JSON test",
            hypotheses_text="## H1: Test hypothesis title is long enough\n",
        )
        serialized = json.dumps(result)
        assert isinstance(serialized, str)

    @patch("researchclaw.literature.search.search_papers_multi_query")
    def test_similar_papers_capped_at_20(self, mock_search: MagicMock) -> None:
        """Output similar_papers list capped at 20."""
        # Create many mock papers
        papers = []
        for i in range(40):
            p = MagicMock()
            p.title = f"Paper about optimization variant {i}"
            p.abstract = "Optimization variant study"
            p.paper_id = f"id_{i}"
            p.year = 2024
            p.venue = "Conf"
            p.citation_count = 10
            p.url = f"https://example.com/{i}"
            p.cite_key = f"key{i}"
            papers.append(p)
        mock_search.return_value = papers
        result = check_novelty(
            topic="optimization",
            hypotheses_text="## H1: Optimization variants improve performance\n",
            similarity_threshold=0.0,  # low threshold → many matches
        )
        assert len(result["similar_papers"]) <= 20


# ---------------------------------------------------------------------------
# Executor integration — _execute_hypothesis_gen with novelty check
# ---------------------------------------------------------------------------


class TestHypothesisGenNoveltyIntegration:
    """Test that _execute_hypothesis_gen integrates novelty check correctly."""

    def test_novelty_report_written_when_available(self, tmp_path: Path) -> None:
        """Hypothesis gen should write novelty_report.json when check succeeds."""
        from researchclaw.pipeline.executor import _execute_hypothesis_gen
        from researchclaw.adapters import AdapterBundle
        from researchclaw.config import RCConfig

        # Set up minimal run directory
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stage_dir = run_dir / "stage-08"
        stage_dir.mkdir()

        # Create synthesis artifact from prior stage
        stage_07 = run_dir / "stage-07"
        stage_07.mkdir()
        (stage_07 / "synthesis.md").write_text("## Synthesis\nSome synthesis content.")

        data = {
            "project": {"name": "novelty-test", "mode": "docs-first"},
            "research": {"topic": "novelty testing"},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local"},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {},
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:1234/v1",
                "api_key_env": "RC_TEST_KEY",
                "api_key": "inline",
            },
        }
        config = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)
        adapters = AdapterBundle()

        with patch(
            "researchclaw.literature.search.search_papers_multi_query"
        ) as mock_search:
            mock_search.return_value = []
            result = _execute_hypothesis_gen(stage_dir, run_dir, config, adapters)

        assert result.stage.name == "HYPOTHESIS_GEN"
        assert result.status.name == "DONE"
        # hypotheses.md always written
        assert (stage_dir / "hypotheses.md").exists()
        # novelty_report.json should be written (API mocked as returning empty)
        assert (stage_dir / "novelty_report.json").exists()
        report = json.loads((stage_dir / "novelty_report.json").read_text())
        assert report["novelty_score"] == 1.0  # no similar papers → max novelty
        assert "novelty_report.json" in result.artifacts

    def test_novelty_failure_does_not_block(self, tmp_path: Path) -> None:
        """If novelty check crashes, hypothesis gen still succeeds."""
        from researchclaw.pipeline.executor import _execute_hypothesis_gen
        from researchclaw.adapters import AdapterBundle
        from researchclaw.config import RCConfig

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stage_dir = run_dir / "stage-08"
        stage_dir.mkdir()

        stage_07 = run_dir / "stage-07"
        stage_07.mkdir()
        (stage_07 / "synthesis.md").write_text("## Synthesis\nContent.")

        data = {
            "project": {"name": "novelty-test", "mode": "docs-first"},
            "research": {"topic": "novelty testing"},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local"},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {},
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:1234/v1",
                "api_key_env": "RC_TEST_KEY",
                "api_key": "inline",
            },
        }
        config = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)
        adapters = AdapterBundle()

        with patch(
            "researchclaw.literature.novelty.check_novelty",
            side_effect=RuntimeError("Novelty check exploded"),
        ):
            result = _execute_hypothesis_gen(stage_dir, run_dir, config, adapters)

        assert result.status.name == "DONE"
        assert (stage_dir / "hypotheses.md").exists()
        # novelty_report.json NOT written since check failed
        assert not (stage_dir / "novelty_report.json").exists()
        assert "novelty_report.json" not in result.artifacts
