"""Tests for researchclaw.assessor — Paper Quality Assessor (Agent D3).

20+ tests covering rubrics, scorer, venue_recommender, and comparator.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from researchclaw.assessor.rubrics import RUBRICS, Rubric
from researchclaw.assessor.scorer import PaperScorer
from researchclaw.assessor.venue_recommender import VenueRecommender
from researchclaw.assessor.comparator import HistoryComparator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sample_paper() -> str:
    return (
        "# Novel Graph Attention Networks\n\n"
        "## Abstract\nWe propose a new method for graph-based learning.\n\n"
        "## Experiments\nWe compare against baseline on CIFAR-10.\n"
        "Results are shown in table 1 and figure 2.\n"
        "Our method achieves 95.2% accuracy.\n"
    ) * 5  # ~500 words


def _sample_scores(overall: float = 7.5) -> dict[str, Any]:
    return {
        "scores": {
            "novelty": 7.0,
            "rigor": 8.0,
            "clarity": 7.0,
            "impact": 7.5,
            "experiments": 8.0,
        },
        "overall": overall,
    }


class MockLLM:
    """Minimal mock LLM client."""

    def __init__(self, response: str = "SCORE: 7\nREASON: Solid contribution"):
        self.response = response

    async def chat_async(self, prompt: str) -> str:
        return self.response


class FailingLLM:
    async def chat_async(self, prompt: str) -> str:
        raise RuntimeError("API error")


# ===================================================================
# Rubric tests
# ===================================================================


class TestRubrics:
    def test_all_five_dimensions_present(self):
        assert set(RUBRICS.keys()) == {
            "novelty", "rigor", "clarity", "impact", "experiments"
        }

    def test_rubric_is_frozen(self):
        r = RUBRICS["novelty"]
        with pytest.raises(AttributeError):
            r.name = "changed"  # type: ignore[misc]

    def test_rubric_has_criteria_and_scale(self):
        for dim, rubric in RUBRICS.items():
            assert rubric.criteria, f"{dim} missing criteria"
            assert rubric.scale, f"{dim} missing scale"

    def test_default_weight(self):
        r = Rubric(name="test", criteria="test criteria", scale="1-10")
        assert r.weight == 1.0


# ===================================================================
# PaperScorer tests
# ===================================================================


class TestPaperScorer:
    def test_score_without_llm(self):
        scorer = PaperScorer()
        result = asyncio.run(scorer.score(_sample_paper()))
        assert "overall" in result
        assert "scores" in result
        assert isinstance(result["overall"], float)
        assert len(result["dimensions_evaluated"]) == 5

    def test_score_with_mock_llm(self):
        llm = MockLLM("SCORE: 8\nREASON: Excellent work")
        scorer = PaperScorer(llm_client=llm)
        result = asyncio.run(scorer.score(_sample_paper()))
        assert result["overall"] == 8.0
        for dim in result["scores"]:
            assert result["scores"][dim] == 8.0

    def test_score_with_failing_llm_falls_back(self):
        scorer = PaperScorer(llm_client=FailingLLM())
        result = asyncio.run(scorer.score(_sample_paper()))
        # Should still return valid scores via heuristic
        assert "overall" in result
        assert result["overall"] > 0

    def test_score_subset_dimensions(self):
        scorer = PaperScorer(dimensions=("novelty", "clarity"))
        result = asyncio.run(scorer.score(_sample_paper()))
        assert len(result["dimensions_evaluated"]) == 2

    def test_parse_score_valid(self):
        score, reason = PaperScorer._parse_score_response(
            "SCORE: 9\nREASON: Breakthrough paper", "novelty"
        )
        assert score == 9.0
        assert reason == "Breakthrough paper"

    def test_parse_score_clamped(self):
        score, _ = PaperScorer._parse_score_response("SCORE: 15", "test")
        assert score == 10.0
        score, _ = PaperScorer._parse_score_response("SCORE: 0", "test")
        assert score == 1.0

    def test_parse_score_missing(self):
        score, reason = PaperScorer._parse_score_response("No format here", "test")
        assert score == 5.0  # default
        assert reason == "No detail provided"

    def test_heuristic_clarity_long_paper(self):
        long_paper = "word " * 4000
        score, detail = PaperScorer._heuristic_score(long_paper, RUBRICS["clarity"])
        assert score == 6.0
        assert "4000" in detail

    def test_heuristic_clarity_short_paper(self):
        short_paper = "word " * 500
        score, _ = PaperScorer._heuristic_score(short_paper, RUBRICS["clarity"])
        assert score == 3.0

    def test_heuristic_experiments_with_table_and_figure(self):
        paper = "Results in table 1 and figure 3 show improvements."
        score, _ = PaperScorer._heuristic_score(paper, RUBRICS["experiments"])
        assert score == 7.0  # 4.0 + 1.5 + 1.5

    def test_heuristic_experiments_no_evidence(self):
        paper = "We discuss theoretical implications."
        score, _ = PaperScorer._heuristic_score(paper, RUBRICS["experiments"])
        assert score == 4.0

    def test_heuristic_default_dimension(self):
        paper = "Some paper content"
        score, reason = PaperScorer._heuristic_score(paper, RUBRICS["novelty"])
        assert score == 5.0
        assert "default" in reason.lower()


# ===================================================================
# VenueRecommender tests
# ===================================================================


class TestVenueRecommender:
    def test_recommend_high_score(self):
        rec = VenueRecommender()
        scores = _sample_scores(overall=9.0)
        results = rec.recommend(scores)
        # Should include tier 1 venues
        tier_1_venues = [r for r in results if r["tier"] == "tier_1"]
        assert len(tier_1_venues) > 0

    def test_recommend_low_score(self):
        rec = VenueRecommender()
        scores = _sample_scores(overall=2.0)
        results = rec.recommend(scores)
        assert len(results) == 0

    def test_recommend_medium_score_no_tier1(self):
        rec = VenueRecommender()
        scores = _sample_scores(overall=5.0)
        results = rec.recommend(scores)
        tier_1 = [r for r in results if r["tier"] == "tier_1"]
        assert len(tier_1) == 0

    def test_recommend_filter_by_domain(self):
        rec = VenueRecommender()
        scores = _sample_scores(overall=9.0)
        results = rec.recommend(scores, domains=["cv"])
        for r in results:
            assert "cv" in r["venue_domains"] or "deep-learning" in r["venue_domains"]

    def test_get_suggestion_weak_dimension(self):
        scores = {"scores": {"novelty": 3, "clarity": 8}, "overall": 5.5}
        suggestion = VenueRecommender._get_suggestion("ICML", scores)
        assert "novelty" in suggestion.lower()
        assert "Strengthen" in suggestion

    def test_get_suggestion_moderate(self):
        scores = {"scores": {"novelty": 6, "clarity": 8}, "overall": 7.0}
        suggestion = VenueRecommender._get_suggestion("ICML", scores)
        assert "improving" in suggestion.lower()

    def test_get_suggestion_strong(self):
        scores = {"scores": {"novelty": 8, "clarity": 9}, "overall": 8.5}
        suggestion = VenueRecommender._get_suggestion("ICML", scores)
        assert "strong" in suggestion.lower()

    def test_get_suggestion_no_scores(self):
        scores = {"overall": 5.0}
        suggestion = VenueRecommender._get_suggestion("ICML", scores)
        assert "Evaluate" in suggestion

    def test_format_recommendations_empty(self):
        rec = VenueRecommender()
        output = rec.format_recommendations([])
        assert "No suitable venues" in output

    def test_format_recommendations_with_data(self):
        rec = VenueRecommender()
        results = rec.recommend(_sample_scores(overall=9.0))
        output = rec.format_recommendations(results)
        assert "Venue Recommendations" in output


# ===================================================================
# HistoryComparator tests
# ===================================================================


class TestHistoryComparator:
    def test_record_and_get_history(self, tmp_path: Path):
        comp = HistoryComparator(history_dir=tmp_path)
        comp.record("run-1", "topic A", _sample_scores(7.5))
        history = comp.get_history()
        assert len(history) == 1
        assert history[0]["run_id"] == "run-1"

    def test_record_persists_to_disk(self, tmp_path: Path):
        comp = HistoryComparator(history_dir=tmp_path)
        comp.record("run-1", "topic A", _sample_scores(7.5))
        # Reload from disk
        comp2 = HistoryComparator(history_dir=tmp_path)
        assert len(comp2.get_history()) == 1

    def test_compare_no_history(self):
        comp = HistoryComparator()
        result = comp.compare(_sample_scores(8.0))
        assert result["comparison"] == "no_history"

    def test_compare_with_previous(self, tmp_path: Path):
        comp = HistoryComparator(history_dir=tmp_path)
        comp.record("run-1", "topic A", _sample_scores(6.0))
        result = comp.compare(_sample_scores(8.0), previous_run_id="run-1")
        assert result["comparison"] == "success"
        assert result["delta"] == 2.0
        assert result["trend"] == "improved"

    def test_compare_stable_trend(self, tmp_path: Path):
        comp = HistoryComparator(history_dir=tmp_path)
        comp.record("run-1", "topic A", _sample_scores(7.5))
        result = comp.compare(_sample_scores(7.5))
        assert result["trend"] == "stable"

    def test_compare_declined_trend(self, tmp_path: Path):
        comp = HistoryComparator(history_dir=tmp_path)
        comp.record("run-1", "topic A", _sample_scores(9.0))
        result = comp.compare(_sample_scores(7.0))
        assert result["trend"] == "declined"

    def test_compare_not_found(self, tmp_path: Path):
        comp = HistoryComparator(history_dir=tmp_path)
        comp.record("run-1", "topic A", _sample_scores(7.0))
        result = comp.compare(_sample_scores(8.0), previous_run_id="nonexistent")
        assert result["comparison"] == "not_found"

    def test_get_best_run(self, tmp_path: Path):
        comp = HistoryComparator(history_dir=tmp_path)
        comp.record("run-1", "topic A", _sample_scores(6.0))
        comp.record("run-2", "topic B", _sample_scores(9.0))
        comp.record("run-3", "topic C", _sample_scores(7.5))
        best = comp.get_best_run()
        assert best is not None
        assert best["run_id"] == "run-2"

    def test_get_best_run_empty(self):
        comp = HistoryComparator()
        assert comp.get_best_run() is None

    def test_dimension_deltas(self, tmp_path: Path):
        comp = HistoryComparator(history_dir=tmp_path)
        scores_old = {
            "scores": {"novelty": 5.0, "clarity": 6.0},
            "overall": 5.5,
        }
        scores_new = {
            "scores": {"novelty": 7.0, "clarity": 8.0},
            "overall": 7.5,
        }
        comp.record("run-1", "topic A", scores_old)
        result = comp.compare(scores_new, previous_run_id="run-1")
        assert result["dimension_deltas"]["novelty"] == 2.0
        assert result["dimension_deltas"]["clarity"] == 2.0
