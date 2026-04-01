"""Automatic research topic generation (ClawZero mode)."""

from __future__ import annotations

import logging
from typing import Any

from researchclaw.trends.opportunity_finder import OpportunityFinder
from researchclaw.trends.trend_analyzer import TrendAnalyzer

logger = logging.getLogger(__name__)


class AutoTopicGenerator:
    """Generate and rank candidate research topics automatically."""

    def __init__(
        self,
        trend_analyzer: TrendAnalyzer,
        opportunity_finder: OpportunityFinder,
        llm_client: Any = None,
    ):
        self.trend_analyzer = trend_analyzer
        self.opportunity_finder = opportunity_finder
        self.llm = llm_client

    async def generate_candidates(
        self,
        domains: list[str],
        papers: list[dict[str, Any]] | None = None,
        count: int = 5,
    ) -> list[dict[str, Any]]:
        """Generate ranked candidate research topics."""
        # 1. Analyze trends
        trend_analysis = self.trend_analyzer.analyze(papers or [])

        # 2. Find opportunities
        opportunities = await self.opportunity_finder.find_opportunities(
            trend_analysis, domains
        )

        # 3. Score and rank candidates
        candidates = []
        for opp in opportunities[:count]:
            score = self._score_candidate(opp, trend_analysis)
            candidates.append({
                "topic": opp["topic"],
                "rationale": opp.get("rationale", ""),
                "feasibility": opp.get("feasibility", "medium"),
                "novelty_score": score["novelty"],
                "feasibility_score": score["feasibility"],
                "impact_score": score["impact"],
                "overall_score": score["overall"],
                "source": opp.get("source", "unknown"),
            })

        candidates.sort(key=lambda c: -c["overall_score"])
        return candidates[:count]

    async def auto_select(
        self,
        domains: list[str],
        papers: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Fully automatic topic selection (Zero-Touch mode)."""
        candidates = await self.generate_candidates(domains, papers, count=5)
        if not candidates:
            return {
                "topic": f"Novel approaches in {domains[0] if domains else 'ML'}",
                "rationale": "Default topic (no trends data available)",
                "overall_score": 0.0,
                "source": "default",
            }
        return candidates[0]

    @staticmethod
    def _score_candidate(
        opportunity: dict[str, Any],
        trend_analysis: dict[str, Any],
    ) -> dict[str, float]:
        """Score a candidate topic on novelty, feasibility, and impact."""
        feasibility_map = {"high": 0.9, "medium": 0.6, "low": 0.3}
        feasibility = feasibility_map.get(
            opportunity.get("feasibility", "medium"), 0.6
        )

        # Novelty: inverse of how much it's already been studied
        topic_words = set(opportunity.get("topic", "").lower().split())
        keyword_overlap = 0
        for kw in trend_analysis.get("rising_keywords", []):
            kw_words = set(kw.get("keyword", "").lower().split())
            if topic_words & kw_words:
                keyword_overlap += 1

        novelty = max(0.3, 1.0 - keyword_overlap * 0.15)

        # Impact: based on trend momentum
        paper_count = trend_analysis.get("paper_count", 0)
        impact = min(1.0, paper_count / 50) if paper_count > 0 else 0.5

        overall = round(
            0.4 * novelty + 0.3 * feasibility + 0.3 * impact, 3
        )

        return {
            "novelty": round(novelty, 3),
            "feasibility": round(feasibility, 3),
            "impact": round(impact, 3),
            "overall": overall,
        }

    def format_candidates(
        self,
        candidates: list[dict[str, Any]],
    ) -> str:
        """Format candidates as a readable string."""
        if not candidates:
            return "No candidate topics generated."

        lines = ["Candidate Research Topics:", "=" * 40, ""]
        for i, c in enumerate(candidates, 1):
            lines.extend([
                f"{i}. {c['topic']}",
                f"   Score: {c['overall_score']:.2f} "
                f"(novelty={c['novelty_score']:.2f}, "
                f"feasibility={c['feasibility_score']:.2f}, "
                f"impact={c['impact_score']:.2f})",
                f"   Rationale: {c.get('rationale', 'N/A')}",
                "",
            ])
        return "\n".join(lines)
