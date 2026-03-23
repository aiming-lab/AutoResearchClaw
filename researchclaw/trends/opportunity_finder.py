"""Research opportunity discovery."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class OpportunityFinder:
    """Identify research opportunities from trend analysis."""

    def __init__(self, llm_client: Any = None):
        self.llm = llm_client

    async def find_opportunities(
        self,
        trend_analysis: dict[str, Any],
        domains: list[str],
    ) -> list[dict[str, Any]]:
        """Identify research gaps and opportunities."""
        if self.llm is not None:
            return await self._llm_find_opportunities(trend_analysis, domains)
        return self._heuristic_find_opportunities(trend_analysis, domains)

    async def _llm_find_opportunities(
        self,
        trend_analysis: dict[str, Any],
        domains: list[str],
    ) -> list[dict[str, Any]]:
        """Use LLM to identify research opportunities."""
        keywords = trend_analysis.get("rising_keywords", [])[:10]
        methods = trend_analysis.get("method_trends", [])[:5]

        prompt = (
            "Based on the following research trends, identify 5 promising "
            "research opportunities:\n\n"
            f"Domains: {', '.join(domains)}\n"
            f"Trending keywords: {[k['keyword'] for k in keywords]}\n"
            f"Popular methods: {[m['method'] for m in methods]}\n\n"
            "For each opportunity, provide:\n"
            "1. A concise research question\n"
            "2. Why it's promising (1 sentence)\n"
            "3. Feasibility estimate (high/medium/low)\n\n"
            "Format each as: TOPIC: ... | WHY: ... | FEASIBILITY: ..."
        )

        try:
            response = await self.llm.chat_async(prompt)
            return self._parse_opportunities(response)
        except Exception as exc:
            logger.warning("LLM opportunity finding failed: %s", exc)
            return self._heuristic_find_opportunities(trend_analysis, domains)

    @staticmethod
    def _parse_opportunities(response: str) -> list[dict[str, Any]]:
        """Parse LLM response into structured opportunities."""
        opportunities = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line or not any(
                marker in line for marker in ("TOPIC:", "topic:", "1.", "2.", "3.")
            ):
                continue

            parts = line.split("|")
            topic = parts[0].split(":", 1)[-1].strip() if parts else line
            why = parts[1].split(":", 1)[-1].strip() if len(parts) > 1 else ""
            feasibility = (
                parts[2].split(":", 1)[-1].strip().lower()
                if len(parts) > 2
                else "medium"
            )

            if topic:
                opportunities.append({
                    "topic": topic,
                    "rationale": why,
                    "feasibility": feasibility,
                    "source": "llm",
                })

        return opportunities[:5]

    @staticmethod
    def _heuristic_find_opportunities(
        trend_analysis: dict[str, Any],
        domains: list[str],
    ) -> list[dict[str, Any]]:
        """Simple heuristic-based opportunity finding."""
        opportunities: list[dict[str, Any]] = []
        keywords = trend_analysis.get("rising_keywords", [])
        methods = trend_analysis.get("method_trends", [])

        # Combine trending keywords with methods for opportunity generation
        for i, kw in enumerate(keywords[:3]):
            for j, method in enumerate(methods[:2]):
                topic = (
                    f"Applying {method['method']} to "
                    f"{kw['keyword']} in {domains[0] if domains else 'ML'}"
                )
                opportunities.append({
                    "topic": topic,
                    "rationale": (
                        f"'{kw['keyword']}' is trending ({kw['count']} mentions) "
                        f"and '{method['method']}' is a popular method"
                    ),
                    "feasibility": "medium",
                    "source": "heuristic",
                })
                if len(opportunities) >= 5:
                    break
            if len(opportunities) >= 5:
                break

        return opportunities
