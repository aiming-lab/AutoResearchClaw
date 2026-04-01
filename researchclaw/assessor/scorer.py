"""Multi-dimensional paper quality scorer."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from researchclaw.assessor.rubrics import RUBRICS, Rubric

logger = logging.getLogger(__name__)


class PaperScorer:
    """Score a paper across multiple quality dimensions using an LLM."""

    def __init__(
        self,
        dimensions: tuple[str, ...] | None = None,
        llm_client: Any = None,
    ):
        self.dimensions = dimensions or tuple(RUBRICS.keys())
        self.llm = llm_client

    async def score(
        self,
        paper_md: str,
        experiment_results: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Score a paper across all configured dimensions."""
        scores: dict[str, float] = {}
        details: dict[str, str] = {}

        for dim in self.dimensions:
            rubric = RUBRICS.get(dim)
            if rubric is None:
                logger.warning("Unknown rubric dimension: %s", dim)
                continue
            score, detail = await self._score_dimension(
                paper_md, experiment_results, rubric
            )
            scores[dim] = score
            details[dim] = detail

        if scores:
            total_weight = sum(
                RUBRICS[d].weight for d in scores if d in RUBRICS
            )
            if total_weight > 0:
                weighted_sum = sum(
                    scores[d] * RUBRICS[d].weight
                    for d in scores
                    if d in RUBRICS
                )
                overall = round(weighted_sum / total_weight, 2)
            else:
                overall = round(sum(scores.values()) / len(scores), 2)
        else:
            overall = 0.0

        return {
            "scores": scores,
            "overall": overall,
            "details": details,
            "dimensions_evaluated": list(scores.keys()),
        }

    async def _score_dimension(
        self,
        paper_md: str,
        experiment_results: dict[str, Any] | None,
        rubric: Rubric,
    ) -> tuple[float, str]:
        """Score a single dimension using the LLM."""
        if self.llm is None:
            return self._heuristic_score(paper_md, rubric)

        exp_context = ""
        if experiment_results:
            exp_context = f"\n\nExperiment results summary:\n{json.dumps(experiment_results, indent=2, default=str)[:2000]}"

        prompt = (
            f"Rate the following research paper on '{rubric.name}' from 1 to 10.\n\n"
            f"Criteria: {rubric.criteria}\n"
            f"Scale: {rubric.scale}\n\n"
            f"Paper content (first 6000 chars):\n{paper_md[:6000]}"
            f"{exp_context}\n\n"
            f"Respond in this exact format:\n"
            f"SCORE: <number 1-10>\n"
            f"REASON: <one sentence explanation>"
        )

        try:
            response = await self.llm.chat_async(prompt)
            return self._parse_score_response(response, rubric.name)
        except Exception as exc:
            logger.warning("LLM scoring failed for %s: %s", rubric.name, exc)
            return self._heuristic_score(paper_md, rubric)

    @staticmethod
    def _parse_score_response(
        response: str,
        dim_name: str,
    ) -> tuple[float, str]:
        """Parse LLM score response."""
        score_match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", response)
        reason_match = re.search(r"REASON:\s*(.+)", response)

        if score_match:
            score = float(score_match.group(1))
            score = max(1.0, min(10.0, score))
        else:
            score = 5.0

        reason = reason_match.group(1).strip() if reason_match else "No detail provided"
        return score, reason

    @staticmethod
    def _heuristic_score(
        paper_md: str,
        rubric: Rubric,
    ) -> tuple[float, str]:
        """Simple heuristic scoring when LLM is unavailable."""
        word_count = len(paper_md.split())

        if rubric.name == "Clarity":
            if word_count > 3000:
                score = 6.0
            elif word_count > 1000:
                score = 5.0
            else:
                score = 3.0
            return score, f"Heuristic: {word_count} words"

        if rubric.name == "Experiments":
            has_table = "table" in paper_md.lower() or "|" in paper_md
            has_figure = "figure" in paper_md.lower() or "fig." in paper_md.lower()
            score = 4.0
            if has_table:
                score += 1.5
            if has_figure:
                score += 1.5
            return min(score, 10.0), "Heuristic: table/figure presence"

        return 5.0, "Heuristic: default score (no LLM)"
