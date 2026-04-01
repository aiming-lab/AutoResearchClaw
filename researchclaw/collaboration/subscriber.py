"""Artifact subscriber — queries and imports shared artifacts."""

from __future__ import annotations

import logging
from typing import Any

from researchclaw.collaboration.repository import ResearchRepository

logger = logging.getLogger(__name__)


class ArtifactSubscriber:
    """Subscribes to and imports artifacts from the shared repository.

    Provides convenience methods for finding and importing relevant
    artifacts for a new pipeline run.
    """

    def __init__(self, repository: ResearchRepository) -> None:
        self._repo = repository

    def find_relevant_literature(
        self,
        topic: str,
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Find literature summaries relevant to a topic.

        Args:
            topic: Research topic to search for.
            max_results: Maximum results.

        Returns:
            List of matching literature artifacts.
        """
        return self._repo.search(
            topic, artifact_type="literature_summary", max_results=max_results
        )

    def find_similar_experiments(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Find experiment results similar to the current task.

        Args:
            query: Search query describing the experiment.
            max_results: Maximum results.

        Returns:
            List of matching experiment artifacts.
        """
        return self._repo.search(
            query, artifact_type="experiment_results", max_results=max_results
        )

    def find_code_templates(
        self,
        query: str,
        max_results: int = 3,
    ) -> list[dict[str, Any]]:
        """Find reusable code templates.

        Args:
            query: Search query.
            max_results: Maximum results.

        Returns:
            List of matching code template artifacts.
        """
        return self._repo.search(
            query, artifact_type="code_template", max_results=max_results
        )

    def import_best_practices(
        self,
        topic: str,
    ) -> str:
        """Compile best practices from historical runs for a topic.

        Args:
            topic: Current research topic.

        Returns:
            Formatted string of best practices for prompt injection.
        """
        parts: list[str] = []

        # Literature insights
        lit_results = self.find_relevant_literature(topic, max_results=3)
        if lit_results:
            parts.append("### Related Literature (from prior runs)")
            for result in lit_results:
                content = result.get("content", "")
                if isinstance(content, str):
                    parts.append(f"- {content[:200]}...")
                run_id = result.get("run_id", "?")
                parts.append(f"  (from run: {run_id})")

        # Experiment insights
        exp_results = self.find_similar_experiments(topic, max_results=3)
        if exp_results:
            parts.append("\n### Related Experiments (from prior runs)")
            for result in exp_results:
                run_id = result.get("run_id", "?")
                parts.append(f"- Experiment from run {run_id}")

        if not parts:
            return ""

        return "\n".join(parts)
