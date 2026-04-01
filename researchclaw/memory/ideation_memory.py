"""Ideation memory — records and retrieves research direction experiences."""

from __future__ import annotations

import logging
from typing import Any

from researchclaw.memory.retriever import MemoryRetriever
from researchclaw.memory.store import MemoryStore

logger = logging.getLogger(__name__)

CATEGORY = "ideation"


class IdeationMemory:
    """Records and retrieves research direction experiences.

    Tracks which topics succeeded or failed, which hypotheses were
    feasible, and builds up anti-patterns to avoid in the future.
    """

    def __init__(
        self,
        store: MemoryStore,
        retriever: MemoryRetriever,
        embed_fn: Any = None,
    ) -> None:
        self._store = store
        self._retriever = retriever
        self._embed_fn = embed_fn

    def record_topic_outcome(
        self,
        topic: str,
        outcome: str,
        quality_score: float,
        run_id: str = "",
    ) -> str:
        """Record the outcome of a research topic.

        Args:
            topic: The research topic description.
            outcome: One of "success", "failure", "abandoned".
            quality_score: Quality score (0-10).
            run_id: Pipeline run identifier.

        Returns:
            The generated memory entry ID.
        """
        content = (
            f"Topic: {topic}\n"
            f"Outcome: {outcome}\n"
            f"Quality: {quality_score:.1f}/10"
        )
        metadata = {
            "type": "topic_outcome",
            "outcome": outcome,
            "quality_score": quality_score,
            "run_id": run_id,
        }

        # Higher quality → higher confidence
        confidence = min(1.0, 0.3 + quality_score / 15.0)
        if outcome == "failure":
            confidence = max(0.5, confidence)  # failures are valuable too

        embedding = self._embed_fn(content) if self._embed_fn else []
        return self._store.add(
            CATEGORY, content, metadata, embedding, confidence
        )

    def record_hypothesis(
        self,
        hypothesis: str,
        feasible: bool,
        reason: str,
        run_id: str = "",
    ) -> str:
        """Record a hypothesis feasibility assessment.

        Args:
            hypothesis: The hypothesis text.
            feasible: Whether it was feasible.
            reason: Reason for the assessment.
            run_id: Pipeline run identifier.

        Returns:
            The generated memory entry ID.
        """
        outcome = "feasible" if feasible else "infeasible"
        content = (
            f"Hypothesis: {hypothesis}\n"
            f"Assessment: {outcome}\n"
            f"Reason: {reason}"
        )
        metadata = {
            "type": "hypothesis",
            "feasible": feasible,
            "run_id": run_id,
        }

        confidence = 0.6 if feasible else 0.7  # infeasible is more informative
        embedding = self._embed_fn(content) if self._embed_fn else []
        return self._store.add(
            CATEGORY, content, metadata, embedding, confidence
        )

    def recall_similar_topics(
        self,
        query: str,
        top_k: int = 5,
    ) -> str:
        """Retrieve similar historical research directions with outcomes.

        Args:
            query: Current research topic or query.
            top_k: Number of results to return.

        Returns:
            Formatted string of similar past topics and their outcomes.
        """
        results = self._retriever.recall_by_text(
            query, category=CATEGORY, top_k=top_k, embed_fn=self._embed_fn
        )
        if not results:
            return ""

        parts = ["### Past Research Directions (from memory)"]
        for i, (entry, score) in enumerate(results, 1):
            outcome = entry.metadata.get("outcome", "unknown")
            quality = entry.metadata.get("quality_score", "?")
            icon = {"success": "+", "failure": "-", "abandoned": "~"}.get(
                outcome, "?"
            )
            parts.append(
                f"{i}. [{icon}] {entry.content.splitlines()[0]} "
                f"(score: {quality}, relevance: {score:.2f})"
            )
        return "\n".join(parts)

    def get_anti_patterns(self) -> list[str]:
        """Get known failure patterns to avoid.

        Returns:
            List of topic descriptions that previously failed.
        """
        entries = self._store.get_all(CATEGORY)
        failures: list[str] = []
        for entry in entries:
            if entry.metadata.get("outcome") == "failure":
                topic_line = entry.content.splitlines()[0]
                reason = entry.metadata.get("reason", "")
                msg = topic_line
                if reason:
                    msg += f" — {reason}"
                failures.append(msg)
        return failures
