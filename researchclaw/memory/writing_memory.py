"""Writing memory — records and retrieves writing experiences."""

from __future__ import annotations

import logging
from typing import Any

from researchclaw.memory.retriever import MemoryRetriever
from researchclaw.memory.store import MemoryStore

logger = logging.getLogger(__name__)

CATEGORY = "writing"


class WritingMemory:
    """Records and retrieves writing experiences.

    Tracks review feedback and resolutions, successful paper structure
    patterns, and writing tips that improved paper quality.
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

    def record_review_feedback(
        self,
        feedback_type: str,
        feedback: str,
        resolution: str,
        run_id: str = "",
    ) -> str:
        """Record review feedback and its resolution.

        Args:
            feedback_type: Type of feedback (e.g., "clarity", "novelty", "methodology").
            feedback: The reviewer's feedback text.
            resolution: How the feedback was addressed.
            run_id: Pipeline run identifier.

        Returns:
            The generated memory entry ID.
        """
        content = (
            f"Feedback type: {feedback_type}\n"
            f"Issue: {feedback}\n"
            f"Resolution: {resolution}"
        )
        metadata = {
            "type": "review_feedback",
            "feedback_type": feedback_type,
            "run_id": run_id,
        }

        confidence = 0.7  # review feedback is usually reliable
        embedding = self._embed_fn(content) if self._embed_fn else []
        return self._store.add(
            CATEGORY, content, metadata, embedding, confidence
        )

    def record_successful_structure(
        self,
        section: str,
        structure: str,
        score: float,
        run_id: str = "",
    ) -> str:
        """Record a high-scoring paper structure pattern.

        Args:
            section: Paper section (e.g., "introduction", "method").
            structure: Description of the structure pattern.
            score: Quality score achieved.
            run_id: Pipeline run identifier.

        Returns:
            The generated memory entry ID.
        """
        content = (
            f"Section: {section}\n"
            f"Structure: {structure}\n"
            f"Quality score: {score:.1f}/10"
        )
        metadata = {
            "type": "structure_pattern",
            "section": section,
            "score": score,
            "run_id": run_id,
        }

        confidence = min(1.0, 0.3 + score / 15.0)
        embedding = self._embed_fn(content) if self._embed_fn else []
        return self._store.add(
            CATEGORY, content, metadata, embedding, confidence
        )

    def recall_writing_tips(
        self,
        section: str,
        context: str,
        top_k: int = 5,
    ) -> str:
        """Retrieve writing tips relevant to the current task.

        Args:
            section: Current paper section being written.
            context: Additional context (topic, methodology, etc.).
            top_k: Number of results.

        Returns:
            Formatted string of writing tips.
        """
        query = f"writing tips for {section} section: {context}"
        results = self._retriever.recall_by_text(
            query, category=CATEGORY, top_k=top_k, embed_fn=self._embed_fn
        )
        if not results:
            return ""

        parts = ["### Writing Experience (from memory)"]
        for i, (entry, score) in enumerate(results, 1):
            entry_type = entry.metadata.get("type", "tip")
            parts.append(
                f"{i}. [{entry_type}] {entry.content.splitlines()[0]} "
                f"(relevance: {score:.2f})"
            )
            # Show resolution for review feedback
            if entry_type == "review_feedback":
                lines = entry.content.splitlines()
                for line in lines:
                    if line.startswith("Resolution:"):
                        parts.append(f"   {line}")
                        break
        return "\n".join(parts)
