"""Similarity-based memory retrieval engine.

Combines cosine similarity with time-decay and confidence weighting
to return the most relevant memory entries for a given query.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

from researchclaw.memory.decay import time_decay_weight
from researchclaw.memory.store import MemoryEntry, MemoryStore

logger = logging.getLogger(__name__)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1, 1], or 0.0 if either vector is zero.
    """
    if len(a) != len(b) or not a:
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


class MemoryRetriever:
    """Retrieves relevant memories using semantic similarity.

    Scoring formula:
        score = sim_weight * cosine_sim
              + decay_weight * time_decay
              + conf_weight * confidence
              + access_weight * normalized_access_count
    """

    def __init__(
        self,
        store: MemoryStore,
        half_life_days: float = 90.0,
        sim_weight: float = 0.5,
        decay_weight: float = 0.2,
        conf_weight: float = 0.2,
        access_weight: float = 0.1,
    ) -> None:
        self._store = store
        self._half_life_days = half_life_days
        self._sim_weight = sim_weight
        self._decay_weight = decay_weight
        self._conf_weight = conf_weight
        self._access_weight = access_weight

    def recall(
        self,
        query_embedding: list[float],
        category: str | None = None,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryEntry, float]]:
        """Retrieve most relevant memories for a query embedding.

        Args:
            query_embedding: Query vector.
            category: Filter by category (None for all).
            top_k: Maximum number of results.
            min_score: Minimum composite score threshold.

        Returns:
            List of (entry, score) tuples sorted by relevance.
        """
        entries = self._store.get_all(category)
        if not entries:
            return []

        # Find max access count for normalization
        max_access = max((e.access_count for e in entries), default=1)
        if max_access == 0:
            max_access = 1

        scored: list[tuple[MemoryEntry, float]] = []
        now = datetime.now(timezone.utc)

        for entry in entries:
            # Cosine similarity
            sim = cosine_similarity(query_embedding, entry.embedding)

            # Time decay
            try:
                created = datetime.fromisoformat(entry.created_at)
            except (ValueError, TypeError):
                created = now
            decay = time_decay_weight(
                created, half_life_days=self._half_life_days, now=now
            )

            # Normalized access frequency
            norm_access = entry.access_count / max_access

            # Composite score
            score = (
                self._sim_weight * sim
                + self._decay_weight * decay
                + self._conf_weight * entry.confidence
                + self._access_weight * norm_access
            )

            if score >= min_score:
                scored.append((entry, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Mark top results as accessed
        for entry, _ in scored[:top_k]:
            self._store.mark_accessed(entry.id)

        return scored[:top_k]

    def recall_by_text(
        self,
        query: str,
        category: str | None = None,
        top_k: int = 5,
        embed_fn: Any = None,
    ) -> list[tuple[MemoryEntry, float]]:
        """Retrieve memories using text query (requires embed function).

        Args:
            query: Text query string.
            category: Filter by category.
            top_k: Maximum results.
            embed_fn: Callable that converts text to embedding vector.

        Returns:
            List of (entry, score) tuples.
        """
        if embed_fn is None:
            logger.warning("No embedding function provided for text recall")
            return []

        query_embedding = embed_fn(query)
        return self.recall(query_embedding, category=category, top_k=top_k)

    def format_for_prompt(
        self,
        results: list[tuple[MemoryEntry, float]],
        max_chars: int = 3000,
    ) -> str:
        """Format retrieval results as prompt injection text.

        Args:
            results: List of (entry, score) tuples from recall().
            max_chars: Maximum character count for output.

        Returns:
            Formatted string suitable for LLM prompt injection.
        """
        if not results:
            return ""

        parts: list[str] = []
        total_len = 0

        for i, (entry, score) in enumerate(results, 1):
            line = f"{i}. [{entry.category}] (relevance: {score:.2f}) {entry.content}"
            if total_len + len(line) > max_chars:
                break
            parts.append(line)
            total_len += len(line)

        return "\n".join(parts)
