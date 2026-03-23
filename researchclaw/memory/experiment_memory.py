"""Experiment memory — records and retrieves experiment experiences."""

from __future__ import annotations

import json
import logging
from typing import Any

from researchclaw.memory.retriever import MemoryRetriever
from researchclaw.memory.store import MemoryStore

logger = logging.getLogger(__name__)

CATEGORY = "experiment"


class ExperimentMemory:
    """Records and retrieves experiment experiences.

    Tracks hyperparameter configurations, model architectures, and
    training tricks that worked (or failed) in past runs.
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

    def record_hyperparams(
        self,
        task_type: str,
        hyperparams: dict[str, Any],
        metric: float,
        metric_name: str = "primary_metric",
        run_id: str = "",
    ) -> str:
        """Record an effective hyperparameter configuration.

        Args:
            task_type: Type of task (e.g., "image_classification").
            hyperparams: Dict of hyperparameter values.
            metric: Achieved metric value.
            metric_name: Name of the metric.
            run_id: Pipeline run identifier.

        Returns:
            The generated memory entry ID.
        """
        hp_str = json.dumps(hyperparams, indent=2, default=str)
        content = (
            f"Task: {task_type}\n"
            f"Hyperparameters:\n{hp_str}\n"
            f"Result: {metric_name}={metric:.4f}"
        )
        metadata = {
            "type": "hyperparams",
            "task_type": task_type,
            "hyperparams": hyperparams,
            "metric": metric,
            "metric_name": metric_name,
            "run_id": run_id,
        }

        # Higher metric → higher confidence (assuming maximize)
        confidence = min(1.0, 0.4 + metric * 0.5)
        embedding = self._embed_fn(content) if self._embed_fn else []
        return self._store.add(
            CATEGORY, content, metadata, embedding, confidence
        )

    def record_architecture(
        self,
        task_type: str,
        architecture: str,
        metric: float,
        run_id: str = "",
    ) -> str:
        """Record a successful model architecture.

        Args:
            task_type: Type of task.
            architecture: Architecture description.
            metric: Achieved metric value.
            run_id: Pipeline run identifier.

        Returns:
            The generated memory entry ID.
        """
        content = (
            f"Task: {task_type}\n"
            f"Architecture: {architecture}\n"
            f"Performance: {metric:.4f}"
        )
        metadata = {
            "type": "architecture",
            "task_type": task_type,
            "architecture": architecture,
            "metric": metric,
            "run_id": run_id,
        }

        confidence = min(1.0, 0.4 + metric * 0.5)
        embedding = self._embed_fn(content) if self._embed_fn else []
        return self._store.add(
            CATEGORY, content, metadata, embedding, confidence
        )

    def record_training_trick(
        self,
        trick: str,
        improvement: float,
        context: str,
        run_id: str = "",
    ) -> str:
        """Record an effective training trick.

        Args:
            trick: Description of the trick.
            improvement: Relative improvement (e.g., 0.05 for 5%).
            context: When/where the trick was applied.
            run_id: Pipeline run identifier.

        Returns:
            The generated memory entry ID.
        """
        content = (
            f"Trick: {trick}\n"
            f"Improvement: {improvement:+.1%}\n"
            f"Context: {context}"
        )
        metadata = {
            "type": "training_trick",
            "trick": trick,
            "improvement": improvement,
            "context": context,
            "run_id": run_id,
        }

        confidence = 0.6 if improvement > 0 else 0.3
        embedding = self._embed_fn(content) if self._embed_fn else []
        return self._store.add(
            CATEGORY, content, metadata, embedding, confidence
        )

    def recall_best_configs(
        self,
        task_type: str,
        top_k: int = 3,
    ) -> str:
        """Retrieve best configurations for a task type.

        Args:
            task_type: Description of the current task.
            top_k: Number of results.

        Returns:
            Formatted string of best configurations.
        """
        query = f"best hyperparameters and architecture for {task_type}"
        results = self._retriever.recall_by_text(
            query, category=CATEGORY, top_k=top_k, embed_fn=self._embed_fn
        )
        if not results:
            return ""

        parts = ["### Best Experiment Configurations (from memory)"]
        for i, (entry, score) in enumerate(results, 1):
            metric = entry.metadata.get("metric", "?")
            parts.append(
                f"{i}. {entry.content.splitlines()[0]} "
                f"(metric: {metric}, relevance: {score:.2f})"
            )
            # Include hyperparams detail if available
            hp = entry.metadata.get("hyperparams")
            if hp:
                parts.append(f"   Config: {json.dumps(hp, default=str)}")
        return "\n".join(parts)
