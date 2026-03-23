"""Knowledge graph query engine."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any

from researchclaw.knowledge.graph.builder import KnowledgeGraphBuilder
from researchclaw.knowledge.graph.entities import EntityType
from researchclaw.knowledge.graph.relations import RelationType

logger = logging.getLogger(__name__)


class KnowledgeGraphQuery:
    """Query engine for the research knowledge graph.

    Provides high-level research-oriented queries like finding gaps,
    trending methods, method comparisons, and topic suggestions.
    """

    def __init__(self, graph: KnowledgeGraphBuilder) -> None:
        self._graph = graph

    def find_research_gaps(self, domain: str = "") -> list[str]:
        """Find research gaps — datasets without many methods applied.

        Args:
            domain: Optional domain filter.

        Returns:
            List of gap descriptions.
        """
        datasets = self._graph.get_entities_by_type(EntityType.DATASET)
        methods = self._graph.get_entities_by_type(EntityType.METHOD)

        if domain:
            datasets = [
                d for d in datasets
                if domain.lower() in d.attributes.get("domain", "").lower()
                or domain.lower() in d.name.lower()
            ]

        gaps: list[str] = []
        for dataset in datasets:
            # Count methods applied to this dataset
            rels = self._graph.get_relations_for(dataset.id, direction="incoming")
            method_rels = [
                r for r in rels if r.relation_type == RelationType.USES_DATASET
            ]
            if len(method_rels) < 2 and methods:
                gaps.append(
                    f"Dataset '{dataset.name}' has only {len(method_rels)} "
                    f"method(s) evaluated — potential research opportunity"
                )

        return gaps

    def find_trending_methods(self, min_citations: int = 2) -> list[str]:
        """Find methods with high citation/usage counts.

        Args:
            min_citations: Minimum citation count to qualify.

        Returns:
            List of trending method descriptions.
        """
        methods = self._graph.get_entities_by_type(EntityType.METHOD)
        trending: list[tuple[int, str]] = []

        for method in methods:
            rels = self._graph.get_relations_for(method.id, direction="incoming")
            citation_count = len([
                r for r in rels
                if r.relation_type in (
                    RelationType.EXTENDS,
                    RelationType.APPLIES_METHOD,
                    RelationType.CITES,
                )
            ])
            if citation_count >= min_citations:
                trending.append((citation_count, method.name))

        trending.sort(reverse=True)
        return [
            f"'{name}' — referenced {count} time(s)"
            for count, name in trending
        ]

    def get_method_comparison(
        self,
        method_a: str,
        method_b: str,
    ) -> dict[str, Any]:
        """Compare two methods across shared datasets.

        Args:
            method_a: Name or ID of first method.
            method_b: Name or ID of second method.

        Returns:
            Dict with comparison results.
        """
        entity_a = self._find_method(method_a)
        entity_b = self._find_method(method_b)

        if not entity_a or not entity_b:
            return {
                "error": "One or both methods not found",
                "method_a": method_a,
                "method_b": method_b,
            }

        # Find datasets used by each method
        datasets_a = self._get_datasets_for_method(entity_a.id)
        datasets_b = self._get_datasets_for_method(entity_b.id)

        shared = set(datasets_a.keys()) & set(datasets_b.keys())

        comparison: dict[str, Any] = {
            "method_a": entity_a.name,
            "method_b": entity_b.name,
            "shared_datasets": list(shared),
            "unique_to_a": list(set(datasets_a.keys()) - shared),
            "unique_to_b": list(set(datasets_b.keys()) - shared),
        }

        # Check outperforms relations
        outperforms_a = []
        outperforms_b = []
        for rel in self._graph.get_relations_for(entity_a.id, direction="outgoing"):
            if rel.relation_type == RelationType.OUTPERFORMS and rel.target_id == entity_b.id:
                outperforms_a.append(rel.attributes)
        for rel in self._graph.get_relations_for(entity_b.id, direction="outgoing"):
            if rel.relation_type == RelationType.OUTPERFORMS and rel.target_id == entity_a.id:
                outperforms_b.append(rel.attributes)

        comparison["a_outperforms_b"] = outperforms_a
        comparison["b_outperforms_a"] = outperforms_b

        return comparison

    def suggest_topics(
        self,
        interests: list[str],
        top_k: int = 5,
    ) -> list[str]:
        """Suggest research topics based on graph structure and interests.

        Args:
            interests: List of interest keywords.
            top_k: Number of suggestions.

        Returns:
            List of suggested topics.
        """
        suggestions: list[tuple[float, str]] = []

        # Score entities by relevance to interests
        for entity in self._graph._entities.values():
            score = 0.0
            name_lower = entity.name.lower()
            desc = entity.attributes.get("description", "").lower()
            abstract = entity.attributes.get("abstract", "").lower()
            combined = f"{name_lower} {desc} {abstract}"

            for interest in interests:
                if interest.lower() in combined:
                    score += 1.0

            if score > 0:
                # Boost by connection count
                rels = self._graph.get_relations_for(entity.id)
                score += len(rels) * 0.1
                suggestions.append((score, entity.name))

        # Find gaps as additional suggestions
        gaps = self.find_research_gaps()
        for gap in gaps[:3]:
            for interest in interests:
                if interest.lower() in gap.lower():
                    suggestions.append((0.5, gap))

        suggestions.sort(reverse=True)
        seen: set[str] = set()
        unique: list[str] = []
        for _, text in suggestions:
            if text not in seen:
                seen.add(text)
                unique.append(text)
            if len(unique) >= top_k:
                break

        return unique

    def _find_method(self, name_or_id: str) -> Any:
        """Find a method entity by name or ID."""
        entity = self._graph.get_entity(name_or_id)
        if entity:
            return entity

        for e in self._graph.get_entities_by_type(EntityType.METHOD):
            if e.name.lower() == name_or_id.lower():
                return e
        return None

    def _get_datasets_for_method(self, method_id: str) -> dict[str, Any]:
        """Get datasets that a method has been evaluated on."""
        datasets: dict[str, Any] = {}
        for rel in self._graph.get_relations_for(method_id, direction="outgoing"):
            if rel.relation_type == RelationType.USES_DATASET:
                entity = self._graph.get_entity(rel.target_id)
                if entity:
                    datasets[entity.name] = rel.attributes
        return datasets
