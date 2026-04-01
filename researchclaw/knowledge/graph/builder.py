"""Knowledge graph builder — constructs graph from literature and experiments."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from researchclaw.knowledge.graph.entities import Entity, EntityType
from researchclaw.knowledge.graph.relations import Relation, RelationType

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """Builds and manages a research knowledge graph.

    Uses dictionaries for storage (compatible with NetworkX-style serialization)
    without requiring networkx as a hard dependency.
    """

    def __init__(self, max_entities: int = 10000) -> None:
        self._entities: dict[str, Entity] = {}
        self._relations: list[Relation] = []
        self._max_entities = max_entities

    @property
    def entity_count(self) -> int:
        """Number of entities in the graph."""
        return len(self._entities)

    @property
    def relation_count(self) -> int:
        """Number of relations in the graph."""
        return len(self._relations)

    def add_entity(self, entity: Entity) -> bool:
        """Add an entity to the graph.

        Args:
            entity: The entity to add.

        Returns:
            True if added, False if capacity reached or duplicate.
        """
        if entity.id in self._entities:
            # Update attributes of existing entity
            existing = self._entities[entity.id]
            merged = {**existing.attributes, **entity.attributes}
            self._entities[entity.id] = Entity(
                id=entity.id,
                entity_type=entity.entity_type,
                name=entity.name or existing.name,
                attributes=merged,
            )
            return True

        if len(self._entities) >= self._max_entities:
            logger.warning("Knowledge graph capacity reached (%d)", self._max_entities)
            return False

        self._entities[entity.id] = entity
        return True

    def add_relation(self, relation: Relation) -> bool:
        """Add a relation to the graph.

        Args:
            relation: The relation to add.

        Returns:
            True if added, False if source or target entity doesn't exist.
        """
        if relation.source_id not in self._entities:
            logger.debug("Source entity not found: %s", relation.source_id)
            return False
        if relation.target_id not in self._entities:
            logger.debug("Target entity not found: %s", relation.target_id)
            return False

        # Check for duplicate
        for existing in self._relations:
            if (
                existing.source_id == relation.source_id
                and existing.target_id == relation.target_id
                and existing.relation_type == relation.relation_type
            ):
                return True  # Already exists

        self._relations.append(relation)
        return True

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        return self._entities.get(entity_id)

    def get_entities_by_type(self, entity_type: EntityType) -> list[Entity]:
        """Get all entities of a specific type."""
        return [
            e for e in self._entities.values() if e.entity_type == entity_type
        ]

    def get_relations_for(
        self,
        entity_id: str,
        direction: str = "both",
    ) -> list[Relation]:
        """Get relations involving an entity.

        Args:
            entity_id: The entity to query.
            direction: "outgoing", "incoming", or "both".

        Returns:
            List of matching relations.
        """
        results: list[Relation] = []
        for rel in self._relations:
            if direction in ("outgoing", "both") and rel.source_id == entity_id:
                results.append(rel)
            if direction in ("incoming", "both") and rel.target_id == entity_id:
                results.append(rel)
        return results

    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity and all its relations."""
        if entity_id not in self._entities:
            return False
        del self._entities[entity_id]
        self._relations = [
            r
            for r in self._relations
            if r.source_id != entity_id and r.target_id != entity_id
        ]
        return True

    def add_paper(
        self,
        paper_id: str,
        title: str,
        year: int | None = None,
        authors: list[str] | None = None,
        abstract: str = "",
    ) -> Entity:
        """Convenience method to add a paper entity.

        Args:
            paper_id: Unique paper ID (e.g., arxiv ID).
            title: Paper title.
            year: Publication year.
            authors: List of author names.
            abstract: Paper abstract.

        Returns:
            The created Entity.
        """
        attrs: dict[str, Any] = {}
        if year:
            attrs["year"] = year
        if authors:
            attrs["authors"] = authors
        if abstract:
            attrs["abstract"] = abstract[:500]

        entity = Entity(
            id=paper_id,
            entity_type=EntityType.PAPER,
            name=title,
            attributes=attrs,
        )
        self.add_entity(entity)
        return entity

    def add_method(
        self,
        method_id: str,
        name: str,
        description: str = "",
    ) -> Entity:
        """Convenience method to add a method entity."""
        entity = Entity(
            id=method_id,
            entity_type=EntityType.METHOD,
            name=name,
            attributes={"description": description} if description else {},
        )
        self.add_entity(entity)
        return entity

    def add_dataset(
        self,
        dataset_id: str,
        name: str,
        domain: str = "",
    ) -> Entity:
        """Convenience method to add a dataset entity."""
        entity = Entity(
            id=dataset_id,
            entity_type=EntityType.DATASET,
            name=name,
            attributes={"domain": domain} if domain else {},
        )
        self.add_entity(entity)
        return entity

    def save(self, path: str | Path) -> None:
        """Save graph to JSON file.

        Args:
            path: File path for output.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "entities": [e.to_dict() for e in self._entities.values()],
            "relations": [r.to_dict() for r in self._relations],
        }
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            "Saved knowledge graph: %d entities, %d relations to %s",
            len(self._entities),
            len(self._relations),
            path,
        )

    def load(self, path: str | Path) -> int:
        """Load graph from JSON file.

        Args:
            path: File path to load from.

        Returns:
            Total number of entities loaded.
        """
        path = Path(path)
        if not path.exists():
            return 0

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load knowledge graph: %s", exc)
            return 0

        for entity_data in data.get("entities", []):
            try:
                entity = Entity.from_dict(entity_data)
                self.add_entity(entity)
            except (ValueError, KeyError) as exc:
                logger.debug("Skipping malformed entity: %s", exc)

        for rel_data in data.get("relations", []):
            try:
                relation = Relation.from_dict(rel_data)
                self.add_relation(relation)
            except (ValueError, KeyError) as exc:
                logger.debug("Skipping malformed relation: %s", exc)

        logger.info(
            "Loaded knowledge graph: %d entities, %d relations",
            self.entity_count,
            self.relation_count,
        )
        return self.entity_count
