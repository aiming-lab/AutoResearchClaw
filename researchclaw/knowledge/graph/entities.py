"""Entity definitions for the research knowledge graph."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class EntityType(str, Enum):
    """Types of entities in the knowledge graph."""

    PAPER = "paper"
    METHOD = "method"
    DATASET = "dataset"
    METRIC = "metric"
    AUTHOR = "author"
    CONCEPT = "concept"


@dataclass
class Entity:
    """A node in the knowledge graph.

    Attributes:
        id: Unique identifier (e.g., arxiv ID, method name hash).
        entity_type: The type of entity.
        name: Display name.
        attributes: Additional key-value attributes.
    """

    id: str
    entity_type: EntityType
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        d = asdict(self)
        d["entity_type"] = self.entity_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Entity:
        """Deserialize from dictionary."""
        return cls(
            id=str(data.get("id", "")),
            entity_type=EntityType(data.get("entity_type", "concept")),
            name=str(data.get("name", "")),
            attributes=data.get("attributes") or {},
        )
