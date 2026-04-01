"""Relation definitions for the research knowledge graph."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class RelationType(str, Enum):
    """Types of relations between entities."""

    CITES = "cites"
    EXTENDS = "extends"
    OUTPERFORMS = "outperforms"
    USES_DATASET = "uses_dataset"
    APPLIES_METHOD = "applies_method"
    EVALUATES_WITH = "evaluates_with"  # metric used for evaluation
    AUTHORED_BY = "authored_by"
    RELATED_TO = "related_to"


@dataclass
class Relation:
    """A directed edge in the knowledge graph.

    Attributes:
        source_id: ID of the source entity.
        target_id: ID of the target entity.
        relation_type: The type of relation.
        attributes: Additional key-value attributes (e.g., metric value).
    """

    source_id: str
    target_id: str
    relation_type: RelationType
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        d = asdict(self)
        d["relation_type"] = self.relation_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Relation:
        """Deserialize from dictionary."""
        return cls(
            source_id=str(data.get("source_id", "")),
            target_id=str(data.get("target_id", "")),
            relation_type=RelationType(data.get("relation_type", "related_to")),
            attributes=data.get("attributes") or {},
        )
