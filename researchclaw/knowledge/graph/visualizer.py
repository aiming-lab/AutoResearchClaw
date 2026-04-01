"""Knowledge graph visualization export."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from researchclaw.knowledge.graph.builder import KnowledgeGraphBuilder
from researchclaw.knowledge.graph.entities import EntityType

logger = logging.getLogger(__name__)

# Color mapping for entity types
_TYPE_COLORS: dict[str, str] = {
    EntityType.PAPER: "#4A90D9",
    EntityType.METHOD: "#E74C3C",
    EntityType.DATASET: "#2ECC71",
    EntityType.METRIC: "#F39C12",
    EntityType.AUTHOR: "#9B59B6",
    EntityType.CONCEPT: "#95A5A6",
}


def export_to_dot(graph: KnowledgeGraphBuilder, path: str | Path) -> None:
    """Export graph to Graphviz DOT format.

    Args:
        graph: The knowledge graph to export.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["digraph KnowledgeGraph {"]
    lines.append("  rankdir=LR;")
    lines.append("  node [shape=box, style=filled, fontsize=10];")

    # Nodes
    for entity in graph._entities.values():
        color = _TYPE_COLORS.get(entity.entity_type, "#CCCCCC")
        label = entity.name[:40].replace('"', '\\"')
        lines.append(
            f'  "{entity.id}" [label="{label}", '
            f'fillcolor="{color}", fontcolor="white"];'
        )

    # Edges
    for rel in graph._relations:
        label = rel.relation_type.value
        lines.append(f'  "{rel.source_id}" -> "{rel.target_id}" [label="{label}"];')

    lines.append("}")
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Exported graph to DOT: %s", path)


def export_to_json_cytoscape(
    graph: KnowledgeGraphBuilder,
    path: str | Path,
) -> None:
    """Export graph to Cytoscape.js compatible JSON.

    Args:
        graph: The knowledge graph to export.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    elements: list[dict[str, Any]] = []

    for entity in graph._entities.values():
        elements.append({
            "data": {
                "id": entity.id,
                "label": entity.name,
                "type": entity.entity_type.value,
                **entity.attributes,
            },
        })

    for i, rel in enumerate(graph._relations):
        elements.append({
            "data": {
                "id": f"edge_{i}",
                "source": rel.source_id,
                "target": rel.target_id,
                "label": rel.relation_type.value,
                **rel.attributes,
            },
        })

    path.write_text(
        json.dumps({"elements": elements}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Exported graph to Cytoscape JSON: %s", path)


def graph_summary(graph: KnowledgeGraphBuilder) -> str:
    """Generate a text summary of the graph.

    Args:
        graph: The knowledge graph.

    Returns:
        Multi-line summary string.
    """
    from collections import Counter

    type_counts: Counter[str] = Counter()
    for entity in graph._entities.values():
        type_counts[entity.entity_type.value] += 1

    rel_counts: Counter[str] = Counter()
    for rel in graph._relations:
        rel_counts[rel.relation_type.value] += 1

    lines = [
        f"Knowledge Graph Summary: {graph.entity_count} entities, "
        f"{graph.relation_count} relations",
        "",
        "Entity types:",
    ]
    for etype, count in type_counts.most_common():
        lines.append(f"  {etype}: {count}")

    lines.append("")
    lines.append("Relation types:")
    for rtype, count in rel_counts.most_common():
        lines.append(f"  {rtype}: {count}")

    return "\n".join(lines)
