"""Tests for the research knowledge graph (20+ tests).

Covers:
- Entity/Relation CRUD
- Graph queries (gaps, trends, comparison)
- JSON serialization/deserialization
- Incremental updates
- Visualizer exports
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchclaw.knowledge.graph.entities import Entity, EntityType
from researchclaw.knowledge.graph.relations import Relation, RelationType
from researchclaw.knowledge.graph.builder import KnowledgeGraphBuilder
from researchclaw.knowledge.graph.query import KnowledgeGraphQuery
from researchclaw.knowledge.graph.visualizer import (
    export_to_dot,
    export_to_json_cytoscape,
    graph_summary,
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def graph() -> KnowledgeGraphBuilder:
    return KnowledgeGraphBuilder(max_entities=100)


@pytest.fixture
def populated_graph(graph: KnowledgeGraphBuilder) -> KnowledgeGraphBuilder:
    # Papers
    graph.add_paper("p1", "ResNet: Deep Residual Learning", year=2016, authors=["He"])
    graph.add_paper("p2", "ViT: An Image is Worth 16x16 Words", year=2021, authors=["Dosovitskiy"])
    graph.add_paper("p3", "DeiT: Training Data-efficient Image Transformers", year=2021, authors=["Touvron"])

    # Methods
    graph.add_method("m1", "ResNet", description="Residual connections for deep networks")
    graph.add_method("m2", "Vision Transformer", description="Transformer for image classification")
    graph.add_method("m3", "Knowledge Distillation", description="Teacher-student learning")

    # Datasets
    graph.add_dataset("d1", "ImageNet", domain="computer vision")
    graph.add_dataset("d2", "CIFAR-10", domain="computer vision")
    graph.add_dataset("d3", "CIFAR-100", domain="computer vision")

    # Relations
    graph.add_relation(Relation("p2", "p1", RelationType.CITES))
    graph.add_relation(Relation("p3", "p2", RelationType.EXTENDS))
    graph.add_relation(Relation("p3", "p1", RelationType.CITES))
    graph.add_relation(Relation("m1", "d1", RelationType.USES_DATASET))
    graph.add_relation(Relation("m1", "d2", RelationType.USES_DATASET))
    graph.add_relation(Relation("m2", "d1", RelationType.USES_DATASET))
    graph.add_relation(Relation("m2", "d2", RelationType.USES_DATASET))
    graph.add_relation(Relation("p1", "m1", RelationType.APPLIES_METHOD))
    graph.add_relation(Relation("p2", "m2", RelationType.APPLIES_METHOD))
    graph.add_relation(Relation("m2", "m1", RelationType.OUTPERFORMS, {"dataset": "ImageNet"}))

    return graph


# ── Entity Tests ─────────────────────────────────────────────────────


class TestEntity:
    def test_create_entity(self) -> None:
        e = Entity("e1", EntityType.PAPER, "Test Paper")
        assert e.id == "e1"
        assert e.entity_type == EntityType.PAPER

    def test_to_dict(self) -> None:
        e = Entity("e1", EntityType.METHOD, "TestMethod", {"key": "val"})
        d = e.to_dict()
        assert d["entity_type"] == "method"
        assert d["attributes"]["key"] == "val"

    def test_from_dict(self) -> None:
        data = {"id": "x", "entity_type": "dataset", "name": "Test", "attributes": {}}
        e = Entity.from_dict(data)
        assert e.entity_type == EntityType.DATASET


class TestRelation:
    def test_create_relation(self) -> None:
        r = Relation("a", "b", RelationType.CITES)
        assert r.source_id == "a"
        assert r.target_id == "b"

    def test_to_dict(self) -> None:
        r = Relation("a", "b", RelationType.OUTPERFORMS, {"margin": 0.05})
        d = r.to_dict()
        assert d["relation_type"] == "outperforms"
        assert d["attributes"]["margin"] == 0.05

    def test_from_dict(self) -> None:
        data = {"source_id": "x", "target_id": "y", "relation_type": "extends"}
        r = Relation.from_dict(data)
        assert r.relation_type == RelationType.EXTENDS


# ── Builder Tests ────────────────────────────────────────────────────


class TestKnowledgeGraphBuilder:
    def test_add_entity(self, graph: KnowledgeGraphBuilder) -> None:
        e = Entity("e1", EntityType.PAPER, "Test")
        assert graph.add_entity(e)
        assert graph.entity_count == 1

    def test_add_duplicate_updates(self, graph: KnowledgeGraphBuilder) -> None:
        graph.add_entity(Entity("e1", EntityType.PAPER, "V1", {"a": 1}))
        graph.add_entity(Entity("e1", EntityType.PAPER, "V2", {"b": 2}))
        assert graph.entity_count == 1
        e = graph.get_entity("e1")
        assert e is not None
        assert e.name == "V2"
        assert e.attributes["a"] == 1  # merged
        assert e.attributes["b"] == 2

    def test_capacity_limit(self) -> None:
        g = KnowledgeGraphBuilder(max_entities=2)
        g.add_entity(Entity("e1", EntityType.PAPER, "P1"))
        g.add_entity(Entity("e2", EntityType.PAPER, "P2"))
        assert not g.add_entity(Entity("e3", EntityType.PAPER, "P3"))
        assert g.entity_count == 2

    def test_add_relation(self, graph: KnowledgeGraphBuilder) -> None:
        graph.add_entity(Entity("a", EntityType.PAPER, "A"))
        graph.add_entity(Entity("b", EntityType.PAPER, "B"))
        assert graph.add_relation(Relation("a", "b", RelationType.CITES))
        assert graph.relation_count == 1

    def test_add_relation_missing_entity(self, graph: KnowledgeGraphBuilder) -> None:
        graph.add_entity(Entity("a", EntityType.PAPER, "A"))
        assert not graph.add_relation(Relation("a", "missing", RelationType.CITES))

    def test_duplicate_relation(self, graph: KnowledgeGraphBuilder) -> None:
        graph.add_entity(Entity("a", EntityType.PAPER, "A"))
        graph.add_entity(Entity("b", EntityType.PAPER, "B"))
        graph.add_relation(Relation("a", "b", RelationType.CITES))
        graph.add_relation(Relation("a", "b", RelationType.CITES))  # duplicate
        assert graph.relation_count == 1

    def test_get_entities_by_type(self, populated_graph: KnowledgeGraphBuilder) -> None:
        papers = populated_graph.get_entities_by_type(EntityType.PAPER)
        assert len(papers) == 3

    def test_get_relations_for(self, populated_graph: KnowledgeGraphBuilder) -> None:
        rels = populated_graph.get_relations_for("p2")
        assert len(rels) >= 2  # outgoing + incoming

    def test_remove_entity(self, populated_graph: KnowledgeGraphBuilder) -> None:
        initial_rels = populated_graph.relation_count
        assert populated_graph.remove_entity("p1")
        assert populated_graph.get_entity("p1") is None
        assert populated_graph.relation_count < initial_rels

    def test_remove_nonexistent_entity(self, graph: KnowledgeGraphBuilder) -> None:
        assert not graph.remove_entity("nope")

    def test_convenience_methods(self, graph: KnowledgeGraphBuilder) -> None:
        paper = graph.add_paper("p1", "Test Paper", year=2024)
        method = graph.add_method("m1", "TestNet", description="A test")
        dataset = graph.add_dataset("d1", "TestSet", domain="cv")
        assert paper.entity_type == EntityType.PAPER
        assert method.entity_type == EntityType.METHOD
        assert dataset.entity_type == EntityType.DATASET


# ── Persistence ──────────────────────────────────────────────────────


class TestGraphPersistence:
    def test_save_and_load(self, populated_graph: KnowledgeGraphBuilder, tmp_path: Path) -> None:
        path = tmp_path / "graph.json"
        populated_graph.save(path)
        assert path.exists()

        new_graph = KnowledgeGraphBuilder()
        loaded = new_graph.load(path)
        assert loaded == populated_graph.entity_count
        assert new_graph.relation_count == populated_graph.relation_count

    def test_load_nonexistent(self, graph: KnowledgeGraphBuilder, tmp_path: Path) -> None:
        assert graph.load(tmp_path / "nope.json") == 0

    def test_load_malformed(self, graph: KnowledgeGraphBuilder, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not json", encoding="utf-8")
        assert graph.load(path) == 0


# ── Query Engine ─────────────────────────────────────────────────────


class TestKnowledgeGraphQuery:
    def test_find_research_gaps(self, populated_graph: KnowledgeGraphBuilder) -> None:
        query = KnowledgeGraphQuery(populated_graph)
        gaps = query.find_research_gaps()
        # CIFAR-100 has no methods using it
        assert any("CIFAR-100" in g for g in gaps)

    def test_find_research_gaps_with_domain(self, populated_graph: KnowledgeGraphBuilder) -> None:
        query = KnowledgeGraphQuery(populated_graph)
        gaps = query.find_research_gaps(domain="computer vision")
        assert isinstance(gaps, list)

    def test_find_trending_methods(self, populated_graph: KnowledgeGraphBuilder) -> None:
        query = KnowledgeGraphQuery(populated_graph)
        trending = query.find_trending_methods(min_citations=1)
        assert len(trending) > 0

    def test_get_method_comparison(self, populated_graph: KnowledgeGraphBuilder) -> None:
        query = KnowledgeGraphQuery(populated_graph)
        comparison = query.get_method_comparison("ResNet", "Vision Transformer")
        assert "method_a" in comparison
        assert "method_b" in comparison
        assert "shared_datasets" in comparison

    def test_get_method_comparison_not_found(self, populated_graph: KnowledgeGraphBuilder) -> None:
        query = KnowledgeGraphQuery(populated_graph)
        comparison = query.get_method_comparison("NonexistentA", "NonexistentB")
        assert "error" in comparison

    def test_suggest_topics(self, populated_graph: KnowledgeGraphBuilder) -> None:
        query = KnowledgeGraphQuery(populated_graph)
        topics = query.suggest_topics(["transformer", "vision"], top_k=3)
        assert isinstance(topics, list)

    def test_suggest_topics_empty_interests(self, populated_graph: KnowledgeGraphBuilder) -> None:
        query = KnowledgeGraphQuery(populated_graph)
        topics = query.suggest_topics([])
        assert isinstance(topics, list)


# ── Visualizer ───────────────────────────────────────────────────────


class TestVisualizer:
    def test_export_dot(self, populated_graph: KnowledgeGraphBuilder, tmp_path: Path) -> None:
        path = tmp_path / "graph.dot"
        export_to_dot(populated_graph, path)
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "digraph" in content
        assert "ResNet" in content

    def test_export_cytoscape(self, populated_graph: KnowledgeGraphBuilder, tmp_path: Path) -> None:
        path = tmp_path / "graph.json"
        export_to_json_cytoscape(populated_graph, path)
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "elements" in data
        assert len(data["elements"]) > 0

    def test_graph_summary(self, populated_graph: KnowledgeGraphBuilder) -> None:
        summary = graph_summary(populated_graph)
        assert "entities" in summary
        assert "relations" in summary
        assert "paper" in summary
