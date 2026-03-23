"""Tests for the collaboration system (15+ tests).

Covers:
- ResearchRepository (publish, search, list)
- ArtifactPublisher (extraction from run dirs)
- ArtifactSubscriber (queries)
- Deduplication (content_hash, deduplicate_artifacts)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchclaw.collaboration.repository import ResearchRepository
from researchclaw.collaboration.publisher import ArtifactPublisher
from researchclaw.collaboration.subscriber import ArtifactSubscriber
from researchclaw.collaboration.dedup import content_hash, deduplicate_artifacts


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def repo(tmp_path: Path) -> ResearchRepository:
    return ResearchRepository(repo_dir=tmp_path / "shared_repo")


@pytest.fixture
def populated_repo(repo: ResearchRepository) -> ResearchRepository:
    repo.publish(
        run_id="run-001",
        artifacts={
            "literature_summary": {"papers": ["Paper A on transformer", "Paper B on vision"]},
            "experiment_results": {"accuracy": 0.95, "model": "ResNet50"},
        },
    )
    repo.publish(
        run_id="run-002",
        artifacts={
            "literature_summary": {"papers": ["Paper C on nlp transformer"]},
            "code_template": "import torch\nmodel = ResNet()\n# pytorch training",
        },
    )
    return repo


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """Create a fake pipeline run directory with stage outputs."""
    d = tmp_path / "run-test"
    d.mkdir()

    # Stage 07 — literature synthesis
    s07 = d / "stage-07-literature_synthesis"
    s07.mkdir()
    (s07 / "synthesis.json").write_text(
        json.dumps({"papers": [{"title": "Test Paper", "year": 2024}]}),
        encoding="utf-8",
    )

    # Stage 10 — code generation
    s10 = d / "stage-10-code_generation"
    s10.mkdir()
    (s10 / "main.py").write_text("print('hello')", encoding="utf-8")

    # Stage 14 — result analysis
    s14 = d / "stage-14-result_analysis"
    s14.mkdir()
    (s14 / "experiment_summary.json").write_text(
        json.dumps({"accuracy": 0.92}), encoding="utf-8"
    )

    # Stage 18 — peer review
    s18 = d / "stage-18-peer_review"
    s18.mkdir()
    (s18 / "review.md").write_text("Good paper overall.", encoding="utf-8")

    return d


# ── Repository Tests ─────────────────────────────────────────────────


class TestResearchRepository:
    def test_publish(self, repo: ResearchRepository) -> None:
        count = repo.publish(
            run_id="run-001",
            artifacts={"literature_summary": {"papers": ["P1"]}},
        )
        assert count == 1

    def test_publish_creates_dirs(self, repo: ResearchRepository) -> None:
        repo.publish(
            run_id="run-new",
            artifacts={"code_template": "print('hi')"},
        )
        assert (repo.repo_dir / "run-new").is_dir()

    def test_publish_unknown_type_skipped(self, repo: ResearchRepository) -> None:
        count = repo.publish(
            run_id="run-bad",
            artifacts={"unknown_type": "data"},
        )
        assert count == 0

    def test_search_by_query(self, populated_repo: ResearchRepository) -> None:
        results = populated_repo.search("transformer")
        assert len(results) >= 2

    def test_search_by_type(self, populated_repo: ResearchRepository) -> None:
        results = populated_repo.search(
            "paper", artifact_type="literature_summary"
        )
        assert len(results) >= 1

    def test_search_no_results(self, populated_repo: ResearchRepository) -> None:
        results = populated_repo.search("quantum_nonexistent_xyz")
        assert len(results) == 0

    def test_search_empty_repo(self, repo: ResearchRepository) -> None:
        results = repo.search("anything")
        assert results == []

    def test_list_runs(self, populated_repo: ResearchRepository) -> None:
        runs = populated_repo.list_runs()
        assert "run-001" in runs
        assert "run-002" in runs

    def test_list_runs_empty(self, repo: ResearchRepository) -> None:
        runs = repo.list_runs()
        assert runs == []

    def test_get_run_artifacts(self, populated_repo: ResearchRepository) -> None:
        artifacts = populated_repo.get_run_artifacts("run-001")
        assert "literature_summary" in artifacts
        assert "experiment_results" in artifacts

    def test_get_run_artifacts_missing(self, populated_repo: ResearchRepository) -> None:
        artifacts = populated_repo.get_run_artifacts("run-999")
        assert artifacts == {}

    def test_import_literature(self, populated_repo: ResearchRepository) -> None:
        lit = populated_repo.import_literature("run-001")
        assert isinstance(lit, list)
        assert len(lit) >= 1

    def test_import_literature_missing_run(self, populated_repo: ResearchRepository) -> None:
        lit = populated_repo.import_literature("run-999")
        assert lit == []

    def test_import_code_template(self, populated_repo: ResearchRepository) -> None:
        code = populated_repo.import_code_template("run-002", "pytorch")
        assert code is not None
        assert "torch" in code

    def test_import_code_template_no_match(self, populated_repo: ResearchRepository) -> None:
        code = populated_repo.import_code_template("run-002", "tensorflow_xyz")
        assert code is None


# ── Publisher Tests ──────────────────────────────────────────────────


class TestArtifactPublisher:
    def test_publish_from_run_dir(self, run_dir: Path, tmp_path: Path) -> None:
        repo = ResearchRepository(repo_dir=tmp_path / "pub_repo")
        publisher = ArtifactPublisher(repo)
        count = publisher.publish_from_run_dir("test-run", run_dir)
        assert count >= 1

    def test_publish_empty_dir(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty_run"
        empty.mkdir()
        repo = ResearchRepository(repo_dir=tmp_path / "pub_repo2")
        publisher = ArtifactPublisher(repo)
        count = publisher.publish_from_run_dir("empty", empty)
        assert count == 0

    def test_publish_nonexistent_dir(self, tmp_path: Path) -> None:
        repo = ResearchRepository(repo_dir=tmp_path / "pub_repo3")
        publisher = ArtifactPublisher(repo)
        count = publisher.publish_from_run_dir("missing", tmp_path / "nope")
        assert count == 0


# ── Subscriber Tests ─────────────────────────────────────────────────


class TestArtifactSubscriber:
    def test_find_relevant_literature(self, populated_repo: ResearchRepository) -> None:
        sub = ArtifactSubscriber(populated_repo)
        results = sub.find_relevant_literature("transformer")
        assert len(results) >= 1

    def test_find_similar_experiments(self, populated_repo: ResearchRepository) -> None:
        sub = ArtifactSubscriber(populated_repo)
        results = sub.find_similar_experiments("resnet")
        assert len(results) >= 1

    def test_find_code_templates(self, populated_repo: ResearchRepository) -> None:
        sub = ArtifactSubscriber(populated_repo)
        results = sub.find_code_templates("pytorch")
        assert len(results) >= 1

    def test_import_best_practices(self, populated_repo: ResearchRepository) -> None:
        sub = ArtifactSubscriber(populated_repo)
        practices = sub.import_best_practices("transformer")
        assert isinstance(practices, str)

    def test_import_best_practices_empty(self, repo: ResearchRepository) -> None:
        sub = ArtifactSubscriber(repo)
        practices = sub.import_best_practices("nonexistent")
        assert practices == ""


# ── Dedup Tests ──────────────────────────────────────────────────────


class TestDedup:
    def test_content_hash_deterministic(self) -> None:
        h1 = content_hash({"a": 1, "b": 2})
        h2 = content_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_content_hash_different(self) -> None:
        h1 = content_hash({"a": 1})
        h2 = content_hash({"a": 2})
        assert h1 != h2

    def test_deduplicate_artifacts(self) -> None:
        artifacts = [
            {"content": {"x": 1}, "tags": ["a"]},
            {"content": {"x": 1}, "tags": ["b"]},  # duplicate content
            {"content": {"y": 2}, "tags": ["c"]},
        ]
        unique = deduplicate_artifacts(artifacts)
        assert len(unique) == 2

    def test_deduplicate_empty(self) -> None:
        assert deduplicate_artifacts([]) == []
