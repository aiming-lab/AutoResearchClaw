"""Tests for multi-project management (C1): ProjectManager, ProjectScheduler, IdeaPool."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from researchclaw.project.models import Idea, Project
from researchclaw.project.manager import ProjectManager
from researchclaw.project.scheduler import ProjectScheduler
from researchclaw.project.idea_pool import IdeaPool


# ── fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def tmp_projects(tmp_path: Path) -> Path:
    return tmp_path / "projects"


@pytest.fixture
def manager(tmp_projects: Path) -> ProjectManager:
    return ProjectManager(tmp_projects)


@pytest.fixture
def config_yaml(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("project:\n  name: test\nresearch:\n  topic: test\n")
    return cfg


@pytest.fixture
def pool_path(tmp_path: Path) -> Path:
    return tmp_path / "ideas.json"


# ══════════════════════════════════════════════════════════════════
# Project model tests
# ══════════════════════════════════════════════════════════════════


class TestProjectModel:
    def test_to_dict_roundtrip(self) -> None:
        p = Project(name="test", config_path="/a/b", run_dir="/c/d", topic="ml")
        d = p.to_dict()
        p2 = Project.from_dict(d)
        assert p2.name == p.name
        assert p2.topic == p.topic
        assert p2.status == "idle"

    def test_from_dict_defaults(self) -> None:
        d = {"name": "x", "config_path": "/a", "run_dir": "/b"}
        p = Project.from_dict(d)
        assert p.status == "idle"
        assert p.last_run_id is None

    def test_from_dict_with_iso_date(self) -> None:
        d = {
            "name": "x",
            "config_path": "/a",
            "run_dir": "/b",
            "created_at": "2024-01-01T00:00:00+00:00",
        }
        p = Project.from_dict(d)
        assert p.created_at.year == 2024


# ══════════════════════════════════════════════════════════════════
# Idea model tests
# ══════════════════════════════════════════════════════════════════


class TestIdeaModel:
    def test_score_calculation(self) -> None:
        idea = Idea(id="1", title="t", description="d", feasibility=1.0, novelty=1.0)
        assert idea.score == pytest.approx(1.0)

    def test_score_weighted(self) -> None:
        idea = Idea(id="1", title="t", description="d", feasibility=0.5, novelty=0.5)
        assert idea.score == pytest.approx(0.5)

    def test_to_dict_roundtrip(self) -> None:
        idea = Idea(id="abc", title="GNN", description="graph stuff", domains=["ml"])
        d = idea.to_dict()
        i2 = Idea.from_dict(d)
        assert i2.id == "abc"
        assert i2.domains == ["ml"]


# ══════════════════════════════════════════════════════════════════
# ProjectManager tests
# ══════════════════════════════════════════════════════════════════


class TestProjectManager:
    def test_create_project(self, manager: ProjectManager, config_yaml: Path) -> None:
        proj = manager.create("my_project", str(config_yaml), topic="RL")
        assert proj.name == "my_project"
        assert proj.topic == "RL"
        assert proj.status == "idle"

    def test_create_sets_active(self, manager: ProjectManager, config_yaml: Path) -> None:
        manager.create("first", str(config_yaml))
        assert manager.active is not None
        assert manager.active.name == "first"

    def test_create_duplicate_raises(self, manager: ProjectManager, config_yaml: Path) -> None:
        manager.create("dup", str(config_yaml))
        with pytest.raises(ValueError, match="already exists"):
            manager.create("dup", str(config_yaml))

    def test_delete_project(self, manager: ProjectManager, config_yaml: Path) -> None:
        manager.create("del_me", str(config_yaml))
        manager.delete("del_me")
        assert "del_me" not in manager.projects

    def test_delete_unknown_raises(self, manager: ProjectManager) -> None:
        with pytest.raises(KeyError):
            manager.delete("nonexistent")

    def test_get_project(self, manager: ProjectManager, config_yaml: Path) -> None:
        manager.create("proj1", str(config_yaml))
        p = manager.get("proj1")
        assert p.name == "proj1"

    def test_get_unknown_raises(self, manager: ProjectManager) -> None:
        with pytest.raises(KeyError):
            manager.get("nope")

    def test_list_all_sorted(self, manager: ProjectManager, config_yaml: Path) -> None:
        manager.create("b_proj", str(config_yaml))
        manager.create("a_proj", str(config_yaml))
        projects = manager.list_all()
        assert len(projects) == 2
        # Sorted by creation time (b_proj first)
        assert projects[0].name == "b_proj"

    def test_get_status(self, manager: ProjectManager, config_yaml: Path) -> None:
        manager.create("s1", str(config_yaml))
        manager.create("s2", str(config_yaml))
        status = manager.get_status()
        assert status["total"] == 2
        assert status["active"] == "s1"

    def test_switch_project(self, manager: ProjectManager, config_yaml: Path) -> None:
        manager.create("a", str(config_yaml))
        manager.create("b", str(config_yaml))
        manager.switch("b")
        assert manager.active is not None
        assert manager.active.name == "b"

    def test_switch_unknown_raises(self, manager: ProjectManager) -> None:
        with pytest.raises(KeyError):
            manager.switch("ghost")

    def test_compare_projects(self, manager: ProjectManager, config_yaml: Path) -> None:
        manager.create("pa", str(config_yaml))
        manager.create("pb", str(config_yaml))
        manager.projects["pa"].metrics = {"acc": 0.9}
        manager.projects["pb"].metrics = {"acc": 0.95}
        result = manager.compare("pa", "pb")
        assert "metric_diff" in result
        assert result["metric_diff"]["acc"]["delta"] == pytest.approx(0.05)

    def test_start_run(self, manager: ProjectManager, config_yaml: Path) -> None:
        manager.create("run_proj", str(config_yaml))
        rid = manager.start_run("run_proj", run_id="rc-123")
        assert rid == "rc-123"
        assert manager.get("run_proj").status == "running"

    def test_finish_run(self, manager: ProjectManager, config_yaml: Path) -> None:
        manager.create("fin_proj", str(config_yaml))
        manager.start_run("fin_proj", run_id="rc-456")
        manager.finish_run("fin_proj", "completed", {"acc": 0.88})
        p = manager.get("fin_proj")
        assert p.status == "completed"
        assert p.metrics["acc"] == 0.88

    def test_registry_persistence(self, tmp_projects: Path, config_yaml: Path) -> None:
        m1 = ProjectManager(tmp_projects)
        m1.create("persist", str(config_yaml), topic="persistence")
        # Load from disk
        m2 = ProjectManager(tmp_projects)
        assert "persist" in m2.projects
        assert m2.projects["persist"].topic == "persistence"

    def test_delete_switches_active(self, manager: ProjectManager, config_yaml: Path) -> None:
        manager.create("first", str(config_yaml))
        manager.create("second", str(config_yaml))
        manager.switch("first")
        manager.delete("first")
        # Should switch active to remaining project
        assert manager.active is not None
        assert manager.active.name == "second"

    def test_config_copied_to_project_dir(self, manager: ProjectManager, config_yaml: Path) -> None:
        proj = manager.create("copy_test", str(config_yaml))
        copied = Path(proj.config_path)
        assert copied.exists()
        assert "test" in copied.read_text()


# ══════════════════════════════════════════════════════════════════
# ProjectScheduler tests
# ══════════════════════════════════════════════════════════════════


class TestProjectScheduler:
    def test_enqueue_and_next(self, manager: ProjectManager, config_yaml: Path) -> None:
        manager.create("proj", str(config_yaml))
        sched = ProjectScheduler(manager, max_concurrent=1)
        sched.enqueue("proj")
        name = sched.next()
        assert name == "proj"

    def test_concurrency_limit(self, manager: ProjectManager, config_yaml: Path) -> None:
        manager.create("a", str(config_yaml))
        manager.create("b", str(config_yaml))
        sched = ProjectScheduler(manager, max_concurrent=1)
        sched.enqueue("a")
        sched.enqueue("b")
        sched.next()  # starts "a"
        assert sched.next() is None  # can't start "b"

    def test_mark_done_frees_slot(self, manager: ProjectManager, config_yaml: Path) -> None:
        manager.create("a", str(config_yaml))
        manager.create("b", str(config_yaml))
        sched = ProjectScheduler(manager, max_concurrent=1)
        sched.enqueue("a")
        sched.enqueue("b")
        sched.next()  # starts "a"
        sched.mark_done("a")
        name = sched.next()
        assert name == "b"

    def test_priority_order(self, manager: ProjectManager, config_yaml: Path) -> None:
        manager.create("low", str(config_yaml))
        manager.create("high", str(config_yaml))
        sched = ProjectScheduler(manager, max_concurrent=2)
        sched.enqueue("low", priority=10)
        sched.enqueue("high", priority=1)
        # Higher priority (lower number) first
        assert sched.next() == "high"
        assert sched.next() == "low"

    def test_enqueue_unknown_raises(self, manager: ProjectManager) -> None:
        sched = ProjectScheduler(manager)
        with pytest.raises(KeyError):
            sched.enqueue("ghost")

    def test_duplicate_enqueue_ignored(self, manager: ProjectManager, config_yaml: Path) -> None:
        manager.create("dup", str(config_yaml))
        sched = ProjectScheduler(manager)
        sched.enqueue("dup")
        sched.enqueue("dup")
        assert sched.queue_size == 1

    def test_get_status(self, manager: ProjectManager, config_yaml: Path) -> None:
        manager.create("s", str(config_yaml))
        sched = ProjectScheduler(manager, max_concurrent=3)
        sched.enqueue("s")
        status = sched.get_status()
        assert status["max_concurrent"] == 3
        assert status["queue_size"] == 1

    def test_can_start_empty_queue(self, manager: ProjectManager) -> None:
        sched = ProjectScheduler(manager)
        assert not sched.can_start()


# ══════════════════════════════════════════════════════════════════
# IdeaPool tests
# ══════════════════════════════════════════════════════════════════


class TestIdeaPool:
    def test_add_idea(self, pool_path: Path) -> None:
        pool = IdeaPool(pool_path)
        idea = pool.add("GNN for proteins", "Apply GNN to protein folding", ["bio", "ml"])
        assert idea.title == "GNN for proteins"
        assert len(idea.id) == 8

    def test_remove_idea(self, pool_path: Path) -> None:
        pool = IdeaPool(pool_path)
        idea = pool.add("remove me", "desc")
        pool.remove(idea.id)
        assert idea.id not in pool.ideas

    def test_remove_unknown_raises(self, pool_path: Path) -> None:
        pool = IdeaPool(pool_path)
        with pytest.raises(KeyError):
            pool.remove("nonexistent")

    def test_get_idea(self, pool_path: Path) -> None:
        pool = IdeaPool(pool_path)
        idea = pool.add("get me", "desc")
        retrieved = pool.get(idea.id)
        assert retrieved.title == "get me"

    def test_evaluate(self, pool_path: Path) -> None:
        pool = IdeaPool(pool_path)
        idea = pool.add("eval", "desc")
        result = pool.evaluate(idea.id, feasibility=0.8, novelty=0.9)
        assert result["feasibility"] == 0.8
        assert result["novelty"] == 0.9
        assert pool.get(idea.id).status == "evaluated"

    def test_evaluate_clamps_values(self, pool_path: Path) -> None:
        pool = IdeaPool(pool_path)
        idea = pool.add("clamp", "desc")
        pool.evaluate(idea.id, feasibility=1.5, novelty=-0.5)
        assert pool.get(idea.id).feasibility == 1.0
        assert pool.get(idea.id).novelty == 0.0

    def test_rank(self, pool_path: Path) -> None:
        pool = IdeaPool(pool_path)
        pool.add("low", "desc")
        pool.add("high", "desc")
        pool.evaluate(pool.list_all()[0].id, 0.1, 0.1)
        pool.evaluate(pool.list_all()[1].id, 0.9, 0.9)
        ranked = pool.rank()
        assert ranked[0].score > ranked[1].score

    def test_list_all(self, pool_path: Path) -> None:
        pool = IdeaPool(pool_path)
        pool.add("a", "desc")
        pool.add("b", "desc")
        assert len(pool.list_all()) == 2

    def test_persistence(self, pool_path: Path) -> None:
        pool1 = IdeaPool(pool_path)
        pool1.add("persist", "desc", ["ml"])
        pool2 = IdeaPool(pool_path)
        assert len(pool2.ideas) == 1
        assert list(pool2.ideas.values())[0].title == "persist"

    def test_to_project(self, pool_path: Path, tmp_path: Path, config_yaml: Path) -> None:
        pool = IdeaPool(pool_path)
        idea = pool.add("my idea", "a nice description")
        projects_dir = tmp_path / "projects"
        proj = pool.to_project(idea.id, str(config_yaml), projects_dir)
        assert proj.topic == "a nice description"
        assert pool.get(idea.id).status == "planned"
