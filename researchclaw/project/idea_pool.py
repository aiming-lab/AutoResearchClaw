"""Idea pool: collect, evaluate, rank, and convert research ideas to projects."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from researchclaw.project.models import Idea, Project

logger = logging.getLogger(__name__)


class IdeaPool:
    """Manage a pool of research ideas with evaluation and ranking."""

    def __init__(self, pool_path: str | Path) -> None:
        self.pool_path = Path(pool_path).expanduser().resolve()
        self.ideas: dict[str, Idea] = {}
        self._load()

    # ── persistence ───────────────────────────────────────────────

    def _load(self) -> None:
        if not self.pool_path.exists():
            return
        try:
            data = json.loads(self.pool_path.read_text(encoding="utf-8"))
            for entry in data.get("ideas", []):
                idea = Idea.from_dict(entry)
                self.ideas[idea.id] = idea
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to load idea pool: %s", exc)

    def _save(self) -> None:
        self.pool_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"ideas": [idea.to_dict() for idea in self.ideas.values()]}
        self.pool_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ── CRUD ──────────────────────────────────────────────────────

    def add(self, title: str, description: str, domains: list[str] | None = None) -> Idea:
        """Add a new idea to the pool."""
        idea_id = uuid.uuid4().hex[:8]
        idea = Idea(
            id=idea_id,
            title=title,
            description=description,
            domains=domains or [],
        )
        self.ideas[idea_id] = idea
        self._save()
        logger.info("Added idea %s: %s", idea_id, title)
        return idea

    def remove(self, idea_id: str) -> None:
        """Remove an idea from the pool."""
        if idea_id not in self.ideas:
            raise KeyError(f"Unknown idea: {idea_id}")
        del self.ideas[idea_id]
        self._save()

    def get(self, idea_id: str) -> Idea:
        """Get an idea by ID."""
        if idea_id not in self.ideas:
            raise KeyError(f"Unknown idea: {idea_id}")
        return self.ideas[idea_id]

    # ── evaluation ────────────────────────────────────────────────

    def evaluate(self, idea_id: str, feasibility: float, novelty: float) -> dict[str, Any]:
        """Set feasibility and novelty scores for an idea."""
        idea = self.get(idea_id)
        idea.feasibility = max(0.0, min(1.0, feasibility))
        idea.novelty = max(0.0, min(1.0, novelty))
        idea.status = "evaluated"
        self._save()
        return {
            "id": idea.id,
            "feasibility": idea.feasibility,
            "novelty": idea.novelty,
            "score": idea.score,
        }

    def rank(self) -> list[Idea]:
        """Return all ideas sorted by composite score (descending)."""
        return sorted(self.ideas.values(), key=lambda i: i.score, reverse=True)

    # ── conversion ────────────────────────────────────────────────

    def to_project(self, idea_id: str, config_path: str, projects_dir: str | Path) -> Project:
        """Convert an idea into a project skeleton."""
        idea = self.get(idea_id)
        from researchclaw.project.manager import ProjectManager

        manager = ProjectManager(projects_dir)
        project = manager.create(
            name=idea.title.lower().replace(" ", "_")[:40],
            config_path=config_path,
            topic=idea.description,
        )
        idea.status = "planned"
        self._save()
        return project

    def list_all(self) -> list[Idea]:
        """Return all ideas sorted by creation time."""
        return sorted(self.ideas.values(), key=lambda i: i.created_at)
