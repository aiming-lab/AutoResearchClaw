"""Project manager: CRUD operations and status tracking for research projects."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from researchclaw.project.models import Project

logger = logging.getLogger(__name__)

_REGISTRY_FILE = "registry.json"


class ProjectManager:
    """Manage multiple research projects with independent directories and configs."""

    def __init__(self, projects_dir: str | Path) -> None:
        self.projects_dir = Path(projects_dir).expanduser().resolve()
        self.projects: dict[str, Project] = {}
        self._active: str | None = None
        self._load_registry()

    # ── persistence ───────────────────────────────────────────────

    def _registry_path(self) -> Path:
        return self.projects_dir / _REGISTRY_FILE

    def _load_registry(self) -> None:
        """Load project registry from disk."""
        path = self._registry_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for entry in data.get("projects", []):
                proj = Project.from_dict(entry)
                self.projects[proj.name] = proj
            self._active = data.get("active")
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to load project registry: %s", exc)

    def _save_registry(self) -> None:
        """Persist project registry to disk."""
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "active": self._active,
            "projects": [p.to_dict() for p in self.projects.values()],
        }
        self._registry_path().write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ── CRUD ──────────────────────────────────────────────────────

    def create(
        self,
        name: str,
        config_path: str,
        topic: str | None = None,
    ) -> Project:
        """Create a new project with an independent directory and config copy."""
        if name in self.projects:
            raise ValueError(f"Project already exists: {name}")

        project_dir = self.projects_dir / name
        project_dir.mkdir(parents=True, exist_ok=True)

        # Copy config to project directory
        src = Path(config_path).expanduser().resolve()
        if src.exists():
            dst = project_dir / "config.yaml"
            shutil.copy2(src, dst)
            stored_config = str(dst)
        else:
            stored_config = config_path

        run_dir = str(project_dir / "artifacts")
        Path(run_dir).mkdir(parents=True, exist_ok=True)

        project = Project(
            name=name,
            config_path=stored_config,
            run_dir=run_dir,
            topic=topic or "",
        )
        self.projects[name] = project
        if self._active is None:
            self._active = name
        self._save_registry()
        logger.info("Created project: %s", name)
        return project

    def delete(self, name: str) -> None:
        """Remove project from registry. Does NOT delete artifacts on disk."""
        if name not in self.projects:
            raise KeyError(f"Unknown project: {name}")
        del self.projects[name]
        if self._active == name:
            self._active = next(iter(self.projects), None)
        self._save_registry()
        logger.info("Deleted project (registry only): %s", name)

    def get(self, name: str) -> Project:
        """Get a single project by name."""
        if name not in self.projects:
            raise KeyError(f"Unknown project: {name}")
        return self.projects[name]

    def list_all(self) -> list[Project]:
        """Return all projects sorted by creation time."""
        return sorted(self.projects.values(), key=lambda p: p.created_at)

    def get_status(self) -> dict[str, Any]:
        """Summary of all project statuses."""
        projects = self.list_all()
        return {
            "total": len(projects),
            "active": self._active,
            "by_status": _count_by(projects, "status"),
            "projects": [
                {"name": p.name, "status": p.status, "topic": p.topic}
                for p in projects
            ],
        }

    # ── project switching ─────────────────────────────────────────

    def switch(self, name: str) -> Project:
        """Set the active project."""
        if name not in self.projects:
            raise KeyError(f"Unknown project: {name}")
        self._active = name
        self._save_registry()
        return self.projects[name]

    @property
    def active(self) -> Project | None:
        """Currently active project."""
        if self._active and self._active in self.projects:
            return self.projects[self._active]
        return None

    # ── comparison ────────────────────────────────────────────────

    def compare(self, name_a: str, name_b: str) -> dict[str, Any]:
        """Compare metrics and status of two projects."""
        a = self.get(name_a)
        b = self.get(name_b)
        return {
            "project_a": {"name": a.name, "status": a.status, "topic": a.topic, "metrics": a.metrics},
            "project_b": {"name": b.name, "status": b.status, "topic": b.topic, "metrics": b.metrics},
            "metric_diff": _metric_diff(a.metrics, b.metrics),
        }

    # ── run lifecycle ─────────────────────────────────────────────

    def start_run(self, name: str, run_id: str) -> str:
        """Mark a project as running with a new run ID."""
        proj = self.get(name)
        proj.status = "running"
        proj.last_run_id = run_id
        self._save_registry()
        return run_id

    def finish_run(self, name: str, status: str, metrics: dict[str, Any] | None = None) -> None:
        """Mark a project run as completed or failed."""
        proj = self.get(name)
        proj.status = status
        if metrics:
            proj.metrics = metrics
        self._save_registry()


def _count_by(projects: list[Project], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in projects:
        val = getattr(p, attr, "unknown")
        counts[val] = counts.get(val, 0) + 1
    return counts


def _metric_diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    all_keys = set(a) | set(b)
    diff: dict[str, Any] = {}
    for key in sorted(all_keys):
        va, vb = a.get(key), b.get(key)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            diff[key] = {"a": va, "b": vb, "delta": round(vb - va, 6)}
        else:
            diff[key] = {"a": va, "b": vb}
    return diff
