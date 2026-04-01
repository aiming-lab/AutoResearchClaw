"""Data models for multi-project management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Project:
    """A research project managed by AutoResearchClaw."""

    name: str
    config_path: str
    run_dir: str
    status: str = "idle"  # idle | running | completed | failed
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_run_id: str | None = None
    topic: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize project to a dictionary."""
        return {
            "name": self.name,
            "config_path": self.config_path,
            "run_dir": self.run_dir,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "last_run_id": self.last_run_id,
            "topic": self.topic,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Project:
        """Deserialize project from a dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)
        return cls(
            name=data["name"],
            config_path=data["config_path"],
            run_dir=data["run_dir"],
            status=data.get("status", "idle"),
            created_at=created_at,
            last_run_id=data.get("last_run_id"),
            topic=data.get("topic", ""),
            metrics=data.get("metrics", {}),
        )


@dataclass
class Idea:
    """A research idea that can be evaluated and converted to a project."""

    id: str
    title: str
    description: str
    status: str = "draft"  # draft | evaluated | planned | running | completed
    feasibility: float = 0.0  # 0-1
    novelty: float = 0.0  # 0-1
    domains: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def score(self) -> float:
        """Composite score: weighted average of feasibility and novelty."""
        return 0.4 * self.feasibility + 0.6 * self.novelty

    def to_dict(self) -> dict[str, Any]:
        """Serialize idea to a dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "feasibility": self.feasibility,
            "novelty": self.novelty,
            "domains": self.domains,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Idea:
        """Deserialize idea from a dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            status=data.get("status", "draft"),
            feasibility=float(data.get("feasibility", 0.0)),
            novelty=float(data.get("novelty", 0.0)),
            domains=data.get("domains", []),
            created_at=created_at,
        )
