"""Project scheduler: priority queue and concurrency control for pipeline runs."""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from typing import Any

from researchclaw.project.manager import ProjectManager

logger = logging.getLogger(__name__)


@dataclass(order=True)
class _QueueEntry:
    """Priority queue entry (lower priority number = higher priority)."""

    priority: int
    project_name: str = field(compare=False)


class ProjectScheduler:
    """Schedule project pipeline runs with priority and concurrency limits."""

    def __init__(self, manager: ProjectManager, max_concurrent: int = 2) -> None:
        self.manager = manager
        self.max_concurrent = max_concurrent
        self._queue: list[_QueueEntry] = []
        self._running: set[str] = set()

    def enqueue(self, project_name: str, priority: int = 0) -> None:
        """Add a project to the run queue."""
        if project_name not in self.manager.projects:
            raise KeyError(f"Unknown project: {project_name}")
        # Avoid duplicate enqueue
        for entry in self._queue:
            if entry.project_name == project_name:
                logger.info("Project %s already in queue", project_name)
                return
        if project_name in self._running:
            logger.info("Project %s already running", project_name)
            return
        heapq.heappush(self._queue, _QueueEntry(priority=priority, project_name=project_name))
        logger.info("Enqueued project %s with priority %d", project_name, priority)

    def dequeue(self) -> str | None:
        """Remove and return the highest-priority project from the queue."""
        if not self._queue:
            return None
        entry = heapq.heappop(self._queue)
        return entry.project_name

    def next(self) -> str | None:
        """Get the next project that should run, if a slot is available."""
        if not self.can_start():
            return None
        name = self.dequeue()
        if name is not None:
            self._running.add(name)
        return name

    def can_start(self) -> bool:
        """Check whether there is capacity to start another run."""
        return len(self._running) < self.max_concurrent and len(self._queue) > 0

    def mark_done(self, project_name: str) -> None:
        """Mark a running project as finished (frees a concurrency slot)."""
        self._running.discard(project_name)

    @property
    def queue_size(self) -> int:
        """Number of projects waiting in the queue."""
        return len(self._queue)

    @property
    def running_count(self) -> int:
        """Number of projects currently running."""
        return len(self._running)

    def get_status(self) -> dict[str, Any]:
        """Scheduler status overview."""
        return {
            "max_concurrent": self.max_concurrent,
            "running": sorted(self._running),
            "running_count": len(self._running),
            "queued": [e.project_name for e in sorted(self._queue)],
            "queue_size": len(self._queue),
        }
