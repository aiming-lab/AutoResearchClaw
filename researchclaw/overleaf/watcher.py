"""File change watcher for Overleaf sync polling."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FileWatcher:
    """Watch a directory for file changes (poll-based)."""

    def __init__(self, watch_dir: Path, extensions: tuple[str, ...] = (".tex", ".bib")) -> None:
        self.watch_dir = watch_dir
        self.extensions = extensions
        self._snapshot: dict[str, float] = {}
        self._take_snapshot()

    def _take_snapshot(self) -> None:
        """Record modification times of all tracked files."""
        self._snapshot = {}
        if not self.watch_dir.exists():
            return
        for ext in self.extensions:
            for f in self.watch_dir.rglob(f"*{ext}"):
                self._snapshot[str(f.relative_to(self.watch_dir))] = f.stat().st_mtime

    def check_changes(self) -> list[str]:
        """Return files that have changed since the last snapshot."""
        changed: list[str] = []
        current: dict[str, float] = {}
        if not self.watch_dir.exists():
            return changed

        for ext in self.extensions:
            for f in self.watch_dir.rglob(f"*{ext}"):
                rel = str(f.relative_to(self.watch_dir))
                mtime = f.stat().st_mtime
                current[rel] = mtime
                old_mtime = self._snapshot.get(rel)
                if old_mtime is None or mtime > old_mtime:
                    changed.append(rel)

        # Check for deleted files
        for rel in self._snapshot:
            if rel not in current:
                changed.append(rel)

        self._snapshot = current
        return changed

    def poll_loop(self, interval_sec: int = 300, callback: Any = None) -> None:
        """Blocking poll loop that calls callback on changes.

        This is meant to be run in a background thread.
        """
        logger.info("Starting file watcher on %s (interval=%ds)", self.watch_dir, interval_sec)
        while True:
            time.sleep(interval_sec)
            changes = self.check_changes()
            if changes and callback:
                callback(changes)
