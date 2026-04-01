"""Unified memory storage engine.

Manages three categories of memory (ideation, experiment, writing) with
JSONL persistence, vector embeddings for semantic retrieval, time-decay
weighting, and confidence scoring.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory record."""

    id: str
    category: str  # "ideation" | "experiment" | "writing"
    content: str
    metadata: dict[str, Any]
    embedding: list[float]
    confidence: float
    created_at: str  # ISO 8601
    last_accessed: str  # ISO 8601
    access_count: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        """Deserialize from dictionary."""
        return cls(
            id=str(data.get("id", "")),
            category=str(data.get("category", "")),
            content=str(data.get("content", "")),
            metadata=data.get("metadata") or {},
            embedding=data.get("embedding") or [],
            confidence=float(data.get("confidence", 0.5)),
            created_at=str(data.get("created_at", "")),
            last_accessed=str(data.get("last_accessed", "")),
            access_count=int(data.get("access_count", 0)),
        )


VALID_CATEGORIES = ("ideation", "experiment", "writing")


class MemoryStore:
    """JSONL-backed persistent memory storage.

    Stores MemoryEntry records organized by category, supporting
    add/recall/update/prune/save/load operations.
    """

    def __init__(
        self,
        store_dir: str | Path,
        max_entries_per_category: int = 500,
        confidence_threshold: float = 0.3,
    ) -> None:
        self._store_dir = Path(store_dir)
        self._max_per_category = max_entries_per_category
        self._confidence_threshold = confidence_threshold
        self._entries: dict[str, list[MemoryEntry]] = {
            cat: [] for cat in VALID_CATEGORIES
        }
        self._dirty = False

    @property
    def store_dir(self) -> Path:
        """Return the storage directory path."""
        return self._store_dir

    def add(
        self,
        category: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        confidence: float = 0.5,
    ) -> str:
        """Add a new memory entry.

        Args:
            category: One of "ideation", "experiment", "writing".
            content: The memory content text.
            metadata: Optional metadata dict (run_id, stage, topic, etc.).
            embedding: Pre-computed embedding vector (or empty).
            confidence: Initial confidence score (0-1).

        Returns:
            The generated entry ID.

        Raises:
            ValueError: If category is invalid.
        """
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{category}'. Must be one of {VALID_CATEGORIES}"
            )

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        entry_id = uuid.uuid4().hex[:12]

        entry = MemoryEntry(
            id=entry_id,
            category=category,
            content=content,
            metadata=metadata or {},
            embedding=embedding or [],
            confidence=confidence,
            created_at=now,
            last_accessed=now,
            access_count=0,
        )

        self._entries[category].append(entry)
        self._dirty = True

        # Enforce capacity limit (remove lowest confidence)
        entries = self._entries[category]
        if len(entries) > self._max_per_category:
            entries.sort(key=lambda e: e.confidence, reverse=True)
            self._entries[category] = entries[: self._max_per_category]

        return entry_id

    def get(self, entry_id: str) -> MemoryEntry | None:
        """Retrieve a single entry by ID."""
        for entries in self._entries.values():
            for entry in entries:
                if entry.id == entry_id:
                    return entry
        return None

    def get_all(self, category: str | None = None) -> list[MemoryEntry]:
        """Return all entries, optionally filtered by category."""
        if category:
            return list(self._entries.get(category, []))
        result: list[MemoryEntry] = []
        for entries in self._entries.values():
            result.extend(entries)
        return result

    def update_confidence(self, entry_id: str, delta: float) -> bool:
        """Update the confidence score of an entry.

        Args:
            entry_id: The entry to update.
            delta: Change amount (+0.1 for success, -0.2 for failure).

        Returns:
            True if entry was found and updated, False otherwise.
        """
        for entries in self._entries.values():
            for i, entry in enumerate(entries):
                if entry.id == entry_id:
                    new_conf = max(0.0, min(1.0, entry.confidence + delta))
                    # Replace with updated entry (frozen-like pattern)
                    entries[i] = MemoryEntry(
                        id=entry.id,
                        category=entry.category,
                        content=entry.content,
                        metadata=entry.metadata,
                        embedding=entry.embedding,
                        confidence=new_conf,
                        created_at=entry.created_at,
                        last_accessed=entry.last_accessed,
                        access_count=entry.access_count,
                    )
                    self._dirty = True
                    return True
        return False

    def mark_accessed(self, entry_id: str) -> bool:
        """Update last_accessed timestamp and increment access count."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for entries in self._entries.values():
            for i, entry in enumerate(entries):
                if entry.id == entry_id:
                    entries[i] = MemoryEntry(
                        id=entry.id,
                        category=entry.category,
                        content=entry.content,
                        metadata=entry.metadata,
                        embedding=entry.embedding,
                        confidence=entry.confidence,
                        created_at=entry.created_at,
                        last_accessed=now,
                        access_count=entry.access_count + 1,
                    )
                    self._dirty = True
                    return True
        return False

    def prune(
        self,
        confidence_threshold: float | None = None,
        max_age_days: float = 365.0,
    ) -> int:
        """Remove expired and low-confidence entries.

        Args:
            confidence_threshold: Minimum confidence (default from init).
            max_age_days: Maximum age in days.

        Returns:
            Number of entries removed.
        """
        threshold = confidence_threshold if confidence_threshold is not None else self._confidence_threshold
        now = datetime.now(timezone.utc)
        removed = 0

        for category in VALID_CATEGORIES:
            before = len(self._entries[category])
            kept: list[MemoryEntry] = []
            for entry in self._entries[category]:
                try:
                    created = datetime.fromisoformat(entry.created_at)
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)
                    age_days = (now - created).total_seconds() / 86400.0
                except (ValueError, TypeError):
                    age_days = 0.0

                if entry.confidence >= threshold and age_days <= max_age_days:
                    kept.append(entry)

            self._entries[category] = kept
            removed += before - len(kept)

        if removed > 0:
            self._dirty = True
            logger.info("Pruned %d memory entries", removed)

        return removed

    def save(self) -> None:
        """Persist all entries to disk in JSONL format."""
        self._store_dir.mkdir(parents=True, exist_ok=True)

        for category in VALID_CATEGORIES:
            path = self._store_dir / f"{category}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for entry in self._entries[category]:
                    f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

        self._dirty = False
        total = sum(len(v) for v in self._entries.values())
        logger.info("Saved %d memory entries to %s", total, self._store_dir)

    def load(self) -> int:
        """Load entries from disk.

        Returns:
            Total number of entries loaded.
        """
        total = 0
        for category in VALID_CATEGORIES:
            path = self._store_dir / f"{category}.jsonl"
            if not path.exists():
                continue
            entries: list[MemoryEntry] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entries.append(MemoryEntry.from_dict(data))
                except (json.JSONDecodeError, TypeError) as exc:
                    logger.warning("Skipping malformed memory entry: %s", exc)
                    continue
            self._entries[category] = entries
            total += len(entries)

        logger.info("Loaded %d memory entries from %s", total, self._store_dir)
        return total

    def count(self, category: str | None = None) -> int:
        """Return total entries, optionally filtered by category."""
        if category:
            return len(self._entries.get(category, []))
        return sum(len(v) for v in self._entries.values())
