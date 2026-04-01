"""Conversation session management."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """A single chat message."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}


@dataclass
class ChatSession:
    """Per-client chat session state."""

    client_id: str
    history: list[ChatMessage] = field(default_factory=list)
    current_project: str = ""
    current_run: str = ""
    created_at: float = field(default_factory=time.time)

    MAX_HISTORY: int = 50

    def add_message(self, role: str, content: str) -> ChatMessage:
        msg = ChatMessage(role=role, content=content)
        self.history.append(msg)
        # Trim to prevent unbounded growth
        if len(self.history) > self.MAX_HISTORY:
            self.history = self.history[-self.MAX_HISTORY:]
        return msg

    def get_context(self, last_n: int = 10) -> list[dict[str, str]]:
        """Get recent messages for LLM context."""
        return [
            {"role": m.role, "content": m.content}
            for m in self.history[-last_n:]
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "current_project": self.current_project,
            "current_run": self.current_run,
            "history": [m.to_dict() for m in self.history],
            "created_at": self.created_at,
        }


class SessionManager:
    """Manage chat sessions."""

    def __init__(self, persist_dir: str = ".researchclaw/sessions") -> None:
        self._sessions: dict[str, ChatSession] = {}
        self._persist_dir = Path(persist_dir)

    def get_or_create(self, client_id: str) -> ChatSession:
        """Get existing session or create new one."""
        if client_id not in self._sessions:
            self._sessions[client_id] = ChatSession(client_id=client_id)
        return self._sessions[client_id]

    def remove(self, client_id: str) -> None:
        """Remove a session."""
        self._sessions.pop(client_id, None)

    def save(self, client_id: str) -> None:
        """Persist session to disk."""
        session = self._sessions.get(client_id)
        if not session:
            return
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        path = self._persist_dir / f"{client_id}.json"
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception:
            logger.debug("Failed to persist session %s", client_id)

    def load(self, client_id: str) -> ChatSession | None:
        """Load session from disk."""
        path = self._persist_dir / f"{client_id}.json"
        if not path.exists():
            return None
        try:
            with path.open() as f:
                data = json.load(f)
            session = ChatSession(
                client_id=data["client_id"],
                current_project=data.get("current_project", ""),
                current_run=data.get("current_run", ""),
                created_at=data.get("created_at", time.time()),
            )
            for m in data.get("history", []):
                session.history.append(
                    ChatMessage(
                        role=m["role"],
                        content=m["content"],
                        timestamp=m.get("timestamp", 0),
                    )
                )
            self._sessions[client_id] = session
            return session
        except Exception:
            logger.debug("Failed to load session %s", client_id)
            return None
