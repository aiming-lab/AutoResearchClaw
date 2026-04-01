"""WebSocket event type definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any
import json
import time


class EventType(str, Enum):
    """All WebSocket event types."""

    # Lifecycle
    CONNECTED = "connected"
    HEARTBEAT = "heartbeat"
    ERROR = "error"

    # Pipeline
    PIPELINE_STARTED = "pipeline_started"
    PIPELINE_COMPLETED = "pipeline_completed"
    STAGE_START = "stage_start"
    STAGE_COMPLETE = "stage_complete"
    STAGE_FAIL = "stage_fail"
    METRIC_UPDATE = "metric_update"
    LOG_LINE = "log_line"
    PAPER_READY = "paper_ready"

    # Chat
    CHAT_RESPONSE = "chat_response"
    CHAT_TYPING = "chat_typing"
    CHAT_SUGGESTION = "chat_suggestion"

    # System
    RUN_DISCOVERED = "run_discovered"
    RUN_STATUS_CHANGED = "run_status_changed"


@dataclass
class Event:
    """A WebSocket event."""

    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(
            {
                "type": self.type.value,
                "data": self.data,
                "timestamp": self.timestamp,
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> Event:
        """Deserialize from JSON string."""
        obj = json.loads(raw)
        return cls(
            type=EventType(obj["type"]),
            data=obj.get("data", {}),
            timestamp=obj.get("timestamp", time.time()),
        )
