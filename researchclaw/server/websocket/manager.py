"""WebSocket connection manager."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import WebSocket

from .events import Event, EventType

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manage WebSocket connections and broadcast events."""

    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}
        self._event_queue: asyncio.Queue[Event] = asyncio.Queue()

    @property
    def active_count(self) -> int:
        return len(self._connections)

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """Accept and register a WebSocket connection."""
        await websocket.accept()
        self._connections[client_id] = websocket
        logger.info("WebSocket connected: %s (total: %d)", client_id, self.active_count)

        await self._send(
            websocket,
            Event(type=EventType.CONNECTED, data={"client_id": client_id}),
        )

    def disconnect(self, client_id: str) -> None:
        """Remove a disconnected client."""
        self._connections.pop(client_id, None)
        logger.info("WebSocket disconnected: %s (total: %d)", client_id, self.active_count)

    async def broadcast(self, event: Event) -> None:
        """Send event to all connected clients."""
        dead: list[str] = []
        for cid, ws in self._connections.items():
            try:
                await self._send(ws, event)
            except Exception:
                dead.append(cid)
        for cid in dead:
            self.disconnect(cid)

    async def send_to(self, client_id: str, event: Event) -> None:
        """Send event to a specific client."""
        ws = self._connections.get(client_id)
        if ws:
            try:
                await self._send(ws, event)
            except Exception:
                self.disconnect(client_id)

    async def _send(self, ws: WebSocket, event: Event) -> None:
        await ws.send_text(event.to_json())

    def publish(self, event: Event) -> None:
        """Non-async publish for use from sync code (thread-safe queue)."""
        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("Event queue full, dropping event: %s", event.type)

    async def drain_queue(self) -> None:
        """Process queued events and broadcast them."""
        while not self._event_queue.empty():
            event = self._event_queue.get_nowait()
            await self.broadcast(event)

    async def heartbeat_loop(self, interval: float = 15.0) -> None:
        """Send periodic heartbeat to all clients."""
        while True:
            await asyncio.sleep(interval)
            await self.broadcast(
                Event(
                    type=EventType.HEARTBEAT,
                    data={"active_clients": self.active_count},
                )
            )
            await self.drain_queue()
