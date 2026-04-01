"""Chat WebSocket endpoint for conversational research."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from researchclaw.server.websocket.events import Event, EventType
from researchclaw.server.websocket.manager import ConnectionManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# Global connection manager (initialized by app.py)
_chat_manager: ConnectionManager | None = None


def set_chat_manager(manager: ConnectionManager) -> None:
    """Set the shared connection manager."""
    global _chat_manager
    _chat_manager = manager


def get_chat_manager() -> ConnectionManager:
    """Get the shared connection manager."""
    if _chat_manager is None:
        raise RuntimeError("Chat manager not initialized")
    return _chat_manager


@router.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket) -> None:
    """WebSocket endpoint for conversational research chat."""
    manager = get_chat_manager()
    client_id = str(uuid.uuid4())[:8]

    await manager.connect(websocket, client_id)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                from researchclaw.server.dialog.router import route_message

                response = await route_message(raw, client_id)
                await manager.send_to(
                    client_id,
                    Event(
                        type=EventType.CHAT_RESPONSE,
                        data={"message": response, "client_id": client_id},
                    ),
                )
            except Exception as exc:
                logger.exception("Chat error for %s", client_id)
                await manager.send_to(
                    client_id,
                    Event(
                        type=EventType.ERROR,
                        data={"error": str(exc), "client_id": client_id},
                    ),
                )
    except WebSocketDisconnect:
        manager.disconnect(client_id)
