"""MCP transport layer: stdio and SSE implementations."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class MCPTransport(Protocol):
    """Protocol for MCP message transport."""

    async def send(self, message: dict[str, Any]) -> None: ...
    async def receive(self) -> dict[str, Any]: ...
    async def close(self) -> None: ...


class StdioTransport:
    """MCP transport over stdin/stdout (for CLI integration)."""

    def __init__(self) -> None:
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    async def start(self) -> None:
        """Initialize stdin/stdout streams for async I/O."""
        loop = asyncio.get_event_loop()
        self._reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(self._reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)
        w_transport, w_protocol = await loop.connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        self._writer = asyncio.StreamWriter(w_transport, w_protocol, self._reader, loop)

    async def send(self, message: dict[str, Any]) -> None:
        """Write a JSON-RPC message to stdout."""
        if self._writer is None:
            raise RuntimeError("Transport not started")
        data = json.dumps(message, ensure_ascii=False)
        header = f"Content-Length: {len(data.encode())}\r\n\r\n"
        self._writer.write(header.encode() + data.encode())
        await self._writer.drain()

    async def receive(self) -> dict[str, Any]:
        """Read a JSON-RPC message from stdin."""
        if self._reader is None:
            raise RuntimeError("Transport not started")
        # Read headers
        content_length = 0
        while True:
            line = await self._reader.readline()
            decoded = line.decode().strip()
            if not decoded:
                break
            if decoded.lower().startswith("content-length:"):
                content_length = int(decoded.split(":")[1].strip())
        if content_length == 0:
            raise EOFError("No content-length header received")
        body = await self._reader.readexactly(content_length)
        return json.loads(body)

    async def close(self) -> None:
        """Close the transport."""
        if self._writer:
            self._writer.close()


class SSETransport:
    """MCP transport over Server-Sent Events (for web integration).

    This is a stub — a full implementation would use aiohttp or similar.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 3000) -> None:
        self.host = host
        self.port = port
        self._running = False

    async def start(self) -> None:
        """Start the SSE server."""
        self._running = True
        logger.info("SSE transport started on %s:%d", self.host, self.port)

    async def send(self, message: dict[str, Any]) -> None:
        """Send an SSE event (stub)."""
        logger.debug("SSE send: %s", json.dumps(message, default=str)[:200])

    async def receive(self) -> dict[str, Any]:
        """Receive from SSE (stub)."""
        raise NotImplementedError("SSE receive not yet implemented")

    async def close(self) -> None:
        """Stop the SSE server."""
        self._running = False
