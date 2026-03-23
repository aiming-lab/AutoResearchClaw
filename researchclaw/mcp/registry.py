"""Registry of connected MCP servers."""

from __future__ import annotations

import logging
from typing import Any

from researchclaw.mcp.client import MCPClient

logger = logging.getLogger(__name__)


class MCPServerRegistry:
    """Track connected external MCP servers."""

    def __init__(self) -> None:
        self._servers: dict[str, MCPClient] = {}

    async def register(self, name: str, uri: str, transport: str = "stdio") -> MCPClient:
        """Register and connect to an external MCP server."""
        client = MCPClient(uri, transport=transport)
        await client.connect()
        self._servers[name] = client
        logger.info("Registered MCP server: %s -> %s", name, uri)
        return client

    async def unregister(self, name: str) -> None:
        """Disconnect and remove an MCP server."""
        client = self._servers.pop(name, None)
        if client:
            await client.disconnect()

    def get(self, name: str) -> MCPClient | None:
        """Get a connected MCP client by name."""
        return self._servers.get(name)

    def list_all(self) -> list[dict[str, Any]]:
        """List all registered MCP servers."""
        return [
            {"name": name, "uri": client.uri, "connected": client.is_connected}
            for name, client in self._servers.items()
        ]

    async def close_all(self) -> None:
        """Disconnect from all servers."""
        for name in list(self._servers):
            await self.unregister(name)

    @property
    def count(self) -> int:
        return len(self._servers)
