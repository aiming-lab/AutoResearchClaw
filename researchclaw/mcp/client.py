"""MCP Client: connect to external MCP servers for enhanced capabilities."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class MCPClient:
    """Connect to an external MCP server and invoke its tools.

    Supports stdio and SSE transports. The actual protocol I/O
    is abstracted so we can add more transports later.
    """

    def __init__(self, server_uri: str, transport: str = "stdio") -> None:
        self.uri = server_uri
        self.transport = transport
        self._connected = False
        self._tools_cache: list[dict[str, Any]] | None = None

    # ── connection ────────────────────────────────────────────────

    async def connect(self) -> None:
        """Establish connection to the MCP server."""
        logger.info("Connecting to MCP server: %s (transport=%s)", self.uri, self.transport)
        self._connected = True

    async def disconnect(self) -> None:
        """Close the connection."""
        self._connected = False
        self._tools_cache = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── tool discovery ────────────────────────────────────────────

    async def list_tools(self) -> list[dict[str, Any]]:
        """List tools available on the remote MCP server."""
        if not self._connected:
            raise ConnectionError("Not connected to MCP server")
        if self._tools_cache is not None:
            return self._tools_cache

        response = await self._send_request("tools/list", {})
        tools = response.get("tools", [])
        self._tools_cache = tools
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a tool on the remote MCP server."""
        if not self._connected:
            raise ConnectionError("Not connected to MCP server")
        return await self._send_request("tools/call", {"name": name, "arguments": arguments})

    # ── resource access ───────────────────────────────────────────

    async def list_resources(self) -> list[dict[str, Any]]:
        """List resources available on the remote MCP server."""
        if not self._connected:
            raise ConnectionError("Not connected to MCP server")
        response = await self._send_request("resources/list", {})
        return response.get("resources", [])

    async def read_resource(self, uri: str) -> str:
        """Read a resource from the remote MCP server."""
        if not self._connected:
            raise ConnectionError("Not connected to MCP server")
        response = await self._send_request("resources/read", {"uri": uri})
        contents = response.get("contents", [])
        if contents:
            return contents[0].get("text", "")
        return ""

    # ── transport layer ───────────────────────────────────────────

    async def _send_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request to the MCP server.

        This is a stub — real implementation delegates to transport.py.
        """
        message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }
        logger.debug("MCP request: %s", json.dumps(message, default=str)[:200])
        # Stub: return empty result
        return {"result": {}}
