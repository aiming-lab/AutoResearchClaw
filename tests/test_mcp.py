"""Tests for MCP integration (C3): Server, Client, Tools, Transport, Registry."""

from __future__ import annotations

import asyncio

import pytest

from researchclaw.mcp.tools import TOOL_DEFINITIONS, get_tool_schema, list_tool_names
from researchclaw.mcp.server import ResearchClawMCPServer
from researchclaw.mcp.client import MCPClient
from researchclaw.mcp.registry import MCPServerRegistry
from researchclaw.mcp.transport import SSETransport


# ══════════════════════════════════════════════════════════════════
# MCP Tools tests
# ══════════════════════════════════════════════════════════════════


class TestMCPTools:
    def test_tool_definitions_not_empty(self) -> None:
        assert len(TOOL_DEFINITIONS) >= 6

    def test_all_tools_have_required_fields(self) -> None:
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"

    def test_get_tool_schema_exists(self) -> None:
        schema = get_tool_schema("run_pipeline")
        assert schema is not None
        assert schema["name"] == "run_pipeline"

    def test_get_tool_schema_missing(self) -> None:
        assert get_tool_schema("nonexistent") is None

    def test_list_tool_names(self) -> None:
        names = list_tool_names()
        assert "run_pipeline" in names
        assert "get_pipeline_status" in names
        assert "search_literature" in names

    def test_run_pipeline_requires_topic(self) -> None:
        schema = get_tool_schema("run_pipeline")
        assert schema is not None
        assert "topic" in schema["inputSchema"]["required"]

    def test_get_paper_has_format_enum(self) -> None:
        schema = get_tool_schema("get_paper")
        assert schema is not None
        props = schema["inputSchema"]["properties"]
        assert "format" in props
        assert "enum" in props["format"]


# ══════════════════════════════════════════════════════════════════
# MCP Server tests
# ══════════════════════════════════════════════════════════════════


class TestMCPServer:
    def test_get_tools(self) -> None:
        server = ResearchClawMCPServer()
        tools = server.get_tools()
        assert len(tools) >= 6
        names = [t["name"] for t in tools]
        assert "run_pipeline" in names

    def test_handle_unknown_tool(self) -> None:
        server = ResearchClawMCPServer()
        result = asyncio.run(server.handle_tool_call("nonexistent", {}))
        assert result["success"] is False
        assert "Unknown tool" in result["error"]

    def test_handle_run_pipeline(self) -> None:
        server = ResearchClawMCPServer()
        result = asyncio.run(server.handle_tool_call("run_pipeline", {"topic": "GNN"}))
        assert result["success"] is True
        assert "GNN" in result["message"]

    def test_handle_get_status_missing_run(self) -> None:
        server = ResearchClawMCPServer()
        result = asyncio.run(server.handle_tool_call("get_pipeline_status", {"run_id": "nonexistent"}))
        assert result["success"] is False

    def test_handle_search_literature(self) -> None:
        server = ResearchClawMCPServer()
        result = asyncio.run(server.handle_tool_call("search_literature", {"query": "transformers"}))
        assert result["success"] is True

    def test_handle_review_paper(self) -> None:
        server = ResearchClawMCPServer()
        result = asyncio.run(server.handle_tool_call("review_paper", {"paper_path": "/tmp/paper.md"}))
        assert result["success"] is True

    def test_start_stop(self) -> None:
        server = ResearchClawMCPServer()
        assert not server.is_running

        async def _run() -> None:
            await server.start()
            assert server.is_running
            await server.stop()
            assert not server.is_running

        asyncio.run(_run())

    def test_handle_get_results_missing(self) -> None:
        server = ResearchClawMCPServer()
        result = asyncio.run(server.handle_tool_call("get_experiment_results", {"run_id": "missing"}))
        assert result["success"] is False

    def test_handle_get_paper_missing(self) -> None:
        server = ResearchClawMCPServer()
        result = asyncio.run(server.handle_tool_call("get_paper", {"run_id": "missing"}))
        assert result["success"] is False


# ══════════════════════════════════════════════════════════════════
# MCP Client tests
# ══════════════════════════════════════════════════════════════════


class TestMCPClient:
    def test_init(self) -> None:
        client = MCPClient("http://localhost:3000")
        assert client.uri == "http://localhost:3000"
        assert not client.is_connected

    def test_connect_disconnect(self) -> None:
        client = MCPClient("http://localhost:3000")

        async def _run() -> None:
            await client.connect()
            assert client.is_connected
            await client.disconnect()
            assert not client.is_connected

        asyncio.run(_run())

    def test_list_tools_not_connected(self) -> None:
        client = MCPClient("http://localhost:3000")
        with pytest.raises(ConnectionError):
            asyncio.run(client.list_tools())

    def test_call_tool_not_connected(self) -> None:
        client = MCPClient("http://localhost:3000")
        with pytest.raises(ConnectionError):
            asyncio.run(client.call_tool("test", {}))

    def test_list_resources_not_connected(self) -> None:
        client = MCPClient("http://localhost:3000")
        with pytest.raises(ConnectionError):
            asyncio.run(client.list_resources())

    def test_read_resource_not_connected(self) -> None:
        client = MCPClient("http://localhost:3000")
        with pytest.raises(ConnectionError):
            asyncio.run(client.read_resource("test://resource"))

    def test_list_tools_connected(self) -> None:
        client = MCPClient("http://localhost:3000")

        async def _run() -> list:
            await client.connect()
            return await client.list_tools()

        tools = asyncio.run(_run())
        assert isinstance(tools, list)

    def test_tools_cached(self) -> None:
        client = MCPClient("http://localhost:3000")

        async def _run() -> tuple:
            await client.connect()
            t1 = await client.list_tools()
            t2 = await client.list_tools()
            return t1, t2

        t1, t2 = asyncio.run(_run())
        assert t1 is t2


# ══════════════════════════════════════════════════════════════════
# MCP Server Registry tests
# ══════════════════════════════════════════════════════════════════


class TestMCPServerRegistry:
    def test_register_and_list(self) -> None:
        async def _run() -> list:
            reg = MCPServerRegistry()
            await reg.register("test", "http://localhost:3000")
            return reg.list_all()

        servers = asyncio.run(_run())
        assert len(servers) == 1
        assert servers[0]["name"] == "test"
        assert servers[0]["connected"] is True

    def test_unregister(self) -> None:
        async def _run() -> int:
            reg = MCPServerRegistry()
            await reg.register("test", "http://localhost:3000")
            await reg.unregister("test")
            return reg.count

        count = asyncio.run(_run())
        assert count == 0

    def test_get(self) -> None:
        async def _run() -> MCPClient | None:
            reg = MCPServerRegistry()
            await reg.register("test", "http://localhost:3000")
            return reg.get("test")

        client = asyncio.run(_run())
        assert client is not None
        assert client.is_connected

    def test_get_missing(self) -> None:
        reg = MCPServerRegistry()
        assert reg.get("nonexistent") is None

    def test_close_all(self) -> None:
        async def _run() -> int:
            reg = MCPServerRegistry()
            await reg.register("a", "http://a:3000")
            await reg.register("b", "http://b:3000")
            await reg.close_all()
            return reg.count

        count = asyncio.run(_run())
        assert count == 0


# ══════════════════════════════════════════════════════════════════
# Transport tests
# ══════════════════════════════════════════════════════════════════


class TestSSETransport:
    def test_start_stop(self) -> None:
        transport = SSETransport(port=9999)

        async def _run() -> None:
            await transport.start()
            assert transport._running is True
            await transport.close()
            assert transport._running is False

        asyncio.run(_run())

    def test_receive_not_implemented(self) -> None:
        transport = SSETransport()
        with pytest.raises(NotImplementedError):
            asyncio.run(transport.receive())
