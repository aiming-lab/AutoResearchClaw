"""ResearchClaw MCP Server: expose pipeline capabilities to external agents."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from researchclaw.mcp.tools import TOOL_DEFINITIONS, list_tool_names

logger = logging.getLogger(__name__)


class ResearchClawMCPServer:
    """MCP Server that exposes AutoResearchClaw capabilities as tools.

    External agents (e.g., Claude, OpenClaw) can connect to this server
    and invoke pipeline operations via the MCP protocol.
    """

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self._handlers: dict[str, Any] = {}
        self._running = False

    def get_tools(self) -> list[dict[str, Any]]:
        """Return the list of available MCP tools."""
        return TOOL_DEFINITIONS

    async def handle_tool_call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle an incoming MCP tool call."""
        if name not in list_tool_names():
            return {"error": f"Unknown tool: {name}", "success": False}

        logger.info("MCP tool call: %s(%s)", name, json.dumps(arguments, default=str)[:200])

        try:
            if name == "run_pipeline":
                return await self._handle_run_pipeline(arguments)
            elif name == "get_pipeline_status":
                return await self._handle_get_status(arguments)
            elif name == "get_experiment_results":
                return await self._handle_get_results(arguments)
            elif name == "search_literature":
                return await self._handle_search_literature(arguments)
            elif name == "review_paper":
                return await self._handle_review_paper(arguments)
            elif name == "get_paper":
                return await self._handle_get_paper(arguments)
            else:
                return {"error": f"Handler not implemented: {name}", "success": False}
        except Exception as exc:
            logger.error("MCP tool call %s failed: %s", name, exc)
            return {"error": str(exc), "success": False}

    async def _handle_run_pipeline(self, args: dict[str, Any]) -> dict[str, Any]:
        """Start a pipeline run."""
        topic = args["topic"]
        # In production, this would invoke the full pipeline asynchronously
        return {
            "success": True,
            "message": f"Pipeline started for topic: {topic}",
            "run_id": f"mcp-stub-{topic[:20]}",
        }

    async def _handle_get_status(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get pipeline status."""
        run_id = args["run_id"]
        run_dir = Path(f"artifacts/{run_id}")
        if not run_dir.exists():
            return {"success": False, "error": f"Run not found: {run_id}"}
        # Read checkpoint if available
        checkpoint_file = run_dir / "checkpoint.json"
        if checkpoint_file.exists():
            data = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            return {"success": True, "run_id": run_id, "checkpoint": data}
        return {"success": True, "run_id": run_id, "status": "no_checkpoint"}

    async def _handle_get_results(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get experiment results."""
        run_id = args["run_id"]
        run_dir = Path(f"artifacts/{run_id}")
        results_file = run_dir / "experiment_results.json"
        if results_file.exists():
            data = json.loads(results_file.read_text(encoding="utf-8"))
            return {"success": True, "results": data}
        return {"success": False, "error": "No results found"}

    async def _handle_search_literature(self, args: dict[str, Any]) -> dict[str, Any]:
        """Search literature (stub — real implementation would use literature/ module)."""
        return {
            "success": True,
            "query": args["query"],
            "results": [],
            "message": "Literature search stub",
        }

    async def _handle_review_paper(self, args: dict[str, Any]) -> dict[str, Any]:
        """Review paper (stub)."""
        return {
            "success": True,
            "paper_path": args["paper_path"],
            "review": "Stub review — not yet implemented",
        }

    async def _handle_get_paper(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get generated paper."""
        run_id = args["run_id"]
        fmt = args.get("format", "markdown")
        run_dir = Path(f"artifacts/{run_id}")
        if fmt == "latex":
            paper_file = run_dir / "paper.tex"
        else:
            paper_file = run_dir / "paper_draft.md"
        if paper_file.exists():
            return {"success": True, "content": paper_file.read_text(encoding="utf-8")}
        return {"success": False, "error": f"Paper not found in {run_dir}"}

    # ── server lifecycle ──────────────────────────────────────────

    async def start(self, transport: str = "stdio") -> None:
        """Start the MCP server (stdio or SSE transport)."""
        self._running = True
        logger.info("MCP server started (transport=%s)", transport)

    async def stop(self) -> None:
        """Stop the MCP server."""
        self._running = False
        logger.info("MCP server stopped")

    @property
    def is_running(self) -> bool:
        return self._running
