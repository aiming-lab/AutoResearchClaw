"""MCP tool definitions for ResearchClaw capabilities."""

from __future__ import annotations

from typing import Any

# Tool schemas exposed by the ResearchClaw MCP Server
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "run_pipeline",
        "description": "Start an autonomous research pipeline run on a given topic.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "The research topic"},
                "config_path": {"type": "string", "description": "Path to config YAML (optional)"},
                "auto_approve": {"type": "boolean", "description": "Auto-approve gate stages"},
            },
            "required": ["topic"],
        },
    },
    {
        "name": "get_pipeline_status",
        "description": "Get the current status of a pipeline run.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "The pipeline run ID"},
            },
            "required": ["run_id"],
        },
    },
    {
        "name": "get_experiment_results",
        "description": "Get experiment results from a completed or running pipeline.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "The pipeline run ID"},
                "stage": {"type": "string", "description": "Specific stage name (optional)"},
            },
            "required": ["run_id"],
        },
    },
    {
        "name": "search_literature",
        "description": "Search academic papers on a topic using Semantic Scholar and arXiv.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Max results (default 10)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "review_paper",
        "description": "Run AI peer review on a paper draft.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "paper_path": {"type": "string", "description": "Path to paper markdown or LaTeX file"},
                "run_id": {"type": "string", "description": "Associated run ID (optional)"},
            },
            "required": ["paper_path"],
        },
    },
    {
        "name": "get_paper",
        "description": "Get the generated paper from a pipeline run.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "The pipeline run ID"},
                "format": {"type": "string", "enum": ["markdown", "latex"], "description": "Output format"},
            },
            "required": ["run_id"],
        },
    },
]


def get_tool_schema(name: str) -> dict[str, Any] | None:
    """Get the schema for a specific tool by name."""
    for tool in TOOL_DEFINITIONS:
        if tool["name"] == name:
            return tool
    return None


def list_tool_names() -> list[str]:
    """Return all available tool names."""
    return [t["name"] for t in TOOL_DEFINITIONS]
