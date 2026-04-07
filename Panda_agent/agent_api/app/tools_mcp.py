from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from .mcp_client import call_mcp_tool, call_mcp_tool_async, list_mcp_tools
from .mcp_config import McpServerConfig, load_mcp_servers
from .tool_registry import ToolRegistry, ToolSpec

logger = logging.getLogger(__name__)


async def _direct_web_search_handler(**kwargs: Any) -> Dict[str, Any]:
    """Call Tavily REST API directly, bypassing MCP subprocess."""
    from mcp_servers.web_search.tavily_client import tavily_search

    query = kwargs.get("query", "")
    max_results = int(kwargs.get("max_results", 5))
    search_depth = str(kwargs.get("search_depth", "basic"))
    hits = await tavily_search(query=query, max_results=max_results, search_depth=search_depth)
    return {"status": "OK", "query": query, "results": hits, "count": len(hits)}


def _make_async_handler(server: McpServerConfig, tool_name: str):
    """Return an async handler that calls the MCP tool natively."""

    async def _handler(**kwargs: Any) -> Dict[str, Any]:
        return await call_mcp_tool_async(server, tool_name, kwargs)

    return _handler


def register_mcp_tools(registry: ToolRegistry) -> Dict[str, Any]:
    """
    Load MCP servers from config and register each MCP tool into ToolRegistry.
    For web_search, uses direct REST API call instead of MCP subprocess.
    """
    servers = load_mcp_servers()
    summary: Dict[str, Any] = {"servers": [], "tools": 0}

    for server in servers:
        if server.name == "web_search" and os.getenv("TAVILY_API_KEY", "").strip():
            name = "mcp.web_search.web_search"
            if registry.get(name) is None:
                registry.register(
                    ToolSpec(
                        name=name,
                        description=(
                            "Perform a real-time web search using Tavily and return "
                            "relevant results (title, URL, content snippet)."
                        ),
                        input_schema={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "max_results": {"type": "integer", "default": 5},
                                "search_depth": {"type": "string", "enum": ["basic", "advanced"], "default": "basic"},
                            },
                            "required": ["query"],
                        },
                        handler=_direct_web_search_handler,
                    )
                )
                summary["tools"] += 1
                logger.info("Registered web_search via direct Tavily REST API (fast path)")
            summary["servers"].append(server.name)
            continue

        try:
            tools = list_mcp_tools(server)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MCP server %s load failed: %s", server.name, exc)
            continue

        for t in tools:
            tool_name = str(t.get("name") or "").strip()
            if not tool_name:
                continue
            name = f"mcp.{server.name}.{tool_name}"
            if registry.get(name) is not None:
                continue

            description = t.get("description") or ""
            input_schema = t.get("input_schema") or {"type": "object", "properties": {}}
            registry.register(
                ToolSpec(
                    name=name,
                    description=f"[mcp:{server.name}] {description}".strip(),
                    input_schema=input_schema,
                    handler=_make_async_handler(server, tool_name),
                )
            )
            summary["tools"] += 1

        summary["servers"].append(server.name)

    return summary
