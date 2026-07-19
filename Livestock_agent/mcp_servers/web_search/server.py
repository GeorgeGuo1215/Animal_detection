"""
Web Search MCP Server for Panda Mind.
Exposes one tool: web_search for real-time panda-related web searches.
"""
from __future__ import annotations

import json
import logging

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .tavily_client import tavily_search

logger = logging.getLogger(__name__)

server = Server(name="web_search")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="web_search",
            description=(
                "Perform a real-time web search using Tavily and return a list of relevant results "
                "(title, URL, content snippet). Useful for up-to-date information: "
                "news, conservation updates, research papers, population data, etc."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, e.g. 'panda population 2026'",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (1-20). Default: 5.",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20,
                    },
                    "search_depth": {
                        "type": "string",
                        "enum": ["basic", "advanced"],
                        "description": "Search depth: 'basic' is faster, 'advanced' yields richer results.",
                        "default": "basic",
                    },
                },
                "required": ["query"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "web_search":
        hits = await tavily_search(
            query=arguments["query"],
            max_results=int(arguments.get("max_results", 5)),
            search_depth=str(arguments.get("search_depth", "basic")),
        )
        result = {"status": "OK", "query": arguments["query"], "results": hits, "count": len(hits)}
    else:
        result = {"error": f"Unknown tool: {name}"}

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]


async def run_server():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
