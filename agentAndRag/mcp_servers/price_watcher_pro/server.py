"""
Price_Watcher_Pro MCP Server.

Exposes two tools:
  - price_compare: Cross-platform price comparison
  - ingredient_check: Ingredient safety analysis against health conditions
"""
from __future__ import annotations

import json
import logging

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .ingredient_checker import check_ingredients
from .price_search import compare_prices

logger = logging.getLogger(__name__)

server = Server("price_watcher_pro")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="price_compare",
            description=(
                "Compare prices of a pet product across multiple e-commerce platforms. "
                "Uses web search to find current prices and provides a budget-aware recommendation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "Full product name, e.g. 'Royal Canin Urinary SO Dry Dog Food 8.8lb'",
                    },
                    "compare_platforms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Platforms to compare, e.g. ['Amazon', 'Chewy', 'JD']",
                        "default": ["Amazon", "Chewy", "JD"],
                    },
                    "user_budget_preference": {
                        "type": "string",
                        "enum": ["budget", "mid-range", "premium"],
                        "description": "User's budget preference",
                        "default": "mid-range",
                    },
                },
                "required": ["product_name"],
            },
        ),
        Tool(
            name="ingredient_check",
            description=(
                "Analyze a pet product's ingredients against the pet's health conditions. "
                "Cross-references ingredient data with a contraindications database. "
                "Returns INSUFFICIENT_DATA if ingredient info cannot be found, "
                "prompting the user to provide the ingredient list manually."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "Full product name",
                    },
                    "current_health_context": {
                        "type": "string",
                        "description": "Pet's current health conditions, e.g. 'Chronic Kidney Disease Stage 1'",
                    },
                },
                "required": ["product_name", "current_health_context"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "price_compare":
        result = compare_prices(
            product_name=arguments["product_name"],
            compare_platforms=arguments.get("compare_platforms"),
            user_budget_preference=arguments.get("user_budget_preference", "mid-range"),
        )
    elif name == "ingredient_check":
        result = check_ingredients(
            product_name=arguments["product_name"],
            current_health_context=arguments["current_health_context"],
        )
    else:
        result = {"error": f"Unknown tool: {name}"}

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]


async def run_server():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
