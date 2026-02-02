from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, List

from .mcp_config import McpServerConfig

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except Exception:  # noqa: BLE001
    ClientSession = None  # type: ignore[assignment]
    StdioServerParameters = None  # type: ignore[assignment]
    stdio_client = None  # type: ignore[assignment]


def _require_mcp() -> None:
    if ClientSession is None or StdioServerParameters is None or stdio_client is None:
        raise RuntimeError("Missing dependency: install 'mcp' to enable MCP tools.")


def _run_sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: Dict[str, Any] = {}
    error: Dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001
            error["err"] = exc

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join()
    if error.get("err"):
        raise error["err"]
    return result.get("value")


def _normalize_tool(tool: Any) -> Dict[str, Any]:
    name = getattr(tool, "name", "") or ""
    description = getattr(tool, "description", "") or ""
    input_schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None) or {}
    if not input_schema:
        input_schema = {"type": "object", "properties": {}}
    return {"name": str(name), "description": str(description), "input_schema": input_schema}


def _dump_content_item(item: Any) -> Dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump()  # type: ignore[no-any-return]
    if hasattr(item, "dict"):
        return item.dict()  # type: ignore[no-any-return]
    if hasattr(item, "__dict__"):
        return dict(item.__dict__)
    return {"value": str(item)}


def _normalize_call_result(result: Any) -> Dict[str, Any]:
    is_error = bool(getattr(result, "isError", False) or getattr(result, "is_error", False))
    content = getattr(result, "content", None)
    if content is None and isinstance(result, dict):
        content = result.get("content")

    if content is None:
        content_out: List[Dict[str, Any]] = []
    elif isinstance(content, list):
        content_out = [_dump_content_item(c) for c in content]
    else:
        content_out = [_dump_content_item(content)]

    return {"is_error": is_error, "content": content_out}


async def _with_session(cfg: McpServerConfig, fn: Any) -> Any:
    _require_mcp()
    if cfg.transport != "stdio":
        raise ValueError(f"Unsupported MCP transport: {cfg.transport}")
    if not cfg.command:
        raise ValueError(f"MCP server '{cfg.name}' missing command for stdio transport.")

    params = StdioServerParameters(
        command=str(cfg.command),
        args=list(cfg.args or []),
        env=dict(cfg.env or {}),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await fn(session)


async def _list_tools_async(cfg: McpServerConfig) -> List[Dict[str, Any]]:
    async def _do(session: Any) -> List[Dict[str, Any]]:
        result = await session.list_tools()
        tools = getattr(result, "tools", None) or []
        return [_normalize_tool(t) for t in tools]

    return await _with_session(cfg, _do)


async def _call_tool_async(cfg: McpServerConfig, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    async def _do(session: Any) -> Dict[str, Any]:
        result = await session.call_tool(tool_name, arguments)
        return _normalize_call_result(result)

    return await _with_session(cfg, _do)


def list_mcp_tools(cfg: McpServerConfig) -> List[Dict[str, Any]]:
    return _run_sync(_list_tools_async(cfg))


def call_mcp_tool(cfg: McpServerConfig, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return _run_sync(_call_tool_async(cfg, tool_name, arguments or {}))
