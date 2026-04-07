from __future__ import annotations

import os
from typing import Any, Dict

from .rag_tools import rag_reindex_tool, rag_search_tool
from .tool_registry import ToolSpec, ToolRegistry

_DEFAULT_DEVICE = os.getenv("AGENT_WARMUP_DEVICE") or None


def register_builtin_tools(registry: ToolRegistry) -> None:
    registry.register(
        ToolSpec(
            name="rag.search",
            description="Search the Giant Panda knowledge base (hybrid/multiroute optional) and return hits/contexts.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50},
                    "index_dir": {"type": ["string", "null"], "default": None},
                    "embedding_model": {"type": "string", "default": "intfloat/multilingual-e5-small"},
                    "device": {"type": ["string", "null"], "default": _DEFAULT_DEVICE,
                               "description": "Inference device (auto-set from AGENT_WARMUP_DEVICE)"},
                    "multi_route": {"type": "boolean", "default": False},
                    "rewrite": {"type": "string", "enum": ["none", "template", "llm"], "default": "template"},
                    "rerank": {"type": "boolean", "default": False},
                    "expand_neighbors": {"type": "integer", "default": 1, "minimum": 0, "maximum": 5},
                },
                "required": ["query"],
            },
            handler=lambda **kwargs: rag_search_tool(**kwargs),
        )
    )
    registry.register(
        ToolSpec(
            name="rag.reindex",
            description="(Re)build the vector index from RAG/data/raw.",
            input_schema={
                "type": "object",
                "properties": {
                    "raw_dir": {"type": ["string", "null"], "default": None},
                    "index_dir": {"type": ["string", "null"], "default": None},
                    "embedding_model": {"type": "string", "default": "intfloat/multilingual-e5-small"},
                    "batch_size": {"type": "integer", "default": 32},
                    "device": {"type": ["string", "null"], "default": _DEFAULT_DEVICE,
                               "description": "Inference device (auto-set from AGENT_WARMUP_DEVICE)"},
                },
                "required": [],
            },
            handler=lambda **kwargs: rag_reindex_tool(**kwargs),
        )
    )


def _debug_echo_tool(**kwargs: Any) -> Dict[str, Any]:
    return {"echo": kwargs}


def register_debug_tools(registry: ToolRegistry) -> None:
    registry.register(
        ToolSpec(
            name="debug.echo",
            description="Echo back the arguments (for debugging).",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=_debug_echo_tool,
        )
    )
