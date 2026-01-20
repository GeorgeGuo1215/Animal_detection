from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .llm_client import OpenAICompatClient, extract_text
from .tool_registry import ToolRegistry


def _safe_json_loads(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Best-effort JSON extraction for LLM outputs.
    Returns (obj, error_message).
    """
    text = (text or "").strip()
    if not text:
        return None, "empty response"
    try:
        return json.loads(text), ""
    except Exception:
        pass

    # try to extract the first JSON object
    l = text.find("{")
    r = text.rfind("}")
    if l >= 0 and r > l:
        snippet = text[l : r + 1]
        try:
            return json.loads(snippet), ""
        except Exception as e:  # noqa: BLE001
            return None, f"json parse failed: {e}"
    return None, "json object not found"


class PlanAndSolveAgent:
    def __init__(self, *, registry: ToolRegistry, llm: OpenAICompatClient) -> None:
        self.registry = registry
        self.llm = llm

    @staticmethod
    def _force_rag_search_defaults(arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Force RAG quality defaults for agent runs.

        Rationale: in practice, planners may omit critical knobs. For this project we want
        multi-route retrieval + reranking + neighbor expansion enabled by default to reduce
        retrieval misses and improve evidence quality.

        This function enforces:
        - multi_route = True
        - rerank = True
        - expand_neighbors >= 1

        It also fills a few safe defaults when missing.
        """
        args = dict(arguments or {})

        # hard-enforce key behaviors
        args["multi_route"] = True
        args["rerank"] = True
        if int(args.get("expand_neighbors") or 0) < 1:
            args["expand_neighbors"] = 1

        # sensible defaults if missing
        args.setdefault("rewrite", "template")
        args.setdefault("top_k", 5)

        # ensure we retrieve enough candidates before rerank
        top_k = int(args.get("top_k") or 5)
        args.setdefault("rerank_candidates", max(10, top_k * 2))
        args.setdefault("rerank_batch_size", 32)
        args.setdefault("rerank_filter_overlap", 0.15)

        return args

    def plan(self, *, query: str, allowed_tools: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        tools = self.registry.list_tools()
        if allowed_tools:
            tools = [t for t in tools if t.name in set(allowed_tools)]

        tool_brief = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools
        ]

        sys = (
            "你是一个严谨的 AI Agent 规划器，采用 Plan-and-Solve。"
            "你必须输出严格 JSON，不要输出任何额外文字。"
            "你的目标是：用尽量少的步骤解决用户问题。"
        )
        user = {
            "query": query,
            "available_tools": tool_brief,
            "output_format": {
                "steps": [
                    {
                        "type": "tool",
                        "tool_name": "rag.search",
                        "arguments": {"query": "..."},
                        "note": "为什么调用这个工具",
                    },
                    {"type": "final", "note": "最后如何组织答案"},
                ]
            },
        }

        resp = self.llm.chat(
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            temperature=0.1,
            max_tokens=512,
            # if supported, nudges JSON output
            response_format={"type": "json_object"},
        )
        text = extract_text(resp)
        obj, err = _safe_json_loads(text)
        if not obj or "steps" not in obj or not isinstance(obj["steps"], list):
            raise RuntimeError(f"Planner output is not valid JSON plan: {err}. raw={text[:800]}")
        return obj["steps"]

    def solve(
        self,
        *,
        query: str,
        plan_steps: List[Dict[str, Any]],
        allowed_tools: Optional[List[str]] = None,
        temperature: float = 0.2,
        max_tokens: int = 768,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        allowed = set(allowed_tools) if allowed_tools else None
        tool_results: List[Dict[str, Any]] = []

        for i, step in enumerate(plan_steps):
            stype = step.get("type")
            if stype == "tool":
                tool_name = str(step.get("tool_name") or "")
                if not tool_name:
                    continue
                if allowed is not None and tool_name not in allowed:
                    raise RuntimeError(f"Tool not allowed: {tool_name}")
                args = step.get("arguments") or {}
                if not isinstance(args, dict):
                    args = {}

                # 打了个补丁，用着先，后面再优化
                if tool_name == "rag.search":
                    args = self._force_rag_search_defaults(args)
                result = self.registry.call(tool_name, args)
                tool_results.append({"step": i, "tool_name": tool_name, "arguments": args, "result": result})
            elif stype == "final":
                break
            else:
                # ignore unknown step types
                continue

        sys = (
            "你是一个面向兽医/动物健康方向的 AI 助手。"
            "请用英语回答，必要时引用你从工具返回的证据片段（简短引用即可）。"
            "如果证据不足，请明确说不确定，并给出下一步建议（例如扩大 top_k 或重建索引）。"
        )
        user = {
            "query": query,
            "plan": plan_steps,
            "tool_results": tool_results,
        }
        resp = self.llm.chat(
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer = extract_text(resp)
        return answer, tool_results

