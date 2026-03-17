from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .llm_client import AsyncOpenAIClient, OpenAICompatClient, extract_text
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
            "你的目标是：用尽量少的步骤解决用户问题。所有的查询使用英语"
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

                if tool_name == "rag.search":
                    args = self._force_rag_search_defaults(args)
                result = self.registry.call_sync(tool_name, args)
                tool_results.append({"step": i, "tool_name": tool_name, "arguments": args, "result": result})
            elif stype == "final":
                break
            else:
                continue

        sys = (
            "你是一个面向兽医/动物健康方向的 AI 助手。"
            "请用中文回答，必要时引用你从工具返回的证据片段（简短引用即可）。"
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


class AsyncPlanAndSolveAgent:
    """Fully async version that uses AsyncOpenAIClient and async registry.call()."""

    def __init__(self, *, registry: ToolRegistry, llm: AsyncOpenAIClient) -> None:
        self.registry = registry
        self.llm = llm

    _force_rag_search_defaults = staticmethod(PlanAndSolveAgent._force_rag_search_defaults)

    async def plan(self, *, query: str, allowed_tools: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        tools = self.registry.list_tools()
        if allowed_tools:
            tools = [t for t in tools if t.name in set(allowed_tools)]

        tool_brief = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools
        ]

        sys = (
            "你是一个严谨的 AI Agent 规划器，采用 Plan-and-Solve。"
            "你必须输出严格 JSON，不要输出任何额外文字。"
            "你的目标是：用尽量少的步骤解决用户问题。\n"
            "工具选择指南：\n"
            "- 健康/医学知识查询 → rag.search (英文query)\n"
            "- 产品价格比较 → mcp.price_watcher.price_compare\n"
            "- 产品成分安全性 → mcp.price_watcher.ingredient_check\n"
            "- 喂食量/热量计算 → mcp.nutritional_planner.calculate_meal_plan\n"
            "- 运动计划 → mcp.nutritional_planner.generate_exercise_plan\n"
            "对于 rag.search 使用英语查询，其他工具按其 input_schema 填写参数。"
        )
        user = {
            "query": query,
            "available_tools": tool_brief,
            "output_format": {
                "steps": [
                    {
                        "type": "tool",
                        "tool_name": "<tool_name>",
                        "arguments": {"<key>": "<value>"},
                        "note": "为什么调用这个工具",
                    },
                    {"type": "final", "note": "最后如何组织答案"},
                ]
            },
        }

        resp = await self.llm.chat(
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            temperature=0.1,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        text = extract_text(resp)
        obj, err = _safe_json_loads(text)
        if not obj or "steps" not in obj or not isinstance(obj["steps"], list):
            raise RuntimeError(f"Planner output is not valid JSON plan: {err}. raw={text[:800]}")
        return obj["steps"]

    async def solve(
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

                if tool_name == "rag.search":
                    args = self._force_rag_search_defaults(args)
                result = await self.registry.call(tool_name, args)
                tool_results.append({"step": i, "tool_name": tool_name, "arguments": args, "result": result})
            elif stype == "final":
                break
            else:
                continue

        sys = (
            "你是一个面向兽医/动物健康方向的 AI 助手。"
            "你拥有以下能力：知识库检索、产品比价与成分分析、营养与运动计划制定。"
            "请用中文回答，必要时引用你从工具返回的证据片段（简短引用即可）。"
            "如果工具返回 INSUFFICIENT_DATA，请明确告知用户需要提供更多信息（如拍摄产品成分表）。"
            "如果工具返回 FEEDING_INQUIRY_NEEDED，请主动询问用户宠物今天是否已经进食。"
            "如果证据不足，请明确说不确定，并给出下一步建议。"
        )
        user = {
            "query": query,
            "plan": plan_steps,
            "tool_results": tool_results,
        }
        resp = await self.llm.chat(
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer = extract_text(resp)
        return answer, tool_results

