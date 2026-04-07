from __future__ import annotations

import json
import os
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from RAG.simple_rag.query_rewrite import is_medical_query

from .llm_client import AsyncOpenAIClient, OpenAICompatClient, extract_text
from .tool_registry import ToolRegistry


_CITATION_INSTRUCTION = (
    "\n\n**引用规范**\n"
    "回答时，请在相关段落末尾标注来源。格式示例：\n"
    "- 中文书籍：（《书名》第 X 页）\n"
    "- 英文书籍：(*Book Title*, p. X)\n"
    "来源信息可从工具返回的 source_path 和文本中的 '--- Page N of M ---' 标记推断。\n"
    "在回答末尾汇总所有引用，使用如下格式：\n"
    "> **参考来源**\n"
    "> - 《书名》第 X 页\n"
    "> - *English Book Title*, p. X\n"
)

_WEB_CITATION_INSTRUCTION = (
    "\n\n**网络搜索结果引用规范**\n"
    "当回答中使用了 web_search 工具返回的信息时：\n"
    "1. 在引用处标注来源编号，如 [1]、[2]\n"
    "2. 在回答末尾的「参考来源」区列出所有网络来源，保留完整 URL，格式如下：\n"
    "> **参考来源（网络）**\n"
    "> 1. [文章标题](https://example.com/url)\n"
    "> 2. [文章标题](https://example.com/url)\n\n"
    "来源的 title 和 url 可从 web_search 工具返回的 results 中提取。\n"
    "务必保留真实 URL，不可编造链接。如果工具未返回 URL，则只标注标题。\n"
)

_MEDICAL_EVIDENCE_LAYERING_INSTRUCTION = (
    "\n\n**医疗类问题回答规则**\n"
    "当用户询问疾病、症状、诊断、治疗、病例、监测、预警或其他医疗主题时，必须执行证据分层：\n"
    "1. 先给出「大熊猫直接证据」：仅能写入当前工具结果中被大熊猫资料、病例、官方机构或研究论文直接支持的事实。\n"
    "2. 再给出「一般兽医学推断/补充」：这一部分只能基于通用兽医资料或跨物种经验，必须明确标注为推断、参考或补充，不能写成已被大熊猫直接证实的事实。\n"
    "3. 如果本地知识库或检索结果缺乏大熊猫直接证据，必须明确说明『当前大熊猫直接证据有限』，然后再谨慎提供一般兽医参考。\n"
    "4. 遇到病因、治疗、手术适应证、预后等高风险信息时，优先使用『可能』『常见于』『可作为参考』这类表述，禁止过度确定化。\n"
    "5. 优先使用如下结构（用加粗文字分节，禁止使用 # 标题语法）：\n"
    "   - **大熊猫直接证据**\n"
    "   - **一般兽医学推断或补充**\n"
    "   - **早期识别与监测建议**\n"
    "   - **证据不足与注意事项**\n"
)


def build_solve_prompt(
    user_role: str = "enthusiast",
    has_web_search: bool = False,
    query: str = "",
) -> str:
    """Build role-aware system prompt for the solve/generation stage."""
    today = date.today().isoformat()

    if user_role == "researcher":
        prompt = (
            f"你是一位濒危野生动物研究领域的高级科研助手。今天是 {today}。\n"
            "你的知识来源涵盖专业书籍与学术论文，专注于以下四个方向：\n"
            "1. 濒危野生动物保护繁殖\n"
            "2. 濒危野生动物遗传学\n"
            "3. 濒危野生动物疾病预防\n"
            "4. 濒危野生动物生态学\n\n"
            "回答要求：\n"
            "- 使用学术化中文，术语准确，逻辑严谨\n"
            "- 回答需结构化呈现，用**加粗文字**作为分节标记，配合列表和段落组织内容\n"
            "- 禁止使用 Markdown 标题语法（# ## ### 等），分节一律用加粗文字代替\n"
            "- 引用具体来源（书名、页码），不可编造\n"
            "- 如果证据不足，明确说明当前知识库尚无定论，建议查阅更多文献\n"
            "- 尽可能给出数据、实验方法或分类学信息\n"
        )
    else:
        prompt = (
            f"你是一位热情的科普讲解员，擅长用生动有趣的方式回答关于大熊猫和濒危野生动物的问题。今天是 {today}。\n"
            "你的知识来源涵盖专业书籍与学术论文。\n\n"
            "回答要求：\n"
            "- 用通俗易懂、活泼有趣的中文回答\n"
            "- 适当使用比喻、趣闻和冷知识让内容更有吸引力\n"
            "- 不需要过于学术化，但信息要准确\n"
            "- 用**加粗文字**分节，配合列表组织内容，不要过度分节\n"
            "- 禁止使用 Markdown 标题语法（# ## ### 等），分节一律用加粗文字代替\n"
            "- 如果证据不足，坦诚告知并引导用户换个角度提问\n"
        )

    prompt += _CITATION_INSTRUCTION

    if has_web_search:
        prompt += _WEB_CITATION_INSTRUCTION
        prompt += (
            "\n\n**使用工具结果的优先级规则**\n"
            "你同时收到了本地知识库（rag.search）和网络搜索（web_search）的结果。\n"
            "**必须遵守以下优先级**：\n"
            "1. 如果本地知识库的结果与用户问题**直接相关**（内容能回答问题），以知识库为主，网络搜索作为补充。\n"
            "2. 如果本地知识库的结果与用户问题**无关或不足**（内容是其他话题），"
            "**必须以网络搜索结果为主**来正面回答用户问题，不要说『知识库没有相关信息』。\n"
            "3. **禁止**在网络搜索结果已能回答问题的情况下，仍然回答『无法找到相关信息』。\n"
            "4. 如果网络搜索结果也不足，才可说明信息有限，但仍需基于已有结果尽力回答。\n"
        )

    if is_medical_query(query):
        prompt += _MEDICAL_EVIDENCE_LAYERING_INSTRUCTION
    return prompt


def _safe_json_loads(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    text = (text or "").strip()
    if not text:
        return None, "empty response"
    try:
        return json.loads(text), ""
    except Exception:
        pass
    l = text.find("{")
    r = text.rfind("}")
    if l >= 0 and r > l:
        snippet = text[l : r + 1]
        try:
            return json.loads(snippet), ""
        except Exception as e:
            return None, f"json parse failed: {e}"
    return None, "json object not found"


class PlanAndSolveAgent:
    def __init__(self, *, registry: ToolRegistry, llm: OpenAICompatClient,
                 solve_llm: Optional[OpenAICompatClient] = None) -> None:
        self.registry = registry
        self.llm = llm
        self.solve_llm = solve_llm or llm

    @staticmethod
    def _force_rag_search_defaults(arguments: Dict[str, Any]) -> Dict[str, Any]:
        args = dict(arguments or {})
        args["multi_route"] = True
        args["rerank"] = True
        if int(args.get("expand_neighbors") or 0) < 1:
            args["expand_neighbors"] = 1
        args.setdefault("rewrite", "template")
        args.setdefault("top_k", 5)
        args.setdefault("device", os.getenv("AGENT_WARMUP_DEVICE") or None)
        top_k = int(args.get("top_k") or 5)
        args.setdefault("rerank_candidates", max(10, top_k * 2))
        args.setdefault("rerank_batch_size", 32)
        args.setdefault("rerank_filter_overlap", 0.15)
        return args

    def plan(self, *, query: str, allowed_tools: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        tools = self.registry.list_tools()
        if allowed_tools:
            tools = [t for t in tools if t.name in set(allowed_tools)]
        tool_brief = [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools]
        sys = (
            f"你是一个严谨的 AI Agent 规划器，采用 Plan-and-Solve。今天是 {date.today().isoformat()}。"
            "你必须输出严格 JSON，不要输出任何额外文字。"
            "你的目标是：用尽量少的步骤解决用户的问题。\n"
            "用户使用的是大熊猫知识服务，不需要在 query 中额外添加'大熊猫'等关键词，知识库已针对该领域。\n"
            "工具选择指南（按优先级）：\n"
            "1. 先用 rag.search 检索本地知识库（同时使用中英文 query 以获得更好的召回效果）\n"
            "2. 如果问题涉及最新动态、实时数据、新闻、知识库可能未覆盖的内容，"
            "同时使用 mcp.web_search.web_search 进行网络搜索\n"
            "3. 对于明确的时效性问题（如'种群数量最新数据'、'最新研究进展'），"
            "必须使用 mcp.web_search.web_search\n"
            "建议：对于不确定知识库是否覆盖的问题，同时规划 rag.search 和 web_search 两个步骤。"
        )
        if is_medical_query(query):
            sys += (
                "\n医疗类问题的额外要求："
                "优先检索疾病、症状、病例、诊断、监测等直接证据；"
                "如果本地知识不足，优先补充官方机构、学术论文或权威兽医资料；"
                "不要把通用兽医经验当作大熊猫已证实事实。"
            )
        user = {
            "query": query,
            "available_tools": tool_brief,
            "output_format": {
                "steps": [
                    {"type": "tool", "tool_name": "rag.search", "arguments": {"query": "..."}, "note": "why"},
                    {"type": "final", "note": "how to compose the answer"},
                ]
            },
        }
        resp = self.llm.chat(
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
            temperature=0.1, max_tokens=512, response_format={"type": "json_object"},
        )
        text = extract_text(resp)
        obj, err = _safe_json_loads(text)
        if not obj or "steps" not in obj or not isinstance(obj["steps"], list):
            raise RuntimeError(f"Planner output is not valid JSON plan: {err}. raw={text[:800]}")
        return obj["steps"]

    def solve(self, *, query: str, plan_steps: List[Dict[str, Any]], allowed_tools: Optional[List[str]] = None,
              temperature: float = 0.2, max_tokens: int = 768, user_role: str = "enthusiast") -> Tuple[str, List[Dict[str, Any]]]:
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
        has_web_search = any(r.get("tool_name", "").startswith("mcp.web_search") for r in tool_results)
        sys = build_solve_prompt(user_role=user_role, has_web_search=has_web_search, query=query)
        user = {"query": query, "plan": plan_steps, "tool_results": tool_results}
        resp = self.solve_llm.chat(
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
            temperature=temperature, max_tokens=max_tokens,
        )
        return extract_text(resp), tool_results


class AsyncPlanAndSolveAgent:
    def __init__(self, *, registry: ToolRegistry, llm: AsyncOpenAIClient,
                 solve_llm: Optional[AsyncOpenAIClient] = None) -> None:
        self.registry = registry
        self.llm = llm
        self.solve_llm = solve_llm or llm

    _force_rag_search_defaults = staticmethod(PlanAndSolveAgent._force_rag_search_defaults)

    async def plan(self, *, query: str, allowed_tools: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        tools = self.registry.list_tools()
        if allowed_tools:
            tools = [t for t in tools if t.name in set(allowed_tools)]
        tool_brief = [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools]
        sys = (
            f"你是一个严谨的 AI Agent 规划器，采用 Plan-and-Solve。今天是 {date.today().isoformat()}。"
            "你必须输出严格 JSON，不要输出任何额外文字。"
            "你的目标是：用尽量少的步骤解决用户的问题。\n"
            "用户使用的是大熊猫知识服务，不需要在 query 中额外添加'大熊猫'等关键词，知识库已针对该领域。\n"
            "工具选择指南（按优先级）：\n"
            "1. 先用 rag.search 检索本地知识库（同时使用中英文 query 以获得更好的召回效果）\n"
            "2. 如果问题涉及最新动态、实时数据、新闻、知识库可能未覆盖的内容，"
            "同时使用 mcp.web_search.web_search 进行网络搜索\n"
            "3. 对于明确的时效性问题（如'种群数量最新数据'、'最新研究进展'），"
            "必须使用 mcp.web_search.web_search\n"
            "建议：对于不确定知识库是否覆盖的问题，同时规划 rag.search 和 web_search 两个步骤。"
        )
        if is_medical_query(query):
            sys += (
                "\n医疗类问题的额外要求："
                "优先检索疾病、症状、病例、诊断、监测等直接证据；"
                "如果本地知识不足，优先补充官方机构、学术论文或权威兽医资料；"
                "不要把通用兽医经验当作大熊猫已证实事实。"
            )
        user = {
            "query": query,
            "available_tools": tool_brief,
            "output_format": {
                "steps": [
                    {"type": "tool", "tool_name": "<tool_name>", "arguments": {"<key>": "<value>"}, "note": "why"},
                    {"type": "final", "note": "how to compose the answer"},
                ]
            },
        }
        resp = await self.llm.chat(
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
            temperature=0.1, max_tokens=512, response_format={"type": "json_object"},
        )
        text = extract_text(resp)
        obj, err = _safe_json_loads(text)
        if not obj or "steps" not in obj or not isinstance(obj["steps"], list):
            raise RuntimeError(f"Planner output is not valid JSON plan: {err}. raw={text[:800]}")
        return obj["steps"]

    async def solve(self, *, query: str, plan_steps: List[Dict[str, Any]], allowed_tools: Optional[List[str]] = None,
                    temperature: float = 0.2, max_tokens: int = 768, user_role: str = "enthusiast") -> Tuple[str, List[Dict[str, Any]]]:
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

        has_web_search = any(r.get("tool_name", "").startswith("mcp.web_search") for r in tool_results)
        sys = build_solve_prompt(user_role=user_role, has_web_search=has_web_search, query=query)
        user = {"query": query, "plan": plan_steps, "tool_results": tool_results}
        resp = await self.solve_llm.chat(
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
            temperature=temperature, max_tokens=max_tokens,
        )
        return extract_text(resp), tool_results
