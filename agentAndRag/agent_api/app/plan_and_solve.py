from __future__ import annotations

import json
import os
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

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


_VET_EVIDENCE_LAYERING_INSTRUCTION = (
    "\n\n**兽医证据分层规范**\n"
    "当回答涉及疾病、症状、诊断、治疗、用药、手术或病例时，必须执行证据分层：\n"
    "1. 先给出「循证兽医学直接证据」：仅写入已被临床研究、教科书、权威指南或病例报告直接支持的事实。\n"
    "2. 再给出「临床经验与推断」：这一部分只能基于临床经验或跨物种推断，必须明确标注为推断或参考。\n"
    "3. 如果检索结果缺乏直接证据，必须明确说明『当前直接证据有限』，然后再谨慎提供临床参考。\n"
    "4. 遇到病因、治疗方案、药物剂量、手术适应证、预后等高风险信息时，优先使用『可能』『常见于』"
    "『建议结合临床评估』等表述，禁止过度确定化。\n"
    "5. 优先使用如下结构（用加粗文字分节）：\n"
    "   - **临床直接证据**\n"
    "   - **临床经验与推断**\n"
    "   - **鉴别诊断与进一步检查建议**\n"
    "   - **证据不足与注意事项**\n"
)


def _is_medical_query(query: str) -> bool:
    """Simple heuristic to detect medical/clinical queries for pets."""
    import re
    medical_keywords = re.compile(
        r"(疾病|症状|诊断|治疗|手术|用药|药物|剂量|病例|感染|炎症|肿瘤|癌|骨折|"
        r"呕吐|腹泻|发烧|咳嗽|抽搐|中毒|过敏|寄生虫|疫苗|免疫|绝育|麻醉|"
        r"disease|symptom|diagnos|treatment|surgery|medication|dose|infection|"
        r"tumor|cancer|fracture|vomit|diarrhea|fever|seizure|poison|parasite|vaccine)",
        re.IGNORECASE,
    )
    return bool(medical_keywords.search(query))


def build_solve_prompt(
    user_role: str = "pet_owner",
    has_web_search: bool = False,
    query: str = "",
) -> str:
    """Build role-aware system prompt for the solve/generation stage."""
    today = date.today().isoformat()

    if user_role == "veterinarian":
        prompt = (
            f"你是一位经验丰富的兽医临床顾问和学术助手。今天是 {today}。\n"
            "你拥有以下能力：知识库检索、网络搜索与成分分析、营养与运动计划制定。\n\n"
            "回答要求：\n"
            "- 使用专业兽医学术中文，术语准确，逻辑严谨\n"
            "- 回答需结构化呈现，用**加粗文字**作为分节标记，配合列表和段落组织内容\n"
            "- 禁止使用 Markdown 标题语法（# ## ### 等），分节一律用加粗文字代替\n"
            "- 引用具体来源（书名、页码、文献），不可编造\n"
            "- 如果证据不足，明确说明当前知识库尚无定论，建议查阅更多文献或结合临床评估\n"
            "- 尽可能给出药物剂量范围、实验室指标参考值、鉴别诊断等专业信息\n"
            "- 如果工具返回 INSUFFICIENT_DATA，请明确告知需补充更多检查结果或病史\n"
            "- 如果工具返回 FEEDING_INQUIRY_NEEDED，请主动询问近期饮食与给药情况\n"
        )
    else:
        prompt = (
            f"你是一位热情的宠物健康顾问，擅长用生动有趣的方式回答关于宠物养护和健康的问题。今天是 {today}。\n"
            "你拥有以下能力：知识库检索、网络搜索与成分分析、营养与运动计划制定。\n\n"
            "回答要求：\n"
            "- 用通俗易懂、活泼亲切的中文回答\n"
            "- 适当使用比喻、趣闻和实用小贴士让内容更有吸引力\n"
            "- 不需要过于学术化，但信息要准确\n"
            "- 用**加粗文字**分节，配合列表组织内容，不要过度分节\n"
            "- 禁止使用 Markdown 标题语法（# ## ### 等），分节一律用加粗文字代替\n"
            "- 如果证据不足，坦诚告知并建议咨询兽医\n"
            "- 如果工具返回 INSUFFICIENT_DATA，请用友好的方式引导用户提供更多信息（如拍照产品成分表）\n"
            "- 如果工具返回 FEEDING_INQUIRY_NEEDED，请用轻松的方式询问宠物今天是否已经进食\n"
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

    if user_role == "veterinarian" and _is_medical_query(query):
        prompt += _VET_EVIDENCE_LAYERING_INSTRUCTION

    return prompt


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
        args.setdefault("device", os.getenv("AGENT_WARMUP_DEVICE") or None)

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
            f"你是一个严谨的 AI Agent 规划器，采用 Plan-and-Solve。今天是 {date.today().isoformat()}。"
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
        user_role: str = "pet_owner",
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

        has_web = any(r.get("tool_name", "").startswith("mcp.web_search") for r in tool_results)
        sys = build_solve_prompt(user_role=user_role, has_web_search=has_web, query=query)
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
            f"你是一个严谨的 AI Agent 规划器，采用 Plan-and-Solve。今天是 {date.today().isoformat()}。"
            "你必须输出严格 JSON，不要输出任何额外文字。"
            "你的目标是：用尽量少的步骤解决用户问题。\n"
            "工具选择指南：\n"
            "- 健康/医学知识查询 → rag.search (英文 query)\n"
            "- 实时网络信息 / 产品信息 / 价格线索 → mcp.web_search.web_search\n"
            "- 产品成分安全性 → mcp.web_search.ingredient_check\n"
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
        user_role: str = "pet_owner",
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

        has_web = any(r.get("tool_name", "").startswith("mcp.web_search") for r in tool_results)
        sys = build_solve_prompt(user_role=user_role, has_web_search=has_web, query=query)
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

