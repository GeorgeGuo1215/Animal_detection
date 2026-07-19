from __future__ import annotations

import json
import os
import re
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
    "1. 先给出「畜牧直接证据」：仅能写入当前工具结果中被牛/猪/羊/马等畜牧资料、病例、官方机构或研究论文直接支持的事实。\n"
    "2. 再给出「一般兽医学推断/补充」：这一部分只能基于通用兽医资料或跨物种经验，必须明确标注为推断、参考或补充，不能写成已被该畜种直接证实的事实。\n"
    "3. 如果本地知识库或检索结果缺乏该畜种直接证据，必须明确说明『当前该畜种直接证据有限』，然后再谨慎提供一般兽医参考。\n"
    "4. 遇到病因、治疗、手术适应证、预后等高风险信息时，优先使用『可能』『常见于』『可作为参考』这类表述，禁止过度确定化。\n"
    "5. 优先使用如下结构（用加粗文字分节，禁止使用 # 标题语法）：\n"
    "   - **畜牧直接证据**\n"
    "   - **一般兽医学推断或补充**\n"
    "   - **早期识别与监测建议**\n"
    "   - **证据不足与注意事项**\n"
)

_VERIFICATION_INSTRUCTION = (
    "\n\n**事实核查/验证类问题回答规则**\n"
    "当用户要求你判断某段陈述是否正确时，必须遵守以下规则：\n"
    "1. **逐条拆解**用户提供的每一个具体声明\n"
    "2. **交叉验证**：综合知识库和网络搜索结果，对每条声明给出「有依据」「部分正确」「缺乏依据」或「错误」的判定\n"
    "3. **证据溯源**：每条判定必须标注来源（知识库 or 网络搜索），不可凭空推断\n"
    "4. **正面回答**：即使知识库信息有限，也必须结合网络搜索结果给出明确判断，不能只说'知识库没有相关信息'\n"
    "5. 优先使用如下结构（用加粗文字分节）：\n"
    "   - **声明拆解与逐条分析**\n"
    "   - **综合判断**\n"
    "   - **补充说明与科学背景**\n"
    "   - **参考来源**\n"
)


def _is_verification_query(query: str) -> bool:
    """Detect fact-checking / verification queries."""
    return bool(re.search(
        r"(正确吗|对吗|真的吗|是真的|是否正确|是否属实|有没有道理|科学吗|准确吗|"
        r"是不是真|可信吗|靠谱吗|验证|核实|辟谣|谣言|误解|误区|真相|"
        r"fact.?check|is.?(?:this|it|that).?(?:true|correct|accurate))",
        query, re.IGNORECASE,
    ))


def _is_timeliness_query(query: str) -> bool:
    return bool(re.search(
        r"(最新|最近|近期|近年|今年|当前|目前|现阶段|recent|latest|current|update|202[3-9]|2030)",
        query,
        re.IGNORECASE,
    ))


def _is_mechanism_query(query: str) -> bool:
    return bool(re.search(
        r"(为什么.*(?:能|会|可以|不会)|原理|机制|怎么做到|如何.*(?:消化|分解|抵抗|解毒)|"
        r"是为了.*吗|是不是为了|是否为了|为了.*(?:吗|么)|适应(?:环境|生态|生存).*(?:吗|么)|"
        r"how.*(?:can|do|does)|why.*(?:can|do|does)|mechanism|reason)",
        query,
        re.IGNORECASE,
    ))


def classify_query(query: str) -> Dict[str, bool]:
    q = query or ""
    return {
        "verification": _is_verification_query(q),
        "timeliness": _is_timeliness_query(q),
        "mechanism": _is_mechanism_query(q),
        "medical": is_medical_query(q),
    }


def harden_plan_steps(
    plan_steps: List[Dict[str, Any]],
    *,
    query: str,
    allowed_tools: Optional[List[str]] = None,
    user_preferences: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Convert prompt-level planning guidance into hard execution rules."""
    prefs = dict(user_preferences or {})
    allowed = set(allowed_tools) if allowed_tools else None
    effective_steps = [dict(step or {}) for step in (plan_steps or []) if isinstance(step, dict)]
    final_step = next((dict(step) for step in effective_steps if step.get("type") == "final"), {"type": "final", "note": "generate answer"})
    tool_steps = [dict(step) for step in effective_steps if step.get("type") == "tool"]

    def _tool_allowed(tool_name: str) -> bool:
        return allowed is None or tool_name in allowed

    def _has_tool(tool_name: str) -> bool:
        return any(str(step.get("tool_name") or "") == tool_name for step in tool_steps)

    if _tool_allowed("rag.search") and not _has_tool("rag.search"):
        tool_steps.insert(0, {"type": "tool", "tool_name": "rag.search", "arguments": {"query": query}, "note": "baseline knowledge-base search"})

    web_policy = str(prefs.get("web_search_policy") or "").strip().lower()
    # 默认先查知识库；仅当用户已确认联网（prefer）时才在计划中保留 web_search
    if web_policy != "prefer":
        tool_steps = [
            step for step in tool_steps
            if str(step.get("tool_name") or "") != "mcp.web_search.web_search"
        ]

    normalized: List[Dict[str, Any]] = []
    for step in tool_steps:
        tool_name = str(step.get("tool_name") or "")
        if not tool_name:
            continue
        args = step.get("arguments") or {}
        if not isinstance(args, dict):
            args = {}
        args.setdefault("query", query)
        normalized.append({**step, "type": "tool", "arguments": args})
    normalized.append(final_step)
    return normalized


def build_solve_prompt(
    user_role: str = "farmer",
    has_web_search: bool = False,
    query: str = "",
    response_lang: str = "auto",
) -> str:
    """Build role-aware system prompt for the solve/generation stage."""
    from .lang_utils import resolve_response_lang

    lang = resolve_response_lang(response_lang, query)
    today = date.today().isoformat()

    if lang == "en":
        if user_role == "veterinarian":
            prompt = (
                f"You are a senior clinical assistant in large-animal veterinary medicine. Today is {today}.\n"
                "Your knowledge covers cattle, pigs, sheep, horses, and related herd management topics:\n"
                "1. Common diseases, diagnosis, and treatment in large livestock\n"
                "2. Herd monitoring, reproduction, and peripartum care\n"
                "3. Nutrition, metabolic disease, and production performance\n"
                "4. Biosecurity, vaccination, and stress management on farms\n\n"
                "Answer requirements:\n"
                "- Respond in clear, professional English with accurate terminology\n"
                "- Structure the answer with **bold section labels**, lists, and short paragraphs\n"
                "- Do NOT use Markdown heading syntax (# ## ###); use bold text for sections\n"
                "- Cite sources when available; do not invent references\n"
                "- If evidence is insufficient, say so explicitly and suggest further reading\n"
                "- Provide actionable diagnostic clues, monitoring metrics, or treatment guidance when possible\n"
            )
        else:
            prompt = (
                f"You are an experienced farm advisor who explains cattle, pig, sheep, and horse topics "
                f"in practical, easy-to-follow English. Today is {today}.\n\n"
                "Answer requirements:\n"
                "- Use plain, actionable English suitable for farm staff\n"
                "- Prioritize on-farm observation, isolation, record-keeping, and when to call a vet\n"
                "- Stay accurate without unnecessary jargon\n"
                "- Use **bold section labels** and lists; avoid Markdown heading syntax (# ## ###)\n"
                "- If evidence is limited, say so and ask for species, age, symptoms, and herd context\n"
            )
        citation = (
            "\n\n**Citation rules**\n"
            "Add source notes at the end of relevant paragraphs. Examples:\n"
            "- Books: (*Book Title*, p. X)\n"
            "- Chinese sources: (《Title》, page X)\n"
            "Infer page numbers from source_path and '--- Page N of M ---' markers in tool results.\n"
            "Summarize all references at the end:\n"
            "> **References**\n"
            "> - *Book Title*, p. X\n"
        )
        web_citation = (
            "\n\n**Web search citation rules**\n"
            "When using web_search results:\n"
            "1. Mark inline citations as [1], [2], etc.\n"
            "2. List all web sources under **References (Web)** with full URLs.\n"
            "Use real URLs from tool results; never invent links.\n"
        )
        web_priority = (
            "\n\n**Tool result priority**\n"
            "You may receive both rag.search (local KB) and web_search results.\n"
            "1. If local KB content directly answers the question, lead with KB and use web as supplement.\n"
            "2. If KB content is off-topic or insufficient, answer primarily from web search.\n"
            "3. Do NOT say 'no relevant KB info' when web results already answer the question.\n"
            "4. Only state limited evidence when both sources are weak.\n"
        )
        medical = (
            "\n\n**Medical question rules**\n"
            "For disease, symptom, diagnosis, treatment, or monitoring questions:\n"
            "1. **Direct livestock evidence**: only facts supported by retrieved cattle/pig/sheep/horse sources.\n"
            "2. **General veterinary inference**: label cross-species or general vet knowledge as inference.\n"
            "3. If direct species evidence is limited, say so before giving cautious general guidance.\n"
            "4. Use hedged language (may, commonly, consider) for high-risk topics.\n"
            "Suggested sections: **Direct evidence**, **General inference**, **Early detection**, **Limitations**.\n"
        )
        verification = (
            "\n\n**Fact-check rules**\n"
            "When the user asks whether a statement is true:\n"
            "1. Break down each claim\n"
            "2. Cross-check KB and web evidence\n"
            "3. Label each claim: supported / partly correct / unsupported / incorrect\n"
            "4. Answer directly even if KB is limited, using web results when needed\n"
        )
    else:
        if user_role == "veterinarian":
            prompt = (
                f"你是一位大型畜牧兽医领域的高级临床助手。今天是 {today}。\n"
                "你的知识来源涵盖专业书籍与学术论文，专注于以下方向：\n"
                "1. 牛、猪、羊、马等家畜与大型畜牧常见疾病诊疗\n"
                "2. 群养监测、繁殖管理与围产期保健\n"
                "3. 饲料营养、代谢病与生产性能\n"
                "4. 牧场生物安全、防疫与应激管理\n\n"
                "回答要求：\n"
                "- 使用学术化中文，术语准确，逻辑严谨\n"
                "- 回答需结构化呈现，用**加粗文字**作为分节标记，配合列表和段落组织内容\n"
                "- 禁止使用 Markdown 标题语法（# ## ### 等），分节一律用加粗文字代替\n"
                "- 引用具体来源（书名、页码），不可编造\n"
                "- 如果证据不足，明确说明当前知识库尚无定论，建议查阅更多文献\n"
                "- 尽可能给出数据、诊断思路、监测指标或处置建议\n"
            )
        else:
            prompt = (
                f"你是一位经验丰富、表达清晰的牧场顾问，擅长用实用易懂的方式回答关于牛、猪、羊、马等大型畜牧的问题。今天是 {today}。\n"
                "你的知识来源涵盖专业书籍与学术论文。\n\n"
                "回答要求：\n"
                "- 用通俗易懂、务实可操作的中文回答\n"
                "- 优先给出牧场现场可执行的观察、隔离、记录与求助建议\n"
                "- 信息要准确，避免过度学术化\n"
                "- 用**加粗文字**分节，配合列表组织内容，不要过度分节\n"
                "- 禁止使用 Markdown 标题语法（# ## ### 等），分节一律用加粗文字代替\n"
                "- 如果证据不足，坦诚告知并引导用户补充畜种、年龄、症状与群养背景\n"
            )
        citation = _CITATION_INSTRUCTION
        web_citation = _WEB_CITATION_INSTRUCTION
        web_priority = (
            "\n\n**使用工具结果的优先级规则**\n"
            "你同时收到了本地知识库（rag.search）和网络搜索（web_search）的结果。\n"
            "**必须遵守以下优先级**：\n"
            "1. 如果本地知识库的结果与用户问题**直接相关**（内容能回答问题），以知识库为主，网络搜索作为补充。\n"
            "2. 如果本地知识库的结果与用户问题**无关或不足**（内容是其他话题），"
            "**必须以网络搜索结果为主**来正面回答用户问题，不要说『知识库没有相关信息』。\n"
            "3. **禁止**在网络搜索结果已能回答问题的情况下，仍然回答『无法找到相关信息』。\n"
            "4. 如果网络搜索结果也不足，才可说明信息有限，但仍需基于已有结果尽力回答。\n"
        )
        medical = _MEDICAL_EVIDENCE_LAYERING_INSTRUCTION
        verification = _VERIFICATION_INSTRUCTION

    prompt += citation

    if has_web_search:
        prompt += web_citation
        prompt += web_priority

    if is_medical_query(query):
        prompt += medical
    if _is_verification_query(query):
        prompt += verification
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
        args.setdefault("rerank_filter_overlap", float(os.getenv("RAG_OVERLAP_THRESHOLD", "0.15")))
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
            "用户使用的是大型畜牧兽医知识服务，不需要在 query 中额外添加'牛''猪'等关键词，知识库已针对该领域。\n"
            "工具选择指南（按优先级）：\n"
            "1. 先用 rag.search 检索本地知识库（同时使用中英文 query 以获得更好的召回效果）\n"
            "2. 若知识库证据不足，系统会先询问用户是否联网；不要默认在计划中直接加入 web_search\n"
            "3. 仅当用户已明确同意联网时，才在计划中包含 mcp.web_search.web_search\n"
            "4. 含心率/呼吸率/体温等时序测量数据 → mcp.vital_signs_analyzer.analyze_vitals\n"
            "注意：仅当用户提供了实际的生理测量数据时才调用 vital_signs_analyzer，普通健康问题不需要它。\n"
            "5. **事实核查、验证类问题**：先 rag.search；证据不足时由系统询问用户是否联网\n"
            "6. **机制/原理类问题**：先 rag.search；需要更全面解释时由系统询问用户是否联网\n"
            "默认计划应只包含 rag.search，然后 final。"
        )
        if is_medical_query(query):
            sys += (
                "\n医疗类问题的额外要求："
                "优先检索疾病、症状、病例、诊断、监测等直接证据；"
                "如果本地知识不足，由系统询问用户是否联网补充官方机构、学术论文或权威兽医资料；"
                "不要把通用兽医经验当作该畜种已证实事实。"
            )
        user = {
            "query": query,
            "available_tools": tool_brief,
            "output_format": {
                "steps": [
                    {"type": "tool", "tool_name": "rag.search", "arguments": {"query": "..."}, "note": "why"},
                    {"type": "ask_user", "question": "clarifying question", "reason": "why clarification is needed"},
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
              temperature: float = 0.2, max_tokens: int = 768, user_role: str = "farmer") -> Tuple[str, List[Dict[str, Any]]]:
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
            elif stype == "ask_user":
                question = str(step.get("question") or step.get("note") or "").strip()
                if question:
                    tool_results.append({"step": i, "type": "ask_user", "question": question, "reason": step.get("reason", "")})
                    return question, tool_results
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
            "用户使用的是大型畜牧兽医知识服务，不需要在 query 中额外添加'牛''猪'等关键词，知识库已针对该领域。\n"
            "工具选择指南（按优先级）：\n"
            "1. 先用 rag.search 检索本地知识库（同时使用中英文 query 以获得更好的召回效果）\n"
            "2. 若知识库证据不足，系统会先询问用户是否联网；不要默认在计划中直接加入 web_search\n"
            "3. 仅当用户已明确同意联网时，才在计划中包含 mcp.web_search.web_search\n"
            "4. 含心率/呼吸率/体温等时序测量数据 → mcp.vital_signs_analyzer.analyze_vitals\n"
            "注意：仅当用户提供了实际的生理测量数据时才调用 vital_signs_analyzer，普通健康问题不需要它。\n"
            "5. **事实核查、验证类问题**：先 rag.search；证据不足时由系统询问用户是否联网\n"
            "6. **机制/原理类问题**：先 rag.search；需要更全面解释时由系统询问用户是否联网\n"
            "默认计划应只包含 rag.search，然后 final。"
        )
        if is_medical_query(query):
            sys += (
                "\n医疗类问题的额外要求："
                "优先检索疾病、症状、病例、诊断、监测等直接证据；"
                "如果本地知识不足，由系统询问用户是否联网补充官方机构、学术论文或权威兽医资料；"
                "不要把通用兽医经验当作该畜种已证实事实。"
            )
        user = {
            "query": query,
            "available_tools": tool_brief,
            "output_format": {
                "steps": [
                    {"type": "tool", "tool_name": "<tool_name>", "arguments": {"<key>": "<value>"}, "note": "why"},
                    {"type": "ask_user", "question": "clarifying question", "reason": "why clarification is needed"},
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
                    temperature: float = 0.2, max_tokens: int = 768, user_role: str = "farmer") -> Tuple[str, List[Dict[str, Any]]]:
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
            elif stype == "ask_user":
                question = str(step.get("question") or step.get("note") or "").strip()
                if question:
                    tool_results.append({"step": i, "type": "ask_user", "question": question, "reason": step.get("reason", "")})
                    return question, tool_results
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
