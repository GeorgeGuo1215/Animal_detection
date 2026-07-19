from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from RAG.simple_rag.query_rewrite import is_medical_query

from .llm_client import AsyncOpenAIClient, extract_text
from .llm_client_stream import AsyncOpenAIStreamClient
from .lang_utils import resolve_response_lang
from .plan_and_solve import (
    AsyncPlanAndSolveAgent,
    _safe_json_loads,
    build_solve_prompt,
    classify_query,
    harden_plan_steps,
)
from .rag_evidence import best_hit_relevance_score, rag_evidence_summary
from .schemas_openai import (
    ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionChunkDelta,
    ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse,
    ChatMessage, UsageInfo,
)
from .qa_store import save_qa_record
from .session_manager import get_session_manager
from .tool_registry import get_registry
from .trace_store import new_trace_id, write_trace

router = APIRouter()
MAX_TOOL_ROUNDS = 5

_REASONER_MODEL = os.getenv("REASONER_MODEL", "deepseek-reasoner")
_CHAT_MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat")

_TIMELINESS_HINTS = re.compile(
    r"(最新|最近|近期|近年|今年|当前|目前最新|202[3-9]|2030|latest|recent|current|update)",
    re.IGNORECASE,
)

_VERIFICATION_HINTS = re.compile(
    r"(正确吗|对吗|真的吗|是真的|是否正确|是否属实|有没有道理|有道理吗|科学吗|准确吗|"
    r"是不是真|是不是对|可信吗|靠谱吗|有依据吗|有根据吗|事实是|实际上是|"
    r"验证|核实|辟谣|谣言|误解|误区|真相|fact.?check|is.?(?:this|it|that).?(?:true|correct|accurate|right))",
    re.IGNORECASE,
)

_KNOWLEDGE_GAP_HINTS = re.compile(
    r"(为什么.*(?:能|会|可以|不会)|原因是|机制|原理|怎么做到|如何.*(?:消化|分解|抵抗|解毒)|"
    r"是为了.*吗|是不是为了|是否为了|为了.*(?:吗|么)|适应(?:环境|生态|生存).*(?:吗|么)|"
    r"how.*(?:can|do|does)|why.*(?:can|do|does)|mechanism|reason)",
    re.IGNORECASE,
)

_WEB_TOOL = "mcp.web_search.web_search"
_FOLLOWUP_ENABLE_WEB = re.compile(
    r"(联网搜索|联网|上网|网络|继续查|继续搜索|继续核实|再搜|"
    r"web search|search online|use web|go online|search the web|web|search)",
    re.IGNORECASE,
)
_FOLLOWUP_KB_ONLY = re.compile(
    r"(仅知识库|只看知识库|只按知识库|不用联网|不要联网|本地即可|只用本地|"
    r"no web|kb only|local only|without web|skip web|knowledge base only)",
    re.IGNORECASE,
)


def _needs_planner(query: str) -> bool:
    """Return True if query likely needs LLM planning.

    Triggers on:
    - Timeliness hints (最新, recent, etc.)
    - Fact-checking / verification questions (正确吗, 真的吗, etc.)
    - Knowledge-gap questions about mechanisms (为什么能, 原理, etc.)
    """
    return bool(
        _TIMELINESS_HINTS.search(query)
        or _VERIFICATION_HINTS.search(query)
        or _KNOWLEDGE_GAP_HINTS.search(query)
    )


def _pick_solve_model(query: str) -> str:
    """Use reasoner for medical/research queries; chat model for general questions."""
    if is_medical_query(query):
        return _REASONER_MODEL
    return _CHAT_MODEL

_HIGH_TRUST_WEB_DOMAINS = (
    "panda.org.cn", "pmc.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov", "msdvetmanual.com", "nature.com",
    "sciencedirect.com", "springer.com", "wiley.com",
)
_LOW_TRUST_WEB_DOMAINS = (
    "hytrans-sh.com", "hanbaoauto.com",
)


def _domain_from_url(url: str) -> str:
    try:
        return urlparse((url or "").strip()).netloc.lower()
    except Exception:
        return ""


def _source_priority(url: str) -> int:
    domain = _domain_from_url(url)
    if not domain:
        return 0
    if any(domain == d or domain.endswith(f".{d}") for d in _HIGH_TRUST_WEB_DOMAINS):
        return 100
    if domain.endswith(".gov") or domain.endswith(".edu") or domain.endswith(".edu.cn") or domain.endswith(".ac.cn"):
        return 90
    if domain.endswith(".org"):
        return 75
    if any(domain == d or domain.endswith(f".{d}") for d in _LOW_TRUST_WEB_DOMAINS):
        return -20
    return 40


def _postprocess_web_results(tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter and re-rank web search results within tool_results by source priority."""
    processed = []
    for tr in tool_results:
        if not (tr.get("tool_name") or "").startswith("mcp.web_search"):
            processed.append(tr)
            continue
        result = tr.get("result")
        if not isinstance(result, dict):
            processed.append(tr)
            continue
        items = result.get("results") or result.get("hits") or []
        if not isinstance(items, list):
            processed.append(tr)
            continue
        filtered = []
        for item in items:
            if not isinstance(item, dict):
                continue
            url = item.get("url") or ""
            prio = _source_priority(url)
            if prio < 0:
                continue
            item["source_priority"] = prio
            item["domain"] = _domain_from_url(url)
            filtered.append(item)
        filtered.sort(key=lambda x: -x.get("source_priority", 0))
        new_result = dict(result)
        if "results" in result:
            new_result["results"] = filtered
        elif "hits" in result:
            new_result["hits"] = filtered
        else:
            new_result["results"] = filtered
        processed.append({**tr, "result": new_result})
    return processed

DEFAULT_ALLOWED_TOOLS = [
    "rag.search",
    "mcp.web_search.web_search",
    "mcp.vital_signs_analyzer.analyze_vitals",
]

_TOOL_DESCRIPTIONS = {
    "rag.search": "Search the knowledge base for information on biology, breeding, conservation, habitat, anatomy, genetics, disease prevention, ecology, etc.",
    "mcp.web_search.web_search": "Perform real-time web search for up-to-date information not covered by the knowledge base.",
}


def _gen_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _now_ts() -> int:
    return int(time.time())


def _extract_user_query(messages: List[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user" and msg.content:
            return msg.content
    return ""


def _build_system_context(messages: List[ChatMessage]) -> str:
    parts = []
    for msg in messages:
        if msg.role == "system" and msg.content:
            parts.append(msg.content)
    return "\n".join(parts) if parts else ""


def _build_conversation_history(messages: List[ChatMessage]) -> List[Dict[str, str]]:
    return [{"role": msg.role, "content": msg.content}
            for msg in messages if msg.role in ("user", "assistant") and msg.content]


def _build_session_context(session_data: Any) -> str:
    if not session_data:
        return ""
    parts: List[str] = []
    prefs = getattr(session_data, "preferences", {}) or {}
    web_policy = str(prefs.get("web_search_policy") or "").strip()
    if web_policy == "kb_only":
        parts.append("用户偏好：默认优先仅按本地知识库回答，除非用户另行要求联网。")
    elif web_policy == "prefer":
        parts.append("用户偏好：当本地证据不足时，允许主动联网核实。")
    recent = getattr(session_data, "messages", []) or []
    if recent:
        tail = recent[-4:]
        parts.append("服务端短期记忆：\n" + "\n".join(f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in tail))
    return "\n".join(parts).strip()


def _interpret_pending_choice(text: str) -> Optional[str]:
    q = (text or "").strip()
    if not q:
        return None
    if _FOLLOWUP_KB_ONLY.search(q):
        return "kb_only"
    if _FOLLOWUP_ENABLE_WEB.search(q):
        return "prefer"
    return None


def _clarification_question(query: str, allowed_tools: List[str], session_prefs: Optional[Dict[str, Any]] = None) -> Optional[str]:
    profile = classify_query(query)
    prefs = dict(session_prefs or {})
    if prefs.get("web_search_policy") == "kb_only":
        return None
    needs_web = profile["verification"] or profile["mechanism"] or profile["timeliness"] or profile["medical"]
    if needs_web and _WEB_TOOL not in (allowed_tools or []):
        return "这个问题更适合结合网络证据核实。要我继续联网搜索，还是只按当前知识库回答？"
    return None


def _rag_evidence_gap_reason(query: str, tool_results: List[Dict[str, Any]]) -> Optional[str]:
    profile = classify_query(query)
    requires_evidence_chain = profile["verification"] or profile["timeliness"]
    summary = rag_evidence_summary(tool_results)
    from .rag_evidence import rag_gap_thresholds

    thresholds = rag_gap_thresholds()
    min_hits = int(thresholds["min_hits"])
    min_strong_hits = int(thresholds["min_strong_hits"])
    min_distinct_sources = int(thresholds["min_distinct_sources"])

    reasons: List[str] = []
    if summary["hits_total"] < min_hits:
        reasons.append(f"RAG only returned {summary['hits_total']} hits")
    if summary["hits_total"] > 0 and summary["strong_hits"] == 0:
        reasons.append(
            f"RAG best score {summary['best_score']:.3f} below relevance threshold "
            f"(rerank>={summary.get('rerank_threshold', 0.35)} or dense>={summary.get('dense_threshold', 0.12)})"
        )
    if requires_evidence_chain:
        if summary["strong_hits"] < min_strong_hits:
            reasons.append(f"only {summary['strong_hits']} strong RAG hits, not enough for an evidence chain")
        if summary["distinct_sources"] < min_distinct_sources:
            reasons.append(
                f"only {summary['distinct_sources']} distinct RAG sources, not enough for cross-source support"
            )

    if not reasons:
        return None
    return "; ".join(reasons)


def _authoritative_web_search_arguments(query: str) -> Dict[str, Any]:
    return {
        "query": query,
        "max_results": int(os.getenv("WEB_FALLBACK_MAX_RESULTS", "8")),
        "search_depth": os.getenv("WEB_FALLBACK_SEARCH_DEPTH", "advanced"),
    }


def _web_search_ask_message(evidence_gap_reason: str, *, response_lang: str = "zh") -> str:
    detail = evidence_gap_reason.strip() or ("relevance too weak" if response_lang == "en" else "相关度不足")
    if response_lang == "en":
        return (
            "I couldn't find a strong enough match in the local knowledge base "
            f"({detail}).\n\n"
            "Would you like me to search the web for more information? "
            "Reply **search online** or **web search** to continue; "
            "reply **no web** or **kb only** to answer from the local knowledge base only."
        )
    return (
        "我在本地知识库中没有找到与您问题足够匹配的内容"
        f"（{detail}）。\n\n"
        "需要我联网搜索补充信息吗？请回复 **联网搜索** 或 **继续搜索**；"
        "若仅基于现有资料回答，请回复 **不用联网**。"
    )


async def _resolve_rag_gap_web_search(
    *,
    query: str,
    tool_results: List[Dict[str, Any]],
    available_tools: List[str],
    session_preferences: Optional[Dict[str, Any]],
    reg: Any,
    response_lang: str = "auto",
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """After RAG: ask user before web search, unless user already opted in (prefer)."""
    evidence_gap_reason = _rag_evidence_gap_reason(query, tool_results)
    already_has_web = any(r.get("tool_name") == _WEB_TOOL for r in tool_results)
    if not evidence_gap_reason or already_has_web or _WEB_TOOL not in available_tools:
        return tool_results, None

    prefs = session_preferences or {}
    if prefs.get("web_search_policy") == "kb_only":
        return tool_results, None

    if prefs.get("web_search_policy") == "prefer":
        web_args = _authoritative_web_search_arguments(query)
        try:
            web_result = await reg.call(_WEB_TOOL, web_args)
            tool_results.append({
                "step": "fallback",
                "tool_name": _WEB_TOOL,
                "arguments": web_args,
                "result": web_result,
            })
        except Exception:
            pass
        return tool_results, None

    return tool_results, _web_search_ask_message(
        evidence_gap_reason,
        response_lang=resolve_response_lang(response_lang, query),
    )


def _answer_quality_flags(query: str, answer: str, tool_results: List[Dict[str, Any]]) -> List[str]:
    flags: List[str] = []
    profile = classify_query(query)
    used_web = any((tr.get("tool_name") or "").startswith("mcp.web_search") for tr in tool_results if isinstance(tr, dict))
    rag_hits = sum(
        len(tr.get("result", {}).get("hits", []))
        for tr in tool_results
        if isinstance(tr, dict) and tr.get("tool_name") == "rag.search" and isinstance(tr.get("result"), dict)
    )
    answer_text = (answer or "").strip()
    if (profile["verification"] or profile["mechanism"] or profile["timeliness"]) and not used_web:
        flags.append("missing_web_cross_check")
    if _rag_evidence_gap_reason(query, tool_results) and not used_web:
        flags.append("rag_evidence_chain_insufficient")
    if rag_hits == 0 and not used_web:
        flags.append("answered_without_evidence")
    if len(answer_text) < 40:
        flags.append("too_short")
    if ("无法证实" in answer_text or "当前知识库" in answer_text) and not used_web:
        flags.append("kb_only_deflection")
    return flags


def _make_decide_prompt(query: str, tool_results: List[Dict[str, Any]], available_tools: List[str]) -> str:
    tools_with_desc = [{"name": t, "description": _TOOL_DESCRIPTIONS.get(t, "")} for t in available_tools]
    return json.dumps({
        "task": "decide next action",
        "instructions": (
            "你是一个智能决策 Agent。根据用户的问题和已有的工具调用结果，"
            "决定是否需要继续调用工具获取更多信息，还是已经可以生成最终回答。"
            "你必须输出严格 JSON。\n"
            "重要规则：\n"
            "- 如果用户的问题是事实核查/验证类（如'正确吗'、'真的吗'、'是否属实'），"
            "必须同时使用 rag.search 和 mcp.web_search.web_search 交叉验证\n"
            "- 如果用户的问题涉及机制/原理（如'为什么能…'、'怎么做到'），"
            "建议同时使用 rag.search 和 web_search 获取更全面的科学解释\n"
            "- 如果 rag.search 虽然有命中，但命中数量少、相关分低、或来源过于单一，"
            "不足以构成证据链，必须继续调用 mcp.web_search.web_search 去权威网站补充证据\n"
            "- 如果已有结果不足以回答问题，主动调用其他工具补充\n"
            "- 如果必须联网核实但当前工具集合里没有 web_search，可输出 ask_user 继续追问用户"
        ),
        "user_query": query,
        "tool_results_so_far": [
            {"tool_name": r.get("tool_name"),
             "hits_count": len(r.get("result", {}).get("hits", [])) if isinstance(r.get("result"), dict) else 0}
            for r in tool_results
        ],
        "available_tools": tools_with_desc,
        "output_format": {"action": "call_tool or ask_user or final_answer", "tool_name": "tool name",
                          "arguments": {}, "question": "clarifying question if needed", "reason": "brief explanation"},
    }, ensure_ascii=False)


async def _stream_multi_turn_agent(
    request_id: str, model: str, query: str, system_context: str,
    conversation_history: List[Dict[str, str]], temperature: float, max_tokens: int,
    allowed_tools: Optional[List[str]], debug_timing: bool = False,
    user_role: str = "farmer",
    session_preferences: Optional[Dict[str, Any]] = None,
    response_lang: str = "auto",
) -> AsyncGenerator[str, None]:
    created = _now_ts()
    reg = get_registry()
    llm = AsyncOpenAIClient()
    solve_model = _pick_solve_model(query)
    stream_llm = AsyncOpenAIStreamClient(model=solve_model)
    agent = AsyncPlanAndSolveAgent(registry=reg, llm=llm)

    _t_start = time.perf_counter()
    _timing: List[Dict[str, Any]] = []

    def _lap(label: str, t0: float, **extra: Any) -> None:
        if debug_timing:
            _timing.append({"step": label, "ms": round((time.perf_counter() - t0) * 1000, 1), **extra})

    def make_chunk(content: str = "", status: str = None, detail: Dict = None, finish: str = None) -> str:
        chunk = ChatCompletionChunk(
            id=request_id, created=created, model=model,
            choices=[ChatCompletionChunkChoice(delta=ChatCompletionChunkDelta(content=content if content else None), finish_reason=finish)],
            agent_status=status, agent_detail=detail,
        )
        return f"data: {chunk.model_dump_json()}\n\n"

    tool_results: List[Dict[str, Any]] = []
    available_tools = list(allowed_tools or ["rag.search"])
    if (session_preferences or {}).get("web_search_policy") == "kb_only":
        available_tools = [t for t in available_tools if t != _WEB_TOOL]

    for round_num in range(MAX_TOOL_ROUNDS):
        yield make_chunk(status="thinking", detail={"message": f"思考中... (第{round_num + 1}轮)", "round": round_num + 1})
        decide_prompt = _make_decide_prompt(query, tool_results, available_tools)
        t0 = time.perf_counter()
        try:
            decide_resp = await llm.chat(
                messages=[{"role": "system", "content": "你是一个智能决策 Agent。输出严格 JSON。"},
                          {"role": "user", "content": decide_prompt}],
                temperature=0.1, max_tokens=256, response_format={"type": "json_object"},
            )
            _lap(f"round_{round_num+1}_decide", t0)
            decision, err = _safe_json_loads(extract_text(decide_resp))
            if not decision:
                yield make_chunk(content=f"Decision parse failed: {err}\n")
                break
        except Exception as e:
            _lap(f"round_{round_num+1}_decide_error", t0)
            yield make_chunk(content=f"Decision failed: {e}\n")
            break

        action = decision.get("action", "final_answer")
        reason = decision.get("reason", "")

        if action == "call_tool":
            tool_name = decision.get("tool_name", "rag.search")
            args = decision.get("arguments", {})
            if tool_name not in available_tools:
                continue
            if tool_name == "rag.search":
                args = agent._force_rag_search_defaults(args)
            yield make_chunk(content=f"\n**Round {round_num + 1}**: {tool_name}\n", status="tool_calling",
                             detail={"tool_name": tool_name, "round": round_num + 1, "reason": reason})
            if args.get("query"):
                yield make_chunk(content=f"   Query: {args['query']}\n")
            t0 = time.perf_counter()
            try:
                result = await reg.call(tool_name, args)
                _lap(f"round_{round_num+1}_tool_{tool_name}", t0)
                tool_results.append({"round": round_num + 1, "tool_name": tool_name, "arguments": args, "result": result})
                _hits = result.get("hits", []) if isinstance(result, dict) else []
                hits_count = len(_hits)
                best_score = best_hit_relevance_score(_hits) if _hits else 0.0
                yield make_chunk(content=f"   Found {hits_count} results\n", status="tool_complete",
                                 detail={"tool_name": tool_name, "hits_count": hits_count,
                                         "best_score": best_score, "round": round_num + 1})
            except Exception as e:
                _lap(f"round_{round_num+1}_tool_error", t0)
                yield make_chunk(content=f"   Tool failed: {e}\n")
                tool_results.append({"round": round_num + 1, "tool_name": tool_name, "arguments": args, "error": str(e)})
        elif action == "final_answer":
            yield make_chunk(content=f"\n**Generating answer** (reason: {reason})\n", status="decided_final",
                             detail={"reason": reason, "total_rounds": round_num + 1})
            break
        elif action == "ask_user":
            question = str(decision.get("question") or "当前证据不足。要我继续联网搜索，还是只按当前知识库回答？").strip()
            yield make_chunk(content=question, status="ask_user", detail={"question": question, "reason": reason}, finish="stop")
            return
        else:
            break

    evidence_gap_reason = _rag_evidence_gap_reason(query, tool_results)
    t0 = time.perf_counter()
    tool_results, ask_msg = await _resolve_rag_gap_web_search(
        query=query,
        tool_results=tool_results,
        available_tools=available_tools,
        session_preferences=session_preferences,
        reg=reg,
        response_lang=response_lang,
    )
    if ask_msg:
        _lap("rag_gap_ask_user", t0, reason=evidence_gap_reason or "")
        yield make_chunk(
            content=ask_msg,
            status="ask_user",
            detail={"question": ask_msg, "reason": "rag_evidence_gap"},
            finish="stop",
        )
        return
    if any(r.get("tool_name") == _WEB_TOOL for r in tool_results):
        _lap("fallback_web_search", t0)

    tool_results = _postprocess_web_results(tool_results)

    yield make_chunk(content="\n", status="generating")

    has_web = any(r.get("tool_name", "").startswith("mcp.web_search") for r in tool_results)
    sys_prompt = build_solve_prompt(
        user_role=user_role, has_web_search=has_web, query=query, response_lang=response_lang,
    )
    if system_context:
        sys_prompt = f"{system_context}\n\n{sys_prompt}"

    user_content_parts = []
    if conversation_history:
        user_content_parts.append("历史对话:\n" + "\n".join(f"{m['role']}: {m['content']}" for m in conversation_history[-6:]))
    user_content_parts.append(json.dumps({"query": query, "tool_results": tool_results}, ensure_ascii=False))
    user_content = "\n\n".join(user_content_parts)

    t0 = time.perf_counter()
    try:
        async for chunk_text in stream_llm.chat_stream(
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_content}],
            temperature=temperature, max_tokens=max_tokens,
        ):
            yield make_chunk(content=chunk_text, status="streaming")
    except Exception as e:
        yield make_chunk(content=f"\nGeneration failed: {e}")
    _lap("final_generation", t0)

    if debug_timing:
        total_ms = round((time.perf_counter() - _t_start) * 1000, 1)
        _timing.append({"step": "total", "ms": total_ms})
        yield make_chunk(status="timing_summary", detail={"timing": _timing})

    yield make_chunk(finish="stop")


async def _stream_plan_and_solve(
    request_id: str, model: str, query: str, system_context: str,
    temperature: float, max_tokens: int, allowed_tools: Optional[List[str]],
    debug_timing: bool = False, user_role: str = "farmer",
    session_preferences: Optional[Dict[str, Any]] = None,
    response_lang: str = "auto",
) -> AsyncGenerator[str, None]:
    created = _now_ts()
    reg = get_registry()
    llm = AsyncOpenAIClient()
    solve_model = _pick_solve_model(query)
    stream_llm = AsyncOpenAIStreamClient(model=solve_model)
    agent = AsyncPlanAndSolveAgent(registry=reg, llm=llm)

    _t_start = time.perf_counter()
    _timing: List[Dict[str, Any]] = []

    def _lap(label: str, t0: float, **extra: Any) -> None:
        if debug_timing:
            _timing.append({"step": label, "ms": round((time.perf_counter() - t0) * 1000, 1), **extra})

    def make_chunk(content: str = "", status: str = None, detail: Dict = None, finish: str = None) -> str:
        chunk = ChatCompletionChunk(
            id=request_id, created=created, model=model,
            choices=[ChatCompletionChunkChoice(delta=ChatCompletionChunkDelta(content=content if content else None), finish_reason=finish)],
            agent_status=status, agent_detail=detail,
        )
        return f"data: {chunk.model_dump_json()}\n\n"

    use_planner = _needs_planner(query)

    if use_planner:
        yield make_chunk(status="planning", detail={"message": "正在制定计划..."})
        t0 = time.perf_counter()
        try:
            plan = await agent.plan(query=query, allowed_tools=allowed_tools)
            _lap("plan_llm", t0)
            yield make_chunk(content="**Plan complete**\n", status="plan_complete", detail={"plan": plan})
        except Exception as e:
            _lap("plan_error", t0)
            yield make_chunk(content=f"Planning failed: {e}", finish="stop")
            return
    else:
        plan = [
            {"type": "tool", "tool_name": "rag.search", "arguments": {"query": query}, "note": "fast path"},
            {"type": "final", "note": "generate answer from RAG results"},
        ]
        yield make_chunk(status="plan_complete", detail={"plan": plan, "fast_path": True})

    plan = harden_plan_steps(plan, query=query, allowed_tools=allowed_tools, user_preferences=session_preferences)
    if any(step.get("type") == "ask_user" for step in plan):
        ask_step = next(step for step in plan if step.get("type") == "ask_user")
        question = str(ask_step.get("question") or "当前证据不足。要我继续联网搜索，还是只按当前知识库回答？").strip()
        yield make_chunk(content=question, status="ask_user", detail={"question": question, "reason": ask_step.get("reason", "")}, finish="stop")
        return

    # --- Collect tool steps and execute in parallel ---
    tool_steps: List[Dict[str, Any]] = []
    allowed = set(allowed_tools) if allowed_tools else None
    for i, step in enumerate(plan):
        if step.get("type") == "tool":
            tool_name = str(step.get("tool_name") or "")
            if not tool_name or (allowed is not None and tool_name not in allowed):
                continue
            args = step.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}
            if tool_name == "rag.search":
                args = agent._force_rag_search_defaults(args)
            tool_steps.append({"step": i, "tool_name": tool_name, "arguments": args})
        elif step.get("type") == "final":
            break

    tool_results: List[Dict[str, Any]] = []

    if len(tool_steps) >= 2:
        yield make_chunk(content=f"\n**Parallel execution**: {', '.join(s['tool_name'] for s in tool_steps)}\n",
                         status="tool_calling", detail={"parallel": True, "count": len(tool_steps)})
        t0 = time.perf_counter()

        async def _run_tool(ts: Dict[str, Any]) -> Dict[str, Any]:
            try:
                result = await reg.call(ts["tool_name"], ts["arguments"])
                return {**ts, "result": result}
            except Exception as exc:
                return {**ts, "error": str(exc)}

        gathered = await asyncio.gather(*[_run_tool(ts) for ts in tool_steps], return_exceptions=False)
        _lap("parallel_tools", t0)

        for tr in gathered:
            tool_results.append(tr)
            if "error" in tr:
                yield make_chunk(content=f"   {tr['tool_name']}: failed ({tr['error']})\n")
            else:
                res = tr.get("result") or {}
                hits = res.get("hits", []) if isinstance(res, dict) else []
                hits_count = len(hits)
                best_score = best_hit_relevance_score(hits) if hits else 0.0
                yield make_chunk(content=f"   {tr['tool_name']}: {hits_count} results\n",
                                 status="tool_complete",
                                 detail={"tool_name": tr["tool_name"], "hits_count": hits_count, "best_score": best_score})
    else:
        for ts in tool_steps:
            yield make_chunk(content=f"\n**Tool**: {ts['tool_name']}\n",
                             status="tool_calling", detail={"tool_name": ts["tool_name"], "step": ts["step"]})
            t0 = time.perf_counter()
            try:
                result = await reg.call(ts["tool_name"], ts["arguments"])
                _lap(f"step_{ts['step']}_{ts['tool_name']}", t0)
                tool_results.append({**ts, "result": result})
                hits = result.get("hits", []) if isinstance(result, dict) else []
                hits_count = len(hits)
                best_score = best_hit_relevance_score(hits) if hits else 0.0
                yield make_chunk(content=f"Found {hits_count} results\n", status="tool_complete",
                                 detail={"tool_name": ts["tool_name"], "hits_count": hits_count, "best_score": best_score})
            except Exception as e:
                _lap(f"step_{ts['step']}_error", t0)
                yield make_chunk(content=f"Tool failed: {e}\n")
                tool_results.append({**ts, "error": str(e)})

    evidence_gap_reason = _rag_evidence_gap_reason(query, tool_results)
    t0 = time.perf_counter()
    tool_results, ask_msg = await _resolve_rag_gap_web_search(
        query=query,
        tool_results=tool_results,
        available_tools=list(allowed_tools or []),
        session_preferences=session_preferences,
        reg=reg,
        response_lang=response_lang,
    )
    if ask_msg:
        _lap("rag_gap_ask_user", t0, reason=evidence_gap_reason or "")
        yield make_chunk(
            content=ask_msg,
            status="ask_user",
            detail={"question": ask_msg, "reason": "rag_evidence_gap"},
            finish="stop",
        )
        return
    if any(r.get("tool_name") == _WEB_TOOL for r in tool_results):
        _lap("fallback_web_search", t0)

    clarification = _clarification_question(query, list(allowed_tools or []), session_prefs=session_preferences)
    has_web_after = any(r.get("tool_name") == _WEB_TOOL for r in tool_results)
    if clarification and not has_web_after:
        yield make_chunk(content=clarification, status="ask_user", detail={"question": clarification, "reason": "missing_web_for_high_risk_query"}, finish="stop")
        return

    tool_results = _postprocess_web_results(tool_results)

    yield make_chunk(content="\n", status="generating")

    has_web = any(r.get("tool_name", "").startswith("mcp.web_search") for r in tool_results)
    sys_prompt = build_solve_prompt(
        user_role=user_role, has_web_search=has_web, query=query, response_lang=response_lang,
    )
    if system_context:
        sys_prompt = f"{system_context}\n\n{sys_prompt}"

    user_content = json.dumps({"query": query, "plan": plan, "tool_results": tool_results}, ensure_ascii=False)
    t0 = time.perf_counter()
    try:
        async for chunk_text in stream_llm.chat_stream(
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_content}],
            temperature=temperature, max_tokens=max_tokens,
        ):
            yield make_chunk(content=chunk_text, status="streaming")
    except Exception as e:
        yield make_chunk(content=f"\nGeneration failed: {e}")
    _lap("final_generation", t0)

    if debug_timing:
        total_ms = round((time.perf_counter() - _t_start) * 1000, 1)
        _timing.append({"step": "total", "ms": total_ms})
        yield make_chunk(status="timing_summary", detail={"timing": _timing})

    yield make_chunk(finish="stop")


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    trace_id = new_trace_id()
    request_id = _gen_id()
    created = _now_ts()
    query = _extract_user_query(req.messages)
    system_context = _build_system_context(req.messages)
    conversation_history = _build_conversation_history(req.messages)
    session_id = (req.session_id or "").strip()
    session_mgr = get_session_manager()
    session_data = await session_mgr.get(session_id) if session_id else None
    session_preferences = dict(getattr(session_data, "preferences", {}) or {})

    pending = getattr(session_data, "pending_clarification", None) if session_data else None
    followup_choice = _interpret_pending_choice(query) if pending else None
    if pending and followup_choice:
        await session_mgr.set_preference(session_id, "web_search_policy", followup_choice)
        session_preferences["web_search_policy"] = followup_choice
        original_query = str(pending.get("original_query") or "").strip()
        if original_query:
            query = original_query
            req.messages = [*req.messages[:-1], ChatMessage(role="user", content=original_query)]
            conversation_history = _build_conversation_history(req.messages)
        await session_mgr.set_pending_clarification(session_id, None)

    session_context = _build_session_context(session_data)
    if session_context:
        system_context = f"{system_context}\n\n{session_context}" if system_context else session_context

    allowed_tools = None
    if req.tools:
        allowed_tools = [t.get("function", {}).get("name") for t in req.tools if t.get("function")]
        allowed_tools = [n for n in allowed_tools if n]
    if not allowed_tools:
        reg = get_registry()
        available_names = {t.name for t in reg.list_tools()}
        allowed_tools = [t for t in DEFAULT_ALLOWED_TOOLS if t in available_names]
        if not allowed_tools:
            allowed_tools = ["rag.search"]
    if session_preferences.get("web_search_policy") == "kb_only":
        allowed_tools = [tool_name for tool_name in allowed_tools if tool_name != _WEB_TOOL]

    use_multi_turn = req.model in ("livestock-multi-turn", "agent-multi-turn", "multi-turn")
    _debug_timing = bool(req.debug_timing)
    user_role = req.user_role or "farmer"
    response_lang = resolve_response_lang(req.response_lang, query)

    source_ip = request.client.host if request.client else ""

    if req.stream:
        async def event_generator():
            t0 = time.monotonic()
            collected_content = []
            collected_tools = []
            collected_timing = []
            collected_rag_hits = 0
            collected_rag_best_score = 0.0
            collected_web_search = False
            collected_ask_user = ""
            source = (
                _stream_multi_turn_agent(
                    request_id=request_id, model=req.model, query=query, system_context=system_context,
                    conversation_history=conversation_history, temperature=req.temperature or 0.2,
                    max_tokens=req.max_tokens or 768, allowed_tools=allowed_tools, debug_timing=_debug_timing,
                    user_role=user_role, session_preferences=session_preferences,
                    response_lang=response_lang,
                ) if use_multi_turn else
                _stream_plan_and_solve(
                    request_id=request_id, model=req.model, query=query, system_context=system_context,
                    temperature=req.temperature or 0.2, max_tokens=req.max_tokens or 768,
                    allowed_tools=allowed_tools, debug_timing=_debug_timing, user_role=user_role,
                    session_preferences=session_preferences,
                    response_lang=response_lang,
                )
            )
            async for chunk in source:
                yield chunk
                if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]":
                    try:
                        obj = json.loads(chunk[6:])
                        delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                        status = obj.get("agent_status")
                        detail = obj.get("agent_detail") or {}
                        if status == "streaming" and delta.get("content"):
                            collected_content.append(delta["content"])
                        elif status == "tool_complete":
                            tool_name = detail.get("tool_name", "")
                            collected_tools.append(tool_name)
                            if "rag" in tool_name:
                                collected_rag_hits += detail.get("hits_count", 0)
                                bs = detail.get("best_score", 0.0) or 0.0
                                if bs > collected_rag_best_score:
                                    collected_rag_best_score = bs
                            if "web_search" in tool_name:
                                collected_web_search = True
                        elif status == "timing_summary":
                            collected_timing = detail.get("timing", [])
                        elif status == "ask_user":
                            collected_ask_user = detail.get("question", "") or delta.get("content") or ""
                    except Exception:
                        pass
            write_trace(trace_id, tool="v1.chat.completions.stream",
                        request={"model": req.model, "query": query, "allowed_tools": allowed_tools},
                        response={"id": request_id, "answer_length": sum(len(s) for s in collected_content),
                                  "tools_called": collected_tools, "timing": collected_timing})
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            assistant_text = "".join(collected_content).strip() or collected_ask_user.strip()
            try:
                await save_qa_record(
                    question=query, answer=assistant_text,
                    model=req.model, tools_used=collected_tools,
                    rag_hit_count=collected_rag_hits,
                    rag_best_score=collected_rag_best_score,
                    used_web_search=collected_web_search,
                    response_time_ms=elapsed_ms,
                    source_ip=source_ip, user_role=user_role,
                    request_id=request_id,
                )
            except Exception as exc:
                print(f"[qa_store] save_qa_record failed for {request_id}: {exc}")
            if session_id:
                try:
                    await session_mgr.append_messages(session_id, [
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": assistant_text},
                    ])
                    if collected_ask_user:
                        await session_mgr.set_pending_clarification(
                            session_id,
                            {"original_query": query, "question": collected_ask_user, "request_id": request_id},
                        )
                    else:
                        await session_mgr.set_pending_clarification(session_id, None)
                except Exception:
                    pass
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
    else:
        t0 = time.monotonic()
        try:
            reg = get_registry()
            llm = AsyncOpenAIClient()
            solve_model = _pick_solve_model(query)
            solve_llm = AsyncOpenAIClient(model=solve_model)
            agent = AsyncPlanAndSolveAgent(registry=reg, llm=llm, solve_llm=solve_llm)
            plan = await agent.plan(query=query, allowed_tools=allowed_tools)
            plan = harden_plan_steps(plan, query=query, allowed_tools=allowed_tools, user_preferences=session_preferences)
            answer, tool_results = await agent.solve(query=query, plan_steps=plan, allowed_tools=allowed_tools,
                                                      temperature=req.temperature or 0.2, max_tokens=req.max_tokens or 768,
                                                      user_role=user_role)
            response = ChatCompletionResponse(
                id=request_id, created=created, model=req.model,
                choices=[ChatCompletionChoice(message=ChatMessage(role="assistant", content=answer), finish_reason="stop")],
                usage=UsageInfo(), plan=plan, tool_results=tool_results,
            )
            write_trace(trace_id, tool="v1.chat.completions", request=req.model_dump(), response=response.model_dump())
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            tools_called = [tr.get("tool_name", "") for tr in (tool_results or []) if isinstance(tr, dict)]
            rag_hits = sum(
                len(tr.get("result", {}).get("hits", []))
                for tr in (tool_results or [])
                if isinstance(tr, dict) and "rag" in tr.get("tool_name", "") and isinstance(tr.get("result"), dict)
            )
            web_used = any("web_search" in tr.get("tool_name", "") for tr in (tool_results or []) if isinstance(tr, dict))
            try:
                await save_qa_record(
                    question=query, answer=answer or "",
                    model=req.model, tools_used=tools_called,
                    rag_hit_count=rag_hits, used_web_search=web_used,
                    response_time_ms=elapsed_ms,
                    source_ip=source_ip, user_role=user_role,
                    request_id=request_id,
                )
            except Exception:
                pass
            if session_id:
                try:
                    await session_mgr.append_messages(session_id, [
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": answer or ""},
                    ])
                    ask_user_step = next((tr for tr in (tool_results or []) if tr.get("type") == "ask_user"), None)
                    if ask_user_step:
                        await session_mgr.set_pending_clarification(
                            session_id,
                            {"original_query": query, "question": ask_user_step.get("question", ""), "request_id": request_id},
                        )
                    else:
                        await session_mgr.set_pending_clarification(session_id, None)
                except Exception:
                    pass
            return response
        except Exception as e:
            error_response = {"error": {"message": str(e), "type": "server_error", "code": "agent_error"}}
            write_trace(trace_id, tool="v1.chat.completions", request=req.model_dump(), response=error_response, error=str(e))
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=error_response)


@router.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "livestock-plan-solve", "object": "model", "created": 1700000000, "owned_by": "livestockmind",
             "description": "Plan-and-Solve agent for LivestockMind veterinary Q&A"},
            {"id": "livestock-multi-turn", "object": "model", "created": 1700000000, "owned_by": "livestockmind",
             "description": "Multi-turn agent with iterative tool calls for livestock veterinary Q&A"},
        ]
    }
