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
from .plan_and_solve import AsyncPlanAndSolveAgent, _safe_json_loads, build_solve_prompt
from .schemas_openai import (
    ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionChunkDelta,
    ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse,
    ChatMessage, UsageInfo,
)
from .qa_store import save_qa_record
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


def _needs_planner(query: str) -> bool:
    """Return True if query likely needs LLM planning (timeliness, multi-step)."""
    return bool(_TIMELINESS_HINTS.search(query))


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


def _make_decide_prompt(query: str, tool_results: List[Dict[str, Any]], available_tools: List[str]) -> str:
    tools_with_desc = [{"name": t, "description": _TOOL_DESCRIPTIONS.get(t, "")} for t in available_tools]
    return json.dumps({
        "task": "decide next action",
        "instructions": (
            "你是一个智能决策 Agent。根据用户的问题和已有的工具调用结果，"
            "决定是否需要继续调用工具获取更多信息，还是已经可以生成最终回答。"
            "你必须输出严格 JSON。"
        ),
        "user_query": query,
        "tool_results_so_far": [
            {"tool_name": r.get("tool_name"),
             "hits_count": len(r.get("result", {}).get("hits", [])) if isinstance(r.get("result"), dict) else 0}
            for r in tool_results
        ],
        "available_tools": tools_with_desc,
        "output_format": {"action": "call_tool or final_answer", "tool_name": "tool name",
                          "arguments": {}, "reason": "brief explanation"},
    }, ensure_ascii=False)


async def _stream_multi_turn_agent(
    request_id: str, model: str, query: str, system_context: str,
    conversation_history: List[Dict[str, str]], temperature: float, max_tokens: int,
    allowed_tools: Optional[List[str]], debug_timing: bool = False,
    user_role: str = "enthusiast",
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
    available_tools = allowed_tools or ["rag.search"]

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
                best_score = max((h.get("score", 0.0) for h in _hits), default=0.0) if _hits else 0.0
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
        else:
            break

    tool_results = _postprocess_web_results(tool_results)

    yield make_chunk(content="\n", status="generating")

    has_web = any(r.get("tool_name", "").startswith("mcp.web_search") for r in tool_results)
    sys_prompt = build_solve_prompt(user_role=user_role, has_web_search=has_web, query=query)
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
    yield "data: [DONE]\n\n"


async def _stream_plan_and_solve(
    request_id: str, model: str, query: str, system_context: str,
    temperature: float, max_tokens: int, allowed_tools: Optional[List[str]],
    debug_timing: bool = False, user_role: str = "enthusiast",
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

    _WEB_TOOL = "mcp.web_search.web_search"
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
            yield "data: [DONE]\n\n"
            return
    else:
        plan = [
            {"type": "tool", "tool_name": "rag.search", "arguments": {"query": query}, "note": "fast path"},
            {"type": "final", "note": "generate answer from RAG results"},
        ]
        yield make_chunk(status="plan_complete", detail={"plan": plan, "fast_path": True})

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
                best_score = max((h.get("score", 0.0) for h in hits), default=0.0) if hits else 0.0
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
                best_score = max((h.get("score", 0.0) for h in hits), default=0.0) if hits else 0.0
                yield make_chunk(content=f"Found {hits_count} results\n", status="tool_complete",
                                 detail={"tool_name": ts["tool_name"], "hits_count": hits_count, "best_score": best_score})
            except Exception as e:
                _lap(f"step_{ts['step']}_error", t0)
                yield make_chunk(content=f"Tool failed: {e}\n")
                tool_results.append({**ts, "error": str(e)})

    # --- RAG fallback: auto web search when RAG results are insufficient or irrelevant ---
    _rag_results = [
        r for r in tool_results
        if r.get("tool_name") == "rag.search" and isinstance(r.get("result"), dict)
    ]
    rag_hits_total = sum(len(r["result"].get("hits", [])) for r in _rag_results)
    # Best rerank score across all RAG hits — low score means semantically irrelevant content
    rag_best_score = max(
        (hit.get("score", 0.0) for r in _rag_results for hit in r["result"].get("hits", [])),
        default=0.0,
    )
    # Trigger web fallback when:
    # 1. Not enough hits (original condition), OR
    # 2. Hits exist but best rerank score is below relevance threshold —
    #    meaning the knowledge base returned content but none of it actually
    #    addresses the query (e.g. asking about cyanide detox gets bamboo-eating hits)
    _RAG_RELEVANCE_THRESHOLD = 0.55
    already_has_web = any(r.get("tool_name") == _WEB_TOOL for r in tool_results)
    _need_web = (rag_hits_total < 2) or (rag_hits_total > 0 and rag_best_score < _RAG_RELEVANCE_THRESHOLD)
    if _need_web and not already_has_web and _WEB_TOOL in (allowed_tools or []):
        _fallback_reason = (
            f"RAG only returned {rag_hits_total} hits"
            if rag_hits_total < 2
            else f"RAG best score {rag_best_score:.3f} < {_RAG_RELEVANCE_THRESHOLD} (irrelevant content)"
        )
        yield make_chunk(
            content=f"\n**Fallback**: web_search ({_fallback_reason})\n",
            status="tool_calling",
            detail={"tool_name": _WEB_TOOL, "step": "fallback", "reason": _fallback_reason},
        )
        t0 = time.perf_counter()
        try:
            web_result = await reg.call(_WEB_TOOL, {"query": query, "max_results": 5, "search_depth": "advanced"})
            _lap("fallback_web_search", t0)
            tool_results.append({"step": "fallback", "tool_name": _WEB_TOOL, "arguments": {"query": query}, "result": web_result})
            web_count = len(web_result.get("results", [])) if isinstance(web_result, dict) else 0
            yield make_chunk(
                content=f"Found {web_count} web results\n",
                status="tool_complete",
                detail={"tool_name": _WEB_TOOL, "hits_count": web_count},
            )
        except Exception as e:
            _lap("fallback_web_search_error", t0)
            yield make_chunk(content=f"Web search fallback failed: {e}\n")

    tool_results = _postprocess_web_results(tool_results)

    yield make_chunk(content="\n", status="generating")

    has_web = any(r.get("tool_name", "").startswith("mcp.web_search") for r in tool_results)
    sys_prompt = build_solve_prompt(user_role=user_role, has_web_search=has_web, query=query)
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
    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    trace_id = new_trace_id()
    request_id = _gen_id()
    created = _now_ts()
    query = _extract_user_query(req.messages)
    system_context = _build_system_context(req.messages)
    conversation_history = _build_conversation_history(req.messages)

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

    use_multi_turn = req.model in ("panda-multi-turn", "agent-multi-turn", "multi-turn")
    _debug_timing = bool(req.debug_timing)
    user_role = req.user_role or "enthusiast"

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
            source = (
                _stream_multi_turn_agent(
                    request_id=request_id, model=req.model, query=query, system_context=system_context,
                    conversation_history=conversation_history, temperature=req.temperature or 0.2,
                    max_tokens=req.max_tokens or 768, allowed_tools=allowed_tools, debug_timing=_debug_timing,
                    user_role=user_role,
                ) if use_multi_turn else
                _stream_plan_and_solve(
                    request_id=request_id, model=req.model, query=query, system_context=system_context,
                    temperature=req.temperature or 0.2, max_tokens=req.max_tokens or 768,
                    allowed_tools=allowed_tools, debug_timing=_debug_timing, user_role=user_role,
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
                    except Exception:
                        pass
            write_trace(trace_id, tool="v1.chat.completions.stream",
                        request={"model": req.model, "query": query, "allowed_tools": allowed_tools},
                        response={"id": request_id, "answer_length": sum(len(s) for s in collected_content),
                                  "tools_called": collected_tools, "timing": collected_timing})
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            try:
                await save_qa_record(
                    question=query, answer="".join(collected_content),
                    model=req.model, tools_used=collected_tools,
                    rag_hit_count=collected_rag_hits,
                    rag_best_score=collected_rag_best_score,
                    used_web_search=collected_web_search,
                    response_time_ms=elapsed_ms,
                    source_ip=source_ip, user_role=user_role,
                    request_id=request_id,
                )
            except Exception:
                pass

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
            {"id": "panda-plan-solve", "object": "model", "created": 1700000000, "owned_by": "panda-mind",
             "description": "Plan-and-Solve agent for Panda Mind Q&A"},
            {"id": "panda-multi-turn", "object": "model", "created": 1700000000, "owned_by": "panda-mind",
             "description": "Multi-turn agent with iterative tool calls"},
        ]
    }
