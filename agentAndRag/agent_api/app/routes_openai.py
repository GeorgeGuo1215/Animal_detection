"""
OpenAI-compatible /v1/chat/completions endpoint with streaming support.

This module provides:
1. Standard OpenAI API format
2. SSE streaming with agent status updates
3. Plan-and-Solve agent integration
4. Multi-turn tool_calls support (Agent decides when to stop)
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from .llm_client import OpenAICompatClient, extract_text
from .llm_client_stream import OpenAIStreamClient
from .plan_and_solve import PlanAndSolveAgent, _safe_json_loads
from .schemas_openai import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    UsageInfo,
)
from .tool_registry import get_registry
from .trace_store import new_trace_id, write_trace


router = APIRouter()

# Maximum number of tool call rounds to prevent infinite loops
MAX_TOOL_ROUNDS = 5


def _gen_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _now_ts() -> int:
    return int(time.time())


def _extract_user_query(messages: List[ChatMessage]) -> str:
    """Extract the last user message as the query."""
    for msg in reversed(messages):
        if msg.role == "user" and msg.content:
            return msg.content
    return ""


def _build_system_context(messages: List[ChatMessage]) -> str:
    """Build system context from messages."""
    parts = []
    for msg in messages:
        if msg.role == "system" and msg.content:
            parts.append(msg.content)
    return "\n".join(parts) if parts else ""


def _build_conversation_history(messages: List[ChatMessage]) -> List[Dict[str, str]]:
    """Build conversation history for multi-turn context."""
    history = []
    for msg in messages:
        if msg.role in ("user", "assistant") and msg.content:
            history.append({"role": msg.role, "content": msg.content})
    return history


def _make_decide_prompt(
    query: str,
    tool_results: List[Dict[str, Any]],
    available_tools: List[str],
) -> str:
    """
    Build a prompt for the LLM to decide whether to call more tools or generate final answer.
    
    Returns a JSON prompt that asks the LLM to output:
    - {"action": "call_tool", "tool_name": "...", "arguments": {...}, "reason": "..."}
    - {"action": "final_answer", "reason": "..."}
    """
    return json.dumps({
        "task": "决定下一步行动",
        "instructions": (
            "你是一个智能 Agent。根据用户问题和已有的工具调用结果，决定是否需要继续调用工具获取更多信息，还是已经可以生成最终回答。\n"
            "如果信息不足，选择调用工具；如果信息足够，选择生成最终回答。\n"
            "你必须输出严格 JSON，不要输出任何额外文字。"
        ),
        "user_query": query,
        "tool_results_so_far": [
            {
                "tool_name": r.get("tool_name"),
                "hits_count": len(r.get("result", {}).get("hits", [])) if isinstance(r.get("result"), dict) else 0,
                "brief": _summarize_tool_result(r.get("result")),
            }
            for r in tool_results
        ],
        "available_tools": available_tools,
        "output_format": {
            "action": "call_tool 或 final_answer",
            "tool_name": "如果 action=call_tool，填写工具名",
            "arguments": {"query": "如果是 rag.search，填写搜索词"},
            "reason": "简短说明为什么做这个决定",
        },
    }, ensure_ascii=False)


def _summarize_tool_result(result: Any, max_chars: int = 500) -> str:
    """Summarize tool result for decision-making context."""
    if not isinstance(result, dict):
        return str(result)[:max_chars]
    
    hits = result.get("hits", [])
    if not hits:
        return "无结果"
    
    summaries = []
    for h in hits[:3]:  # Only first 3 hits
        text = h.get("text", "")[:200]
        summaries.append(text)
    
    return " | ".join(summaries)[:max_chars]


async def _stream_multi_turn_agent(
    request_id: str,
    model: str,
    query: str,
    system_context: str,
    conversation_history: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    allowed_tools: Optional[List[str]],
) -> AsyncGenerator[str, None]:
    """
    Generator for SSE streaming of multi-turn agent.
    
    The agent iteratively:
    1. Decides whether to call a tool or generate final answer
    2. If call_tool: execute tool, add result, loop back to step 1
    3. If final_answer: stream the final response
    
    Maximum MAX_TOOL_ROUNDS iterations to prevent infinite loops.
    """
    created = _now_ts()
    reg = get_registry()
    llm = OpenAICompatClient()
    stream_llm = OpenAIStreamClient()
    agent = PlanAndSolveAgent(registry=reg, llm=llm)

    def make_chunk(content: str = "", status: str = None, detail: Dict = None, finish: str = None) -> str:
        chunk = ChatCompletionChunk(
            id=request_id,
            created=created,
            model=model,
            choices=[ChatCompletionChunkChoice(
                delta=ChatCompletionChunkDelta(content=content if content else None),
                finish_reason=finish,
            )],
            agent_status=status,
            agent_detail=detail,
        )
        return f"data: {chunk.model_dump_json()}\n\n"

    tool_results: List[Dict[str, Any]] = []
    available_tools = allowed_tools or ["rag.search"]
    
    # Multi-turn loop
    for round_num in range(MAX_TOOL_ROUNDS):
        yield make_chunk(
            status="thinking",
            detail={"message": f"思考中... (第{round_num + 1}轮)", "round": round_num + 1}
        )
        
        # Ask LLM to decide next action
        decide_prompt = _make_decide_prompt(query, tool_results, available_tools)
        
        try:
            decide_resp = llm.chat(
                messages=[
                    {"role": "system", "content": "你是一个智能决策 Agent。输出严格 JSON。"},
                    {"role": "user", "content": decide_prompt},
                ],
                temperature=0.1,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            decide_text = extract_text(decide_resp)
            decision, err = _safe_json_loads(decide_text)
            
            if not decision:
                yield make_chunk(content=f"⚠️ 决策解析失败: {err}\n")
                break
                
        except Exception as e:
            yield make_chunk(content=f"❌ 决策失败: {e}\n")
            break
        
        action = decision.get("action", "final_answer")
        reason = decision.get("reason", "")
        
        if action == "call_tool":
            tool_name = decision.get("tool_name", "rag.search")
            args = decision.get("arguments", {})
            
            if tool_name not in available_tools:
                yield make_chunk(content=f"⚠️ 工具 {tool_name} 不可用，跳过\n")
                continue
            
            # Force RAG defaults
            if tool_name == "rag.search":
                args = agent._force_rag_search_defaults(args)
            
            yield make_chunk(
                content=f"\n🔍 **第{round_num + 1}轮工具调用**: {tool_name}\n",
                status="tool_calling",
                detail={"tool_name": tool_name, "round": round_num + 1, "reason": reason}
            )
            
            if args.get("query"):
                yield make_chunk(content=f"   搜索词: {args['query']}\n")
            
            try:
                result = reg.call(tool_name, args)
                tool_results.append({
                    "round": round_num + 1,
                    "tool_name": tool_name,
                    "arguments": args,
                    "result": result,
                })
                
                hits_count = len(result.get("hits", [])) if isinstance(result, dict) else 0
                yield make_chunk(
                    content=f"   ✅ 找到 {hits_count} 条相关信息\n",
                    status="tool_complete",
                    detail={"tool_name": tool_name, "hits_count": hits_count, "round": round_num + 1}
                )
            except Exception as e:
                yield make_chunk(content=f"   ⚠️ 工具调用失败: {e}\n")
                tool_results.append({
                    "round": round_num + 1,
                    "tool_name": tool_name,
                    "arguments": args,
                    "error": str(e),
                })
        
        elif action == "final_answer":
            yield make_chunk(
                content=f"\n💡 **决定生成最终回答** (原因: {reason})\n",
                status="decided_final",
                detail={"reason": reason, "total_rounds": round_num + 1}
            )
            break
        
        else:
            # Unknown action, treat as final
            yield make_chunk(content=f"⚠️ 未知决策: {action}，将生成回答\n")
            break
    
    # Phase: Generate final answer (streaming)
    yield make_chunk(
        content="\n💭 **正在生成回答...**\n\n",
        status="generating"
    )

    sys_prompt = (
        "你是一个面向兽医/动物健康方向的 AI 助手。"
        "请用中文回答，必要时引用你从工具返回的证据片段（简短引用即可）。"
        "如果证据不足，请明确说不确定，并给出下一步建议。"
    )
    if system_context:
        sys_prompt = f"{system_context}\n\n{sys_prompt}"

    # Build context with conversation history and tool results
    user_content_parts = []
    if conversation_history:
        user_content_parts.append("历史对话:\n" + "\n".join(
            f"{m['role']}: {m['content']}" for m in conversation_history[-6:]  # Last 6 messages
        ))
    
    user_content_parts.append(json.dumps({
        "query": query,
        "tool_results": tool_results,
    }, ensure_ascii=False))
    
    user_content = "\n\n".join(user_content_parts)

    try:
        for chunk_text in stream_llm.chat_stream(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield make_chunk(content=chunk_text, status="streaming")
    except Exception as e:
        yield make_chunk(content=f"\n❌ 生成失败: {e}")

    yield make_chunk(finish="stop")
    yield "data: [DONE]\n\n"


# Keep the old single-turn implementation for backward compatibility
async def _stream_plan_and_solve(
    request_id: str,
    model: str,
    query: str,
    system_context: str,
    temperature: float,
    max_tokens: int,
    allowed_tools: Optional[List[str]],
) -> AsyncGenerator[str, None]:
    """
    Generator for SSE streaming of plan-and-solve agent (single-turn, backward compatible).
    
    Yields SSE-formatted chunks with agent status updates.
    """
    created = _now_ts()
    reg = get_registry()
    llm = OpenAICompatClient()
    stream_llm = OpenAIStreamClient()
    agent = PlanAndSolveAgent(registry=reg, llm=llm)

    def make_chunk(content: str = "", status: str = None, detail: Dict = None, finish: str = None) -> str:
        chunk = ChatCompletionChunk(
            id=request_id,
            created=created,
            model=model,
            choices=[ChatCompletionChunkChoice(
                delta=ChatCompletionChunkDelta(content=content if content else None),
                finish_reason=finish,
            )],
            agent_status=status,
            agent_detail=detail,
        )
        return f"data: {chunk.model_dump_json()}\n\n"

    # Phase 1: Planning
    yield make_chunk(status="planning", detail={"message": "正在制定计划..."})
    
    try:
        plan = agent.plan(query=query, allowed_tools=allowed_tools)
        yield make_chunk(
            content="📋 **计划制定完成**\n",
            status="plan_complete",
            detail={"plan": plan}
        )
    except Exception as e:
        yield make_chunk(content=f"❌ 计划失败: {e}", finish="stop")
        yield "data: [DONE]\n\n"
        return

    # Phase 2: Execute tools (RAG search etc.)
    tool_results: List[Dict[str, Any]] = []
    allowed = set(allowed_tools) if allowed_tools else None

    for i, step in enumerate(plan):
        stype = step.get("type")
        if stype == "tool":
            tool_name = str(step.get("tool_name") or "")
            if not tool_name:
                continue
            if allowed is not None and tool_name not in allowed:
                continue
            
            args = step.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}
            
            # Force RAG defaults
            if tool_name == "rag.search":
                args = agent._force_rag_search_defaults(args)
            
            yield make_chunk(
                content=f"\n🔍 **执行工具**: {tool_name}\n",
                status="tool_calling",
                detail={"tool_name": tool_name, "step": i}
            )
            
            try:
                result = reg.call(tool_name, args)
                tool_results.append({
                    "step": i,
                    "tool_name": tool_name,
                    "arguments": args,
                    "result": result
                })
                
                # Show brief result info
                hits_count = len(result.get("hits", [])) if isinstance(result, dict) else 0
                yield make_chunk(
                    content=f"✅ 找到 {hits_count} 条相关信息\n",
                    status="tool_complete",
                    detail={"tool_name": tool_name, "hits_count": hits_count}
                )
            except Exception as e:
                yield make_chunk(content=f"⚠️ 工具调用失败: {e}\n")
        
        elif stype == "final":
            break

    # Phase 3: Generate final answer (streaming)
    yield make_chunk(
        content="\n💭 **正在生成回答...**\n\n",
        status="generating"
    )

    sys_prompt = (
        "你是一个面向兽医/动物健康方向的 AI 助手。"
        "请用中文回答，必要时引用你从工具返回的证据片段（简短引用即可）。"
        "如果证据不足，请明确说不确定，并给出下一步建议。"
    )
    if system_context:
        sys_prompt = f"{system_context}\n\n{sys_prompt}"

    user_content = json.dumps({
        "query": query,
        "plan": plan,
        "tool_results": tool_results,
    }, ensure_ascii=False)

    try:
        for chunk_text in stream_llm.chat_stream(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield make_chunk(content=chunk_text, status="streaming")
    except Exception as e:
        yield make_chunk(content=f"\n❌ 生成失败: {e}")

    yield make_chunk(finish="stop")
    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    """
    OpenAI-compatible chat completions endpoint.
    
    Supports both streaming (stream=true) and non-streaming modes.
    
    Models:
    - "agent-plan-solve": Single-turn plan-and-solve (backward compatible)
    - "agent-multi-turn": Multi-turn agent with iterative tool calls
    """
    trace_id = new_trace_id()
    request_id = _gen_id()
    created = _now_ts()
    
    query = _extract_user_query(req.messages)
    system_context = _build_system_context(req.messages)
    conversation_history = _build_conversation_history(req.messages)
    
    # Determine allowed tools from request
    allowed_tools = None
    if req.tools:
        allowed_tools = [t.get("function", {}).get("name") for t in req.tools if t.get("function")]
        allowed_tools = [n for n in allowed_tools if n]
    if not allowed_tools:
        allowed_tools = ["rag.search"]  # Default to RAG search

    # Choose agent mode based on model name
    use_multi_turn = req.model in ("agent-multi-turn", "agent-multiturn", "multi-turn")

    if req.stream:
        # Streaming response
        async def event_generator():
            if use_multi_turn:
                async for chunk in _stream_multi_turn_agent(
                    request_id=request_id,
                    model=req.model,
                    query=query,
                    system_context=system_context,
                    conversation_history=conversation_history,
                    temperature=req.temperature or 0.2,
                    max_tokens=req.max_tokens or 768,
                    allowed_tools=allowed_tools,
                ):
                    yield chunk
            else:
                async for chunk in _stream_plan_and_solve(
                    request_id=request_id,
                    model=req.model,
                    query=query,
                    system_context=system_context,
                    temperature=req.temperature or 0.2,
                    max_tokens=req.max_tokens or 768,
                    allowed_tools=allowed_tools,
                ):
                    yield chunk

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    
    else:
        # Non-streaming response (uses single-turn for simplicity)
        try:
            reg = get_registry()
            llm = OpenAICompatClient()
            agent = PlanAndSolveAgent(registry=reg, llm=llm)
            
            plan = agent.plan(query=query, allowed_tools=allowed_tools)
            answer, tool_results = agent.solve(
                query=query,
                plan_steps=plan,
                allowed_tools=allowed_tools,
                temperature=req.temperature or 0.2,
                max_tokens=req.max_tokens or 768,
            )
            
            response = ChatCompletionResponse(
                id=request_id,
                created=created,
                model=req.model,
                choices=[ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=answer),
                    finish_reason="stop",
                )],
                usage=UsageInfo(),
                plan=plan,
                tool_results=tool_results,
            )
            
            write_trace(trace_id, tool="v1.chat.completions", request=req.model_dump(), response=response.model_dump())
            return response
            
        except Exception as e:
            # Return error in OpenAI format
            error_response = {
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": "agent_error",
                }
            }
            write_trace(trace_id, tool="v1.chat.completions", request=req.model_dump(), response=error_response, error=str(e))
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=error_response)


@router.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "agent-plan-solve",
                "object": "model",
                "created": 1700000000,
                "owned_by": "pethealthai",
                "description": "Single-turn plan-and-solve agent",
            },
            {
                "id": "agent-multi-turn",
                "object": "model",
                "created": 1700000000,
                "owned_by": "pethealthai",
                "description": "Multi-turn agent with iterative tool calls",
            },
        ]
    }
