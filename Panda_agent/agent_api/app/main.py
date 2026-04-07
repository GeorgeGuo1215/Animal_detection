from __future__ import annotations

import os
import traceback
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from .auth import APIKeyAuthMiddleware, load_api_keys
from .qa_store import (
    get_feedback_stats, get_knowledge_gaps, get_qa_stats,
    init_db, query_qa_history, submit_feedback,
)
from .rate_limit import RateLimitMiddleware
from .session_manager import get_session_manager
from .rag_tools import rag_reindex_tool, rag_search_tool, warmup_rag_cache
from .tool_registry import get_registry
from .tools_builtin import register_builtin_tools, register_debug_tools
from .tools_mcp import register_mcp_tools
from .routes_openai import router as openai_router
from .routes_chat_ui import router as chat_ui_router
from .schemas import (
    AgentPlanAndSolveRequest, AgentPlanAndSolveResponse,
    RagReindexRequest, RagSearchRequest, ToolCallRequest,
    ToolListResponse, ToolSpecOut, ToolResponse,
)
from .llm_client import AsyncOpenAIClient
from .plan_and_solve import AsyncPlanAndSolveAgent
from .trace_store import new_trace_id, write_trace


app = FastAPI(title="Panda Mind Agent API", version="1.0.0",
              description="Giant Panda Knowledge Q&A System — RAG + Agent")

app.include_router(openai_router)
app.include_router(chat_ui_router)

app.add_middleware(APIKeyAuthMiddleware)

_rl_rate = float(os.getenv("AGENT_RATE_LIMIT", "30"))
_rl_burst = int(os.getenv("AGENT_RATE_BURST", str(int(_rl_rate))))
app.add_middleware(RateLimitMiddleware, rate=_rl_rate, burst=_rl_burst)

if os.getenv("AGENT_ENABLE_CORS", "0") == "1":
    _origins = [o.strip() for o in os.getenv("AGENT_CORS_ORIGINS", "*").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins if _origins != ["*"] else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["Content-Type", "Authorization", "Accept", "X-Requested-With"],
    )


@app.on_event("startup")
def _startup() -> None:
    load_api_keys()
    init_db()

    reg = get_registry()
    if reg.get("rag.search") is None:
        register_builtin_tools(reg)
        register_debug_tools(reg)
        if os.getenv("AGENT_ENABLE_MCP", "1") == "1":
            register_mcp_tools(reg)

    if os.getenv("AGENT_WARMUP_RAG", "1") == "1":
        from pathlib import Path
        from RAG.simple_rag.config import default_config

        repo_root = Path(__file__).resolve().parents[2]
        cfg = default_config(repo_root)
        device = os.getenv("AGENT_WARMUP_DEVICE") or None
        embedding_model = os.getenv("AGENT_WARMUP_EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
        rerank_model = os.getenv("AGENT_WARMUP_RERANK_MODEL", "BAAI/bge-reranker-large")
        enable_bm25 = os.getenv("AGENT_WARMUP_BM25", "1") == "1"
        enable_reranker = os.getenv("AGENT_WARMUP_RERANKER", "1") == "1"

        try:
            stats = warmup_rag_cache(
                index_dir=cfg.index_dir, embedding_model=embedding_model, device=device,
                enable_bm25=enable_bm25, enable_reranker=enable_reranker, rerank_model=rerank_model,
            )

            actual_device = device or "cpu"
            try:
                import torch
                if torch.cuda.is_available() and actual_device != "cpu":
                    gpu_name = torch.cuda.get_device_name(0)
                    print(f"[startup] Device: {actual_device} ({gpu_name})")
                else:
                    print(f"[startup] Device: cpu")
            except ImportError:
                print(f"[startup] Device: {actual_device}")
            print(f"[startup] Embedding: {embedding_model}")
            print(f"[startup] Reranker: {rerank_model if enable_reranker else 'disabled'}")
            print(f"[startup] BM25: {'enabled' if enable_bm25 else 'disabled'}")
            print(f"[startup] Index: {stats['index_size']} chunks from panda books")
        except Exception as exc:  # noqa: BLE001
            print("[startup] RAG warmup failed; continuing without preloaded cache.")
            print(f"[startup] Warmup error: {exc}")
            print(traceback.format_exc())


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "Panda Mind"}


@app.post("/tools/rag/search", response_model=ToolResponse)
def rag_search(req: RagSearchRequest) -> ToolResponse:
    trace_id = new_trace_id()
    try:
        data = rag_search_tool(**req.model_dump())
        resp = ToolResponse(ok=True, trace_id=trace_id, data=data)
        write_trace(trace_id, tool="rag.search", request=req.model_dump(), response=resp.model_dump())
        return resp
    except Exception as e:
        err = {"code": "RAG_SEARCH_FAILED", "message": str(e), "detail": {"traceback": traceback.format_exc()}}
        resp = ToolResponse(ok=False, trace_id=trace_id, error=err)
        write_trace(trace_id, tool="rag.search", request=req.model_dump(), response=resp.model_dump(), error=str(e))
        return resp


@app.post("/tools/rag/reindex", response_model=ToolResponse)
def rag_reindex(req: RagReindexRequest) -> ToolResponse:
    trace_id = new_trace_id()
    try:
        data = rag_reindex_tool(**req.model_dump())
        resp = ToolResponse(ok=True, trace_id=trace_id, data=data)
        write_trace(trace_id, tool="rag.reindex", request=req.model_dump(), response=resp.model_dump())
        return resp
    except Exception as e:
        err = {"code": "RAG_REINDEX_FAILED", "message": str(e), "detail": {"traceback": traceback.format_exc()}}
        resp = ToolResponse(ok=False, trace_id=trace_id, error=err)
        write_trace(trace_id, tool="rag.reindex", request=req.model_dump(), response=resp.model_dump(), error=str(e))
        return resp


@app.get("/tools", response_model=ToolListResponse)
def tools_list() -> ToolListResponse:
    trace_id = new_trace_id()
    reg = get_registry()
    tools = [ToolSpecOut(name=t.name, description=t.description, input_schema=t.input_schema) for t in reg.list_tools()]
    return ToolListResponse(ok=True, trace_id=trace_id, tools=tools)


@app.post("/tools/call", response_model=ToolResponse)
async def tools_call(req: ToolCallRequest) -> ToolResponse:
    trace_id = new_trace_id()
    reg = get_registry()
    try:
        data = await reg.call(req.tool_name, req.arguments)
        resp = ToolResponse(ok=True, trace_id=trace_id, data=data)
        write_trace(trace_id, tool="tools.call", request=req.model_dump(), response=resp.model_dump())
        return resp
    except Exception as e:
        err = {"code": "TOOL_CALL_FAILED", "message": str(e), "detail": {"traceback": traceback.format_exc()}}
        resp = ToolResponse(ok=False, trace_id=trace_id, error=err)
        write_trace(trace_id, tool="tools.call", request=req.model_dump(), response=resp.model_dump(), error=str(e))
        return resp


@app.post("/agent/plan_and_solve", response_model=AgentPlanAndSolveResponse)
async def agent_plan_and_solve(req: AgentPlanAndSolveRequest) -> AgentPlanAndSolveResponse:
    trace_id = new_trace_id()
    try:
        reg = get_registry()
        llm = AsyncOpenAIClient(base_url=req.llm_base_url, api_key=req.llm_api_key, model=req.llm_model)
        agent = AsyncPlanAndSolveAgent(registry=reg, llm=llm)
        plan = await agent.plan(query=req.query, allowed_tools=req.allowed_tools)
        answer, tool_results = await agent.solve(
            query=req.query, plan_steps=plan, allowed_tools=req.allowed_tools,
            temperature=req.temperature, max_tokens=req.max_tokens,
        )
        resp = AgentPlanAndSolveResponse(ok=True, trace_id=trace_id, answer=answer, plan=plan, tool_results=tool_results)
        write_trace(trace_id, tool="agent.plan_and_solve", request=req.model_dump(), response=resp.model_dump())
        return resp
    except Exception as e:
        err = {"code": "AGENT_PLAN_SOLVE_FAILED", "message": str(e), "detail": {"traceback": traceback.format_exc()}}
        resp = AgentPlanAndSolveResponse(ok=False, trace_id=trace_id, error=err)
        write_trace(trace_id, tool="agent.plan_and_solve", request=req.model_dump(), response=resp.model_dump(), error=str(e))
        return resp


@app.post("/sessions")
async def create_session():
    mgr = get_session_manager()
    sess = await mgr.create()
    return {"ok": True, "session_id": sess.session_id}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    mgr = get_session_manager()
    sess = await mgr.get(session_id)
    if not sess:
        return {"ok": False, "error": "session not found or expired"}
    return {"ok": True, "session_id": sess.session_id, "messages": sess.messages,
            "created_at": sess.created_at, "last_active": sess.last_active}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    mgr = get_session_manager()
    deleted = await mgr.delete(session_id)
    return {"ok": deleted}


# ---------------------------------------------------------------------------
# QA management endpoints — localhost only
# ---------------------------------------------------------------------------

_QA_ADMIN_TOKEN = os.getenv("QA_ADMIN_TOKEN", "")


async def _require_admin(request: Request) -> None:
    """Verify admin access via X-Admin-Token header.

    Separate from the general API key auth so it is never bypassed
    by AGENT_DISABLE_AUTH.  Rejects requests without the correct token
    with 403 — external users through frp do not know this token.
    """
    if not _QA_ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="QA admin token not configured on server")
    token = request.headers.get("X-Admin-Token", "").strip()
    if token != _QA_ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing admin token")


@app.get("/qa/history", dependencies=[Depends(_require_admin)])
async def qa_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    date_from: Optional[str] = Query(None, description="YYYY-MM-DD"),
    date_to: Optional[str] = Query(None, description="YYYY-MM-DD"),
    keyword: Optional[str] = Query(None),
) -> Dict[str, Any]:
    data = await query_qa_history(
        page=page, page_size=page_size,
        date_from=date_from, date_to=date_to, keyword=keyword,
    )
    return {"ok": True, **data}


@app.get("/qa/stats", dependencies=[Depends(_require_admin)])
async def qa_stats(
    date_from: Optional[str] = Query(None, description="YYYY-MM-DD"),
    date_to: Optional[str] = Query(None, description="YYYY-MM-DD"),
) -> Dict[str, Any]:
    data = await get_qa_stats(date_from=date_from, date_to=date_to)
    return {"ok": True, **data}


@app.get("/qa/knowledge-gaps", dependencies=[Depends(_require_admin)])
async def qa_knowledge_gaps(
    date_from: Optional[str] = Query(None, description="YYYY-MM-DD"),
    date_to: Optional[str] = Query(None, description="YYYY-MM-DD"),
    min_occurrences: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
) -> Dict[str, Any]:
    data = await get_knowledge_gaps(
        date_from=date_from, date_to=date_to,
        min_occurrences=min_occurrences, limit=limit,
    )
    return {"ok": True, **data}


# ---------------------------------------------------------------------------
# Feedback endpoints
# ---------------------------------------------------------------------------

@app.post("/qa/feedback")
async def qa_feedback(request: Request) -> Dict[str, Any]:
    """Public endpoint — any user can submit feedback for an answer."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    request_id = (body.get("request_id") or "").strip()
    rating = body.get("rating")
    comment = (body.get("comment") or "").strip()
    if not request_id:
        raise HTTPException(status_code=400, detail="request_id is required")
    if not isinstance(rating, int) or rating < 1 or rating > 5:
        raise HTTPException(status_code=400, detail="rating must be an integer between 1 and 5")
    ok = await submit_feedback(request_id=request_id, rating=rating, comment=comment)
    if not ok:
        raise HTTPException(status_code=404, detail="Record not found or already rated")
    return {"ok": True}


@app.get("/qa/feedback-stats", dependencies=[Depends(_require_admin)])
async def qa_feedback_stats(
    date_from: Optional[str] = Query(None, description="YYYY-MM-DD"),
    date_to: Optional[str] = Query(None, description="YYYY-MM-DD"),
) -> Dict[str, Any]:
    data = await get_feedback_stats(date_from=date_from, date_to=date_to)
    return {"ok": True, **data}
