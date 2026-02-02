from __future__ import annotations

import os
import traceback
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .jobs import create_job, get_job
from .rag_tools import rag_reindex_tool, rag_search_tool, warmup_rag_cache
from .tool_registry import get_registry
from .tools_builtin import register_builtin_tools, register_debug_tools
from .tools_mcp import register_mcp_tools
from .schemas import (
    AgentPlanAndSolveRequest,
    AgentPlanAndSolveResponse,
    JobCreateResponse,
    JobStatusResponse,
    LoraEvalCheckpointsRequest,
    RagEvalSweepRequest,
    RagReindexRequest,
    RagSearchRequest,
    ToolCallRequest,
    ToolListResponse,
    ToolSpecOut,
    ToolResponse,
)
from .llm_client import OpenAICompatClient
from .plan_and_solve import PlanAndSolveAgent
from .trace_store import new_trace_id, write_trace


app = FastAPI(title="DeepSeek-OCR Agent Tools API", version="0.1.0")

# Frontend is a static page and may be served from a different origin (nginx/n8n).
# For early-stage deployment we allow CORS. You can tighten this later by setting
# AGENT_CORS_ORIGINS="https://your-domain,https://another-domain".
if os.getenv("AGENT_DISABLE_CORS", "0") != "1":
    _origins = [o.strip() for o in os.getenv("AGENT_CORS_ORIGINS", "*").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins if _origins != ["*"] else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        # Explicit headers to avoid wildcard incompatibility in some proxies/browsers
        allow_headers=["Content-Type", "Authorization", "Accept", "X-Requested-With"],
    )


@app.on_event("startup")
def _startup() -> None:
    # register tools once
    reg = get_registry()
    if reg.get("rag.search") is None:
        register_builtin_tools(reg)
        register_debug_tools(reg)
        if os.getenv("AGENT_ENABLE_MCP", "1") == "1":
            register_mcp_tools(reg)

    # Optional warmup to avoid slow first request.
    # Set env vars to enable:
    #   AGENT_WARMUP_RAG=1
    #   AGENT_WARMUP_BM25=1
    #   AGENT_WARMUP_RERANKER=0/1
    #   AGENT_WARMUP_DEVICE=cuda/cpu (optional)
    # Preload embedding + reranker on startup to avoid first-request latency.
    if os.getenv("AGENT_WARMUP_RAG", "1") == "1":
        from pathlib import Path
        from RAG.simple_rag.config import default_config

        repo_root = Path(__file__).resolve().parents[2]
        cfg = default_config(repo_root)
        warmup_rag_cache(
            index_dir=cfg.index_dir,
            embedding_model=os.getenv("AGENT_WARMUP_EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
            device=os.getenv("AGENT_WARMUP_DEVICE") or None,
            enable_bm25=os.getenv("AGENT_WARMUP_BM25", "1") == "1",
            enable_reranker=os.getenv("AGENT_WARMUP_RERANKER", "1") == "1",
            rerank_model=os.getenv("AGENT_WARMUP_RERANK_MODEL", "BAAI/bge-reranker-large"),
        )


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True}


@app.post("/tools/rag/search", response_model=ToolResponse)
def rag_search(req: RagSearchRequest) -> ToolResponse:
    trace_id = new_trace_id()
    try:
        data = rag_search_tool(**req.model_dump())
        resp = ToolResponse(ok=True, trace_id=trace_id, data=data)
        write_trace(trace_id, tool="rag.search", request=req.model_dump(), response=resp.model_dump())
        return resp
    except Exception as e:  # noqa: BLE001
        err = {"code": "RAG_SEARCH_FAILED", "message": str(e), "detail": {"traceback": traceback.format_exc()}}
        resp = ToolResponse(ok=False, trace_id=trace_id, error=err)  # type: ignore[arg-type]
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
    except Exception as e:  # noqa: BLE001
        err = {"code": "RAG_REINDEX_FAILED", "message": str(e), "detail": {"traceback": traceback.format_exc()}}
        resp = ToolResponse(ok=False, trace_id=trace_id, error=err)  # type: ignore[arg-type]
        write_trace(trace_id, tool="rag.reindex", request=req.model_dump(), response=resp.model_dump(), error=str(e))
        return resp


@app.get("/tools", response_model=ToolListResponse)
def tools_list() -> ToolListResponse:
    trace_id = new_trace_id()
    reg = get_registry()
    tools = [ToolSpecOut(name=t.name, description=t.description, input_schema=t.input_schema) for t in reg.list_tools()]
    resp = ToolListResponse(ok=True, trace_id=trace_id, tools=tools)
    write_trace(trace_id, tool="tools.list", request={}, response=resp.model_dump())
    return resp


@app.post("/tools/call", response_model=ToolResponse)
def tools_call(req: ToolCallRequest) -> ToolResponse:
    trace_id = new_trace_id()
    reg = get_registry()
    try:
        data = reg.call(req.tool_name, req.arguments)
        resp = ToolResponse(ok=True, trace_id=trace_id, data=data)
        write_trace(trace_id, tool="tools.call", request=req.model_dump(), response=resp.model_dump())
        return resp
    except Exception as e:  # noqa: BLE001
        err = {"code": "TOOL_CALL_FAILED", "message": str(e), "detail": {"traceback": traceback.format_exc()}}
        resp = ToolResponse(ok=False, trace_id=trace_id, error=err)  # type: ignore[arg-type]
        write_trace(trace_id, tool="tools.call", request=req.model_dump(), response=resp.model_dump(), error=str(e))
        return resp


@app.post("/agent/plan_and_solve", response_model=AgentPlanAndSolveResponse)
def agent_plan_and_solve(req: AgentPlanAndSolveRequest) -> AgentPlanAndSolveResponse:
    trace_id = new_trace_id()
    try:
        reg = get_registry()
        llm = OpenAICompatClient(base_url=req.llm_base_url, api_key=req.llm_api_key, model=req.llm_model)
        agent = PlanAndSolveAgent(registry=reg, llm=llm)
        plan = agent.plan(query=req.query, allowed_tools=req.allowed_tools)
        answer, tool_results = agent.solve(
            query=req.query,
            plan_steps=plan,
            allowed_tools=req.allowed_tools,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        resp = AgentPlanAndSolveResponse(ok=True, trace_id=trace_id, answer=answer, plan=plan, tool_results=tool_results)
        write_trace(trace_id, tool="agent.plan_and_solve", request=req.model_dump(), response=resp.model_dump())
        return resp
    except Exception as e:  # noqa: BLE001
        err = {"code": "AGENT_PLAN_SOLVE_FAILED", "message": str(e), "detail": {"traceback": traceback.format_exc()}}
        resp = AgentPlanAndSolveResponse(ok=False, trace_id=trace_id, error=err)  # type: ignore[arg-type]
        write_trace(trace_id, tool="agent.plan_and_solve", request=req.model_dump(), response=resp.model_dump(), error=str(e))
        return resp


@app.post("/tools/rag/eval_sweep", response_model=JobCreateResponse)
def rag_eval_sweep(req: RagEvalSweepRequest) -> JobCreateResponse:
    """
    最小可用：先起一个后台 job，避免阻塞 n8n。
    当前实现不注入 experiments/CONFIG，只是一个“占位 wrapper”。
    后续再把 CONFIG 参数化，真正做到 n8n 可配置 sweep。
    """

    trace_id = new_trace_id()

    def _run() -> Dict[str, Any]:
        # 先留空，避免误跑很久；后续按你的论文设计把 CONFIG 变成请求参数
        return {"note": req.note, "status": "not_implemented_yet"}

    job = create_job(_run)
    write_trace(trace_id, tool="rag.eval_sweep", request=req.model_dump(), response={"job_id": job.job_id})
    return JobCreateResponse(ok=True, trace_id=trace_id, job_id=job.job_id)


@app.post("/tools/lora/eval_checkpoints", response_model=JobCreateResponse)
def lora_eval_checkpoints(req: LoraEvalCheckpointsRequest) -> JobCreateResponse:
    """
    最小可用：后台 job 运行 RAG/tools/eval_lora_checkpoints_metrics.py 逻辑（已实现为脚本）。
    后续可升级为：直接 import 并返回结构化结果 + 支持并行/缓存。
    """

    trace_id = new_trace_id()

    def _run() -> Dict[str, Any]:
        # 这里先用“子进程”方式，避免在 API 进程里长时间占用 GPU/内存
        import subprocess
        import sys

        repo_root = __import__("pathlib").Path(__file__).resolve().parents[2]
        script = repo_root / "RAG" / "tools" / "eval_lora_checkpoints_metrics.py"
        cmd = [
            sys.executable,
            str(script),
            "--model_id",
            req.model_id,
            "--output_dir",
            req.output_dir,
            "--checkpoints",
            ",".join(req.checkpoints),
            "--device_map",
            req.device_map,
            "--max_new_tokens",
            str(req.max_new_tokens),
            "--bertscore_lang",
            req.bertscore_lang,
        ]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
        if p.returncode != 0:
            raise RuntimeError(f"eval_checkpoints failed: {p.stderr}\n{p.stdout}")
        # 结果默认写到 output_dir/metrics.csv
        return {"stdout": p.stdout[-4000:], "metrics_csv": str((repo_root / req.output_dir / "metrics.csv").resolve())}

    job = create_job(_run)
    write_trace(trace_id, tool="lora.eval_checkpoints", request=req.model_dump(), response={"job_id": job.job_id})
    return JobCreateResponse(ok=True, trace_id=trace_id, job_id=job.job_id)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str) -> JobStatusResponse:
    trace_id = new_trace_id()
    job = get_job(job_id)
    if not job:
        return JobStatusResponse(
            ok=False, trace_id=trace_id, job_id=job_id, status="failed", error="job not found", result=None
        )
    return JobStatusResponse(
        ok=True,
        trace_id=trace_id,
        job_id=job_id,
        status=job.status,
        result=job.result,
        error=job.error,
    )


