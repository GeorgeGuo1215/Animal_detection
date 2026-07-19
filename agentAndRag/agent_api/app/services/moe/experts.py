"""MoE 下游专家委员会：人设、授权工具子集与单专家会诊。

P1 约束：4 个专家共享现有 RAG 索引（无分库/物种元数据过滤），物种安全用人设
prompt 强约束；每个专家一次 rag.search（自有查询）+ 一次结构化意见 LLM 调用，
并行执行。专家工具子集复用 registry 的 allowed_tools 机制，便于后续接入各自 MCP。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...context.request_context import ANIMAL_REQUIRED_TOOLS, get_request_animal_id
from ...llm.llm_client import AsyncOpenAIClient, extract_text
from ...tools.tool_registry import ToolRegistry
from ..plan_and_solve import AsyncPlanAndSolveAgent, execute_tool_steps, _safe_json_loads
from .trace import MoETrace, extract_usage


# 每个专家最多执行的工具步数（含保底 rag.search），控制并行专家场景下的总调用量。
MAX_EXPERT_TOOL_STEPS = 3


@dataclass(frozen=True)
class ExpertConfig:
    key: str
    name_zh: str
    persona: str
    allowed_tools: List[str] = field(default_factory=list)
    rag_query_hint: str = ""


_SPECIES_GUARD = (
    "严格遵守物种安全：犬、猫的生理与药理差异巨大，禁止把某一物种的方案直接用于另一物种；"
    "若用户未说明物种，需提示该差异并按通用/谨慎口径作答。"
)

_OUTPUT_CONTRACT = (
    "你必须只输出严格 JSON（不要任何额外文字、不要 markdown 代码块），结构如下：\n"
    "{\n"
    '  "conclusion": "你的核心结论（中文，2-4 句）",\n'
    '  "evidence": ["支撑该结论的依据，尽量引用检索内容里的来源/页码"],\n'
    '  "risks": ["与本专业相关的风险提示或禁忌"],\n'
    '  "confidence": 0.0\n'
    "}\n"
    "confidence 为 0~1 的自评置信度：检索证据充分且与问题高度相关时高，证据不足时低。"
)


EXPERTS: Dict[str, ExpertConfig] = {
    "clinical": ExpertConfig(
        key="clinical",
        name_zh="兽医临床专家",
        persona=(
            "你是一位经验丰富的兽医临床专家，负责症状分诊、疾病鉴别与就医建议。"
            "你基于循证兽医学谨慎推断，区分『直接证据』与『临床经验推断』，"
            "不做确定性诊断，必要时建议线下就医与进一步检查。"
        ),
        allowed_tools=[
            "rag.search",
            "sql.search",
            "vitals.summary",
            "mcp.web_search.web_search",
        ],
        rag_query_hint="clinical signs differential diagnosis treatment",
    ),
    "nutrition": ExpertConfig(
        key="nutrition",
        name_zh="兽医营养专家",
        persona=(
            "你是一位兽医营养专家，负责膳食配方、体重与慢病饮食管理。"
            "你依据 NRC/AAFCO 等营养标准给出热量与配方建议，关注个体体重与病史。"
        ),
        allowed_tools=[
            "rag.search",
            "sql.search",
            "mcp.nutritional_planner.calculate_meal_plan",
            "mcp.nutritional_planner.generate_exercise_plan",
            "mcp.web_search.ingredient_check",
            "mcp.web_search.web_search",
        ],
        rag_query_hint="nutrition diet calorie requirement feeding",
    ),
    "pharmacy": ExpertConfig(
        key="pharmacy",
        name_zh="兽医药剂师",
        persona=(
            "你是一位兽医药剂师，负责用药安全、剂量、相互作用与禁忌。"
            "你尤其关注犬猫物种特异性毒性，对剂量与禁忌保持高度谨慎，"
            "对任何不确定的用药一律给出警示并建议遵医嘱。"
        ),
        allowed_tools=["rag.search", "mcp.web_search.web_search"],
        rag_query_hint="drug dosage contraindication toxicity interaction",
    ),
    "behavior": ExpertConfig(
        key="behavior",
        name_zh="行为安抚老师",
        persona=(
            "你是一位动物行为与安抚专家，负责焦虑/应激识别、行为矫正与宠主沟通。"
            "你用温和、可执行的方式给出训练与安抚建议，并照顾宠主的情绪与依从性。"
        ),
        allowed_tools=["rag.search", "mcp.web_search.web_search"],
        rag_query_hint="animal behavior anxiety stress training",
    ),
}


def _rag_metrics(result: Dict[str, Any]) -> tuple[int, float]:
    hits = result.get("hits") if isinstance(result, dict) else None
    if not isinstance(hits, list) or not hits:
        return 0, 0.0
    best = max((float(h.get("score", 0.0)) for h in hits if isinstance(h, dict)), default=0.0)
    return len(hits), best


def _build_evidence_block(result: Dict[str, Any], max_chars: int = 2400) -> str:
    """把 rag.search 命中拼成精简证据块，供专家阅读。"""
    if not isinstance(result, dict):
        return "（无检索结果）"
    hits = result.get("hits") or []
    if not hits:
        return "（知识库未命中相关内容）"
    parts: List[str] = []
    for i, h in enumerate(hits, start=1):
        if not isinstance(h, dict):
            continue
        src = h.get("source_path") or "unknown"
        text = (h.get("text") or "").strip()
        parts.append(f"[{i}] 来源: {src}\n{text}")
    block = "\n\n".join(parts)
    return block[:max_chars]


async def run_expert(
    *,
    expert: ExpertConfig,
    query: str,
    weight: float,
    registry: ToolRegistry,
    llm: AsyncOpenAIClient,
    rag_top_k: int = 5,
    device: Optional[str] = None,
    species_en: Optional[str] = None,
    species_zh: Optional[str] = None,
    recorder: Optional[MoETrace] = None,
) -> Dict[str, Any]:
    """单专家会诊：在授权工具子集内自主规划并调用工具（含 SQL/vitals/MCP），
    再据全部工具结果产出结构化意见。返回意见 dict。"""

    species_term = f" {species_en}" if species_en else ""
    rag_query = f"{query}{species_term} {expert.rag_query_hint}".strip()

    # 1a) 本专家可见工具子集：无 animal_id 时隐藏需绑定宠物的工具（sql.search / vitals.summary）
    visible_tools = list(expert.allowed_tools)
    if not get_request_animal_id():
        visible_tools = [t for t in visible_tools if t not in ANIMAL_REQUIRED_TOOLS]

    # 1b) 规划：在子集内决定调用哪些工具（planner LLM）
    tool_steps: List[Dict[str, Any]] = []
    try:
        planner = AsyncPlanAndSolveAgent(registry=registry, llm=llm)
        steps = await planner.plan(
            query=rag_query,
            allowed_tools=visible_tools,
            recorder=recorder,
            stage=f"expert:{expert.key}:plan",
        )
        # 仅保留属于本专家授权子集的工具步：planner 偶尔会建议越权/虚构工具，
        # 若直接交给 execute_tool_steps 会因 "Tool not allowed" 抛错并连带丢掉保底 rag。
        _visible = set(visible_tools)
        tool_steps = [
            s for s in steps
            if isinstance(s, dict) and s.get("type") == "tool" and str(s.get("tool_name") or "") in _visible
        ]
    except Exception:  # noqa: BLE001 - 规划失败时降级为仅保底 rag.search
        tool_steps = []

    # 1c) 保底 rag.search：确保 RAG 始终运行，并为 rag 步补默认查询/设备
    if not any(s.get("tool_name") == "rag.search" for s in tool_steps):
        tool_steps.insert(0, {"type": "tool", "tool_name": "rag.search", "arguments": {}})
    for s in tool_steps:
        if s.get("tool_name") == "rag.search":
            args = s.get("arguments")
            if not isinstance(args, dict):
                args = {}
            args.setdefault("query", rag_query)
            args.setdefault("top_k", rag_top_k)
            if device is not None:
                args.setdefault("device", device)
            s["arguments"] = args

    # 1d) 执行（封顶步数，rag 保底步在最前）
    tool_steps = tool_steps[:MAX_EXPERT_TOOL_STEPS]
    try:
        tool_results = await execute_tool_steps(
            registry=registry, plan_steps=tool_steps, allowed_tools=visible_tools
        )
    except Exception:  # noqa: BLE001
        tool_results = []

    # 1e) 汇总证据 + 记录 trace（rag → record_rag；其他工具 → record_tool）
    hits_count, best_score = 0, 0.0
    evidence_parts: List[str] = []
    tools_used: List[str] = []
    for tr in tool_results:
        name = tr.get("tool_name") or ""
        tools_used.append(name)
        result = tr.get("result")
        if name == "rag.search":
            hc, bs = _rag_metrics(result)
            hits_count += hc
            best_score = max(best_score, bs)
            if recorder is not None:
                recorder.record_rag(
                    stage=f"expert:{expert.key}",
                    query=str((tr.get("arguments") or {}).get("query") or rag_query),
                    hits_count=hc,
                    best_score=bs,
                    latency_ms=tr.get("latency_ms", 0.0),
                )
            evidence_parts.append(_build_evidence_block(result))
        else:
            if recorder is not None:
                recorder.record_tool(
                    stage=f"expert:{expert.key}",
                    tool_name=name,
                    arguments=tr.get("arguments") or {},
                    ok=bool(tr.get("ok", True)),
                    latency_ms=tr.get("latency_ms", 0.0),
                    error=tr.get("error"),
                )
            try:
                snippet = json.dumps(result, ensure_ascii=False)
            except Exception:  # noqa: BLE001
                snippet = str(result)
            evidence_parts.append(f"[工具 {name}] 结果：\n{snippet[:1500]}")

    tools_used = sorted({t for t in tools_used if t})
    evidence = "\n\n".join(p for p in evidence_parts if p) or "（无检索结果）"

    # 2) 结构化意见 LLM 调用
    sys_prompt = f"{expert.persona}\n{_SPECIES_GUARD}\n\n{_OUTPUT_CONTRACT}"
    expert_payload = {"user_question": query, "retrieved_evidence": evidence}
    if species_zh:
        expert_payload["species"] = species_zh
    user_payload = json.dumps(expert_payload, ensure_ascii=False)
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_payload},
    ]

    t0 = time.perf_counter()
    conclusion = ""
    evidence_list: List[str] = []
    risks: List[str] = []
    confidence = 0.0
    try:
        resp = await llm.chat(
            messages=messages,
            temperature=0.2,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        latency = (time.perf_counter() - t0) * 1000.0
        text = extract_text(resp)
        if recorder is not None:
            recorder.record_llm(
                stage=f"expert:{expert.key}",
                model=getattr(llm, "model", ""),
                messages=messages,
                output=text,
                latency_ms=latency,
                usage=extract_usage(resp),
                meta={"weight": weight, "rag_hits": hits_count},
            )
        obj, _err = _safe_json_loads(text)
        if isinstance(obj, dict):
            conclusion = str(obj.get("conclusion") or "").strip()
            ev = obj.get("evidence")
            evidence_list = [str(x) for x in ev] if isinstance(ev, list) else ([str(ev)] if ev else [])
            rk = obj.get("risks")
            risks = [str(x) for x in rk] if isinstance(rk, list) else ([str(rk)] if rk else [])
            try:
                confidence = max(0.0, min(1.0, float(obj.get("confidence", 0.0))))
            except (TypeError, ValueError):
                confidence = 0.0
        if not conclusion:
            conclusion = text.strip()[:800]
    except Exception as exc:  # noqa: BLE001
        latency = (time.perf_counter() - t0) * 1000.0
        if recorder is not None:
            recorder.record_llm(
                stage=f"expert:{expert.key}",
                model=getattr(llm, "model", ""),
                messages=messages,
                output=f"[error] {exc}",
                latency_ms=latency,
                meta={"weight": weight, "error": str(exc)},
            )
        conclusion = f"（该专家意见生成失败：{exc}）"

    return {
        "expert": expert.key,
        "name_zh": expert.name_zh,
        "weight": round(float(weight), 4),
        "conclusion": conclusion,
        "evidence": evidence_list,
        "risks": risks,
        "confidence": round(float(confidence), 3),
        "rag_hits": hits_count,
        "rag_best_score": round(float(best_score), 4),
        "tools_used": tools_used,
        "plan_steps": tool_steps,
        "tool_results": tool_results,
    }
