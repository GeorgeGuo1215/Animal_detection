"""MoE 编排器：串联 Router → 并行加权专家 → Critic → 流式融合生成。

两个入口共享同一组内部阶段函数：
- `stream(...)`  生产路径：异步产出事件 dict（content/status/detail/finish），仅最终答案 token 流式；
- `run(...)`     评测路径：非流式跑完，返回 (final_answer, MoETrace)。
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from ...llm.llm_client import AsyncOpenAIClient, extract_text, get_shared_async_client
from ...llm.llm_client_stream import AsyncOpenAIStreamClient, get_shared_async_stream_client
from ...tools.tool_registry import ToolRegistry, get_registry
from ..plan_and_solve import build_solve_prompt
from .critic import CriticResult, review
from .experts import EXPERTS, run_expert
from .router import RouterConfig, RouterDecision, route
from .trace import MoETrace, extract_usage
from ...sql_search import fetch_animal_profile, species_label
from ...context.request_context import get_request_animal_id


_OUT_OF_SCOPE_TEXT = (
    "抱歉，这个问题似乎与宠物的健康、养护、营养、用药或行为无关，"
    "我是宠物健康助手，暂时无法回答。如果你有关于猫狗等宠物健康的问题，欢迎随时问我～"
)

_BLOCK_FALLBACK_TEXT = (
    "出于安全考虑，我不能直接给出该建议。你描述的情况可能涉及较高风险，"
    "强烈建议尽快联系或前往专业兽医进行线下评估与处理。\n\n"
    "**免责声明**：以上内容仅供健康管理参考，不能替代执业兽医的诊断与治疗。"
)


@dataclass
class OrchestratorConfig:
    router: RouterConfig = field(default_factory=RouterConfig)
    rag_top_k: int = 5
    temperature: float = 0.3
    max_tokens: int = 900
    user_role: str = "pet_owner"
    device: Optional[str] = None
    animal_id: Optional[str] = None


def _event(content: str = "", status: Optional[str] = None,
           detail: Optional[Dict[str, Any]] = None, finish: Optional[str] = None) -> Dict[str, Any]:
    return {"content": content, "status": status, "detail": detail, "finish": finish}


class MoEOrchestrator:
    def __init__(
        self,
        *,
        registry: Optional[ToolRegistry] = None,
        llm: Optional[AsyncOpenAIClient] = None,
        stream_llm: Optional[AsyncOpenAIStreamClient] = None,
        config: Optional[OrchestratorConfig] = None,
    ) -> None:
        self.registry = registry or get_registry()
        self.llm = llm or get_shared_async_client()
        self.stream_llm = stream_llm or get_shared_async_stream_client()
        self.config = config or OrchestratorConfig()

    # ------------------------------------------------------------------ stages
    def _resolve_species(self):
        # Resolve (species_en, species_zh) for the request-scoped animal; (None, None) if unknown.
        animal_id = self.config.animal_id or get_request_animal_id()
        if not animal_id:
            return None, None
        profile = fetch_animal_profile(animal_id)
        if not profile:
            return None, None
        return (profile.get("species") or None), species_label(profile)

    async def _route(self, query: str, recorder: Optional[MoETrace]) -> RouterDecision:
        _, species_zh = self._resolve_species()
        return await route(
            query=query,
            user_role=self.config.user_role,
            llm=self.llm,
            config=self.config.router,
            species_zh=species_zh,
            recorder=recorder,
        )

    async def _run_experts(
        self, query: str, decision: RouterDecision, recorder: Optional[MoETrace]
    ) -> List[Dict[str, Any]]:
        species_en, species_zh = self._resolve_species()
        tasks = []
        for key in decision.selected_experts:
            expert = EXPERTS.get(key)
            if expert is None:
                continue
            weight = decision.weights.get(key, 0.0)
            tasks.append(
                run_expert(
                    expert=expert,
                    query=query,
                    weight=weight,
                    registry=self.registry,
                    llm=self.llm,
                    rag_top_k=self.config.rag_top_k,
                    device=self.config.device,
                    species_en=species_en,
                    species_zh=species_zh,
                    recorder=recorder,
                )
            )
        if not tasks:
            return []
        opinions = await asyncio.gather(*tasks)
        opinions = list(opinions)
        opinions.sort(key=lambda o: o.get("weight", 0.0), reverse=True)
        if recorder is not None:
            recorder.expert_opinions = opinions
        return opinions

    async def _critique(
        self, query: str, opinions: List[Dict[str, Any]], emergency: bool, recorder: Optional[MoETrace]
    ) -> CriticResult:
        return await review(
            query=query,
            expert_opinions=opinions,
            emergency=emergency,
            llm=self.llm,
            recorder=recorder,
        )

    def _build_synthesis_messages(
        self,
        *,
        query: str,
        opinions: List[Dict[str, Any]],
        critic: CriticResult,
        decision: RouterDecision,
        system_context: str = "",
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        base = build_solve_prompt(user_role=self.config.user_role, has_web_search=False, query=query)
        agg = (
            "\n\n**多专家融合规范**\n"
            "下面是多位兽医专家针对该问题给出的加权意见（weight 越高越重要）。请你作为融合器：\n"
            "- 按权重与各专家自评置信度综合，形成一致、连贯的最终答复；\n"
            "- 显式标注专家间的冲突点（若有），不要简单拼接；\n"
            "- 用如下结构分节（加粗文字，不用 Markdown 标题）：**结论** / **依据** / **风险提示** / **建议行动** / **何时必须就医**；\n"
        )
        if decision.emergency:
            agg += "- 当前疑似急症：务必把『立即就医』放在最前并加粗强调。\n"
        if critic.constraints:
            agg += "\n**审核专家（Critic）下达的硬性约束，必须全部满足：**\n"
            agg += "\n".join(f"- {c}" for c in critic.constraints)
        sys_prompt = base + agg
        if system_context:
            sys_prompt = f"{system_context}\n\n{sys_prompt}"

        payload = {
            "query": query,
            "router": {"weights": decision.weights, "emergency": decision.emergency},
            "expert_opinions": opinions,
            "critic_verdict": critic.verdict,
        }
        parts: List[str] = []
        if conversation_history:
            parts.append("历史对话:\n" + "\n".join(
                f"{m['role']}: {m['content']}" for m in conversation_history[-6:]
            ))
        parts.append(json.dumps(payload, ensure_ascii=False))
        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "\n\n".join(parts)},
        ]

    # ------------------------------------------------------------------ stream
    async def stream(
        self,
        *,
        query: str,
        system_context: str = "",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        recorder: Optional[MoETrace] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        # 1) 路由
        yield _event(status="routing", detail={"message": "正在分诊与路由…"})
        decision = await self._route(query, recorder)

        if decision.out_of_scope:
            if recorder is not None:
                recorder.out_of_scope = True
                recorder.final_answer = _OUT_OF_SCOPE_TEXT
                recorder.finalize()
            yield _event(
                status="routing",
                detail={"out_of_scope": True, "reason": decision.reason, "scores": decision.scores},
            )
            yield _event(content=_OUT_OF_SCOPE_TEXT, status="streaming")
            return

        yield _event(
            content=f"\n**路由完成**：{', '.join(EXPERTS[k].name_zh for k in decision.selected_experts)}\n",
            status="routing",
            detail={
                "weights": decision.weights,
                "raw_weights": decision.raw_weights,
                "scores": decision.scores,
                "selected_experts": decision.selected_experts,
                "emergency": decision.emergency,
                "reason": decision.reason,
            },
        )

        # 2) 并行专家会诊
        for key in decision.selected_experts:
            yield _event(
                content=f"\n**{EXPERTS[key].name_zh} 会诊中**（权重 {decision.weights.get(key, 0):.2f}）\n",
                status="expert_calling",
                detail={"expert": key, "name_zh": EXPERTS[key].name_zh, "weight": decision.weights.get(key, 0)},
            )
        opinions = await self._run_experts(query, decision, recorder)
        for o in opinions:
            _tu = o.get("tools_used") or []
            _tools_txt = f"，工具 {', '.join(_tu)}" if _tu else ""
            yield _event(
                content=f"   {o['name_zh']}：置信度 {o['confidence']:.2f}，RAG 命中 {o['rag_hits']}{_tools_txt}\n",
                status="expert_complete",
                detail={
                    "expert": o["expert"],
                    "name_zh": o["name_zh"],
                    "weight": o.get("weight"),
                    "confidence": o["confidence"],
                    "hits_count": o["rag_hits"],
                    "best_score": o["rag_best_score"],
                    "tools_used": o.get("tools_used", []),
                    "opinion": o,
                },
            )

        # 3) Critic 审核
        yield _event(content="\n**边界审核中…**\n", status="reviewing", detail={"message": "安全与边界校验"})
        critic = await self._critique(query, opinions, decision.emergency, recorder)
        yield _event(
            status="reviewing",
            detail={"verdict": critic.verdict, "issues": critic.issues, "reason": critic.reason},
        )

        if critic.blocked:
            if recorder is not None:
                recorder.blocked = True
                recorder.final_answer = _BLOCK_FALLBACK_TEXT
                recorder.finalize()
            yield _event(content="\n**生成回答…**\n\n", status="generating")
            yield _event(content=_BLOCK_FALLBACK_TEXT, status="streaming")
            return

        # 4) 流式融合生成
        yield _event(content="\n**生成回答…**\n\n", status="generating")
        messages = self._build_synthesis_messages(
            query=query, opinions=opinions, critic=critic, decision=decision,
            system_context=system_context, conversation_history=conversation_history,
        )
        collected: List[str] = []
        t0 = time.perf_counter()
        try:
            async for piece in self.stream_llm.chat_stream(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            ):
                collected.append(piece)
                yield _event(content=piece, status="streaming")
        except Exception as exc:  # noqa: BLE001
            yield _event(content=f"\n生成失败：{exc}")
        latency = (time.perf_counter() - t0) * 1000.0
        final_answer = "".join(collected)
        if recorder is not None:
            recorder.record_llm(
                stage="aggregator",
                model=getattr(self.stream_llm, "model", ""),
                messages=messages,
                output=final_answer,
                latency_ms=latency,
                meta={"streamed": True},
            )
            recorder.final_answer = final_answer
            recorder.finalize()

    # --------------------------------------------------------------- non-stream
    async def run(
        self,
        *,
        query: str,
        system_context: str = "",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        recorder: Optional[MoETrace] = None,
    ) -> Tuple[str, Optional[MoETrace]]:
        decision = await self._route(query, recorder)

        if decision.out_of_scope:
            if recorder is not None:
                recorder.out_of_scope = True
                recorder.final_answer = _OUT_OF_SCOPE_TEXT
                recorder.finalize()
            return _OUT_OF_SCOPE_TEXT, recorder

        opinions = await self._run_experts(query, decision, recorder)
        critic = await self._critique(query, opinions, decision.emergency, recorder)

        if critic.blocked:
            if recorder is not None:
                recorder.blocked = True
                recorder.final_answer = _BLOCK_FALLBACK_TEXT
                recorder.finalize()
            return _BLOCK_FALLBACK_TEXT, recorder

        messages = self._build_synthesis_messages(
            query=query, opinions=opinions, critic=critic, decision=decision,
            system_context=system_context, conversation_history=conversation_history,
        )
        t0 = time.perf_counter()
        resp = await self.llm.chat(
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        latency = (time.perf_counter() - t0) * 1000.0
        final_answer = extract_text(resp)
        if recorder is not None:
            recorder.record_llm(
                stage="aggregator",
                model=getattr(self.llm, "model", ""),
                messages=messages,
                output=final_answer,
                latency_ms=latency,
                usage=extract_usage(resp),
                meta={"streamed": False},
            )
            recorder.final_answer = final_answer
            recorder.finalize()
        return final_answer, recorder
