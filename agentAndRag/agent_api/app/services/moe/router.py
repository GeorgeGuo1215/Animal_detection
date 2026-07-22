"""MoE 路由器：规则粗筛 + LLM 软路由权重 + 门控选专家 + 急症硬规则。

设计要点（与项目文档一致，全程 MoE、无独立"简单回复"分支）：
- LLM 对 4 个专家各打 0~10 相关性分数；
- 若最高分 < `min_relevance` 且非急症 → 判定与宠物健康无关 → `out_of_scope`（拒答）；
- 否则对分数做 softmax 得权重，激活 `weight >= threshold` 的专家（上限 `max_experts`），
  命中 1 个即单专家、命中多个即加权委员会；选中后权重重归一化；
- 急症关键词命中 → 强制激活临床+药剂（覆盖门控），并置 `emergency=True`。
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ...llm.llm_client import AsyncOpenAIClient, extract_text
from ..plan_and_solve import _safe_json_loads
from .experts import EXPERTS
from .trace import MoETrace, extract_usage


_EMERGENCY_RE = re.compile(
    r"(抽搐|痉挛|昏迷|休克|大出血|出血不止|呼吸困难|窒息|喘不上气|中毒|误食|误吞|"
    r"中暑|热射病|无法站立|瘫痪|大量呕吐|持续呕吐|血便|血尿|难产|车祸|坠楼|骨折|"
    r"seizure|convulsion|unconscious|coma|shock|bleeding|choking|poison|toxic|"
    r"heatstroke|paralysis|can'?t breathe|emergency)",
    re.IGNORECASE,
)

# 急症强制激活的核心专家
_EMERGENCY_EXPERTS = ["clinical", "pharmacy"]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


@dataclass
class RouterConfig:
    gating_threshold: float = field(default_factory=lambda: _env_float("MOE_GATING_THRESHOLD", 0.15))
    max_experts: int = field(default_factory=lambda: _env_int("MOE_MAX_EXPERTS", 3))
    min_relevance: float = field(default_factory=lambda: _env_float("MOE_MIN_RELEVANCE", 3.0))
    softmax_temp: float = field(default_factory=lambda: _env_float("MOE_SOFTMAX_TEMP", 1.0))


@dataclass
class RouterDecision:
    scores: Dict[str, float]          # LLM 原始相关性分（0~10）
    raw_weights: Dict[str, float]     # 全部专家 softmax 权重
    weights: Dict[str, float]         # 选中专家的重归一化权重
    selected_experts: List[str]
    emergency: bool
    out_of_scope: bool
    reason: str


def _softmax(scores: Dict[str, float], temp: float) -> Dict[str, float]:
    keys = list(scores.keys())
    temp = max(1e-6, float(temp))
    vals = [scores[k] / temp for k in keys]
    m = max(vals) if vals else 0.0
    exps = [math.exp(v - m) for v in vals]
    s = sum(exps) or 1.0
    return {k: exps[i] / s for i, k in enumerate(keys)}


def _build_router_messages(
    query: str,
    user_role: str,
    species_zh: Optional[str] = None,
    breed: Optional[str] = None,
) -> List[Dict[str, str]]:
    expert_desc = "\n".join(f"- {k} ({c.name_zh})" for k, c in EXPERTS.items())
    sys = (
        "你是宠物健康多专家系统的路由器。给定用户问题，为每位专家打 0~10 的相关性分数"
        "（该专家对回答此问题的贡献度），并判断是否为危及生命的急症。\n"
        f"专家列表：\n{expert_desc}\n\n"
        "重要：如果问题与宠物健康/养护/行为/营养/用药完全无关（例如闲聊、编程、时事、人类医学等），"
        "请把所有专家分数都打到 0~2 的低分。\n"
        "你必须只输出严格 JSON（无额外文字、无代码块），结构：\n"
        '{\n'
        '  "scores": {"clinical": 0, "nutrition": 0, "pharmacy": 0, "behavior": 0},\n'
        '  "emergency": false,\n'
        '  "reason": "简要中文说明"\n'
        "}"
    )
    payload = {"user_question": query, "user_role": user_role}
    if species_zh:
        payload["species"] = species_zh
    if breed:
        payload["breed"] = breed
    user = json.dumps(payload, ensure_ascii=False)
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]


async def route(
    *,
    query: str,
    user_role: str,
    llm: AsyncOpenAIClient,
    config: Optional[RouterConfig] = None,
    species_zh: Optional[str] = None,
    breed: Optional[str] = None,
    recorder: Optional[MoETrace] = None,
) -> RouterDecision:
    cfg = config or RouterConfig()
    expert_keys = list(EXPERTS.keys())

    emergency_rule = bool(_EMERGENCY_RE.search(query or ""))

    messages = _build_router_messages(query, user_role, species_zh, breed)
    scores: Dict[str, float] = {k: 0.0 for k in expert_keys}
    emergency_llm = False
    reason = ""

    t0 = time.perf_counter()
    try:
        resp = await llm.chat(
            messages=messages,
            temperature=0.1,
            max_tokens=256,
            response_format={"type": "json_object"},
        )
        latency = (time.perf_counter() - t0) * 1000.0
        text = extract_text(resp)
        if recorder is not None:
            recorder.record_llm(
                stage="router",
                model=getattr(llm, "model", ""),
                messages=messages,
                output=text,
                latency_ms=latency,
                usage=extract_usage(resp),
            )
        obj, _err = _safe_json_loads(text)
        if isinstance(obj, dict):
            raw_scores = obj.get("scores") or {}
            if isinstance(raw_scores, dict):
                for k in expert_keys:
                    try:
                        scores[k] = max(0.0, min(10.0, float(raw_scores.get(k, 0.0))))
                    except (TypeError, ValueError):
                        scores[k] = 0.0
            emergency_llm = bool(obj.get("emergency", False))
            reason = str(obj.get("reason") or "")
    except Exception as exc:  # noqa: BLE001
        latency = (time.perf_counter() - t0) * 1000.0
        if recorder is not None:
            recorder.record_llm(
                stage="router",
                model=getattr(llm, "model", ""),
                messages=messages,
                output=f"[error] {exc}",
                latency_ms=latency,
                meta={"error": str(exc)},
            )
        # 路由失败的安全回退：默认走临床专家，避免直接拒答
        scores["clinical"] = 6.0
        reason = f"路由 LLM 失败，回退到临床专家：{exc}"

    emergency = emergency_rule or emergency_llm
    raw_weights = _softmax(scores, cfg.softmax_temp)

    max_score = max(scores.values()) if scores else 0.0
    out_of_scope = (max_score < cfg.min_relevance) and not emergency

    selected: List[str] = []
    weights: Dict[str, float] = {}

    if not out_of_scope:
        # 门控：weight >= 阈值，按权重降序取 Top-K
        ranked = sorted(raw_weights.items(), key=lambda kv: kv[1], reverse=True)
        for k, w in ranked:
            if w >= cfg.gating_threshold and len(selected) < cfg.max_experts:
                selected.append(k)
        # 兜底：门控过严导致空选时，取最高分专家
        if not selected and ranked:
            selected.append(ranked[0][0])

        # 急症强制激活核心专家（覆盖门控）
        if emergency:
            for k in _EMERGENCY_EXPERTS:
                if k not in selected:
                    selected.append(k)

        # 选中专家权重重归一化
        sub = {k: raw_weights[k] for k in selected}
        s = sum(sub.values()) or 1.0
        weights = {k: round(v / s, 4) for k, v in sub.items()}

    decision = RouterDecision(
        scores={k: round(v, 3) for k, v in scores.items()},
        raw_weights={k: round(v, 4) for k, v in raw_weights.items()},
        weights=weights,
        selected_experts=selected,
        emergency=emergency,
        out_of_scope=out_of_scope,
        reason=reason,
    )
    if recorder is not None:
        recorder.router_decision = {
            "scores": decision.scores,
            "raw_weights": decision.raw_weights,
            "weights": decision.weights,
            "selected_experts": decision.selected_experts,
            "emergency": decision.emergency,
            "emergency_rule_hit": emergency_rule,
            "out_of_scope": decision.out_of_scope,
            "reason": decision.reason,
            "config": {
                "gating_threshold": cfg.gating_threshold,
                "max_experts": cfg.max_experts,
                "min_relevance": cfg.min_relevance,
                "softmax_temp": cfg.softmax_temp,
            },
        }
    return decision
