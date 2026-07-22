"""边界审核专家 Critic：召回前对专家草案做安全与边界校验。

审核维度：责任边界 / 事实一致性 / 用药安全 / 数据一致性 / 合规与免责。
输出 verdict：
- pass   : 草案安全，可直接合成；
- revise : 存在可修补问题，把 constraints 注入最终生成 prompt（强制免责/降确定性/标红用药）；
- block  : 命中硬性安全红线，一票否决 → 编排层走安全兜底话术。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...llm.llm_client import AsyncOpenAIClient, extract_text
from ..plan_and_solve import _safe_json_loads
from .trace import MoETrace, extract_usage


_CRITIC_SYS = (
    "你是宠物健康智能体的『边界审核专家』，是医疗安全的最后守门人。"
    "你将审核多位专家给出的意见草案，从以下维度把关：\n"
    "1) 责任边界：是否用『确诊为…』等绝对措辞替代线下检查，或劝阻就医；\n"
    "2) 事实一致性：结论是否有检索依据支撑、是否疑似幻觉；\n"
    "3) 用药安全：剂量/禁忌/物种特异性毒性（犬猫差异）是否存在危险；\n"
    "4) 数据一致性：建议之间是否相互矛盾；\n"
    "5) 合规与免责：是否需要补充免责声明与紧急情形指引。\n\n"
    "重要区分：\n"
    "- 用户（尤其 veterinarian）若明确要求『做出诊断/鉴别诊断/病例整理』，"
    "专家给出『高度怀疑 / 鉴别诊断排序 / 优先排除尿道阻塞』并强调立即就医，属于正常临床工作流，"
    "**不得**仅因出现诊断相关表述就 block；最多用 revise 要求改成鉴别诊断措辞并补免责/急诊指引。\n"
    "- block 仅用于硬性红线：危险用药剂量、明确鼓励自行用药替代就医、"
    "或在无任何就医提示下给出绝对确诊并误导宠主。\n\n"
    "判定规则：硬性安全红线 → verdict=block；"
    "仅需补免责/降确定性/补充就医提示等可修补问题 → verdict=revise；"
    "草案安全 → verdict=pass。\n"
    "你必须只输出严格 JSON（无额外文字、无代码块），结构：\n"
    "{\n"
    '  "verdict": "pass | revise | block",\n'
    '  "issues": ["发现的问题"],\n'
    '  "constraints": ["写最终答案时必须遵守的硬性约束，如必须加入的免责/就医提示/用药警示"],\n'
    '  "reason": "简要中文说明"\n'
    "}"
)


@dataclass
class CriticResult:
    verdict: str = "pass"
    issues: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    reason: str = ""

    @property
    def blocked(self) -> bool:
        return self.verdict == "block"


async def review(
    *,
    query: str,
    expert_opinions: List[Dict[str, Any]],
    emergency: bool,
    llm: AsyncOpenAIClient,
    user_role: str = "pet_owner",
    recorder: Optional[MoETrace] = None,
) -> CriticResult:
    opinions_brief = [
        {
            "expert": o.get("name_zh") or o.get("expert"),
            "weight": o.get("weight"),
            "confidence": o.get("confidence"),
            "conclusion": o.get("conclusion"),
            "risks": o.get("risks"),
        }
        for o in expert_opinions
    ]
    user_payload = json.dumps(
        {
            "user_question": query,
            "user_role": user_role,
            "emergency": emergency,
            "expert_opinions": opinions_brief,
        },
        ensure_ascii=False,
    )
    messages = [
        {"role": "system", "content": _CRITIC_SYS},
        {"role": "user", "content": user_payload},
    ]

    t0 = time.perf_counter()
    try:
        resp = await llm.chat(
            messages=messages,
            temperature=0.1,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        latency = (time.perf_counter() - t0) * 1000.0
        text = extract_text(resp)
        if recorder is not None:
            recorder.record_llm(
                stage="critic",
                model=getattr(llm, "model", ""),
                messages=messages,
                output=text,
                latency_ms=latency,
                usage=extract_usage(resp),
            )
        obj, _err = _safe_json_loads(text)
        result = CriticResult()
        if isinstance(obj, dict):
            verdict = str(obj.get("verdict") or "pass").strip().lower()
            if verdict not in ("pass", "revise", "block"):
                verdict = "revise"
            result.verdict = verdict
            iss = obj.get("issues")
            result.issues = [str(x) for x in iss] if isinstance(iss, list) else ([str(iss)] if iss else [])
            cons = obj.get("constraints")
            result.constraints = [str(x) for x in cons] if isinstance(cons, list) else ([str(cons)] if cons else [])
            result.reason = str(obj.get("reason") or "")
    except Exception as exc:  # noqa: BLE001
        latency = (time.perf_counter() - t0) * 1000.0
        if recorder is not None:
            recorder.record_llm(
                stage="critic",
                model=getattr(llm, "model", ""),
                messages=messages,
                output=f"[error] {exc}",
                latency_ms=latency,
                meta={"error": str(exc)},
            )
        # 审核失败的安全回退：要求最终答案补充免责与就医提示
        result = CriticResult(
            verdict="revise",
            issues=[f"审核 LLM 失败：{exc}"],
            constraints=["务必补充免责声明，并提示如有异常及时线下就医。"],
            reason="critic 调用失败的安全回退",
        )

    if recorder is not None:
        recorder.critic_result = {
            "verdict": result.verdict,
            "issues": result.issues,
            "constraints": result.constraints,
            "reason": result.reason,
        }
    return result
