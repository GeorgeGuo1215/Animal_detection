"""离线单测：兽医病例工作流门控与 Aggregator/答风分节文案。

Run: pytest tests/moe/test_vet_case_prompt.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.services.plan_and_solve import (
    build_solve_prompt,
    has_concrete_case_narrative,
    is_concrete_vet_case,
)
from app.services.moe.orchestrator import MoEOrchestrator, OrchestratorConfig
from app.services.moe.router import RouterDecision
from app.services.moe.critic import CriticResult


_FLUTD_NARRATIVE = (
    "5 岁的英短，公猫，已经绝育。今天从早上开始就一直往猫砂盆跑，差不多十几分钟去一次，"
    "每次蹲很久，但是宠主看不出来到底有没有尿出来。猫砂盆里好像只有一点点尿团，比平时少很多。"
    "它今天不太爱吃东西，平时早上会主动来要罐头，今天只舔了几口。精神也差一些，老是趴着，还会舔下面。"
    "没有明显呕吐，但是刚才好像干呕了一下。昨天晚上还挺正常的。最近没有换粮，喝水感觉和平时差不多。"
    "宠主家里最近来了客人，它有点紧张，躲了两天。疫苗应该是去年打的，驱虫不太记得。"
    "之前有过一次尿血，可能是膀胱炎，吃药后好了。今天还没有去医院，也没有吃药。"
)

_FLUTD_DIAGNOSIS = _FLUTD_NARRATIVE + "\n请对以上病例做出诊断。"
_FLUTD_ORGANIZE = _FLUTD_NARRATIVE + "\n请对以上病例做出整理，输出统一的病例格式。"

_BULLDOG_EXERCISE = (
    "3岁法斗（法国斗牛犬）公犬，已绝育，体重12kg。"
    "主人想每天带它剧烈跑步或追球1小时，最近热天遛弯就张口喘、不愿走。"
    "从兽医角度评估运动建议与风险边界，并说明与普通中型犬的差异。"
)


def test_narrative_alone_is_not_case_workup():
    assert has_concrete_case_narrative(_FLUTD_NARRATIVE)
    assert not is_concrete_vet_case(_FLUTD_NARRATIVE)


def test_diagnosis_intent_triggers():
    assert is_concrete_vet_case(_FLUTD_DIAGNOSIS)


def test_organize_intent_triggers():
    assert is_concrete_vet_case(_FLUTD_ORGANIZE)


def test_vet_eval_golden_prompt_triggers():
    """兽医学生评估「指令一」原文应触发病例工作流。"""
    q = (
        "请帮我看看下面这个病例应该怎么整理和诊断：\n"
        "3 岁多金毛，公的，已经绝育了。昨天晚上开始有点不太对劲，平时特别爱吃，昨天晚饭就没怎么吃。"
        "半夜吐了一次，是黄色的水，今天早上又吐了两次。"
    )
    assert has_concrete_case_narrative(q)
    assert is_concrete_vet_case(q)


def test_focused_exercise_ask_does_not_trigger():
    assert has_concrete_case_narrative(_BULLDOG_EXERCISE)
    assert not is_concrete_vet_case(_BULLDOG_EXERCISE)


def test_concrete_case_rejects_vague_vet_question():
    assert not is_concrete_vet_case("猫尿血怎么办")
    assert not is_concrete_vet_case("犬呕吐常见原因")


def test_concrete_case_rejects_short_text():
    assert not is_concrete_vet_case("英短公猫")


def test_build_solve_prompt_diagnosis_has_case_structure():
    prompt = build_solve_prompt(user_role="veterinarian", query=_FLUTD_DIAGNOSIS)
    assert "病例整理" in prompt
    assert "基本信息" in prompt
    assert "主诉" in prompt
    assert "现病史" in prompt
    assert "既往史" in prompt
    assert "问题列表" in prompt
    assert "向宠主追问的问题清单" in prompt
    assert "临床表现或指标异常" in prompt
    assert "不要**写成「问题1：急性胰腺炎」" in prompt or "问题1：急性胰腺炎" in prompt
    assert "检查与治疗方案" in prompt
    assert "紧急处理" in prompt
    assert "风险提示" in prompt
    assert "器官系统" in prompt
    assert "个体信号" not in prompt
    assert "红旗信号" not in prompt
    assert "条目化归纳" not in prompt


def test_build_solve_prompt_exercise_no_forced_case_structure():
    prompt = build_solve_prompt(user_role="veterinarian", query=_BULLDOG_EXERCISE)
    assert "兽医病例工作流答风" not in prompt
    assert "兽医证据分层规范" in prompt


def test_build_solve_prompt_vet_vague_keeps_evidence_layering():
    prompt = build_solve_prompt(user_role="veterinarian", query="猫尿血怎么办")
    assert "病例整理" not in prompt or "兽医病例工作流答风" not in prompt
    assert "兽医证据分层规范" in prompt


def test_build_solve_prompt_owner_no_case_structure():
    prompt = build_solve_prompt(user_role="pet_owner", query=_FLUTD_DIAGNOSIS)
    assert "兽医病例工作流答风" not in prompt


def test_aggregator_synthesis_structure_for_diagnosis():
    orch = MoEOrchestrator(config=OrchestratorConfig(user_role="veterinarian"))
    decision = RouterDecision(
        scores={"clinical": 8},
        raw_weights={"clinical": 1.0},
        weights={"clinical": 1.0},
        selected_experts=["clinical"],
        emergency=False,
        out_of_scope=False,
        reason="test",
    )
    msgs = orch._build_synthesis_messages(
        query=_FLUTD_DIAGNOSIS,
        opinions=[{
            "expert": "clinical", "name_zh": "兽医临床专家", "weight": 1.0,
            "conclusion": "疑似下尿路问题", "evidence": [], "risks": [], "confidence": 0.7,
        }],
        critic=CriticResult(verdict="pass", issues=[], constraints=[], reason="ok"),
        decision=decision,
    )
    sys = msgs[0]["content"]
    assert "病例整理" in sys
    assert "问题列表" in sys
    assert "症状/异常指标" in sys
    assert "检查与治疗方案" in sys
    assert "风险提示" in sys
    assert "基本信息" in sys
    assert "个体信号" not in sys
    assert "红旗信号" not in sys


def test_aggregator_exercise_does_not_force_case_sections():
    orch = MoEOrchestrator(config=OrchestratorConfig(user_role="veterinarian"))
    decision = RouterDecision(
        scores={"clinical": 8},
        raw_weights={"clinical": 1.0},
        weights={"clinical": 1.0},
        selected_experts=["clinical"],
        emergency=False,
        out_of_scope=False,
        reason="test",
    )
    msgs = orch._build_synthesis_messages(
        query=_BULLDOG_EXERCISE,
        opinions=[{
            "expert": "clinical", "name_zh": "兽医临床专家", "weight": 1.0,
            "conclusion": "限制剧烈运动", "evidence": [], "risks": ["热射病"], "confidence": 0.8,
        }],
        critic=CriticResult(verdict="pass", issues=[], constraints=[], reason="ok"),
        decision=decision,
    )
    sys = msgs[0]["content"]
    assert "不要仅因叙述中含品种/症状就强行输出病例整理" in sys
    assert '"concrete_vet_case": false' in msgs[1]["content"]


def test_aggregator_max_tokens_bumped_only_for_workup():
    orch = MoEOrchestrator(config=OrchestratorConfig(user_role="veterinarian", max_tokens=900))
    assert orch._aggregator_max_tokens(_FLUTD_DIAGNOSIS) == 1500
    assert orch._aggregator_max_tokens(_BULLDOG_EXERCISE) == 900
    assert orch._aggregator_max_tokens("猫尿血怎么办") == 900
