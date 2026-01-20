from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Protocol


class QueryRewriter(Protocol):
    """
    Query 重写器接口：输入原始 query，输出多个候选 query（用于多路召回/扩展检索）。
    """

    def rewrite(self, query: str) -> List[str]: ...


@dataclass(frozen=True)
class NoRewrite(QueryRewriter):
    def rewrite(self, query: str) -> List[str]:
        q = (query or "").strip()
        return [q] if q else []


@dataclass(frozen=True)
class TemplateRewriter(QueryRewriter):
    """
    简单的“模板化”重写：把一个 query 扩展成多种问法/关键词组合。
    - 不调用 LLM，完全可复现
    - 适合毕业设计里做“query rewrite / query expansion”的对比维度
    """

    templates: tuple[str, ...] = (
        "{q}",
        "definition of {q}",
        "what is {q}",
        "what is the function of {q}",
        "indications of {q}",
        "contraindications of {q}",
        "treatment for {q}",
        "diagnosis of {q}",
        "symptoms of {q}",
        "dose of {q}",
    )
    max_out: int = 8

    def rewrite(self, query: str) -> List[str]:
        q0 = (query or "").strip()
        if not q0:
            return []

        # 去掉选择题选项行（如果用户把完整题干扔进来）
        q = q0.split("\n\n", 1)[0].strip()
        q = re.sub(r"\s+", " ", q)

        out: List[str] = []
        seen = set()
        for t in self.templates:
            cand = t.format(q=q).strip()
            cand = re.sub(r"\s+", " ", cand)
            if cand and cand not in seen:
                out.append(cand)
                seen.add(cand)
            if len(out) >= int(self.max_out):
                break
        return out




