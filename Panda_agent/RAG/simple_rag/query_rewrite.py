from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Protocol

import httpx


class QueryRewriter(Protocol):
    def rewrite(self, query: str) -> List[str]: ...


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def _normalize_query(query: str) -> str:
    q0 = (query or "").strip()
    if not q0:
        return ""
    q = q0.split("\n\n", 1)[0].strip()
    return re.sub(r"\s+", " ", q)


_MEDICAL_QUERY_HINTS = (
    "疾病", "病症", "诊断", "治疗", "病例", "病因", "症状", "临床", "监测", "预警",
    "感染", "炎症", "腹痛", "腹胀", "便秘", "腹泻", "呕吐", "厌食", "食欲下降",
    "寄生虫", "梗阻", "肠阻塞", "肠梗阻", "肠套叠", "肠扭转", "肠系膜", "急腹症",
    "麻醉", "手术", "粪石", "粪团", "抬尾", "努责", "无粪", "排便异常",
    "disease", "diagnosis", "treatment", "clinical", "symptom", "symptoms",
    "prevention", "monitoring", "obstruction", "ileus", "constipation",
    "vomiting", "diarrhea", "abdominal pain", "infection", "parasite",
)

_MEDICAL_ALIAS_MAP = {
    "肠梗阻": ("肠阻塞", "肠道梗阻", "intestinal obstruction", "ileus"),
    "肠阻塞": ("肠梗阻", "肠道梗阻", "intestinal obstruction", "ileus"),
    "便秘": ("排便困难", "无粪排出", "constipation"),
    "腹痛": ("腹部疼痛", "abdominal pain"),
    "腹胀": ("腹围增大", "腹部膨隆", "abdominal distention"),
    "呕吐": ("反胃", "vomiting"),
    "腹泻": ("diarrhea",),
    "寄生虫": ("蛔虫", "parasite", "parasites"),
    "急腹症": ("腹痛", "腹胀", "急诊", "acute abdomen"),
}


def is_medical_query(query: str) -> bool:
    q = _normalize_query(query).lower()
    if not q:
        return False
    return any(token.lower() in q for token in _MEDICAL_QUERY_HINTS)


def _medical_rewrite_candidates(query: str) -> List[str]:
    q = _normalize_query(query)
    if not q or not is_medical_query(q):
        return []

    out = [
        q,
        f"{q} 症状 诊断 病因",
        f"{q} 临床表现 早期识别",
        f"{q} 疾病预防 监测",
        f"{q} symptoms diagnosis causes",
        f"{q} clinical signs early detection",
    ]
    if any(k in q for k in ("前期", "早期", "识别", "异常", "监测")):
        out.extend([
            f"{q} 抬尾 努责 无粪排出",
            f"{q} 食欲下降 腹痛 腹胀",
            f"{q} constipation abdominal pain agitation",
        ])
    for src, aliases in _MEDICAL_ALIAS_MAP.items():
        if src in q:
            for alias in aliases:
                out.append(q.replace(src, alias))
    return out


@dataclass(frozen=True)
class NoRewrite(QueryRewriter):
    def rewrite(self, query: str) -> List[str]:
        q = _normalize_query(query)
        return [q] if q else []


_ZH_EN_TOPIC_MAP: dict[str, str] = {
    "食性": "diet feeding",
    "竹子": "bamboo",
    "栖息地": "habitat",
    "繁殖": "breeding reproduction",
    "遗传": "genetics genomics",
    "保护": "conservation",
    "行为": "behavior ethology",
    "解剖": "anatomy morphology",
    "生理": "physiology",
    "生态": "ecology",
    "进化": "evolution evolutionary",
    "消化": "digestion digestive",
    "肠道": "intestinal gut",
    "微生物": "microbiome microbial",
    "氰化物": "cyanide cyanogenic",
    "解毒": "detoxification detox",
    "种群": "population",
    "基因": "gene genetic",
    "分类": "taxonomy classification",
    "圈养": "captive captivity",
    "野化": "reintroduction rewilding",
    "饲养": "husbandry feeding management",
    "疫苗": "vaccine vaccination",
    "免疫": "immune immunology",
    "寿命": "lifespan longevity",
    "体重": "body weight",
    "幼崽": "cub infant",
    "发情": "estrus mating",
    "妊娠": "pregnancy gestation",
    "哺乳": "lactation nursing",
}


def _generate_en_variant(query: str) -> Optional[str]:
    """Generate an English keyword expansion for a Chinese query by mapping
    recognized Chinese terms to their English equivalents."""
    q = query.lower()
    en_parts: List[str] = []
    for zh, en in _ZH_EN_TOPIC_MAP.items():
        if zh in q:
            en_parts.append(en)
    if not en_parts:
        return None
    return "giant panda " + " ".join(en_parts[:4])


@dataclass(frozen=True)
class TemplateRewriter(QueryRewriter):
    templates: tuple[str, ...] = (
        "{q}",
        "{q} conservation",
        "{q} breeding",
        "{q} habitat",
        "{q} biology",
        "{q} genetics",
        "{q} disease prevention",
    )
    max_out: int = 10

    def rewrite(self, query: str) -> List[str]:
        q = _normalize_query(query)
        if not q:
            return []
        out: List[str] = []
        seen: set[str] = set()

        def _add(cand: str) -> None:
            cand = re.sub(r"\s+", " ", (cand or "").strip())
            if cand and cand not in seen:
                out.append(cand)
                seen.add(cand)

        is_med = is_medical_query(q)
        budget = max(int(self.max_out), 12) if is_med else int(self.max_out)

        _add(q)

        en_variant = _generate_en_variant(q)
        if en_variant:
            _add(en_variant)

        if is_med:
            for cand in _medical_rewrite_candidates(q):
                _add(cand)
                if len(out) >= budget:
                    return out

        for t in self.templates:
            _add(t.format(q=q))
            if len(out) >= budget:
                break
        return out


@dataclass(frozen=True)
class LLMRewriter(QueryRewriter):
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    timeout_s: float = 60.0
    max_out: int = 5
    temperature: float = 0.0

    def _chat_json(self, messages: List[dict]) -> dict:
        base_url = (self.base_url or _env("OPENAI_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        api_key = self.api_key or _env("OPENAI_API_KEY") or _env("DEEPSEEK_API_KEY") or ""
        model = self.model or _env("OPENAI_MODEL") or _env("DEEPSEEK_MODEL") or "deepseek-chat"
        if not api_key:
            raise RuntimeError("Missing API key for LLMRewriter.")
        url = f"{base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(self.temperature),
            "max_tokens": 256,
            "response_format": {"type": "json_object"},
        }
        with httpx.Client(timeout=float(self.timeout_s)) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()

    @staticmethod
    def _extract_queries(text: str, max_out: int) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        try:
            obj = json.loads(text)
        except Exception:
            l = text.find("{")
            r = text.rfind("}")
            if l >= 0 and r > l:
                try:
                    obj = json.loads(text[l : r + 1])
                except Exception:
                    return []
            else:
                return []
        qs = obj.get("queries")
        if not isinstance(qs, list):
            return []
        out: List[str] = []
        seen = set()
        for item in qs:
            cand = re.sub(r"\s+", " ", str(item or "").strip())
            if cand and cand not in seen:
                out.append(cand)
                seen.add(cand)
            if len(out) >= int(max_out):
                break
        return out

    def rewrite(self, query: str) -> List[str]:
        q = _normalize_query(query)
        if not q:
            return []
        system = (
            "You are a retrieval query rewriting assistant for a wildlife knowledge RAG system. "
            "Generate concise search queries that preserve the original meaning, "
            "expand key terms when useful, and improve recall. "
            "The knowledge base already focuses on the relevant domain, so do NOT prepend domain keywords to every query. "
            "Support both Chinese and English queries. Return strict JSON only."
        )
        user = {
            "task": "rewrite_query_for_retrieval",
            "query": q,
            "requirements": [
                "Keep the original meaning unchanged.",
                "Include both Chinese and English variants when possible.",
                "Prefer short retrieval-friendly phrasing.",
                "Include the original query as one candidate.",
                f"Return at most {int(self.max_out)} queries.",
            ],
            "output_format": {"queries": ["original query", "rewritten query 1", "rewritten query 2"]},
        }
        try:
            resp = self._chat_json(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
                ]
            )
            content = (resp.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
            llm_queries = self._extract_queries(content, max_out=int(self.max_out))
        except Exception:
            llm_queries = []
        out: List[str] = []
        seen = set()
        for cand in [q] + llm_queries:
            norm = re.sub(r"\s+", " ", (cand or "").strip())
            if norm and norm not in seen:
                out.append(norm)
                seen.add(norm)
            if len(out) >= int(self.max_out):
                break
        return out
