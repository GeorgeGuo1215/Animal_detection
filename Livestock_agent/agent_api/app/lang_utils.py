from __future__ import annotations

import re
from typing import Literal, Optional

ResponseLang = Literal["en", "zh"]

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def detect_response_lang(text: str) -> ResponseLang:
    q = (text or "").strip()
    if not q:
        return "en"
    cjk = len(_CJK_RE.findall(q))
    if cjk >= 2 or (cjk >= 1 and len(q) <= 24):
        return "zh"
    return "en"


def resolve_response_lang(requested: Optional[str], query: str) -> ResponseLang:
    req = (requested or "auto").strip().lower()
    if req in {"zh", "cn", "chinese", "zh-cn", "zh_cn"}:
        return "zh"
    if req in {"en", "english"}:
        return "en"
    return detect_response_lang(query)
