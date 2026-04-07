from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import httpx


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def _is_reasoner_model(model: str) -> bool:
    return "reasoner" in model.lower()


class OpenAICompatClient:
    def __init__(
        self, *, base_url: Optional[str] = None, api_key: Optional[str] = None,
        model: Optional[str] = None, timeout_s: float = 300.0,
    ) -> None:
        self.base_url = (base_url or _env("OPENAI_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        self.api_key = api_key or _env("OPENAI_API_KEY") or _env("DEEPSEEK_API_KEY") or ""
        self.model = model or _env("OPENAI_MODEL") or _env("DEEPSEEK_MODEL") or "deepseek-chat"
        self.timeout_s = float(timeout_s)

    def chat(self, *, messages: List[Dict[str, Any]], temperature: float = 0.2,
             max_tokens: int = 768, response_format: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("Missing API key: set OPENAI_API_KEY (or DEEPSEEK_API_KEY).")
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload: Dict[str, Any] = {
            "model": self.model, "messages": messages,
            "max_tokens": int(max_tokens),
        }
        if _is_reasoner_model(self.model):
            pass
        else:
            payload["temperature"] = float(temperature)
            if response_format:
                payload["response_format"] = response_format
        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            return r.json()


class AsyncOpenAIClient:
    def __init__(self, *, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 model: Optional[str] = None) -> None:
        self.base_url = (base_url or _env("OPENAI_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        self.api_key = api_key or _env("OPENAI_API_KEY") or _env("DEEPSEEK_API_KEY") or ""
        self.model = model or _env("OPENAI_MODEL") or _env("DEEPSEEK_MODEL") or "deepseek-chat"
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10, read=300, write=10, pool=30),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    async def chat(self, *, messages: List[Dict[str, Any]], temperature: float = 0.2,
                   max_tokens: int = 768, response_format: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("Missing API key: set OPENAI_API_KEY (or DEEPSEEK_API_KEY).")
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload: Dict[str, Any] = {
            "model": self.model, "messages": messages,
            "max_tokens": int(max_tokens),
        }
        if _is_reasoner_model(self.model):
            pass
        else:
            payload["temperature"] = float(temperature)
            if response_format:
                payload["response_format"] = response_format
        r = await self._client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()

    async def close(self) -> None:
        await self._client.aclose()


def extract_text(resp: Dict[str, Any]) -> str:
    try:
        msg = resp["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or "").strip()
        if reasoning and content:
            return f"<details><summary>💭 思考过程</summary>\n\n{reasoning}\n\n</details>\n\n{content}"
        return content or reasoning
    except Exception:
        return json.dumps(resp, ensure_ascii=False)[:2000]
