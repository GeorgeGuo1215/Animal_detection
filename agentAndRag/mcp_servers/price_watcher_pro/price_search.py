"""
Price comparison via DuckDuckGo search + LLM extraction.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

_LLM_BASE = (os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com").rstrip("/")
_LLM_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or ""
_LLM_MODEL = os.getenv("OPENAI_MODEL") or os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"


def _llm_extract(prompt: str, max_tokens: int = 1024) -> str:
    """Call LLM to extract structured info from raw search snippets."""
    if not _LLM_KEY:
        return "{}"
    url = f"{_LLM_BASE}/chat/completions"
    payload = {
        "model": _LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a data extraction assistant. Output strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {_LLM_KEY}", "Content-Type": "application/json"}
    with httpx.Client(timeout=60) as client:
        r = client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return "{}"


def _search_duckduckgo(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search DuckDuckGo and return snippets."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [{"title": r.get("title", ""), "body": r.get("body", ""), "href": r.get("href", "")} for r in results]
    except ImportError:
        logger.warning("duckduckgo-search / ddgs not installed, returning empty results")
        return []
    except Exception as exc:
        logger.warning("DuckDuckGo search failed: %s", exc)
        return []


def compare_prices(
    product_name: str,
    compare_platforms: Optional[List[str]] = None,
    user_budget_preference: str = "mid-range",
) -> Dict[str, Any]:
    """
    Search for product prices across platforms using DuckDuckGo + LLM extraction.
    """
    platforms = compare_platforms or ["Amazon", "Chewy", "JD"]
    all_snippets: List[Dict[str, Any]] = []

    for platform in platforms:
        query = f"{product_name} price {platform}"
        results = _search_duckduckgo(query, max_results=3)
        for r in results:
            r["platform"] = platform
        all_snippets.extend(results)

    if not all_snippets:
        return {
            "status": "NO_RESULTS",
            "product_name": product_name,
            "prices": [],
            "recommendation": None,
            "confidence": "low",
            "message": "No search results found. Please check the product name or try again later.",
        }

    snippets_text = json.dumps(all_snippets, ensure_ascii=False)[:4000]

    prompt = f"""Given these search results about "{product_name}" prices, extract structured price data.

Search results:
{snippets_text}

User budget preference: {user_budget_preference}

Output a JSON object with this structure:
{{
  "prices": [
    {{
      "platform": "platform name",
      "price": "price string (e.g. $45.99)",
      "price_numeric": 45.99,
      "currency": "USD",
      "url": "product URL if found",
      "notes": "any relevant notes (size, variant, etc.)"
    }}
  ],
  "recommendation": "which option best matches the budget preference and why",
  "confidence": "high/medium/low based on data quality"
}}

If price data is unclear or not found for a platform, set price to null.
"""
    raw = _llm_extract(prompt)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"prices": [], "confidence": "low"}

    result["product_name"] = product_name
    result["status"] = "OK"
    result.setdefault("prices", [])
    result.setdefault("confidence", "medium")
    return result
