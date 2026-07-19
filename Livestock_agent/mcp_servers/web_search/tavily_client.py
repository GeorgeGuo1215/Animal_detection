from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

_TAVILY_REST_URL = "https://api.tavily.com/search"

_http_pool: httpx.AsyncClient | None = None


def _get_http_pool() -> httpx.AsyncClient:
    global _http_pool
    if _http_pool is None or _http_pool.is_closed:
        _http_pool = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5, read=30, write=5, pool=10),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
    return _http_pool

_HIGH_TRUST_DOMAINS = (
    "panda.org.cn", "pmc.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov", "msdvetmanual.com", "nature.com",
    "sciencedirect.com", "springer.com", "wiley.com",
)
_LOW_TRUST_DOMAINS = (
    "hytrans-sh.com", "hanbaoauto.com",
)


def _domain_of(url: str) -> str:
    try:
        return urlparse((url or "").strip()).netloc.lower()
    except Exception:
        return ""


def _source_priority(url: str) -> int:
    domain = _domain_of(url)
    if not domain:
        return 0
    if any(domain == d or domain.endswith(f".{d}") for d in _HIGH_TRUST_DOMAINS):
        return 100
    if domain.endswith(".gov") or domain.endswith(".edu") or domain.endswith(".edu.cn") or domain.endswith(".ac.cn"):
        return 90
    if domain.endswith(".org"):
        return 75
    if any(domain == d or domain.endswith(f".{d}") for d in _LOW_TRUST_DOMAINS):
        return -20
    return 40


def _rank_and_filter(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, ranked = set(), []
    for item in items:
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        content = (item.get("content") or "").strip()
        key = (url or title).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        priority = _source_priority(url)
        if priority < 0:
            continue
        ranked.append({
            "title": title, "url": url,
            "content": content[:1500],
            "domain": _domain_of(url),
            "source_priority": priority,
        })
    ranked.sort(key=lambda x: -x.get("source_priority", 0))
    return ranked


def _extract_items_from_content(content_list: list) -> List[Dict[str, Any]]:
    """Extract search result items from MCP content, handling multiple formats."""
    items: List[Dict[str, Any]] = []

    for c in content_list:
        if hasattr(c, "model_dump"):
            data = c.model_dump()
        elif isinstance(c, dict):
            data = c
        else:
            data = {"type": getattr(c, "type", "text"), "text": str(c)}

        text = data.get("text") or data.get("value") or ""
        meta = data.get("metadata") or {}

        if meta.get("url") or meta.get("title"):
            items.append({
                "title": meta.get("title") or "",
                "url": meta.get("url") or meta.get("source") or "",
                "content": text,
            })
            continue

        # Tavily MCP often returns a single TextContent whose text is a JSON string
        # containing the actual search results array or object.
        parsed = _try_parse_json_text(text)
        if parsed is not None:
            if isinstance(parsed, list):
                for entry in parsed:
                    if isinstance(entry, dict):
                        items.append(_normalize_result_entry(entry))
            elif isinstance(parsed, dict):
                for entry in parsed.get("results", []):
                    if isinstance(entry, dict):
                        items.append(_normalize_result_entry(entry))
                if not items and parsed.get("url"):
                    items.append(_normalize_result_entry(parsed))
        else:
            items.append({"title": "", "url": "", "content": text})

    return items


def _try_parse_json_text(text: str):
    """Attempt to parse text as JSON; return None on failure."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _normalize_result_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a single search result dict to {title, url, content}."""
    return {
        "title": entry.get("title") or "",
        "url": entry.get("url") or entry.get("link") or entry.get("source") or "",
        "content": entry.get("content") or entry.get("snippet") or entry.get("text") or "",
    }


async def tavily_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
) -> List[Dict[str, Any]]:
    """
    Call the Tavily REST API directly (no MCP subprocess) and return search results.

    Returns:
        [{"title": "...", "url": "...", "content": "..."}, ...]
    """
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        logger.warning("TAVILY_API_KEY not set, Tavily search disabled.")
        return []

    payload = {
        "query": query,
        "max_results": int(max_results),
        "search_depth": search_depth,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        client = _get_http_pool()
        resp = await client.post(_TAVILY_REST_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Tavily REST search failed: %s", exc)
        return []

    results = data.get("results") or []
    items = [_normalize_result_entry(r) for r in results if isinstance(r, dict)]
    return _rank_and_filter(items)