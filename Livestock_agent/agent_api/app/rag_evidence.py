from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from RAG.simple_rag.config import default_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _score_thresholds() -> Dict[str, float]:
    dense = float(os.getenv("RAG_DENSE_RELEVANCE_THRESHOLD",
                           os.getenv("RAG_RELEVANCE_THRESHOLD", "0.10")))
    rerank = float(os.getenv("RAG_RERANK_RELEVANCE_THRESHOLD", "0.35"))
    return {"dense": dense, "rerank": rerank}


def hit_relevance_score(hit: Dict[str, Any]) -> float:
    """Dense retrieval score for gap messages (rerank logits are not comparable)."""
    if hit.get("score_retrieval") is not None:
        return float(hit["score_retrieval"])
    return float(hit.get("score") or 0.0)


def hit_is_relevant(hit: Dict[str, Any]) -> bool:
    """Relevant if dense cosine OR rerank score clears its own bar."""
    thr = _score_thresholds()
    dense = hit_relevance_score(hit)
    rerank = float(hit.get("score") or 0.0)
    if hit.get("score_retrieval") is not None:
        return dense >= thr["dense"] or rerank >= thr["rerank"]
    return dense >= thr["dense"]


def best_hit_relevance_score(hits: List[Dict[str, Any]]) -> float:
    if not hits:
        return 0.0
    return max(hit_relevance_score(h) for h in hits if isinstance(h, dict))


@lru_cache(maxsize=1)
def rag_index_chunk_count() -> int:
    try:
        meta_path = default_config(_repo_root()).index_dir / "meta.jsonl"
        if not meta_path.is_file():
            return 0
        with meta_path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def rag_gap_thresholds() -> Dict[str, float | int]:
    thr = _score_thresholds()
    index_size = rag_index_chunk_count()
    return {
        "score_threshold": thr["dense"],
        "rerank_threshold": thr["rerank"],
        "dense_threshold": thr["dense"],
        "min_hits": int(os.getenv("RAG_WEB_FALLBACK_MIN_HITS", "1")),
        "min_strong_hits": int(os.getenv("RAG_MIN_STRONG_HITS", "1")),
        "min_distinct_sources": int(os.getenv("RAG_MIN_DISTINCT_SOURCES", "1")),
        "index_size": index_size,
        "small_index": index_size > 0 and index_size <= int(os.getenv("RAG_SMALL_INDEX_MAX_CHUNKS", "50")),
    }


def rag_evidence_summary(tool_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    rag_hits: List[Dict[str, Any]] = []
    rag_contexts: List[Dict[str, Any]] = []
    for tr in tool_results:
        if not (isinstance(tr, dict) and tr.get("tool_name") == "rag.search"):
            continue
        result = tr.get("result") or {}
        if not isinstance(result, dict):
            continue
        hits = result.get("hits") or []
        contexts = result.get("contexts") or []
        if isinstance(hits, list):
            rag_hits.extend(h for h in hits if isinstance(h, dict))
        if isinstance(contexts, list):
            rag_contexts.extend(c for c in contexts if isinstance(c, dict))

    thresholds = rag_gap_thresholds()
    thr = _score_thresholds()
    distinct_sources = {
        str(item.get("source_path") or "").strip()
        for item in [*rag_hits, *rag_contexts]
        if str(item.get("source_path") or "").strip()
    }
    strong_hits = sum(hit_is_relevant(h) for h in rag_hits)
    best_score = best_hit_relevance_score(rag_hits)
    return {
        "hits_total": len(rag_hits),
        "contexts_total": len(rag_contexts),
        "best_score": best_score,
        "strong_hits": strong_hits,
        "distinct_sources": len(distinct_sources),
        "score_threshold": thresholds["score_threshold"],
        "rerank_threshold": thr["rerank"],
        "dense_threshold": thr["dense"],
        "index_size": thresholds["index_size"],
        "small_index": thresholds["small_index"],
    }
