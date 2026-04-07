from __future__ import annotations

import hashlib
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading

_RAG_DEVICE: Optional[str] = os.getenv("AGENT_WARMUP_DEVICE") or None

from RAG.simple_rag.config import RagConfig, default_config
from RAG.simple_rag.context_utils import build_neighbor_contexts, build_source_index
from RAG.simple_rag.embeddings import Embedder
from RAG.simple_rag.pipeline import build_or_update_index
from RAG.simple_rag.query_rewrite import LLMRewriter, NoRewrite, TemplateRewriter, is_medical_query
from RAG.simple_rag.retrieval import BM25Retriever, MultiRouteRetriever, RetrievedChunk
from RAG.simple_rag.reranker import CrossEncoderReranker
from RAG.simple_rag.vector_store import NumpyVectorStore


_RE_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\\-]{2,}")

_MEDICAL_TEXT_HINTS = (
    "疾病", "病症", "诊断", "治疗", "病例", "病因", "症状", "临床", "监测",
    "肠梗阻", "肠阻塞", "寄生虫", "感染", "炎症", "手术", "麻醉", "药物",
    "便秘", "腹泻", "呕吐", "脱水", "体温", "白细胞", "血常规", "B超",
    "影像", "X线", "超声", "病理", "尸检", "犬瘟热", "蛔虫", "绦虫",
    "intestinal", "obstruction", "ileus", "diagnosis", "treatment",
    "disease", "clinical", "symptom", "pathology",
)

_TOC_PATTERNS = re.compile(
    r"(目\s*录|第[一二三四五六七八九十\d]+[章节篇]|CONTENTS|Table\s+of\s+Contents)",
    re.IGNORECASE,
)


def _looks_like_toc(text: str) -> bool:
    if not text or len(text) < 20:
        return True
    lines = text.strip().split("\n")
    if len(lines) < 3:
        return False
    short_lines = sum(1 for ln in lines if len(ln.strip()) < 15)
    if short_lines / max(len(lines), 1) > 0.7:
        return True
    if _TOC_PATTERNS.search(text[:200]):
        ellipsis_count = text.count("…") + text.count("...") + text.count("·" * 3)
        if ellipsis_count >= 3:
            return True
    return False


def _medical_text_score(text: str) -> int:
    t = (text or "").lower()
    return sum(1 for hint in _MEDICAL_TEXT_HINTS if hint.lower() in t)


def _prioritize_medical_hits(hits: List[dict], query: str) -> List[dict]:
    if not is_medical_query(query) or not hits:
        return hits
    filtered = [h for h in hits if not _looks_like_toc(h.get("text") or "")]
    if not filtered:
        filtered = hits
    for h in filtered:
        h["_med_score"] = _medical_text_score(h.get("text") or "")
    filtered.sort(key=lambda x: (-x.get("_med_score", 0), -x.get("score", 0)))
    for h in filtered:
        h.pop("_med_score", None)
    return filtered


def overlap_score(query: str, ctx: str) -> float:
    tq = set(t.lower() for t in _RE_WORD.findall(query or ""))
    if not tq:
        return 0.0
    tc = set(t.lower() for t in _RE_WORD.findall(ctx or ""))
    if not tc:
        return 0.0
    return len(tq & tc) / float(len(tq))


_LOCK = threading.RLock()
_STORE_CACHE: Dict[str, NumpyVectorStore] = {}
_EMBEDDER_CACHE: Dict[tuple[str, Optional[str]], Embedder] = {}
_BM25_CACHE: Dict[str, BM25Retriever] = {}
_RERANKER_CACHE: Dict[tuple[str, Optional[str]], CrossEncoderReranker] = {}
_SOURCE_INDEX_CACHE: Dict[str, Dict[str, Dict[int, str]]] = {}
_QUERY_EMB_CACHE_SIZE = 128
_QUERY_EMB_CACHE: OrderedDict[str, Any] = OrderedDict()


def _query_emb_key(query: str, model: str) -> str:
    return hashlib.md5(f"{model}||{query}".encode()).hexdigest()


def _get_store(index_dir: Path) -> NumpyVectorStore:
    key = str(index_dir.resolve())
    with _LOCK:
        st = _STORE_CACHE.get(key)
        if st is not None:
            return st
        st = NumpyVectorStore(index_dir)
        st.load()
        _STORE_CACHE[key] = st
        return st


def _get_embedder(embedding_model: str, device: Optional[str]) -> Embedder:
    key = (str(embedding_model), device)
    with _LOCK:
        em = _EMBEDDER_CACHE.get(key)
        if em is not None:
            return em
        em = Embedder(embedding_model, device=device)
        _EMBEDDER_CACHE[key] = em
        return em


def _get_bm25(index_dir: Path) -> BM25Retriever:
    key = str(index_dir.resolve())
    with _LOCK:
        bm = _BM25_CACHE.get(key)
        if bm is not None:
            return bm
        st = _get_store(index_dir)
        bm = BM25Retriever(metas=st._meta)
        _BM25_CACHE[key] = bm
        return bm


def _get_reranker(rerank_model: str, device: Optional[str]) -> CrossEncoderReranker:
    key = (str(rerank_model), device)
    with _LOCK:
        rr = _RERANKER_CACHE.get(key)
        if rr is not None:
            return rr
        rr = CrossEncoderReranker(rerank_model, device=device)
        _RERANKER_CACHE[key] = rr
        return rr


def _get_source_index(index_dir: Path) -> Dict[str, Dict[int, str]]:
    key = str(index_dir.resolve())
    with _LOCK:
        si = _SOURCE_INDEX_CACHE.get(key)
        if si is not None:
            return si
        st = _get_store(index_dir)
        si = build_source_index(st._meta)
        _SOURCE_INDEX_CACHE[key] = si
        return si


def _embed_query_cached(embedder: Embedder, query: str, model_name: str):
    ck = _query_emb_key(query, model_name)
    with _LOCK:
        if ck in _QUERY_EMB_CACHE:
            _QUERY_EMB_CACHE.move_to_end(ck)
            return _QUERY_EMB_CACHE[ck]
    vec = embedder.embed_queries([query], batch_size=1, normalize=True).vectors[0]
    with _LOCK:
        _QUERY_EMB_CACHE[ck] = vec
        if len(_QUERY_EMB_CACHE) > _QUERY_EMB_CACHE_SIZE:
            _QUERY_EMB_CACHE.popitem(last=False)
    return vec


def _invalidate_index_cache(index_dir: Path) -> None:
    key = str(index_dir.resolve())
    with _LOCK:
        _STORE_CACHE.pop(key, None)
        _BM25_CACHE.pop(key, None)
        _SOURCE_INDEX_CACHE.pop(key, None)
        _QUERY_EMB_CACHE.clear()


def warmup_rag_cache(
    *, index_dir: Path, embedding_model: str = "intfloat/multilingual-e5-small",
    device: Optional[str] = None, enable_bm25: bool = True,
    enable_reranker: bool = False, rerank_model: str = "BAAI/bge-reranker-large",
) -> Dict[str, Any]:
    st = _get_store(index_dir)
    _get_embedder(embedding_model, device)
    _get_source_index(index_dir)
    if enable_bm25:
        _get_bm25(index_dir)
    if enable_reranker:
        _get_reranker(rerank_model, device)
    return {
        "index_dir": str(index_dir), "embedding_model": embedding_model,
        "device": device, "enable_bm25": bool(enable_bm25),
        "enable_reranker": bool(enable_reranker), "index_size": int(st.size),
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


class _CachedDenseRetriever:
    def __init__(self, *, store: NumpyVectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, *, top_k: int) -> List[RetrievedChunk]:
        q = (query or "").strip()
        if not q:
            return []
        q_emb = _embed_query_cached(self.embedder, q, self.embedder.model_name_or_path)
        hits = self.store.search(q_emb, top_k=int(top_k))
        return [RetrievedChunk(chunk_id=str(m.get("chunk_id")), score=float(s), meta=m) for m, s in hits]


def _build_cached_multiroute(
    *, index_dir: Path, embedding_model: str, device: Optional[str],
    rewrite: str, rewrite_base_url: Optional[str], rewrite_api_key: Optional[str],
    rewrite_model: Optional[str], rewrite_max_out: int, rewrite_timeout_s: float,
) -> MultiRouteRetriever:
    st = _get_store(index_dir)
    em = _get_embedder(embedding_model, device)
    dense = _CachedDenseRetriever(store=st, embedder=em)
    bm25 = _get_bm25(index_dir)
    if rewrite == "none":
        rewriter = NoRewrite()
    elif rewrite == "llm":
        rewriter = LLMRewriter(
            base_url=rewrite_base_url, api_key=rewrite_api_key,
            model=rewrite_model, max_out=int(rewrite_max_out), timeout_s=float(rewrite_timeout_s),
        )
    else:
        rewriter = TemplateRewriter(max_out=int(rewrite_max_out))
    return MultiRouteRetriever(retrievers=[("dense", dense), ("bm25", bm25)], rewriter=rewriter, top_k_per_route=20)


def rag_search_tool(
    *, query: str, top_k: int = 5, index_dir: Optional[str] = None,
    embedding_model: str = "intfloat/multilingual-e5-small", device: Optional[str] = _RAG_DEVICE,
    multi_route: bool = False, rewrite: str = "template",
    rewrite_base_url: Optional[str] = None, rewrite_api_key: Optional[str] = None,
    rewrite_model: Optional[str] = None, rewrite_max_out: int = 5, rewrite_timeout_s: float = 60.0,
    rerank: bool = False, rerank_model: str = "BAAI/bge-reranker-large",
    rerank_candidates: int = 10, rerank_batch_size: int = 32,
    rerank_keep_topn: int = 0, rerank_filter_overlap: float = 0.15,
    expand_neighbors: int = 1, per_text_max_chars: int = 5000,
    include_hits_text: bool = True, include_contexts_text: bool = True,
) -> Dict[str, Any]:
    repo_root = _repo_root()
    cfg0 = default_config(repo_root)
    cfg = RagConfig(
        raw_dir=cfg0.raw_dir,
        index_dir=Path(index_dir) if index_dir else cfg0.index_dir,
        embedding_model=embedding_model,
    )
    retrieve_k = int(top_k)
    if rerank:
        retrieve_k = max(retrieve_k, int(rerank_candidates))
    if not multi_route:
        st = _get_store(cfg.index_dir)
        em = _get_embedder(cfg.embedding_model, device)
        q_emb = _embed_query_cached(em, query, cfg.embedding_model)
        raw_hits = st.search(q_emb, top_k=retrieve_k)
        hits = [
            {"score": float(score), "source_path": meta.get("source_path"),
             "chunk_index": meta.get("chunk_index"), "n_words": meta.get("n_words"),
             "text": meta.get("text"), "chunk_id": meta.get("chunk_id")}
            for meta, score in raw_hits
        ]
    else:
        mr = _build_cached_multiroute(
            index_dir=cfg.index_dir, embedding_model=cfg.embedding_model, device=device,
            rewrite=rewrite, rewrite_base_url=rewrite_base_url, rewrite_api_key=rewrite_api_key,
            rewrite_model=rewrite_model, rewrite_max_out=int(rewrite_max_out),
            rewrite_timeout_s=float(rewrite_timeout_s),
        )
        hits = [
            {"score": float(h.score), "source_path": h.meta.get("source_path"),
             "chunk_index": h.meta.get("chunk_index"), "n_words": h.meta.get("n_words"),
             "text": h.meta.get("text"), "chunk_id": h.meta.get("chunk_id")}
            for h in mr.retrieve(query, top_k=retrieve_k)
        ]
    hits = _prioritize_medical_hits(hits, query)

    if rerank and hits:
        passages = [(h.get("text") or "").strip() for h in hits]
        rr = _get_reranker(rerank_model, device)
        order = rr.rerank(query=query, passages=passages, top_k=int(top_k), batch_size=int(rerank_batch_size))
        new_hits: List[dict] = []
        for r in order:
            h = dict(hits[int(r.index)])
            h["score_retrieval"] = float(h.get("score") or 0.0)
            h["score"] = float(r.score)
            new_hits.append(h)
        hits = new_hits
        thr = float(rerank_filter_overlap or 0.0)
        if thr > 0.0:
            kept = [dict(h, overlap=float(overlap_score(query, h.get("text") or "")))
                    for h in hits if overlap_score(query, h.get("text") or "") >= thr]
            if kept:
                hits = kept
        topn = int(rerank_keep_topn or 0)
        if topn > 0 and len(hits) > topn:
            hits = hits[:topn]

    hits = _prioritize_medical_hits(hits, query)
    contexts: List[dict] = []
    if int(expand_neighbors) > 0 and hits:
        store = _get_store(cfg.index_dir)
        src_idx = _get_source_index(cfg.index_dir)
        contexts = build_neighbor_contexts(metas=store._meta, hits=hits, neighbor_n=int(expand_neighbors), _source_index=src_idx)

    def _clip(s: str) -> str:
        s = (s or "").strip()
        if per_text_max_chars > 0 and len(s) > per_text_max_chars:
            return s[:per_text_max_chars] + "\n...(truncated)..."
        return s

    if not include_hits_text:
        for h in hits:
            h.pop("text", None)
    else:
        for h in hits:
            if "text" in h:
                h["text"] = _clip(h["text"])
    if not include_contexts_text:
        for c in contexts:
            c.pop("text", None)
    else:
        for c in contexts:
            if "text" in c:
                c["text"] = _clip(c["text"])
    return {"query": query, "params": {"top_k": int(top_k), "multi_route": bool(multi_route), "rewrite": rewrite,
            "rerank": bool(rerank), "expand_neighbors": int(expand_neighbors),
            "index_dir": str(cfg.index_dir), "embedding_model": cfg.embedding_model},
            "hits": hits, "contexts": contexts}


def rag_reindex_tool(
    *, raw_dir: Optional[str] = None, index_dir: Optional[str] = None,
    embedding_model: str = "intfloat/multilingual-e5-small",
    batch_size: int = 32, limit_books: Optional[int] = None, device: Optional[str] = _RAG_DEVICE,
) -> Dict[str, Any]:
    repo_root = _repo_root()
    cfg0 = default_config(repo_root)
    cfg = RagConfig(
        raw_dir=Path(raw_dir) if raw_dir else cfg0.raw_dir,
        index_dir=Path(index_dir) if index_dir else cfg0.index_dir,
        embedding_model=embedding_model,
    )
    out = build_or_update_index(cfg, limit_books=limit_books, batch_size=int(batch_size), device=device)
    _invalidate_index_cache(cfg.index_dir)
    return out
