from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import threading

from RAG.simple_rag.config import RagConfig, default_config
from RAG.simple_rag.context_utils import build_neighbor_contexts
from RAG.simple_rag.embeddings import Embedder
from RAG.simple_rag.pipeline import build_or_update_index
from RAG.simple_rag.query_rewrite import NoRewrite, TemplateRewriter
from RAG.simple_rag.retrieval import BM25Retriever, MultiRouteRetriever, RetrievedChunk
from RAG.simple_rag.reranker import CrossEncoderReranker
from RAG.simple_rag.vector_store import NumpyVectorStore

from RAG.query import overlap_score  # 复用 query.py 的轻量 overlap


_LOCK = threading.RLock()
_STORE_CACHE: Dict[str, NumpyVectorStore] = {}
_EMBEDDER_CACHE: Dict[tuple[str, Optional[str]], Embedder] = {}
_BM25_CACHE: Dict[str, BM25Retriever] = {}
_RERANKER_CACHE: Dict[tuple[str, Optional[str]], CrossEncoderReranker] = {}


def _get_store(index_dir: Path) -> NumpyVectorStore:
    """
    Cache-loaded vector store. Loading embeddings.npy/meta.jsonl repeatedly is slow.
    """
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
    """
    Cache SentenceTransformer model instance. Cold start is slow.
    """
    key = (str(embedding_model), device)
    with _LOCK:
        em = _EMBEDDER_CACHE.get(key)
        if em is not None:
            return em
        em = Embedder(embedding_model, device=device)
        _EMBEDDER_CACHE[key] = em
        return em


def _get_bm25(index_dir: Path) -> BM25Retriever:
    """
    Cache BM25 statistics (tf/df) for the whole corpus. Building it per request is very slow.
    """
    key = str(index_dir.resolve())
    with _LOCK:
        bm = _BM25_CACHE.get(key)
        if bm is not None:
            return bm
        st = _get_store(index_dir)
        bm = BM25Retriever(metas=st._meta)  # noqa: SLF001
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


def _invalidate_index_cache(index_dir: Path) -> None:
    """
    After reindex, embeddings/meta changed. Drop related caches so next query reloads.
    """
    key = str(index_dir.resolve())
    with _LOCK:
        _STORE_CACHE.pop(key, None)
        _BM25_CACHE.pop(key, None)


def warmup_rag_cache(
    *,
    index_dir: Path,
    embedding_model: str = "intfloat/multilingual-e5-small",
    device: Optional[str] = None,
    enable_bm25: bool = True,
    enable_reranker: bool = False,
    rerank_model: str = "BAAI/bge-reranker-large",
) -> Dict[str, Any]:
    """
    Preload heavy objects on service startup to avoid slow first request.

    Returns a small stats dict for logging/trace.
    """
    st = _get_store(index_dir)
    _get_embedder(embedding_model, device)
    if enable_bm25:
        _get_bm25(index_dir)
    if enable_reranker:
        _get_reranker(rerank_model, device)
    return {
        "index_dir": str(index_dir),
        "embedding_model": embedding_model,
        "device": device,
        "enable_bm25": bool(enable_bm25),
        "enable_reranker": bool(enable_reranker),
        "rerank_model": rerank_model if enable_reranker else None,
        "index_size": int(st.size),
    }


def _repo_root() -> Path:
    # agent_api/app/rag_tools.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


class _CachedDenseRetriever:
    def __init__(self, *, store: NumpyVectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, *, top_k: int) -> List[RetrievedChunk]:
        q = (query or "").strip()
        if not q:
            return []
        q_emb = self.embedder.embed_queries([q], batch_size=1, normalize=True).vectors[0]
        hits = self.store.search(q_emb, top_k=int(top_k))
        return [RetrievedChunk(chunk_id=str(m.get("chunk_id")), score=float(s), meta=m) for m, s in hits]


def _build_cached_multiroute(
    *,
    index_dir: Path,
    embedding_model: str,
    device: Optional[str],
    rewrite: str,
) -> MultiRouteRetriever:
    st = _get_store(index_dir)
    em = _get_embedder(embedding_model, device)
    dense = _CachedDenseRetriever(store=st, embedder=em)
    bm25 = _get_bm25(index_dir)
    rewriter = NoRewrite() if rewrite == "none" else TemplateRewriter()
    return MultiRouteRetriever(retrievers=[("dense", dense), ("bm25", bm25)], rewriter=rewriter, top_k_per_route=20)


def rag_search_tool(
    *,
    query: str,
    top_k: int = 5,
    index_dir: Optional[str] = None,
    embedding_model: str = "intfloat/multilingual-e5-small",
    device: Optional[str] = None,
    multi_route: bool = False,
    rewrite: str = "template",
    rerank: bool = False,
    rerank_model: str = "BAAI/bge-reranker-large",
    rerank_candidates: int = 10,
    rerank_batch_size: int = 32,
    rerank_keep_topn: int = 0,
    rerank_filter_overlap: float = 0.15,
    expand_neighbors: int = 1,
    per_text_max_chars: int = 5000,
    include_hits_text: bool = True,
    include_contexts_text: bool = True,
) -> Dict[str, Any]:
    repo_root = _repo_root()
    cfg0 = default_config(repo_root)
    cfg = RagConfig(
        raw_dir=cfg0.raw_dir,
        index_dir=Path(index_dir) if index_dir else cfg0.index_dir,
        embedding_model=embedding_model,
        chunk_words=cfg0.chunk_words,
        chunk_overlap_words=cfg0.chunk_overlap_words,
        min_chunk_words=cfg0.min_chunk_words,
    )

    retrieve_k = int(top_k)
    if rerank:
        retrieve_k = max(retrieve_k, int(rerank_candidates))

    # 1) 召回 hits（小块）
    if not multi_route:
        # Cached dense retrieval: avoid loading store + model for each request.
        st = _get_store(cfg.index_dir)
        em = _get_embedder(cfg.embedding_model, device)
        q_emb = em.embed_queries([query], batch_size=1, normalize=True).vectors[0]
        raw_hits = st.search(q_emb, top_k=retrieve_k)
        hits = [
            {
                "score": float(score),
                "source_path": meta.get("source_path"),
                "chunk_index": meta.get("chunk_index"),
                "n_words": meta.get("n_words"),
                "text": meta.get("text"),
                "chunk_id": meta.get("chunk_id"),
            }
            for meta, score in raw_hits
        ]
    else:
        mr = _build_cached_multiroute(index_dir=cfg.index_dir, embedding_model=cfg.embedding_model, device=device, rewrite=rewrite)
        hits = [
            {
                "score": float(h.score),
                "source_path": h.meta.get("source_path"),
                "chunk_index": h.meta.get("chunk_index"),
                "n_words": h.meta.get("n_words"),
                "text": h.meta.get("text"),
                "chunk_id": h.meta.get("chunk_id"),
            }
            for h in mr.retrieve(query, top_k=retrieve_k)
        ]

    # 2) rerank（可选）
    if rerank and hits:
        passages = [(h.get("text") or "").strip() for h in hits]
        rr = _get_reranker(rerank_model, device)
        order = rr.rerank(query=query, passages=passages, top_k=int(top_k), batch_size=int(rerank_batch_size))
        new_hits: List[dict] = []
        for r in order:
            h = dict(hits[int(r.index)])
            h["score_retrieval"] = float(h.get("score") or 0.0)
            h["score"] = float(r.score)
            h["score_rerank"] = float(r.score)
            new_hits.append(h)
        hits = new_hits

        thr = float(rerank_filter_overlap or 0.0)
        if thr > 0.0:
            kept = []
            for h in hits:
                ov = overlap_score(query, (h.get("text") or ""))
                hh = dict(h)
                hh["overlap"] = float(ov)
                if ov >= thr:
                    kept.append(hh)
            if kept:
                hits = kept

        topn = int(rerank_keep_topn or 0)
        if topn > 0 and len(hits) > topn:
            hits = hits[:topn]

    # 3) 邻居拼接 contexts（可选）
    contexts: List[dict] = []
    if int(expand_neighbors) > 0 and hits:
        store = _get_store(cfg.index_dir)
        contexts = build_neighbor_contexts(
            metas=store._meta,  # noqa: SLF001（tool 允许读内部 meta）
            hits=hits,
            neighbor_n=int(expand_neighbors),
        )

    # 4) 截断（避免响应太大）
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

    return {
        "query": query,
        "params": {
            "top_k": int(top_k),
            "multi_route": bool(multi_route),
            "rewrite": rewrite,
            "rerank": bool(rerank),
            "expand_neighbors": int(expand_neighbors),
            "index_dir": str(cfg.index_dir),
            "embedding_model": cfg.embedding_model,
        },
        "hits": hits,
        "contexts": contexts,
    }


def rag_reindex_tool(
    *,
    raw_dir: Optional[str] = None,
    index_dir: Optional[str] = None,
    embedding_model: str = "intfloat/multilingual-e5-small",
    batch_size: int = 32,
    limit_books: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    repo_root = _repo_root()
    cfg0 = default_config(repo_root)
    cfg = RagConfig(
        raw_dir=Path(raw_dir) if raw_dir else cfg0.raw_dir,
        index_dir=Path(index_dir) if index_dir else cfg0.index_dir,
        embedding_model=embedding_model,
        chunk_words=cfg0.chunk_words,
        chunk_overlap_words=cfg0.chunk_overlap_words,
        min_chunk_words=cfg0.min_chunk_words,
    )
    out = build_or_update_index(cfg, limit_books=limit_books, batch_size=int(batch_size), device=device)
    _invalidate_index_cache(cfg.index_dir)
    return out


