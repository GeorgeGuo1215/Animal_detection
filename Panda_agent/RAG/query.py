from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import List

from simple_rag.config import RagConfig, default_config
from simple_rag.pipeline import search
from simple_rag.query_rewrite import LLMRewriter, NoRewrite, TemplateRewriter
from simple_rag.retrieval import build_default_multiroute
from simple_rag.context_utils import build_neighbor_contexts
from simple_rag.reranker import CrossEncoderReranker
from simple_rag.vector_store import NumpyVectorStore


_RE_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\\-]{2,}")


def overlap_score(query: str, ctx: str) -> float:
    tq = set(t.lower() for t in _RE_WORD.findall(query or ""))
    if not tq:
        return 0.0
    tc = set(t.lower() for t in _RE_WORD.findall(ctx or ""))
    if not tc:
        return 0.0
    return len(tq & tc) / float(len(tq))


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the Panda RAG index")
    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--as-json", action="store_true")
    parser.add_argument("--multi-route", action="store_true")
    parser.add_argument("--rewrite", type=str, default="template", choices=["none", "template", "llm"])
    parser.add_argument("--expand-neighbors", type=int, default=1)
    parser.add_argument("--embedding-model", type=str, default="intfloat/multilingual-e5-small")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[0].parent
    cfg0 = default_config(repo_root)
    cfg = RagConfig(
        raw_dir=cfg0.raw_dir,
        index_dir=cfg0.index_dir,
        embedding_model=args.embedding_model,
    )

    if not args.multi_route:
        hits = search(cfg, args.query, top_k=args.top_k, device=args.device)
    else:
        if args.rewrite == "none":
            rewriter = NoRewrite()
        elif args.rewrite == "llm":
            rewriter = LLMRewriter()
        else:
            rewriter = TemplateRewriter()
        mr = build_default_multiroute(
            index_dir=str(cfg.index_dir), embedding_model=cfg.embedding_model,
            device=args.device, enable_bm25=True, rewriter=rewriter,
        )
        hits = [
            {"score": float(h.score), "source_path": h.meta.get("source_path"),
             "chunk_index": h.meta.get("chunk_index"), "n_words": h.meta.get("n_words"),
             "text": h.meta.get("text"), "chunk_id": h.meta.get("chunk_id")}
            for h in mr.retrieve(args.query, top_k=args.top_k)
        ]

    contexts: List[dict] = []
    if int(args.expand_neighbors) > 0 and hits:
        store = NumpyVectorStore(cfg.index_dir)
        store.load()
        contexts = build_neighbor_contexts(metas=store._meta, hits=hits, neighbor_n=int(args.expand_neighbors))

    if args.as_json:
        print(json.dumps({"hits": hits, "contexts": contexts} if contexts else hits, ensure_ascii=False, indent=2))
        return

    target = contexts if contexts else hits
    for i, h in enumerate(target[:args.top_k], 1):
        print(f"\n[{i}] score={h.get('score', 0):.4f}  source: {h.get('source_path', '')}")
        text = (h.get("text") or "").strip()
        if len(text) > 1200:
            text = text[:1200] + "\n...(truncated)..."
        print(text)


if __name__ == "__main__":
    main()
