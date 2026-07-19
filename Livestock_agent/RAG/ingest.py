from __future__ import annotations

import argparse
import json
from pathlib import Path

from simple_rag.config import RagConfig, default_config
from simple_rag.pipeline import build_or_update_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vector index from extracted panda book text files")
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--index-dir", type=str, default=None)
    parser.add_argument("--embedding-model", type=str, default="intfloat/multilingual-e5-small")
    parser.add_argument("--chunk-words", type=int, default=800)
    parser.add_argument("--chunk-overlap-words", type=int, default=150)
    parser.add_argument("--min-chunk-words", type=int, default=40)
    parser.add_argument("--limit-books", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[0].parent
    cfg0 = default_config(repo_root)
    cfg = RagConfig(
        raw_dir=Path(args.raw_dir) if args.raw_dir else cfg0.raw_dir,
        index_dir=Path(args.index_dir) if args.index_dir else cfg0.index_dir,
        embedding_model=args.embedding_model,
        chunk_words=args.chunk_words,
        chunk_overlap_words=args.chunk_overlap_words,
        min_chunk_words=args.min_chunk_words,
    )
    stats = build_or_update_index(
        cfg, limit_books=args.limit_books, batch_size=args.batch_size, device=args.device,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
