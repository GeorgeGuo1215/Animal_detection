from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RagConfig:
    raw_dir: Path
    index_dir: Path
    embedding_model: str = "intfloat/multilingual-e5-small"
    chunk_words: int = 800
    chunk_overlap_words: int = 150
    min_chunk_words: int = 40


def default_config(repo_root: Path) -> RagConfig:
    rag_dir = repo_root / "RAG"
    return RagConfig(
        raw_dir=rag_dir / "data" / "raw",
        index_dir=rag_dir / "data" / "rag_index_e5",
    )
