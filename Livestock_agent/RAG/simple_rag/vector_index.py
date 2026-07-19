from __future__ import annotations

from .text_utils import TextChunk


def chunk_to_meta(chunk: TextChunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "source_path": chunk.source_path,
        "source_sha1": chunk.source_sha1,
        "chunk_index": chunk.chunk_index,
        "n_words": chunk.n_words,
        "text": chunk.text,
    }
