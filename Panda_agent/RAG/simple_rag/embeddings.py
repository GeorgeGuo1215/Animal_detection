from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: np.ndarray
    dim: int


class Embedder:
    def __init__(self, model_name_or_path: str, device: Optional[str] = None, *, auto_e5_prefix: bool = True) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name_or_path = model_name_or_path
        self.auto_e5_prefix = bool(auto_e5_prefix)
        self.model = SentenceTransformer(model_name_or_path, device=device)

    def _looks_like_e5(self) -> bool:
        if not self.auto_e5_prefix:
            return False
        name = (self.model_name_or_path or "").lower()
        return "e5" in name

    @staticmethod
    def _with_prefix(texts: List[str], *, prefix: str) -> List[str]:
        out: List[str] = []
        pfx = prefix.strip().lower()
        for t in texts:
            s = (t or "").strip()
            s_low = s.lower()
            if s_low.startswith("query:") or s_low.startswith("passage:"):
                out.append(s)
            else:
                out.append(f"{pfx} {s}")
        return out

    def embed_texts(
        self,
        texts: List[str],
        *,
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> EmbeddingResult:
        if self._looks_like_e5():
            texts = self._with_prefix(texts, prefix="passage:")
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
        )
        arr = np.asarray(vecs, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"embedding dim error: expected 2D, got {arr.shape}")
        return EmbeddingResult(vectors=arr, dim=int(arr.shape[1]))

    def embed_queries(
        self,
        queries: Iterable[str],
        *,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> EmbeddingResult:
        qs = list(queries)
        if self._looks_like_e5():
            qs = self._with_prefix(qs, prefix="query:")
        return self.embed_texts(qs, batch_size=batch_size, normalize=normalize, show_progress=False)
