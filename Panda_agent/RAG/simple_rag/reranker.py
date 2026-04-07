from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RerankResult:
    index: int
    score: float


class CrossEncoderReranker:
    def __init__(self, model_name_or_path: str, *, device: Optional[str] = None) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name_or_path = model_name_or_path
        self.model = CrossEncoder(model_name_or_path, device=device)

    def score_pairs(
        self,
        *,
        query: str,
        passages: Sequence[str],
        batch_size: int = 32,
    ) -> List[float]:
        q = (query or "").strip()
        pairs: List[Tuple[str, str]] = [(q, (p or "").strip()) for p in passages]
        scores = self.model.predict(pairs, batch_size=int(batch_size), show_progress_bar=False)
        return [float(s) for s in scores]

    def rerank(
        self,
        *,
        query: str,
        passages: Sequence[str],
        top_k: int,
        batch_size: int = 32,
    ) -> List[RerankResult]:
        if top_k <= 0 or not passages:
            return []
        scores = self.score_pairs(query=query, passages=passages, batch_size=batch_size)
        results = [RerankResult(index=i, score=float(s)) for i, s in enumerate(scores)]
        results.sort(key=lambda x: x.score, reverse=True)
        return results[: int(top_k)]
