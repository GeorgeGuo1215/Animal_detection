from __future__ import annotations

from typing import Dict, List, Tuple


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    合并同一 source_path 下重叠/相邻的 [l, r] 区间。
    """
    if not intervals:
        return []
    ivs = sorted((int(l), int(r)) for l, r in intervals)
    out: List[Tuple[int, int]] = []
    cur_l, cur_r = ivs[0]
    for l, r in ivs[1:]:
        if l <= cur_r + 1:  # 相邻也合并，避免重复上下文
            cur_r = max(cur_r, r)
        else:
            out.append((cur_l, cur_r))
            cur_l, cur_r = l, r
    out.append((cur_l, cur_r))
    return out


def build_neighbor_contexts(
    *,
    metas: List[dict],
    hits: List[dict],
    neighbor_n: int,
) -> List[dict]:
    """
    小块检索命中后，把同一 source_path 的邻居 chunk 拼接成更完整的 context。

    入参约定：
    - metas: 来自 meta.jsonl 的列表（每项至少含 source_path/chunk_index/text）
    - hits: 检索命中列表（每项至少含 source_path/chunk_index/score）

    返回：
    - contexts: List[dict]
      - score: float（该 context 区间内命中 hit 的 max score）
      - source_path: str
      - chunk_index_start/end: int
      - n_chunks: int（拼接包含多少个 chunk）
      - text: str（拼接后的文本）
    """
    n = int(neighbor_n)
    if n <= 0 or not hits:
        return []

    # source_path -> chunk_index -> text
    by_src: Dict[str, Dict[int, str]] = {}
    for m in metas:
        sp = m.get("source_path")
        ci = m.get("chunk_index")
        txt = m.get("text")
        if not isinstance(sp, str) or not sp:
            continue
        try:
            cii = int(ci)
        except Exception:
            continue
        if not isinstance(txt, str) or not txt.strip():
            continue
        by_src.setdefault(sp, {})[cii] = txt

    # 1) 由命中 hits 构造“窗口区间”
    win_by_src: Dict[str, List[Tuple[int, int]]] = {}
    for h in hits:
        sp = h.get("source_path")
        ci = h.get("chunk_index")
        if not isinstance(sp, str) or not sp:
            continue
        try:
            cii = int(ci)
        except Exception:
            continue
        l, r = cii - n, cii + n
        win_by_src.setdefault(sp, []).append((l, r))

    # 2) 合并窗口、拼接文本
    contexts: List[dict] = []
    for sp, ivs in win_by_src.items():
        merged = _merge_intervals(ivs)
        chunks = by_src.get(sp, {})
        for l, r in merged:
            parts: List[str] = []
            for idx in range(int(l), int(r) + 1):
                t = chunks.get(idx)
                if t:
                    parts.append(t.strip())
            if not parts:
                continue
            text = "\n\n".join(parts).strip()

            # 区间分数：取该 source_path 下落在这个合并区间里的命中 hit 的 max score
            best = float("-inf")
            for h in hits:
                if h.get("source_path") != sp:
                    continue
                try:
                    cii = int(h.get("chunk_index"))
                except Exception:
                    continue
                if int(l) <= cii <= int(r):
                    try:
                        best = max(best, float(h.get("score") or 0.0))
                    except Exception:
                        pass
            if best == float("-inf"):
                best = 0.0

            contexts.append(
                {
                    "score": float(best),
                    "source_path": sp,
                    "chunk_index_start": int(l),
                    "chunk_index_end": int(r),
                    "n_chunks": int(len(parts)),
                    "text": text,
                }
            )

    contexts.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    return contexts


