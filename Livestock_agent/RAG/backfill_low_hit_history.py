from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from agent_api.app.qa_store import _get_low_hit_backfill_records_sync
from simple_rag.config import default_config
from simple_rag.pipeline import build_or_update_index


DEFAULT_OUTPUT_NAME = "历史低命中问题回灌语料_自动生成.txt"


def _role_label(role: str) -> str:
    if role == "researcher":
        return "科研人员"
    if role == "enthusiast":
        return "爱好者"
    return "未标注"


def _render_backfill_corpus(records: List[Dict[str, Any]]) -> str:
    role_counts = Counter(_role_label(str(rec.get("user_role") or "")) for rec in records)
    role_summary = "；".join(f"{role} {count} 条" for role, count in sorted(role_counts.items()))
    lines = [
        "历史低命中问题回灌语料（自动生成）",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "用途：将问答历史中 RAG 自动命中率低、但已有较完整回答的问题回灌到知识库，提升后续检索覆盖率。",
        f"回灌条目数：{len(records)}",
        f"角色分布：{role_summary or '暂无'}",
        "",
    ]
    for idx, rec in enumerate(records, start=1):
        question = str(rec.get("question") or "").strip()
        answer = str(rec.get("answer") or "").strip()
        ts = str(rec.get("ts") or "")
        role = _role_label(str(rec.get("user_role") or ""))
        rag_hits = int(rec.get("rag_hit_count") or 0)
        rag_score = float(rec.get("rag_best_score") or 0.0)
        lines.extend(
            [
                f"### 条目 {idx}",
                f"标准问题：{question}",
                f"适用角色：{role}",
                f"来源时间：{ts}",
                f"低命中信号：RAG 命中 {rag_hits}，最高相关分 {rag_score:.4f}",
                "参考回答：",
                answer,
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def sync_low_hit_history_to_corpus(
    *,
    limit: int = 100,
    sample_size: int = 400,
    min_answer_chars: int = 120,
    output_path: str | None = None,
    reindex: bool = False,
    batch_size: int = 32,
    device: str | None = None,
) -> Dict[str, Any]:
    cfg = default_config(REPO_ROOT)
    out_path = Path(output_path) if output_path else (cfg.raw_dir / DEFAULT_OUTPUT_NAME)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = _get_low_hit_backfill_records_sync(
        limit=limit,
        sample_size=sample_size,
        min_answer_chars=min_answer_chars,
    )
    corpus_text = _render_backfill_corpus(records)
    out_path.write_text(corpus_text, encoding="utf-8")

    role_distribution = Counter(str(rec.get("user_role") or "") for rec in records)
    result: Dict[str, Any] = {
        "output_path": str(out_path),
        "record_count": len(records),
        "role_distribution": dict(role_distribution),
        "source_ids": [int(rec.get("id") or 0) for rec in records],
    }

    if reindex:
        if cfg.index_dir.exists():
            shutil.rmtree(cfg.index_dir)
        result["reindex"] = build_or_update_index(
            cfg,
            batch_size=batch_size,
            device=device,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill low-hit QA history into the RAG raw corpus")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=400)
    parser.add_argument("--min-answer-chars", type=int, default=120)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    result = sync_low_hit_history_to_corpus(
        limit=args.limit,
        sample_size=args.sample_size,
        min_answer_chars=args.min_answer_chars,
        output_path=args.output_path,
        reindex=args.reindex,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
