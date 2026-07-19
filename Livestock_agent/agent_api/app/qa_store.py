"""SQLite-backed Q&A record storage for LivestockMind.

Stores every question/answer pair with metadata (tools used, RAG hits,
response time, etc.) for admin review and knowledge-gap analysis.
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import re


def _default_db_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "agent_api_logs" / "livestock_qa.db"


_DB_PATH: Optional[str] = None


def _get_db_path() -> str:
    global _DB_PATH
    if _DB_PATH is None:
        _DB_PATH = os.getenv("QA_DB_PATH", str(_default_db_path()))
    return _DB_PATH


def _connect() -> sqlite3.Connection:
    path = _get_db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS qa_records (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,
    date_key        TEXT    NOT NULL,
    question        TEXT    NOT NULL,
    answer          TEXT    NOT NULL DEFAULT '',
    model           TEXT    NOT NULL DEFAULT '',
    tools_used      TEXT    NOT NULL DEFAULT '[]',
    rag_hit_count   INTEGER NOT NULL DEFAULT 0,
    rag_best_score  REAL    NOT NULL DEFAULT 0.0,
    used_web_search INTEGER NOT NULL DEFAULT 0,
    response_time_ms INTEGER NOT NULL DEFAULT 0,
    source_ip       TEXT    NOT NULL DEFAULT '',
    user_role       TEXT    NOT NULL DEFAULT ''
);
"""

_MIGRATE_SQL = [
    "ALTER TABLE qa_records ADD COLUMN rag_best_score REAL NOT NULL DEFAULT 0.0",
    "ALTER TABLE qa_records ADD COLUMN request_id TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE qa_records ADD COLUMN feedback_rating INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE qa_records ADD COLUMN feedback_comment TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE qa_records ADD COLUMN feedback_ts TEXT NOT NULL DEFAULT ''",
]

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_qa_date_key ON qa_records(date_key);",
    "CREATE INDEX IF NOT EXISTS idx_qa_ts ON qa_records(ts);",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_qa_request_id ON qa_records(request_id) WHERE request_id != '';",
]


def init_db() -> None:
    conn = _connect()
    try:
        conn.execute(_CREATE_TABLE)
        for sql in _MIGRATE_SQL:
            try:
                conn.execute(sql)
            except sqlite3.OperationalError:
                pass  # column already exists
        for idx_sql in _CREATE_INDEXES:
            conn.execute(idx_sql)
        conn.commit()
        print(f"[qa_store] DB ready at {_get_db_path()}")
    finally:
        conn.close()


def _save_record_sync(
    question: str,
    answer: str,
    model: str = "",
    tools_used: Optional[List[str]] = None,
    rag_hit_count: int = 0,
    rag_best_score: float = 0.0,
    used_web_search: bool = False,
    response_time_ms: int = 0,
    source_ip: str = "",
    user_role: str = "",
    request_id: str = "",
) -> int:
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    date_key = time.strftime("%Y-%m-%d")
    conn = _connect()
    try:
        cur = conn.execute(
            """INSERT INTO qa_records
               (ts, date_key, question, answer, model, tools_used,
                rag_hit_count, rag_best_score, used_web_search, response_time_ms,
                source_ip, user_role, request_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                now, date_key, question, answer, model,
                json.dumps(tools_used or [], ensure_ascii=False),
                rag_hit_count, round(rag_best_score, 4), int(used_web_search),
                response_time_ms, source_ip, user_role, request_id,
            ),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]
    finally:
        conn.close()


async def save_qa_record(**kwargs: Any) -> int:
    """Async wrapper — runs the blocking SQLite insert in a thread."""
    return await asyncio.to_thread(_save_record_sync, **kwargs)


def _query_history_sync(
    page: int = 1,
    page_size: int = 20,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    keyword: Optional[str] = None,
    user_role: Optional[str] = None,
) -> Dict[str, Any]:
    conditions: List[str] = []
    params: List[Any] = []

    if date_from:
        conditions.append("date_key >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("date_key <= ?")
        params.append(date_to)
    if keyword:
        conditions.append("(question LIKE ? OR answer LIKE ?)")
        params.extend([f"%{keyword}%", f"%{keyword}%"])
    if user_role:
        conditions.append("user_role = ?")
        params.append(user_role)

    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""

    conn = _connect()
    try:
        total = conn.execute(f"SELECT COUNT(*) FROM qa_records{where}", params).fetchone()[0]
        offset = (max(page, 1) - 1) * page_size
        rows = conn.execute(
            f"SELECT * FROM qa_records{where} ORDER BY id DESC LIMIT ? OFFSET ?",
            params + [page_size, offset],
        ).fetchall()
        records = [dict(r) for r in rows]
        for rec in records:
            try:
                rec["tools_used"] = json.loads(rec.get("tools_used") or "[]")
            except Exception:
                pass
        return {"total": total, "page": page, "page_size": page_size, "records": records}
    finally:
        conn.close()


async def query_qa_history(**kwargs: Any) -> Dict[str, Any]:
    return await asyncio.to_thread(_query_history_sync, **kwargs)


def _get_stats_sync(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    user_role: Optional[str] = None,
) -> Dict[str, Any]:
    conditions: List[str] = []
    params: List[Any] = []
    if date_from:
        conditions.append("date_key >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("date_key <= ?")
        params.append(date_to)
    if user_role:
        conditions.append("user_role = ?")
        params.append(user_role)
    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""

    conn = _connect()
    try:
        row = conn.execute(
            f"""SELECT
                    COUNT(*)            AS total,
                    AVG(response_time_ms) AS avg_response_ms,
                    SUM(used_web_search)  AS web_search_count,
                    AVG(rag_hit_count)    AS avg_rag_hits
                FROM qa_records{where}""",
            params,
        ).fetchone()

        daily = conn.execute(
            f"""SELECT date_key, COUNT(*) AS count
                FROM qa_records{where}
                GROUP BY date_key ORDER BY date_key DESC LIMIT 30""",
            params,
        ).fetchall()

        model_dist = conn.execute(
            f"""SELECT model, COUNT(*) AS count
                FROM qa_records{where}
                GROUP BY model ORDER BY count DESC""",
            params,
        ).fetchall()

        role_dist = conn.execute(
            f"""SELECT user_role, COUNT(*) AS count
                FROM qa_records{where}
                GROUP BY user_role ORDER BY count DESC""",
            params,
        ).fetchall()

        return {
            "total": row["total"],
            "avg_response_ms": round(row["avg_response_ms"] or 0, 1),
            "web_search_count": row["web_search_count"] or 0,
            "avg_rag_hits": round(row["avg_rag_hits"] or 0, 2),
            "daily": [dict(d) for d in daily],
            "model_distribution": [dict(m) for m in model_dist],
            "role_distribution": [dict(r) for r in role_dist],
        }
    finally:
        conn.close()


async def get_qa_stats(**kwargs: Any) -> Dict[str, Any]:
    return await asyncio.to_thread(_get_stats_sync, **kwargs)


_EXAMPLE_VERIFICATION_HINTS = re.compile(r"(正确吗|对吗|是否正确|是否属实|真的假的|真的吗)")
_EXAMPLE_PREFIX_PATTERNS = [
    re.compile(r"^请问(?:一下)?[，,:：\s]*"),
    re.compile(r"^麻烦问下[，,:：\s]*"),
    re.compile(r"^想请教一下[，,:：\s]*"),
    re.compile(r"^以下(?:引号里|引号里面)?的内容(?:是否)?正确吗[？?:：\s]*"),
    re.compile(r"^请问以下(?:引号里|引号里面)?的内容(?:是否)?正确吗[？?:：\s]*"),
    re.compile(r"^请问引号里(?:面)?的内容(?:是否)?正确吗[？?:：\s]*"),
]
_EXAMPLE_QUOTE_PATTERN = re.compile(r"[“\"]([^”\"]{2,240})[”\"]")
_EXAMPLE_JUNK_PATTERN = re.compile(r"^[a-z0-9 _-]{1,12}$", re.IGNORECASE)


def _trim_example_text(text: str, limit: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip(" ，。；;、：:!?！？")
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip(" ，。；;、：:") + "…"


def _simplify_example_question(question: str, limit: int = 32) -> Optional[str]:
    text = re.sub(r"\s+", " ", (question or "")).strip()
    if not text:
        return None
    if _EXAMPLE_JUNK_PATTERN.fullmatch(text) and len(text) <= 8:
        return None

    is_verification = bool(_EXAMPLE_VERIFICATION_HINTS.search(text))
    quoted = _EXAMPLE_QUOTE_PATTERN.search(text)
    if quoted:
        core = quoted.group(1).strip()
    else:
        core = text
        for pattern in _EXAMPLE_PREFIX_PATTERNS:
            core = pattern.sub("", core, count=1).strip()
        if "：" in core:
            prefix, suffix = core.split("：", 1)
            if _EXAMPLE_VERIFICATION_HINTS.search(prefix):
                core = suffix.strip()
        elif ":" in core:
            prefix, suffix = core.split(":", 1)
            if _EXAMPLE_VERIFICATION_HINTS.search(prefix):
                core = suffix.strip()

    core = core.strip("“”\"' ")
    core = re.sub(r"^[，,:：\s]+", "", core)
    core = re.sub(r"[?？]+$", "", core)
    if not core:
        return None

    if is_verification:
        for delim in ("?", "？", "。", "！", "!", "；", ";"):
            if delim in core:
                head = core.split(delim, 1)[0].strip(" ，。；;、：:")
                if len(head) >= 4:
                    core = head
                    break
        if len(core) > limit and "，" in core:
            head = core.split("，", 1)[0].strip(" ，。；;、：:")
            if len(head) >= 4:
                core = head
        suffix = "，对吗？"
        core = _trim_example_text(core, max(4, limit - len(suffix)))
        return core + suffix

    for prefix in ("你好，", "你好,", "您好，", "您好,"):
        if core.startswith(prefix):
            core = core[len(prefix):].strip()
            break
    return _trim_example_text(core, limit)


def _get_recent_example_questions_sync(
    limit: int = 6,
    sample_size: int = 30,
    user_role: Optional[str] = None,
) -> List[str]:
    conn = _connect()
    try:
        conditions = ["TRIM(question) != ''"]
        params: List[Any] = []
        if user_role:
            conditions.append("user_role = ?")
            params.append(user_role)
        where = " WHERE " + " AND ".join(conditions)
        rows = conn.execute(
            f"""SELECT question
               FROM qa_records
               {where}
               ORDER BY id DESC
               LIMIT ?""",
            params + [max(limit * 3, sample_size)],
        ).fetchall()
        results: List[str] = []
        seen: set[str] = set()
        for row in rows:
            simplified = _simplify_example_question(str(row["question"] or ""))
            if not simplified:
                continue
            if simplified in seen:
                continue
            seen.add(simplified)
            results.append(simplified)
            if len(results) >= limit:
                break
        return results
    finally:
        conn.close()


async def get_recent_example_questions(
    limit: int = 6,
    sample_size: int = 30,
    user_role: Optional[str] = None,
) -> List[str]:
    return await asyncio.to_thread(
        _get_recent_example_questions_sync,
        limit=limit,
        sample_size=sample_size,
        user_role=user_role,
    )


_RAG_GAP_SCORE_THRESHOLD = float(os.getenv("RAG_DENSE_RELEVANCE_THRESHOLD", "0.12"))
_QUALITY_VERIFICATION_HINTS = re.compile(
    r"(正确吗|对吗|真的吗|是否正确|是否属实|验证|核实|辟谣|谣言|fact.?check)",
    re.IGNORECASE,
)
_QUALITY_MECHANISM_HINTS = re.compile(
    r"(为什么.*(?:能|会|可以|不会)|原理|机制|怎么做到|如何.*(?:分解|解毒|抵抗|消化)|"
    r"是为了.*吗|是不是为了|是否为了|为了.*(?:吗|么)|适应(?:环境|生态|生存).*(?:吗|么)|"
    r"mechanism|reason)",
    re.IGNORECASE,
)
_QUALITY_TIMELINESS_HINTS = re.compile(
    r"(最新|最近|近期|当前|目前|latest|recent|current|update|202[3-9])",
    re.IGNORECASE,
)
_QUALITY_KB_DEFLECTION_HINTS = re.compile(
    r"(无法证实|无法确认|当前知识库|现有资料|缺乏依据|暂无依据)",
    re.IGNORECASE,
)


def _normalize_backfill_question(question: str) -> str:
    text = re.sub(r"\s+", " ", (question or "")).strip()
    for prefix in ("你好，", "你好,", "您好，", "您好,"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    text = text.rstrip("？?。！!；; ")
    return text.lower()


def _rank_backfill_candidate(rec: Dict[str, Any]) -> tuple[int, int, int, float, int]:
    return (
        int(rec.get("used_web_search") or 0),
        int(rec.get("feedback_rating") or 0),
        len(str(rec.get("answer") or "")),
        float(rec.get("rag_best_score") or 0.0),
        int(rec.get("id") or 0),
    )


def _get_low_hit_backfill_records_sync(
    *,
    limit: int = 100,
    sample_size: int = 400,
    min_answer_chars: int = 120,
) -> List[Dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            f"""SELECT id, ts, date_key, question, answer, user_role,
                       rag_hit_count, rag_best_score, used_web_search, feedback_rating
                FROM qa_records
                WHERE (rag_hit_count = 0 OR (rag_hit_count > 0 AND rag_best_score < {_RAG_GAP_SCORE_THRESHOLD}))
                  AND TRIM(question) != ''
                  AND TRIM(answer) != ''
                ORDER BY id DESC
                LIMIT ?""",
            (max(limit * 3, sample_size),),
        ).fetchall()
        best_by_question: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            rec = dict(row)
            question = str(rec.get("question") or "").strip()
            answer = str(rec.get("answer") or "").strip()
            if not question or not answer:
                continue
            if _EXAMPLE_JUNK_PATTERN.fullmatch(question) and len(question) <= 8:
                continue
            if len(answer) < min_answer_chars:
                continue
            if _QUALITY_KB_DEFLECTION_HINTS.search(answer):
                continue
            normalized = _normalize_backfill_question(question)
            if not normalized:
                continue
            chosen = best_by_question.get(normalized)
            if chosen is None or _rank_backfill_candidate(rec) > _rank_backfill_candidate(chosen):
                best_by_question[normalized] = rec

        results = sorted(best_by_question.values(), key=lambda item: int(item.get("id") or 0), reverse=True)
        return results[:limit]
    finally:
        conn.close()


async def get_low_hit_backfill_records(
    *,
    limit: int = 100,
    sample_size: int = 400,
    min_answer_chars: int = 120,
) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(
        _get_low_hit_backfill_records_sync,
        limit=limit,
        sample_size=sample_size,
        min_answer_chars=min_answer_chars,
    )


def _get_knowledge_gaps_sync(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    min_occurrences: int = 1,
    limit: int = 50,
) -> Dict[str, Any]:
    """Find questions where RAG had no useful hits — potential knowledge-base gaps.

    A gap is defined as:
    - RAG returned zero hits, OR
    - RAG returned hits but the best rerank score was below the relevance
      threshold (content was retrieved but not actually relevant to the query).
    """
    conditions = [f"(rag_hit_count = 0 OR (rag_hit_count > 0 AND rag_best_score < {_RAG_GAP_SCORE_THRESHOLD}))"]
    params: List[Any] = []
    if date_from:
        conditions.append("date_key >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("date_key <= ?")
        params.append(date_to)
    where = " WHERE " + " AND ".join(conditions)

    conn = _connect()
    try:
        rows = conn.execute(
            f"""SELECT question, COUNT(*) AS occurrences,
                       MAX(used_web_search) AS ever_web_searched,
                       MAX(ts) AS last_asked,
                       MAX(rag_best_score) AS max_rag_score
                FROM qa_records{where}
                GROUP BY question
                HAVING COUNT(*) >= ?
                ORDER BY occurrences DESC
                LIMIT ?""",
            params + [min_occurrences, limit],
        ).fetchall()
        return {
            "total_gap_questions": len(rows),
            "gaps": [dict(r) for r in rows],
        }
    finally:
        conn.close()


async def get_knowledge_gaps(**kwargs: Any) -> Dict[str, Any]:
    return await asyncio.to_thread(_get_knowledge_gaps_sync, **kwargs)


def _quality_flags(question: str, answer: str, tools_used: List[str], rag_hit_count: int, rag_best_score: float) -> List[str]:
    flags: List[str] = []
    q = question or ""
    a = answer or ""
    used_web = any("web_search" in t for t in (tools_used or []))
    high_risk = bool(
        _QUALITY_VERIFICATION_HINTS.search(q)
        or _QUALITY_MECHANISM_HINTS.search(q)
        or _QUALITY_TIMELINESS_HINTS.search(q)
    )
    if high_risk and not used_web:
        flags.append("missing_web_cross_check")
    if rag_hit_count <= 0 and not used_web:
        flags.append("no_evidence_path")
    if rag_hit_count > 0 and rag_best_score < _RAG_GAP_SCORE_THRESHOLD and not used_web:
        flags.append("weak_rag_without_web")
    if len(a.strip()) < 40:
        flags.append("answer_too_short")
    if _QUALITY_KB_DEFLECTION_HINTS.search(a) and not used_web:
        flags.append("kb_only_deflection")
    return flags


def _get_quality_report_sync(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    user_role: Optional[str] = None,
    limit: int = 200,
    sample_size: int = 30,
) -> Dict[str, Any]:
    conditions: List[str] = []
    params: List[Any] = []
    if date_from:
        conditions.append("date_key >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("date_key <= ?")
        params.append(date_to)
    if user_role:
        conditions.append("user_role = ?")
        params.append(user_role)
    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""

    conn = _connect()
    try:
        rows = conn.execute(
            f"""SELECT id, ts, date_key, request_id, question, answer, model, tools_used,
                       rag_hit_count, rag_best_score, used_web_search, feedback_rating
                FROM qa_records{where}
                ORDER BY id DESC LIMIT ?""",
            params + [limit],
        ).fetchall()
        samples: List[Dict[str, Any]] = []
        flag_counts: Dict[str, int] = {}
        total_score = 0.0
        problematic_records = 0
        trend_map: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            rec = dict(row)
            try:
                rec["tools_used"] = json.loads(rec.get("tools_used") or "[]")
            except Exception:
                rec["tools_used"] = []
            flags = _quality_flags(
                question=str(rec.get("question") or ""),
                answer=str(rec.get("answer") or ""),
                tools_used=rec.get("tools_used") or [],
                rag_hit_count=int(rec.get("rag_hit_count") or 0),
                rag_best_score=float(rec.get("rag_best_score") or 0.0),
            )
            for flag in flags:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
            score = max(0, 100 - len(flags) * 20)
            if int(rec.get("feedback_rating") or 0) > 0:
                score = max(0, min(100, score + (int(rec["feedback_rating"]) - 3) * 5))
            total_score += score
            if flags:
                problematic_records += 1
            day = str(rec.get("date_key") or "")
            if day:
                bucket = trend_map.setdefault(day, {
                    "date_key": day,
                    "evaluated_count": 0,
                    "problem_count": 0,
                    "total_score": 0.0,
                    "missing_web_cross_check": 0,
                })
                bucket["evaluated_count"] += 1
                bucket["total_score"] += score
                if flags:
                    bucket["problem_count"] += 1
                if "missing_web_cross_check" in flags:
                    bucket["missing_web_cross_check"] += 1
            if flags and len(samples) < sample_size:
                samples.append({
                    "id": rec["id"],
                    "ts": rec["ts"],
                    "request_id": rec.get("request_id", ""),
                    "question": rec["question"],
                    "model": rec["model"],
                    "tools_used": rec["tools_used"],
                    "rag_hit_count": rec["rag_hit_count"],
                    "rag_best_score": rec["rag_best_score"],
                    "feedback_rating": rec.get("feedback_rating", 0),
                    "quality_score": score,
                    "flags": flags,
                })

        total = len(rows)
        trend = []
        for day in sorted(trend_map.keys(), reverse=True):
            bucket = trend_map[day]
            count = int(bucket["evaluated_count"] or 0)
            problem_count = int(bucket["problem_count"] or 0)
            avg_score = round(float(bucket["total_score"] or 0.0) / count, 2) if count else 0.0
            trend.append({
                "date_key": day,
                "evaluated_count": count,
                "problem_count": problem_count,
                "problem_rate": round((problem_count / count) * 100, 2) if count else 0.0,
                "avg_quality_score": avg_score,
                "missing_web_cross_check": int(bucket["missing_web_cross_check"] or 0),
            })

        def _window_stats(items: List[Dict[str, Any]]) -> Dict[str, Any]:
            evaluated = sum(int(x.get("evaluated_count") or 0) for x in items)
            problems = sum(int(x.get("problem_count") or 0) for x in items)
            total_window_score = sum(float(x.get("avg_quality_score") or 0.0) * int(x.get("evaluated_count") or 0) for x in items)
            return {
                "evaluated_count": evaluated,
                "problem_count": problems,
                "problem_rate": round((problems / evaluated) * 100, 2) if evaluated else 0.0,
                "avg_quality_score": round(total_window_score / evaluated, 2) if evaluated else 0.0,
            }

        recent_window = _window_stats(trend[:7])
        previous_window = _window_stats(trend[7:14])
        trend_summary = {
            "recent_7d": recent_window,
            "previous_7d": previous_window,
            "quality_score_delta": round(recent_window["avg_quality_score"] - previous_window["avg_quality_score"], 2) if previous_window["evaluated_count"] else 0.0,
            "problem_rate_delta": round(recent_window["problem_rate"] - previous_window["problem_rate"], 2) if previous_window["evaluated_count"] else 0.0,
        }
        return {
            "evaluated_count": total,
            "avg_quality_score": round(total_score / total, 2) if total else 0.0,
            "problem_count": problematic_records,
            "problem_rate": round((problematic_records / total) * 100, 2) if total else 0.0,
            "flag_counts": flag_counts,
            "trend": trend[:30],
            "trend_summary": trend_summary,
            "samples": samples,
        }
    finally:
        conn.close()


async def get_quality_report(**kwargs: Any) -> Dict[str, Any]:
    return await asyncio.to_thread(_get_quality_report_sync, **kwargs)


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

def _submit_feedback_sync(
    request_id: str,
    rating: int,
    comment: str = "",
) -> str:
    """Update feedback columns for the record matching *request_id*.

    Returns one of: "ok", "not_found", "already_rated".
    """
    if not request_id:
        return "not_found"
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id, feedback_rating FROM qa_records WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if not row:
            return "not_found"
        if row["feedback_rating"] != 0:
            return "already_rated"
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        conn.execute(
            "UPDATE qa_records SET feedback_rating = ?, feedback_comment = ?, feedback_ts = ? WHERE id = ?",
            (rating, comment, now, row["id"]),
        )
        conn.commit()
        return "ok"
    finally:
        conn.close()


async def submit_feedback(**kwargs: Any) -> str:
    return await asyncio.to_thread(_submit_feedback_sync, **kwargs)


def _get_feedback_stats_sync(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    user_role: Optional[str] = None,
) -> Dict[str, Any]:
    conditions: List[str] = []
    params: List[Any] = []
    if date_from:
        conditions.append("date_key >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("date_key <= ?")
        params.append(date_to)
    if user_role:
        conditions.append("user_role = ?")
        params.append(user_role)
    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""

    rated_where = (where + " AND " if where else " WHERE ") + "feedback_rating > 0"

    conn = _connect()
    try:
        total_row = conn.execute(f"SELECT COUNT(*) AS c FROM qa_records{where}", params).fetchone()
        total = total_row["c"]

        rated_row = conn.execute(
            f"SELECT COUNT(*) AS c, AVG(feedback_rating) AS avg_r FROM qa_records{rated_where}", params,
        ).fetchone()
        rated_count = rated_row["c"]
        avg_rating = round(rated_row["avg_r"] or 0, 2)

        dist_rows = conn.execute(
            f"SELECT feedback_rating AS r, COUNT(*) AS c FROM qa_records{rated_where} GROUP BY feedback_rating ORDER BY r",
            params,
        ).fetchall()
        distribution = {str(r["r"]): r["c"] for r in dist_rows}

        recent = conn.execute(
            f"""SELECT request_id, question, feedback_rating, feedback_comment, feedback_ts, user_role
                FROM qa_records{rated_where}
                ORDER BY feedback_ts DESC LIMIT 50""",
            params,
        ).fetchall()

        return {
            "total_answers": total,
            "rated_count": rated_count,
            "avg_rating": avg_rating,
            "distribution": distribution,
            "recent_feedback": [dict(r) for r in recent],
        }
    finally:
        conn.close()


async def get_feedback_stats(**kwargs: Any) -> Dict[str, Any]:
    return await asyncio.to_thread(_get_feedback_stats_sync, **kwargs)
