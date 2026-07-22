"""兽医视角 MoE：对已启动的后端发 HTTP 请求，落盘 SSE 事件与终答。

用法：
    # 先启动服务（agentAndRag 目录）
    #   python -m uvicorn agent_api.app.main:app --host 127.0.0.1 --port 8000
    # 或 start_agent.bat

    python tests/moe/run_moe_vet_live.py
    python tests/moe/run_moe_vet_live.py --base-url http://127.0.0.1:8000 --case bulldog

输出：tests/moe/reports/vet_live_<timestamp>/
  - case_*.md（请求、SSE 进度、终答、风格验收）
  - case_*_answer.txt
  - SUMMARY.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

_THIS = Path(__file__).resolve()
_AGENT_API = _THIS.parents[2]
_AGENTANDRAG = _THIS.parents[3]
for _p in (str(_AGENT_API), str(_AGENTANDRAG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_dotenv() -> None:
    env_path = _AGENTANDRAG / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:  # noqa: BLE001
        pass


_load_dotenv()
os.environ.setdefault("HTTPX_TRUST_ENV", "0")

from app.services.plan_and_solve import is_concrete_vet_case  # noqa: E402

_FLUTD = (
    "5 岁的英短，公猫，已经绝育。今天从早上开始就一直往猫砂盆跑，差不多十几分钟去一次，"
    "每次蹲很久，但是宠主看不出来到底有没有尿出来。猫砂盆里好像只有一点点尿团，比平时少很多。"
    "它今天不太爱吃东西，平时早上会主动来要罐头，今天只舔了几口。精神也差一些，老是趴着，还会舔下面。"
    "没有明显呕吐，但是刚才好像干呕了一下。昨天晚上还挺正常的。最近没有换粮，喝水感觉和平时差不多。"
    "宠主家里最近来了客人，它有点紧张，躲了两天。疫苗应该是去年打的，驱虫不太记得。"
    "之前有过一次尿血，可能是膀胱炎，吃药后好了。今天还没有去医院，也没有吃药。"
)

CASES: List[Dict[str, Any]] = [
    {
        "id": "flutd_diagnosis",
        "title": "英短病例 + 请做出诊断（应出病例工作流）",
        "expect_concrete": True,
        "expect_sections": ["病例整理", "问题列表", "检查与治疗方案", "风险提示"],
        "question": _FLUTD + "\n请对以上病例做出诊断。",
    },
    {
        "id": "flutd_organize",
        "title": "英短病例 + 请整理病例格式（应出病例工作流）",
        "expect_concrete": True,
        "expect_sections": ["病例整理", "问题列表", "风险提示"],
        "question": _FLUTD + "\n请对以上病例做出整理，输出统一的病例格式。",
    },
    {
        "id": "bulldog",
        "title": "法斗运动评估（不应强制病例三件套）",
        "expect_concrete": False,
        "forbid_required_sections": True,
        "expect_keywords": ["短吻", "热", "运动", "呼吸"],
        "question": (
            "3岁法斗（法国斗牛犬）公犬，已绝育，体重12kg。"
            "主人想每天带它剧烈跑步或追球1小时，最近热天遛弯就张口喘、不愿走。"
            "从兽医角度评估运动建议与风险边界，并说明与普通中型犬的差异。"
        ),
    },
    {
        "id": "vague",
        "title": "笼统对照：猫尿血怎么办",
        "expect_concrete": False,
        "forbid_required_sections": True,
        "question": "猫尿血怎么办",
    },
]


def _default_api_key() -> str:
    keys_path = _AGENT_API / "keys.txt"
    if keys_path.exists():
        for line in keys_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    return os.getenv("AGENT_API_KEY") or "sk-petmind-default-key-2026"


def _style_checks(case: Dict[str, Any], answer: str) -> List[str]:
    issues: List[str] = []
    q = case["question"]
    concrete = is_concrete_vet_case(q)
    if case.get("expect_concrete") and not concrete:
        issues.append("启发式未识别为病例工作流意图（与 expect_concrete 不符）")
    if case.get("expect_concrete") is False and concrete:
        issues.append("启发式误判为病例工作流意图")
    for sec in case.get("expect_sections") or []:
        if sec not in answer:
            issues.append(f"终答缺少分节关键词: {sec}")
    if case.get("forbid_required_sections"):
        if all(s in answer for s in ("病例整理", "问题列表", "检查与治疗方案")):
            issues.append("非诊断/整理意图不应强制完整病例三件套")
    kws = case.get("expect_keywords") or []
    if kws and not any(k in answer for k in kws):
        issues.append(f"终答未命中任一特异化关键词: {kws}")
    return issues


def _parse_sse_block(block: str) -> Optional[Dict[str, Any]]:
    data_lines = []
    for line in block.splitlines():
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if not data_lines:
        return None
    payload = "\n".join(data_lines).strip()
    if not payload or payload == "[DONE]":
        return {"done": True}
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {"raw": payload}


def _chat_stream(
    *,
    base_url: str,
    api_key: str,
    question: str,
    max_tokens: int,
    timeout_s: float,
) -> Tuple[str, List[Dict[str, Any]]]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": "agent-moe",
        "stream": True,
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "user_role": "veterinarian",
        "messages": [{"role": "user", "content": question}],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    events: List[Dict[str, Any]] = []
    answer_parts: List[str] = []
    with httpx.Client(timeout=httpx.Timeout(connect=10, read=timeout_s, write=30, pool=30), trust_env=False) as client:
        with client.stream("POST", url, headers=headers, json=body) as resp:
            if resp.status_code >= 400:
                err = resp.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {resp.status_code}: {err[:800]}")
            buf = ""
            for chunk in resp.iter_text():
                buf += chunk
                while "\n\n" in buf:
                    block, buf = buf.split("\n\n", 1)
                    obj = _parse_sse_block(block)
                    if not obj:
                        continue
                    if obj.get("done"):
                        events.append({"type": "done"})
                        continue
                    events.append(obj)
                    # OpenAI chunk: content under streaming status; also collect any delta content
                    status = obj.get("agent_status")
                    detail = obj.get("agent_detail") or {}
                    choices = obj.get("choices") or []
                    content = ""
                    if choices:
                        content = ((choices[0].get("delta") or {}).get("content")) or ""
                    if status == "streaming" and content:
                        answer_parts.append(content)
                    elif status and status != "streaming":
                        # keep progress breadcrumbs in events only
                        _ = detail
    return "".join(answer_parts), events


def _render_report(
    case: Dict[str, Any],
    *,
    question: str,
    answer: str,
    events: List[Dict[str, Any]],
    issues: List[str],
    base_url: str,
) -> str:
    lines: List[str] = []
    lines.append("# Vet MoE HTTP Live Report")
    lines.append("")
    lines.append(f"- 时间: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- case_id: {case['id']}")
    lines.append(f"- title: {case['title']}")
    lines.append(f"- base_url: {base_url}")
    lines.append(f"- user_role: veterinarian")
    lines.append(f"- concrete_gate: {is_concrete_vet_case(question)}")
    lines.append("")
    lines.append("## 请求问题")
    lines.append("")
    lines.append("```text")
    lines.append(question)
    lines.append("```")
    lines.append("")
    lines.append("## SSE 进度摘要")
    lines.append("")
    lines.append("| # | agent_status | detail 摘要 |")
    lines.append("| --- | --- | --- |")
    for i, ev in enumerate(events, 1):
        if ev.get("type") == "done":
            lines.append(f"| {i} | done | |")
            continue
        st = ev.get("agent_status") or ""
        detail = ev.get("agent_detail") or {}
        summary = json.dumps(detail, ensure_ascii=False)
        if len(summary) > 160:
            summary = summary[:160] + "…"
        summary = summary.replace("|", "\\|")
        lines.append(f"| {i} | {st} | {summary} |")
    lines.append("")
    lines.append("## 最终答案")
    lines.append("")
    lines.append(answer or "_（空）_")
    lines.append("")
    lines.append("## 风格验收")
    lines.append("")
    if issues:
        lines.extend(f"- FAIL: {x}" for x in issues)
    else:
        lines.append("- PASS")
    lines.append("")
    lines.append("## 原始 SSE 事件（截断保存）")
    lines.append("")
    lines.append("```json")
    # Drop huge content deltas; keep status events + short streaming markers
    slim: List[Any] = []
    for ev in events:
        if ev.get("type") == "done":
            slim.append(ev)
            continue
        st = ev.get("agent_status")
        if st == "streaming":
            continue
        slim.append({
            "agent_status": st,
            "agent_detail": ev.get("agent_detail"),
            "finish_reason": ((ev.get("choices") or [{}])[0].get("finish_reason")),
        })
    lines.append(json.dumps(slim, ensure_ascii=False, indent=2)[:20000])
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="兽医 MoE：HTTP 请求已启动后端")
    p.add_argument("--base-url", default=os.getenv("AGENT_BASE_URL") or "http://127.0.0.1:8000")
    p.add_argument("--api-key", default=None, help="默认读 agent_api/keys.txt")
    p.add_argument("--case", action="append", help="只跑指定 case id；可多次")
    p.add_argument("--max-tokens", type=int, default=1500)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    api_key = args.api_key or _default_api_key()
    base = args.base_url.rstrip("/")

    # health check
    try:
        with httpx.Client(timeout=5.0, trust_env=False) as c:
            r = c.get(base + "/health")
            r.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"[错误] 无法连接 {base}/health：{exc}")
        print("请先启动：python -m uvicorn agent_api.app.main:app --host 127.0.0.1 --port 8000")
        raise SystemExit(2)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (_THIS.parent / "reports" / f"vet_live_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = CASES
    if args.case:
        ids = set(args.case)
        selected = [c for c in CASES if c["id"] in ids]
        if not selected:
            print(f"[错误] 无匹配 case: {args.case}")
            raise SystemExit(2)

    summary: List[str] = []
    all_ok = True
    for case in selected:
        print(f"\n=== RUN {case['id']}: {case['title']} ===")
        q = case["question"]
        try:
            answer, events = _chat_stream(
                base_url=base,
                api_key=api_key,
                question=q,
                max_tokens=args.max_tokens,
                timeout_s=args.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] 请求异常: {exc}")
            all_ok = False
            summary.append(f"- {case['id']}: ERROR ({exc})")
            continue
        issues = _style_checks(case, answer)
        md = _render_report(case, question=q, answer=answer, events=events, issues=issues, base_url=base)
        out_path = out_dir / f"case_{case['id']}.md"
        out_path.write_text(md, encoding="utf-8")
        (out_dir / f"case_{case['id']}_answer.txt").write_text(answer or "", encoding="utf-8")
        ok = not issues
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] -> {out_path}")
        for iss in issues:
            print(f"  - {iss}")
        summary.append(f"- {case['id']}: {status} ({out_path.name})")

    (out_dir / "SUMMARY.md").write_text("# Vet MoE Live Summary\n\n" + "\n".join(summary) + "\n", encoding="utf-8")
    print(f"\n报告目录: {out_dir}")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
