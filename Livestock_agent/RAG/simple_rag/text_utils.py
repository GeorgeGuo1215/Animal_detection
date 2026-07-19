from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


_RE_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_PAGE_MARKER = re.compile(r"---\s*Page\s+\d+\s+of\s+\d+\s*---", re.IGNORECASE)
_RE_WHITESPACE = re.compile(r"[ \t]+")
_RE_SENT_SPLIT = re.compile(r"(?<=[。！？.!?])(?:\s+|(?=[\u4e00-\u9fff\u3400-\u4dbf]))")


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def read_text_lossy(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    return raw.decode("utf-8", errors="ignore"), sha1_bytes(raw)


def cleanup_text(text: str) -> str:
    """
    Clean extracted PDF text:
    - Remove page markers (--- Page N of M ---)
    - Remove markdown image refs and HTML tags
    - Normalize whitespace
    """
    text = _RE_PAGE_MARKER.sub("\n\n", text)
    text = _RE_MD_IMAGE.sub("", text)
    text = _RE_HTML_TAG.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _RE_WHITESPACE.sub(" ", text)

    lines = [ln.strip() for ln in text.split("\n")]
    out: list[str] = []
    blank = 0
    for ln in lines:
        if not ln:
            blank += 1
            if blank <= 2:
                out.append("")
            continue
        blank = 0
        out.append(ln)
    return "\n".join(out).strip()


def iter_pages(clean_text: str) -> Iterator[str]:
    buf: list[str] = []
    for ln in clean_text.split("\n"):
        if ln.strip() == "":
            if buf:
                yield "\n".join(buf).strip()
                buf = []
            continue
        buf.append(ln)
    if buf:
        yield "\n".join(buf).strip()


_RE_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")


def word_count(text: str) -> int:
    """Count words, handling CJK characters (each ~= 1 word) and space-separated tokens."""
    s = text.strip()
    if not s:
        return 0
    cjk_chars = len(_RE_CJK.findall(s))
    non_cjk = _RE_CJK.sub(" ", s)
    latin_words = len([w for w in re.split(r"\s+", non_cjk.strip()) if w])
    return cjk_chars + latin_words


def split_sentences(text: str) -> list[str]:
    s = re.sub(r"\s+", " ", (text or "").strip())
    if not s:
        return []
    parts = _RE_SENT_SPLIT.split(s)
    out = [p.strip() for p in parts if p and p.strip()]
    return out


def recursive_sentence_chunks(
    *,
    text: str,
    chunk_words: int,
    chunk_overlap_words: int,
    min_chunk_words: int,
) -> list[str]:
    assert chunk_words > 0
    assert 0 <= chunk_overlap_words < chunk_words

    def split_fallback(s: str) -> list[str]:
        s = s.strip()
        if not s:
            return []
        parts = re.split(r"(?<=[,;，；])(?:\s+|(?=[\u4e00-\u9fff]))", s)
        parts = [p.strip() for p in parts if p and p.strip()]
        if len(parts) >= 2:
            return parts
        if len(s) > chunk_words * 2:
            step = max(chunk_words // 2, 200)
            return [s[i:i + step].strip() for i in range(0, len(s), step) if s[i:i + step].strip()]
        return [s]

    sentences: list[str] = []
    for sent in split_sentences(text):
        if word_count(sent) > chunk_words * 1.2:
            sentences.extend(split_fallback(sent))
        else:
            sentences.append(sent)

    merged: list[str] = []
    cur: list[str] = []
    cur_words = 0
    for s in sentences:
        w = word_count(s)
        if cur and (cur_words + w) > chunk_words:
            merged.append("\n".join(cur).strip())
            overlap_sents: list[str] = []
            overlap_wc = 0
            if chunk_overlap_words > 0:
                for prev_s in reversed(cur):
                    pw = word_count(prev_s)
                    if overlap_wc + pw > chunk_overlap_words:
                        break
                    overlap_sents.insert(0, prev_s)
                    overlap_wc += pw
            cur = overlap_sents + [s]
            cur_words = overlap_wc + w
        else:
            cur.append(s)
            cur_words += w
    if cur:
        merged.append("\n".join(cur).strip())

    return [block for block in merged if word_count(block) >= min_chunk_words]


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    source_path: str
    source_sha1: str
    chunk_index: int
    text: str
    n_words: int


def chunk_text(
    *,
    source_path: Path,
    source_sha1: str,
    clean_text: str,
    chunk_words: int,
    chunk_overlap_words: int,
    min_chunk_words: int,
) -> list[TextChunk]:
    assert chunk_words > 0
    assert 0 <= chunk_overlap_words < chunk_words

    paragraphs = [p for p in iter_pages(clean_text) if p]

    merged: list[str] = []
    for p in paragraphs:
        merged.extend(
            recursive_sentence_chunks(
                text=p,
                chunk_words=chunk_words,
                chunk_overlap_words=chunk_overlap_words,
                min_chunk_words=min_chunk_words,
            )
        )

    chunks: list[TextChunk] = []
    for i, block in enumerate(merged):
        text = block.strip()
        n_words = word_count(text)
        if n_words < min_chunk_words:
            continue

        chunk_id = hashlib.sha1(
            (str(source_path).lower() + ":" + source_sha1 + ":" + str(i)).encode("utf-8")
        ).hexdigest()
        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                source_path=str(source_path),
                source_sha1=source_sha1,
                chunk_index=i,
                text=text,
                n_words=n_words,
            )
        )
    return chunks


def iter_txt_files(raw_dir: Path) -> Iterable[Path]:
    """Glob .txt files (extracted from PDFs) instead of .mmd."""
    return sorted(raw_dir.rglob("*.txt"))
