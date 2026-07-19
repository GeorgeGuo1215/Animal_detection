from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Session:
    session_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    pending_clarification: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)


class SessionManager:
    def __init__(self, max_sessions: int = 1000, ttl_seconds: float = 3600.0):
        self._sessions: Dict[str, Session] = {}
        self._max = max_sessions
        self._ttl = ttl_seconds

    def _cleanup(self) -> None:
        now = time.time()
        expired = [k for k, v in self._sessions.items() if now - v.last_active > self._ttl]
        for k in expired:
            del self._sessions[k]

    async def create(self) -> Session:
        self._cleanup()
        sess = Session(session_id=uuid.uuid4().hex)
        self._sessions[sess.session_id] = sess
        return sess

    async def get(self, session_id: str) -> Optional[Session]:
        self._cleanup()
        sess = self._sessions.get(session_id)
        if sess:
            sess.last_active = time.time()
        return sess

    async def append_messages(self, session_id: str, new_messages: List[Dict[str, Any]], *, keep_last: int = 20) -> Optional[Session]:
        sess = await self.get(session_id)
        if not sess:
            return None
        for msg in new_messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "").strip()
            content = str(msg.get("content") or "").strip()
            if not role or not content:
                continue
            sess.messages.append({"role": role, "content": content})
        if keep_last > 0 and len(sess.messages) > keep_last:
            sess.messages = sess.messages[-keep_last:]
        sess.last_active = time.time()
        return sess

    async def set_preference(self, session_id: str, key: str, value: Any) -> Optional[Session]:
        sess = await self.get(session_id)
        if not sess or not key:
            return None
        sess.preferences[key] = value
        sess.last_active = time.time()
        return sess

    async def set_pending_clarification(self, session_id: str, payload: Optional[Dict[str, Any]]) -> Optional[Session]:
        sess = await self.get(session_id)
        if not sess:
            return None
        sess.pending_clarification = dict(payload or {}) if payload else None
        sess.last_active = time.time()
        return sess

    async def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None


_MGR: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    global _MGR
    if _MGR is None:
        _MGR = SessionManager()
    return _MGR
