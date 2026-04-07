from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Session:
    session_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
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

    async def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None


_MGR: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    global _MGR
    if _MGR is None:
        _MGR = SessionManager()
    return _MGR
