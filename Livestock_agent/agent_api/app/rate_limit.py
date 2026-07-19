from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, Tuple

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate: float = 30.0, burst: int = 30):
        super().__init__(app)
        self.rate = float(rate)
        self.burst = int(burst)
        self._buckets: Dict[str, Tuple[float, float]] = defaultdict(lambda: (float(burst), 0.0))

    def _get_key(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"

    async def dispatch(self, request: Request, call_next):
        key = self._get_key(request)
        now = time.monotonic()
        tokens, last = self._buckets[key]
        elapsed = now - last if last else 0.0
        tokens = min(float(self.burst), tokens + elapsed * self.rate)
        if tokens < 1.0:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        tokens -= 1.0
        self._buckets[key] = (tokens, now)
        return await call_next(request)
