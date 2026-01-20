from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, Optional


JobStatus = Literal["queued", "running", "succeeded", "failed"]


@dataclass
class Job:
    job_id: str
    status: JobStatus = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


_LOCK = threading.Lock()
_JOBS: Dict[str, Job] = {}


def create_job(fn: Callable[[], Dict[str, Any]]) -> Job:
    job_id = uuid.uuid4().hex
    job = Job(job_id=job_id)
    with _LOCK:
        _JOBS[job_id] = job

    def _runner():
        job.status = "running"
        job.started_at = time.time()
        try:
            job.result = fn()
            job.status = "succeeded"
        except Exception as e:  # noqa: BLE001
            job.status = "failed"
            job.error = str(e)
        finally:
            job.finished_at = time.time()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    return job


def get_job(job_id: str) -> Optional[Job]:
    with _LOCK:
        return _JOBS.get(job_id)


