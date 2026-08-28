"""Фоновые задачи: индексация эталонов и анализ тендера."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


@dataclass
class JobRecord:
    id: str
    kind: str
    status: JobStatus = JobStatus.pending
    progress: float = 0.0
    message: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def create(self, kind: str) -> JobRecord:
        job = JobRecord(id=uuid.uuid4().hex, kind=kind)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        progress: float | None = None,
        message: str | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = progress
            if message is not None:
                job.message = message
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error

    def run_in_background(self, job_id: str, fn: Callable[[], dict[str, Any] | None]) -> None:
        def worker() -> None:
            self.update(job_id, status=JobStatus.running, progress=0.01, message="Запуск…")
            try:
                result = fn()
                self.update(
                    job_id,
                    status=JobStatus.done,
                    progress=1.0,
                    message="Готово",
                    result=result or {},
                )
            except Exception as exc:
                self.update(
                    job_id,
                    status=JobStatus.failed,
                    progress=1.0,
                    message="Ошибка",
                    error=str(exc),
                )

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def make_progress(self, job_id: str) -> Callable[[str, float], None]:
        def update(message: str, value: float) -> None:
            self.update(
                job_id,
                status=JobStatus.running,
                progress=min(max(value, 0.0), 1.0),
                message=message,
            )

        return update


job_manager = JobManager()
