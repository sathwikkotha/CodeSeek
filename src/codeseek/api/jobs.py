"""In-process background job tracking for long-running requests (currently
just /ingest): the HTTP handler returns a job id immediately instead of
blocking the request for the full clone+chunk+embed duration, which can run
tens of minutes for a large repo -- the caller polls GET /ingest/{job_id}
for progress instead of holding one connection open the whole time.

In-memory and single-process on purpose: this matches the project's current
single-machine deployment (see docs/scaling_design.md). A multi-worker
deployment would need job state shared across processes (Redis, a DB row)
instead of this dict -- a real, known limitation of this simple version, not
an oversight."""

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

JobState = Literal["pending", "running", "done", "error"]


@dataclass
class IngestJob:
    id: str
    state: JobState = "pending"
    repo: str | None = None
    chunks_indexed: int | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class JobStore:
    """Thread-safe in-memory registry of IngestJob records."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, IngestJob] = {}

    def create(self) -> IngestJob:
        job = IngestJob(id=str(uuid.uuid4()))
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> IngestJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def run(self, job_id: str, fn: Callable[[], tuple[str, int]]) -> None:
        """Run fn() (the real clone+index work) on a background thread and
        update the job's state as it progresses. fn returns (repo_name, chunks_indexed)
        on success, or raises -- the exception's message becomes job.error."""

        def _set(**fields: object) -> None:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    for key, value in fields.items():
                        setattr(job, key, value)

        def _worker() -> None:
            _set(state="running")
            try:
                repo_name, total = fn()
            except Exception as exc:
                _set(state="error", error=str(exc))
                return
            _set(state="done", repo=repo_name, chunks_indexed=total)

        threading.Thread(target=_worker, daemon=True).start()
