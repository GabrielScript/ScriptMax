"""Fila de processamento em segundo plano.

Um worker só: serializa o uso das APIs (limites do plano gratuito da Groq) e
o Chromium do PDF. O processamento sobrevive a refresh/fechamento da aba.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

from scriptmax.pipeline import ProcessRequest, ReportBuildError, ReportPipeline, Stage

logger = logging.getLogger(__name__)

FINISHED_JOBS_KEPT = 50
STALE_UPLOAD_SECONDS = 3 * 24 * 3600

JobWork = Callable[[Callable[[Stage, float, str], None]], tuple[str, str]]


class JobNotFound(LookupError):
    """Job inexistente."""


class JobNotRetryable(RuntimeError):
    """Job não falhou ou não guarda o áudio para nova tentativa."""


@dataclass(frozen=True)
class JobView:
    id: str
    title: str
    stage: Stage
    progress: float
    message: str
    report_id: str | None
    error: str | None
    retryable: bool
    created_at: float


@dataclass
class _Job:
    view: JobView
    work: JobWork
    upload: Path | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class JobQueue:
    def __init__(self, pipeline: ReportPipeline) -> None:
        self._pipeline = pipeline
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scriptmax-job")
        self._jobs: dict[str, _Job] = {}
        self._registry_lock = threading.Lock()

    def submit_process(self, request: ProcessRequest) -> JobView:
        return self._submit(
            title=f"{request.subject} — {request.source_name}",
            work=lambda report: self._pipeline.process(request, report),
            upload=request.audio_path,
        )

    def submit_regenerate(self, report_id: str, title: str) -> JobView:
        return self._submit(title=f"Regerar: {title}", work=lambda report: self._pipeline.regenerate(report_id, report))

    def retry(self, job_id: str) -> JobView:
        job = self._get(job_id)
        if job.view.stage is not Stage.FAILED or (job.upload is not None and not job.upload.exists()):
            raise JobNotRetryable(job_id)
        self._update(job, stage=Stage.QUEUED, progress=0.0, message="Na fila (nova tentativa)", error=None, retryable=False)
        self._executor.submit(self._run, job)
        return job.view

    def get(self, job_id: str) -> JobView:
        return self._get(job_id).view

    def list(self) -> list[JobView]:
        with self._registry_lock:
            views = [job.view for job in self._jobs.values()]
        return sorted(views, key=lambda view: view.created_at, reverse=True)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _submit(self, title: str, work: JobWork, upload: Path | None = None) -> JobView:
        view = JobView(
            id=uuid.uuid4().hex, title=title, stage=Stage.QUEUED, progress=0.0, message="Na fila",
            report_id=None, error=None, retryable=False, created_at=time.time(),
        )
        job = _Job(view=view, work=work, upload=upload)
        with self._registry_lock:
            self._jobs[view.id] = job
            self._forget_old_jobs()
        self._executor.submit(self._run, job)
        return view

    def _run(self, job: _Job) -> None:
        def report(stage: Stage, progress: float, message: str) -> None:
            self._update(job, stage=stage, progress=max(0.0, min(progress, 1.0)), message=message)

        try:
            report_id, message = job.work(report)
        except ReportBuildError as error:
            logger.exception("Job %s: relatório falhou após a transcrição", job.view.id)
            self._discard_upload(job)
            self._update(job, stage=Stage.FAILED, message="Falhou", error=str(error), report_id=error.report_id)
            return
        except Exception as error:  # noqa: BLE001 - qualquer falha precisa virar estado visível do job
            logger.exception("Job %s falhou", job.view.id)
            self._update(job, stage=Stage.FAILED, message="Falhou", error=str(error) or type(error).__name__, retryable=True)
            return
        self._discard_upload(job)
        self._update(job, stage=Stage.DONE, progress=1.0, message=message, report_id=report_id)

    @staticmethod
    def _discard_upload(job: _Job) -> None:
        if job.upload is not None:
            job.upload.unlink(missing_ok=True)

    def _update(self, job: _Job, **changes: object) -> None:
        with job.lock:
            job.view = replace(job.view, **changes)  # type: ignore[arg-type]

    def _get(self, job_id: str) -> _Job:
        with self._registry_lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFound(job_id)
        return job

    def _forget_old_jobs(self) -> None:
        finished = [job for job in self._jobs.values() if job.view.stage is Stage.DONE]
        finished.sort(key=lambda job: job.view.created_at)
        for job in finished[:-FINISHED_JOBS_KEPT]:
            del self._jobs[job.view.id]


def purge_stale_uploads(uploads_dir: Path, max_age_seconds: int = STALE_UPLOAD_SECONDS) -> int:
    """Apaga uploads órfãos (jobs que falharam e nunca foram refeitos antes de um reinício)."""
    if not uploads_dir.is_dir():
        return 0
    cutoff = time.time() - max_age_seconds
    removed = 0
    for path in uploads_dir.iterdir():
        if path.is_file() and path.stat().st_mtime < cutoff:
            path.unlink(missing_ok=True)
            removed += 1
    return removed
