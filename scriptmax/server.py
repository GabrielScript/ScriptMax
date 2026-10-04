"""API HTTP (FastAPI) + frontend estático (HTML/JS)."""
from __future__ import annotations

import logging
import os
import re
import sys
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from scriptmax.categories import PROFILES, Category
from scriptmax.jobs import JobNotFound, JobNotRetryable, JobQueue, JobView, purge_stale_uploads
from scriptmax.library import InvalidFolder, PdfLibrary, normalize_folder, safe_segment
from scriptmax.pipeline import ProcessRequest, ReportPipeline
from scriptmax.rendering import REPORT_CSP
from scriptmax.security import (
    APP_CSP, BASE_SECURITY_HEADERS, SESSION_COOKIE, SESSION_MAX_AGE_SECONDS,
    LoginRateLimiter, SessionAuth, is_direct_local_request, origin_is_trusted,
)
from scriptmax.storage import ReportFile, ReportMeta, ReportNotFound, ReportStore

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
ALLOWED_AUDIO_EXTENSIONS = frozenset(
    {".mp3", ".m4a", ".wav", ".ogg", ".oga", ".opus", ".webm", ".flac", ".mp4", ".aac", ".mpeg", ".mpga"}
)
MAX_SUBJECT_CHARS = 120
MAX_SOURCE_NAME_CHARS = 120
COPY_BUFFER_BYTES = 1024 * 1024
MULTIPART_OVERHEAD_BYTES = 64 * 1024
UPLOAD_PATH = "/api/jobs"
CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")
SERVED_FILES = {
    "html": (ReportFile.HTML, "text/html; charset=utf-8", "html"),
    "pdf": (ReportFile.PDF, "application/pdf", "pdf"),
    "transcript": (ReportFile.TRANSCRIPT_TEXT, "text/plain; charset=utf-8", "txt"),
    "markdown": (ReportFile.MARKDOWN, "text/markdown; charset=utf-8", "md"),
}


@dataclass(frozen=True)
class AppServices:
    store: ReportStore
    pipeline: ReportPipeline
    jobs: JobQueue
    library: PdfLibrary
    auth: SessionAuth
    uploads_dir: Path
    max_upload_bytes: int
    email_enabled: bool


class LoginBody(BaseModel):
    token: str = Field(min_length=1, max_length=512)


class MoveBody(BaseModel):
    category: Category
    folder: str = Field(default="", max_length=300)


def clean_text(raw: str, max_chars: int) -> str:
    return re.sub(r"\s+", " ", CONTROL_CHARACTERS.sub(" ", raw)).strip()[:max_chars]


def create_app(services: AppServices) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        services.uploads_dir.mkdir(parents=True, exist_ok=True)
        removed = purge_stale_uploads(services.uploads_dir)
        if removed:
            logger.info("Removidos %d upload(s) antigos.", removed)
        yield
        services.jobs.shutdown()

    app = FastAPI(title="ScriptMax", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def request_guard(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        request_id = uuid.uuid4().hex[:12]
        try:
            response = _reject_upload_before_parsing(request, services) or await call_next(request)
        except Exception:
            # Erro inesperado: registra com contexto, responde genérico (sem stack trace ao cliente).
            logger.exception("Erro não tratado [%s] %s %s", request_id, request.method, request.url.path)
            response = JSONResponse({"detail": f"Erro interno. Código: {request_id}"}, status_code=500)
        response.headers["X-Request-ID"] = request_id
        for header, value in BASE_SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        response.headers.setdefault("Content-Security-Policy", APP_CSP)
        return response

    app.include_router(_public_router(services))
    app.include_router(_protected_router(services))
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    return app


def _public_router(services: AppServices) -> APIRouter:
    router = APIRouter(prefix="/api")
    rate_limiter = LoginRateLimiter()

    @router.get("/config")
    def config(request: Request) -> dict[str, object]:
        return {
            "token_required": services.auth.token_required,
            "authenticated": services.auth.is_authorized(request),
            "email_enabled": services.email_enabled,
            "max_upload_mb": services.max_upload_bytes // (1024 * 1024),
            "categories": [{"id": category.value, "label": profile.label} for category, profile in PROFILES.items()],
            "library_dir": str(services.library.root),
            "can_open_library": sys.platform == "win32" and is_direct_local_request(request),
        }

    @router.post("/login", status_code=204)
    def login(body: LoginBody, request: Request, response: Response) -> None:
        if not origin_is_trusted(request):
            raise HTTPException(403, "Origem não permitida.")
        client_key = request.client.host if request.client else "desconhecido"
        if not rate_limiter.allow(client_key):
            raise HTTPException(429, "Muitas tentativas. Aguarde 15 minutos.")
        if not services.auth.token_matches(body.token):
            raise HTTPException(401, "Token inválido.")
        response.set_cookie(
            SESSION_COOKIE, services.auth.session_cookie_value(), max_age=SESSION_MAX_AGE_SECONDS,
            httponly=True, samesite="strict", secure=request.url.scheme == "https", path="/",
        )

    @router.post("/logout", status_code=204)
    def logout(response: Response) -> None:
        response.delete_cookie(SESSION_COOKIE, path="/")

    return router


def _require_session(services: AppServices) -> Callable[[Request], None]:
    def dependency(request: Request) -> None:
        if not services.auth.is_authorized(request):
            raise HTTPException(401, "Faça login.")
        if request.method not in {"GET", "HEAD", "OPTIONS"} and not origin_is_trusted(request):
            raise HTTPException(403, "Origem não permitida.")

    return dependency


def _protected_router(services: AppServices) -> APIRouter:
    router = APIRouter(prefix="/api", dependencies=[Depends(_require_session(services))])
    _add_job_routes(router, services)
    _add_report_routes(router, services)
    return router


def _add_job_routes(router: APIRouter, services: AppServices) -> None:
    @router.get("/jobs")
    def list_jobs() -> list[JobView]:
        return services.jobs.list()

    @router.post("/jobs", status_code=202)
    def create_job(
        file: Annotated[UploadFile, File()],
        subject: Annotated[str, Form()],
        category: Annotated[Category, Form()],
        folder: Annotated[str, Form(max_length=300)] = "",
        send_email: Annotated[bool, Form()] = False,
    ) -> JobView:
        clean_subject = clean_text(subject, MAX_SUBJECT_CHARS)
        if not clean_subject:
            raise HTTPException(400, "Informe o título/assunto.")
        try:
            clean_folder = normalize_folder(folder)
        except InvalidFolder as error:
            raise HTTPException(400, str(error)) from error
        source_name = clean_text(Path(file.filename or "gravacao").name, MAX_SOURCE_NAME_CHARS)
        extension = Path(source_name).suffix.lower()
        if extension not in ALLOWED_AUDIO_EXTENSIONS:
            raise HTTPException(400, f"Formato não suportado: {extension or 'sem extensão'}.")

        audio_path = _save_upload(file, services.uploads_dir / f"{uuid.uuid4().hex}{extension}", services.max_upload_bytes)
        return services.jobs.submit_process(
            ProcessRequest(
                audio_path=audio_path, source_name=source_name, subject=clean_subject, category=category,
                folder=clean_folder, send_email=send_email and services.email_enabled,
            )
        )

    @router.get("/jobs/{job_id}")
    def get_job(job_id: str) -> JobView:
        try:
            return services.jobs.get(job_id)
        except JobNotFound as error:
            raise HTTPException(404, "Processamento não encontrado.") from error

    @router.post("/jobs/{job_id}/retry", status_code=202)
    def retry_job(job_id: str) -> JobView:
        try:
            return services.jobs.retry(job_id)
        except JobNotFound as error:
            raise HTTPException(404, "Processamento não encontrado.") from error
        except JobNotRetryable as error:
            raise HTTPException(409, "Este processamento não pode ser refeito.") from error


def _add_report_routes(router: APIRouter, services: AppServices) -> None:
    def load_meta(report_id: str) -> ReportMeta:
        try:
            return services.store.get(report_id)
        except ReportNotFound as error:
            raise HTTPException(404, "Relatório não encontrado.") from error

    @router.get("/reports")
    def list_reports() -> list[ReportMeta]:
        return services.store.list()

    @router.patch("/reports/{report_id}")
    def move_report(report_id: str, body: MoveBody) -> ReportMeta:
        load_meta(report_id)
        try:
            return services.pipeline.move(report_id, body.category, body.folder)
        except InvalidFolder as error:
            raise HTTPException(400, str(error)) from error

    @router.delete("/reports/{report_id}", status_code=204)
    def delete_report(report_id: str) -> None:
        load_meta(report_id)
        services.pipeline.delete(report_id)

    @router.post("/reports/{report_id}/regenerate", status_code=202)
    def regenerate_report(report_id: str) -> JobView:
        meta = load_meta(report_id)
        return services.jobs.submit_regenerate(report_id, meta.subject)

    @router.get("/reports/{report_id}/files/{kind}")
    def report_file(report_id: str, kind: str, download: bool = False) -> FileResponse:
        meta = load_meta(report_id)
        if kind not in SERVED_FILES:
            raise HTTPException(404, "Arquivo desconhecido.")
        report_file_kind, media_type, extension = SERVED_FILES[kind]
        try:
            path = services.store.file_path(report_id, report_file_kind)
        except ReportNotFound as error:
            raise HTTPException(404, "Arquivo ainda não gerado.") from error
        headers = {"Content-Security-Policy": REPORT_CSP} if kind == "html" else {}
        return FileResponse(
            path, media_type=media_type, headers=headers,
            filename=f"{safe_segment(meta.subject) or 'relatorio'}.{extension}",
            content_disposition_type="attachment" if download else "inline",
        )

    @router.post("/library/open", status_code=204)
    def open_library(request: Request) -> None:
        if sys.platform != "win32" or not is_direct_local_request(request):
            raise HTTPException(403, "Disponível só no próprio PC (Windows).")
        os.startfile(services.library.root)  # type: ignore[attr-defined]  # noqa: S606 - caminho fixo da config


def _reject_upload_before_parsing(request: Request, services: AppServices) -> Response | None:
    """O FastAPI grava o multipart em disco ANTES das dependências: autentica e mede aqui."""
    if request.method != "POST" or request.url.path != UPLOAD_PATH:
        return None
    if not services.auth.is_authorized(request):
        return JSONResponse({"detail": "Faça login."}, status_code=401)
    if not origin_is_trusted(request):
        return JSONResponse({"detail": "Origem não permitida."}, status_code=403)
    declared = request.headers.get("content-length", "")
    if not declared.isdigit():
        return JSONResponse({"detail": "Envio sem Content-Length."}, status_code=411)
    if int(declared) > services.max_upload_bytes + MULTIPART_OVERHEAD_BYTES:
        limit_mb = services.max_upload_bytes // (1024 * 1024)
        return JSONResponse({"detail": f"Arquivo maior que {limit_mb} MB."}, status_code=413)
    return None


def _save_upload(upload: UploadFile, destination: Path, max_bytes: int) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with destination.open("wb") as target:
            while block := upload.file.read(COPY_BUFFER_BYTES):
                written += len(block)
                if written > max_bytes:
                    raise HTTPException(413, f"Arquivo maior que {max_bytes // (1024 * 1024)} MB.")
                target.write(block)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    if written == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(400, "Arquivo vazio.")
    return destination
