"""Ponto de entrada: `python -m scriptmax`."""
from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from scriptmax.config import ConfigError, Settings, load_settings
from scriptmax.emailer import EmailSender
from scriptmax.jobs import JobQueue
from scriptmax.library import PdfLibrary
from scriptmax.pipeline import ReportPipeline
from scriptmax.security import SessionAuth
from scriptmax.server import AppServices, create_app
from scriptmax.storage import ReportStore
from scriptmax.summarization import Summarizer
from scriptmax.transcription import GroqTranscriber

if TYPE_CHECKING:
    from fastapi import FastAPI


def build_services(settings: Settings) -> AppServices:
    store = ReportStore(settings.reports_dir)
    library = PdfLibrary(settings.library_dir)
    pipeline = ReportPipeline(
        transcriber=GroqTranscriber.from_api_key(
            settings.groq_api_key, settings.transcription_model, settings.ffmpeg_path, settings.transcript_cache_dir
        ),
        summarizer=Summarizer.from_api_key(settings.deepseek_api_key, settings.summary_model),
        store=store,
        library=library,
        emailer=EmailSender(settings.email) if settings.email else None,
    )
    return AppServices(
        store=store,
        pipeline=pipeline,
        jobs=JobQueue(pipeline),
        library=library,
        auth=SessionAuth(settings.app_token),
        uploads_dir=settings.uploads_dir,
        max_upload_bytes=settings.max_upload_bytes,
        email_enabled=settings.email is not None,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        settings = load_settings()
    except ConfigError as error:
        logging.error("Configuração inválida: %s", error)
        return 1
    app = create_app(build_services(settings))
    logging.info("ScriptMax em http://%s:%d", settings.host, settings.port)
    if settings.behind_proxy:
        _serve_behind_proxy(app, settings)
    else:
        import uvicorn

        # proxy_headers só confia em 127.0.0.1 (ex.: ngrok local) para obter o esquema https.
        uvicorn.run(app, host=settings.host, port=settings.port, proxy_headers=True, forwarded_allow_ips="127.0.0.1")
    return 0


def _serve_behind_proxy(app: FastAPI, settings: Settings) -> None:
    """Cloud Run: Hypercorn fala HTTP/2 em texto puro (h2c), o que remove o limite
    de 32 MB por requisição do HTTP/1; ProxyFix confia só no último salto do
    X-Forwarded-For/Proto, adicionado pelo front-end do Google."""
    import asyncio

    from hypercorn.asyncio import serve
    from hypercorn.config import Config
    from hypercorn.middleware import ProxyFixMiddleware

    config = Config()
    config.bind = [f"{settings.host}:{settings.port}"]
    asyncio.run(serve(ProxyFixMiddleware(app, mode="legacy", trusted_hops=1), config))


if __name__ == "__main__":
    sys.exit(main())
