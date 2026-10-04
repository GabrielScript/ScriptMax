from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from scriptmax.jobs import JobQueue
from scriptmax.library import PdfLibrary
from scriptmax.pipeline import ReportPipeline
from scriptmax.security import SessionAuth
from scriptmax.server import AppServices
from scriptmax.storage import ReportStore
from scriptmax.summarization import SummaryRequest, SummaryResult
from scriptmax.transcription import Segment, Transcript

FFMPEG = shutil.which("ffmpeg")
requires_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg não instalado")


class FakeTranscriber:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str]] = []

    def transcribe(self, audio_path: Path, context_hint: str, on_progress=None) -> Transcript:
        self.calls.append((audio_path, context_hint))
        if on_progress:
            on_progress(1, 1)
        return Transcript(segments=[Segment(0.0, 5.0, "Hoje vamos estudar o produto interno.")], duration_seconds=5.0)


class FakeSummarizer:
    def __init__(self) -> None:
        self.requests: list[SummaryRequest] = []

    def summarize(self, transcript_text: str, request: SummaryRequest, on_progress=None) -> SummaryResult:
        self.requests.append(request)
        return SummaryResult(markdown="# Produto interno\n\nDefinição: $\\langle u, v \\rangle$. Custa R$ 10.", approach="A")


@pytest.fixture()
def fake_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evita subir o Chromium nos testes da API."""

    def write_pdf(html_path: Path, pdf_path: Path, page) -> bool:
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        return True

    monkeypatch.setattr("scriptmax.pipeline.write_pdf", write_pdf)


def build_services(tmp_path: Path, app_token: str | None = None) -> AppServices:
    store = ReportStore(tmp_path / "reports")
    library = PdfLibrary(tmp_path / "biblioteca")
    pipeline = ReportPipeline(FakeTranscriber(), FakeSummarizer(), store, library, emailer=None)
    return AppServices(
        store=store,
        pipeline=pipeline,
        jobs=JobQueue(pipeline),
        library=library,
        auth=SessionAuth(app_token),
        uploads_dir=tmp_path / "uploads",
        max_upload_bytes=1024 * 1024,
        email_enabled=False,
    )
