"""Orquestra: áudio -> transcrição -> relatório (HTML/PDF) -> biblioteca -> e-mail.

A transcrição é salva ANTES da sumarização: se a DeepSeek falhar, o relatório
pode ser regerado sem pagar a transcrição de novo.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from scriptmax.categories import Category, profile_for
from scriptmax.emailer import EmailError, EmailSender
from scriptmax.library import PdfLibrary, normalize_folder
from scriptmax.memory import MemoryBlock, collect_memory
from scriptmax.rendering import ReportPage, render_html, write_pdf
from scriptmax.storage import ReportFile, ReportMeta, ReportStore
from scriptmax.summarization import ProgressCallback, SummaryError, SummaryRequest, SummaryResult
from scriptmax.transcription import Transcript, format_timestamp

logger = logging.getLogger(__name__)


class Stage(str, Enum):
    QUEUED = "queued"
    TRANSCRIBING = "transcribing"
    SUMMARIZING = "summarizing"
    RENDERING = "rendering"
    EMAILING = "emailing"
    DONE = "done"
    FAILED = "failed"


ProgressReporter = Callable[[Stage, float, str], None]

CARD_WARNING = "Aviso: a ficha de memória não foi gerada e será refeita no próximo item."
MEMORY_WARNING = "Aviso: a memória da pasta não pôde ser montada; o relatório foi gerado sem ela."


class ReportBuildError(RuntimeError):
    """A transcrição foi salva, mas o relatório falhou: deve ser regerado, não reprocessado."""

    def __init__(self, report_id: str, cause: Exception) -> None:
        super().__init__(f"Transcrição salva, mas o relatório falhou: {cause}. Use 'Gerar relatório' na biblioteca.")
        self.report_id = report_id


class TranscriberPort(Protocol):
    def transcribe(self, audio_path: Path, context_hint: str, on_progress: ProgressCallback | None = None) -> Transcript: ...


class SummarizerPort(Protocol):
    def summarize(self, transcript_text: str, request: SummaryRequest, on_progress: ProgressCallback | None = None) -> SummaryResult: ...

    def write_memory_card(self, request: SummaryRequest, approach: str, report_markdown: str) -> str: ...


@dataclass(frozen=True)
class ProcessRequest:
    audio_path: Path
    source_name: str
    subject: str
    category: Category
    folder: str
    send_email: bool


class ReportPipeline:
    def __init__(
        self,
        transcriber: TranscriberPort,
        summarizer: SummarizerPort,
        store: ReportStore,
        library: PdfLibrary,
        emailer: EmailSender | None,
    ) -> None:
        self._transcriber = transcriber
        self._summarizer = summarizer
        self._store = store
        self._library = library
        self._emailer = emailer

    def process(self, request: ProcessRequest, report: ProgressReporter) -> tuple[str, str]:
        """Retorna (id do relatório, mensagem final)."""
        meta = ReportMeta.new(request.subject, request.source_name, request.category.value, normalize_folder(request.folder))
        context_hint = f"{profile_for(request.category).whisper_hint} {request.subject}"
        report(Stage.TRANSCRIBING, 0.0, "Transcrevendo com Whisper large-v3 (Groq)...")
        transcript = self._transcriber.transcribe(
            request.audio_path,
            context_hint,
            on_progress=lambda done, total: report(Stage.TRANSCRIBING, done / total, f"Transcrevendo bloco {done}/{total}"),
        )
        meta.duration_seconds = round(transcript.duration_seconds, 1)
        self._store.write_text(meta.id, ReportFile.TRANSCRIPT_JSON, transcript.to_json())
        self._store.write_text(meta.id, ReportFile.TRANSCRIPT_TEXT, transcript.to_timestamped_text())
        self._store.save_meta(meta)
        try:
            return meta.id, self._build_report(meta, transcript.text, report, request.send_email)
        except Exception as error:
            raise ReportBuildError(meta.id, error) from error

    def regenerate(self, report_id: str, report: ProgressReporter) -> tuple[str, str]:
        meta = self._store.get(report_id)
        transcript_json = self._store.file_path(report_id, ReportFile.TRANSCRIPT_JSON).read_text(encoding="utf-8")
        transcript = Transcript.from_json(transcript_json)
        return meta.id, self._build_report(meta, transcript.text, report, send_email=False)

    def move(self, report_id: str, category: Category, folder: str) -> ReportMeta:
        meta = self._store.get(report_id)
        meta.category = category.value
        meta.folder = normalize_folder(folder)
        if meta.report_ready:
            pdf_path = self._store.file_path(report_id, ReportFile.PDF)
            self._publish_to_library(meta, pdf_path)
        self._store.save_meta(meta)
        return meta

    def delete(self, report_id: str) -> None:
        meta = self._store.get(report_id)
        self._library.remove(meta.library_pdf)
        self._store.delete(report_id)

    def _build_report(self, meta: ReportMeta, transcript_text: str, report: ProgressReporter, send_email: bool) -> str:
        category = Category(meta.category)
        warnings: list[str] = []
        # Uma ficha velha nunca vale depois de regerar: ela é refeita junto com o relatório.
        self._store.remove_file(meta.id, ReportFile.MEMORY)
        memory = self._collect_memory(meta, report, warnings)
        report(Stage.SUMMARIZING, 0.0, "Gerando relatório com DeepSeek...")
        summary = self._summarizer.summarize(
            transcript_text,
            SummaryRequest(subject=meta.subject, category=category, memory=memory.text),
            on_progress=lambda done, total: report(Stage.SUMMARIZING, done / total, f"Relatório: trecho {done}/{total}"),
        )
        meta.memory_items = memory.items
        self._write_card(meta, category, memory, summary, report, warnings)
        report(Stage.RENDERING, 0.0, "Gerando HTML e PDF...")
        page = ReportPage(title=meta.subject, subtitle=_subtitle(meta, category), markdown_text=summary.markdown)
        self._store.write_text(meta.id, ReportFile.MARKDOWN, summary.markdown)
        html_path = self._store.write_text(meta.id, ReportFile.HTML, render_html(page))
        pdf_path = self._store.directory(meta.id) / ReportFile.PDF.value
        meta.math_rendered = write_pdf(html_path, pdf_path, page)
        meta.approach = summary.approach
        meta.complete = summary.is_complete
        meta.usage = asdict(summary.usage)
        meta.report_ready = True
        self._publish_to_library(meta, pdf_path)
        self._store.save_meta(meta)
        logger.info("Relatório %s pronto. Tokens: %s", meta.id, meta.usage)
        return " ".join([self._notify(meta, [pdf_path, html_path], send_email), *warnings])

    def _collect_memory(self, meta: ReportMeta, report: ProgressReporter, warnings: list[str]) -> MemoryBlock:
        """Memória é aditiva: qualquer falha inesperada gera o relatório sem ela, com aviso visível."""
        if not meta.folder:
            return MemoryBlock()
        report(Stage.SUMMARIZING, 0.0, "Montando memória da pasta...")
        try:
            return collect_memory(
                self._store,
                meta,
                self._summarizer,
                on_progress=lambda done, total: report(Stage.SUMMARIZING, 0.0, f"Memória da pasta: item {done}/{total}"),
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            logger.warning("Memória da pasta de %s não montada: %s", meta.id, error)
            warnings.append(MEMORY_WARNING)
            return MemoryBlock()

    def _write_card(
        self,
        meta: ReportMeta,
        category: Category,
        memory: MemoryBlock,
        summary: SummaryResult,
        report: ProgressReporter,
        warnings: list[str],
    ) -> None:
        """Ficha do item atual, com o mesmo system prompt dos trechos. Falhar não derruba o relatório."""
        report(Stage.SUMMARIZING, 1.0, "Gerando ficha de memória...")
        request = SummaryRequest(subject=meta.subject, category=category, memory=memory.text)
        try:
            card = self._summarizer.write_memory_card(request, summary.approach, summary.markdown)
            self._store.write_text(meta.id, ReportFile.MEMORY, card)
        except (SummaryError, OSError) as error:
            logger.warning("Ficha de memória de %s não gerada: %s", meta.id, error)
            warnings.append(CARD_WARNING)

    def _publish_to_library(self, meta: ReportMeta, pdf_path: Path) -> None:
        target = self._library.target_path(Category(meta.category), meta.folder, meta.library_title, meta.library_suffix)
        new_relative = self._library.publish(pdf_path, target)
        if meta.library_pdf and meta.library_pdf != new_relative:
            self._library.remove(meta.library_pdf)
        meta.library_pdf = new_relative

    def _notify(self, meta: ReportMeta, attachments: list[Path], send_email: bool) -> str:
        status = "Relatório pronto." if meta.complete else "Relatório parcial: algum trecho falhou — use Regerar."
        if not send_email or self._emailer is None:
            return status
        if not meta.complete:
            return f"{status} E-mail não enviado (relatório incompleto)."
        try:
            self._emailer.send(meta.subject, attachments)
        except EmailError as error:
            logger.error("%s", error)
            return f"{status} Porém o e-mail falhou: {error}"
        return f"{status} Enviado por e-mail."


def _subtitle(meta: ReportMeta, category: Category) -> str:
    duration = format_timestamp(meta.duration_seconds) if meta.duration_seconds else ""
    memory = f"memória: {meta.memory_items} {'item' if meta.memory_items == 1 else 'itens'}" if meta.memory_items else ""
    parts = [profile_for(category).label, meta.created_at[:10], f"duração {duration}" if duration else "", memory]
    return " · ".join(part for part in parts if part)
