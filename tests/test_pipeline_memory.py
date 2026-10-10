from __future__ import annotations

import json
from pathlib import Path

import pytest

from scriptmax.categories import Category
from scriptmax.library import PdfLibrary
from scriptmax.pipeline import ProcessRequest, ReportPipeline
from scriptmax.storage import ReportFile, ReportMeta, ReportStore
from scriptmax.summarization import SummaryError
from tests.conftest import FakeSummarizer, FakeTranscriber

NO_PROGRESS = lambda stage, progress, message: None  # noqa: E731


@pytest.fixture()
def parts(tmp_path: Path, fake_pdf: None) -> tuple[ReportPipeline, ReportStore, FakeSummarizer]:
    store = ReportStore(tmp_path / "reports")
    summarizer = FakeSummarizer()
    pipeline = ReportPipeline(FakeTranscriber(), summarizer, store, PdfLibrary(tmp_path / "biblioteca"), emailer=None)
    return pipeline, store, summarizer


def request(tmp_path: Path, subject: str, folder: str = "Kubernetes") -> ProcessRequest:
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"x")
    return ProcessRequest(audio_path=audio, source_name="a.m4a", subject=subject,
                          category=Category.TECH, folder=folder, send_email=False)


def test_first_item_has_no_memory_and_gets_its_card(parts, tmp_path: Path) -> None:
    pipeline, store, summarizer = parts

    report_id, status = pipeline.process(request(tmp_path, "Aula 1"), NO_PROGRESS)

    assert summarizer.requests[0].memory == ""
    assert store.get(report_id).memory_items == 0
    assert store.file_path(report_id, ReportFile.MEMORY).read_text(encoding="utf-8") == "FICHA de Aula 1"
    assert status == "Relatório pronto."
    card_request, approach, markdown = summarizer.card_calls[0]
    assert card_request.memory == "" and approach == "A" and markdown.startswith("# Produto interno")


def test_second_item_receives_first_card_and_writes_its_own_with_the_same_memory(parts, tmp_path: Path) -> None:
    pipeline, store, summarizer = parts
    first_id, _ = pipeline.process(request(tmp_path, "Aula 1"), NO_PROGRESS)

    second_id, _ = pipeline.process(request(tmp_path, "Aula 2"), NO_PROGRESS)

    memory = summarizer.requests[1].memory
    assert "FICHA de Aula 1" in memory and "Aula 2" not in memory
    assert store.get(second_id).memory_items == 1
    assert summarizer.card_calls[-1][0].memory == memory
    assert len(summarizer.card_calls) == 2
    assert store.file_path(first_id, ReportFile.MEMORY).exists()


def test_root_folder_never_uses_memory(parts, tmp_path: Path) -> None:
    pipeline, _, summarizer = parts
    pipeline.process(request(tmp_path, "Solta 1", folder=""), NO_PROGRESS)

    pipeline.process(request(tmp_path, "Solta 2", folder=""), NO_PROGRESS)

    assert summarizer.requests[1].memory == ""


def test_card_failure_does_not_fail_the_report_and_is_visible(parts, tmp_path: Path) -> None:
    pipeline, store, summarizer = parts
    summarizer.card_error = SummaryError("api caiu")

    report_id, status = pipeline.process(request(tmp_path, "Aula 1"), NO_PROGRESS)

    assert store.get(report_id).report_ready is True
    assert not (store.directory(report_id) / ReportFile.MEMORY.value).exists()
    assert "ficha de memória não foi gerada" in status


def test_missing_card_is_backfilled_on_next_item(parts, tmp_path: Path) -> None:
    pipeline, _, summarizer = parts
    summarizer.card_error = SummaryError("api caiu")
    pipeline.process(request(tmp_path, "Aula 1"), NO_PROGRESS)
    summarizer.card_error = None

    pipeline.process(request(tmp_path, "Aula 2"), NO_PROGRESS)

    assert "FICHA de Aula 1" in summarizer.requests[1].memory


def test_regenerate_drops_stale_card_and_rewrites_it_seeing_only_earlier_items(parts, tmp_path: Path) -> None:
    pipeline, store, summarizer = parts
    pipeline.process(request(tmp_path, "Aula 1"), NO_PROGRESS)
    second_id, _ = pipeline.process(request(tmp_path, "Aula 2"), NO_PROGRESS)
    pipeline.process(request(tmp_path, "Aula 3"), NO_PROGRESS)
    store.write_text(second_id, ReportFile.MEMORY, "FICHA VELHA")

    pipeline.regenerate(second_id, NO_PROGRESS)

    assert store.file_path(second_id, ReportFile.MEMORY).read_text(encoding="utf-8") == "FICHA de Aula 2"
    memory = summarizer.requests[-1].memory
    assert "Aula 1" in memory and "Aula 3" not in memory


def test_regenerate_with_failing_card_leaves_no_stale_card(parts, tmp_path: Path) -> None:
    pipeline, store, summarizer = parts
    report_id, _ = pipeline.process(request(tmp_path, "Aula 1"), NO_PROGRESS)
    summarizer.card_error = SummaryError("api caiu")

    _, status = pipeline.regenerate(report_id, NO_PROGRESS)

    assert not (store.directory(report_id) / ReportFile.MEMORY.value).exists()
    assert "ficha de memória não foi gerada" in status


def test_report_subtitle_mentions_memory_items(parts, tmp_path: Path) -> None:
    pipeline, store, _ = parts
    pipeline.process(request(tmp_path, "Aula 1"), NO_PROGRESS)

    second_id, _ = pipeline.process(request(tmp_path, "Aula 2"), NO_PROGRESS)

    assert "memória: 1 item" in store.file_path(second_id, ReportFile.HTML).read_text(encoding="utf-8")


def test_progress_reports_memory_stage(parts, tmp_path: Path) -> None:
    pipeline, _, _ = parts
    pipeline.process(request(tmp_path, "Aula 1"), NO_PROGRESS)
    messages: list[str] = []

    pipeline.process(request(tmp_path, "Aula 2"), lambda stage, progress, message: messages.append(message))

    assert any("memória" in message.lower() for message in messages)


def test_old_meta_without_memory_items_still_loads(tmp_path: Path) -> None:
    store = ReportStore(tmp_path / "reports")
    meta = ReportMeta.new("x", "a.m4a", "tech", "K")
    store.save_meta(meta)
    path = store.directory(meta.id) / "meta.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("memory_items", None)
    path.write_text(json.dumps(data), encoding="utf-8")

    assert store.get(meta.id).memory_items == 0
