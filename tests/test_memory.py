from __future__ import annotations

from pathlib import Path

import pytest

from scriptmax import memory
from scriptmax.categories import Category
from scriptmax.memory import MemoryBlock, collect_memory
from scriptmax.storage import ReportFile, ReportMeta, ReportStore
from scriptmax.summarization import SummaryError, SummaryRequest


class FakeCardWriter:
    def __init__(self, failing: set[str] | None = None) -> None:
        self.calls: list[tuple[SummaryRequest, str, str]] = []
        self.failing = failing or set()

    def write_memory_card(self, request: SummaryRequest, approach: str, report_markdown: str) -> str:
        self.calls.append((request, approach, report_markdown))
        if request.subject in self.failing:
            raise SummaryError("falhou")
        return f"FICHA de {request.subject}"


def add_report(
    store: ReportStore,
    subject: str,
    day: int,
    *,
    category: str = Category.TECH.value,
    folder: str = "Kubernetes",
    ready: bool = True,
    complete: bool = True,
    card: str | None = None,
) -> ReportMeta:
    meta = ReportMeta.new(subject, "a.m4a", category, folder)
    meta.created_at = f"2026-10-{day:02d}T10:00:00+00:00"
    meta.report_ready = ready
    meta.complete = complete
    store.save_meta(meta)
    if ready:
        store.write_text(meta.id, ReportFile.MARKDOWN, f"# Relatório de {subject}")
    if card is not None:
        store.write_text(meta.id, ReportFile.MEMORY, card)
    return meta


@pytest.fixture()
def store(tmp_path: Path) -> ReportStore:
    return ReportStore(tmp_path / "reports")


def test_collects_only_earlier_ready_siblings_of_same_category_and_folder_oldest_first(store: ReportStore) -> None:
    add_report(store, "Aula 2", 2)
    add_report(store, "Aula 1", 1)
    current = add_report(store, "Aula 3", 3, ready=False)
    add_report(store, "Aula 4 (futura)", 4)
    add_report(store, "Outra pasta", 1, folder="Docker")
    add_report(store, "Outra categoria", 1, category=Category.WORK.value)
    add_report(store, "Sem relatório", 1, ready=False)

    block = collect_memory(store, current, FakeCardWriter())

    assert block.items == 2
    assert block.omitted == 0
    assert block.text.index("Aula 1") < block.text.index("Aula 2")
    for excluded in ("futura", "Outra pasta", "Outra categoria", "Sem relatório", "Aula 3"):
        assert excluded not in block.text


def test_root_folder_has_no_memory(store: ReportStore) -> None:
    add_report(store, "Solta 1", 1, folder="")
    current = add_report(store, "Solta 2", 2, folder="", ready=False)
    writer = FakeCardWriter()

    assert collect_memory(store, current, writer) == MemoryBlock()
    assert writer.calls == []


def test_no_siblings_gives_empty_block(store: ReportStore) -> None:
    current = add_report(store, "Primeira", 1, ready=False)
    assert collect_memory(store, current, FakeCardWriter()) == MemoryBlock()


def test_missing_card_is_generated_saved_and_reused(store: ReportStore) -> None:
    old = add_report(store, "Aula 1", 1)
    current = add_report(store, "Aula 2", 2, ready=False)
    writer = FakeCardWriter()

    first = collect_memory(store, current, writer)
    assert "FICHA de Aula 1" in first.text
    assert store.file_path(old.id, ReportFile.MEMORY).read_text(encoding="utf-8") == "FICHA de Aula 1"
    request, approach, markdown = writer.calls[0]
    assert request.memory == "" and request.subject == "Aula 1" and request.category is Category.TECH
    assert markdown == "# Relatório de Aula 1"

    second = collect_memory(store, current, writer)
    assert second == first
    assert len(writer.calls) == 1


def test_existing_card_is_used_without_calling_the_writer(store: ReportStore) -> None:
    add_report(store, "Aula 1", 1, card="Ficha pronta.")
    current = add_report(store, "Aula 2", 2, ready=False)
    writer = FakeCardWriter()

    block = collect_memory(store, current, writer)

    assert "Ficha pronta." in block.text
    assert writer.calls == []


def test_blank_card_file_counts_as_missing(store: ReportStore) -> None:
    add_report(store, "Aula 1", 1, card="  \n")
    current = add_report(store, "Aula 2", 2, ready=False)

    block = collect_memory(store, current, FakeCardWriter())

    assert "FICHA de Aula 1" in block.text


def test_sibling_whose_card_fails_is_skipped_not_fatal(store: ReportStore) -> None:
    add_report(store, "Aula 1", 1, card="Ficha 1")
    add_report(store, "Aula 2", 2)
    add_report(store, "Aula 3", 3, card="Ficha 3")
    current = add_report(store, "Aula 4", 4, ready=False)

    block = collect_memory(store, current, FakeCardWriter(failing={"Aula 2"}))

    assert block.items == 2
    assert "Ficha 1" in block.text and "Ficha 3" in block.text
    assert "Aula 2" not in block.text


def test_sibling_without_report_file_is_skipped(store: ReportStore) -> None:
    broken = add_report(store, "Quebrado", 1)
    store.file_path(broken.id, ReportFile.MARKDOWN).unlink()
    current = add_report(store, "Aula 2", 2, ready=False)

    assert collect_memory(store, current, FakeCardWriter()) == MemoryBlock()


def test_partial_sibling_is_tagged(store: ReportStore) -> None:
    add_report(store, "Aula 1", 1, complete=False, card="Ficha parcial")
    current = add_report(store, "Aula 2", 2, ready=False)

    block = collect_memory(store, current, FakeCardWriter())

    assert "(relatório parcial)" in block.text


def test_labels_follow_position_and_header_carries_the_rules(store: ReportStore) -> None:
    add_report(store, "Aula 1", 1, card="a")
    add_report(store, "Aula 2", 2, card="b")
    current = add_report(store, "Aula 3", 3, ready=False)

    text = collect_memory(store, current, FakeCardWriter()).text

    assert text.startswith("## MEMÓRIA DA PASTA")
    assert "### Item 1 — Aula 1 (2026-10-01)" in text
    assert "### Item 2 — Aula 2 (2026-10-02)" in text
    assert "vale a transcrição" in text
    assert 'marcadas como "Análise:"' in text


def test_over_budget_keeps_newest_and_reports_omitted(store: ReportStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory, "MEMORY_MAX_CHARS", 250)
    for day in (1, 2, 3, 4):
        add_report(store, f"Aula {day}", day, card="x" * 100)
    current = add_report(store, "Aula 5", 5, ready=False)

    block = collect_memory(store, current, FakeCardWriter())

    assert 0 < block.items < 4
    assert block.omitted == 4 - block.items
    assert "Aula 4" in block.text and "Aula 1" not in block.text
    assert f"{block.omitted} itens mais antigos omitidos" in block.text
    assert f"### Item 4 — Aula 4" in block.text


def test_oversized_single_card_is_truncated_but_never_dropped(store: ReportStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory, "MAX_CARD_CHARS", 50)
    add_report(store, "Aula 1", 1, card="y" * 500)
    current = add_report(store, "Aula 2", 2, ready=False)

    block = collect_memory(store, current, FakeCardWriter())

    assert block.items == 1
    assert "y" * 51 not in block.text


def test_progress_callback_counts_siblings(store: ReportStore) -> None:
    add_report(store, "Aula 1", 1, card="a")
    add_report(store, "Aula 2", 2, card="b")
    current = add_report(store, "Aula 3", 3, ready=False)
    seen: list[tuple[int, int]] = []

    collect_memory(store, current, FakeCardWriter(), on_progress=lambda done, total: seen.append((done, total)))

    assert seen == [(1, 2), (2, 2)]
