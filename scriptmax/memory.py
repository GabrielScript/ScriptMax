"""Memória entre os relatórios de uma mesma pasta.

Cada relatório pronto tem uma ficha curta (memory.md). Ao gerar o item N, as
fichas dos itens ANTERIORES da mesma categoria e pasta exata viram um bloco no
prompt de sistema. A memória é sempre derivada do estado atual da pasta: mover,
apagar ou regerar um relatório nunca deixa nada desatualizado.

Ordem do mais antigo ao mais novo: um item novo entra no fim e o começo do
prompt se mantém, o que preserva o cache da DeepSeek entre episódios.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from scriptmax.categories import Category
from scriptmax.storage import ReportFile, ReportMeta, ReportNotFound, ReportStore
from scriptmax.summarization import SummaryError, SummaryRequest

logger = logging.getLogger(__name__)

# ~9 mil tokens: cerca de 10 fichas. Acima disso ficam as mais novas.
MEMORY_MAX_CHARS = 30_000
MAX_CARD_CHARS = 6_000
PARTIAL_TAG = "(relatório parcial)"

MEMORY_HEADER = """## MEMÓRIA DA PASTA (itens anteriores desta sequência, do mais antigo ao mais recente)
Use as fichas só como contexto:
- Quando o trecho tiver relação real com um item anterior, cite-o pelo rótulo (ex.: "retoma o conceito X do Item 3"). Não force relações.
- A memória não é fonte de fatos do trecho atual: se ela discordar da transcrição, vale a transcrição.
- Não copie o conteúdo das fichas para o relatório. Inferências ficam marcadas como "Análise:"."""

ProgressCallback = Callable[[int, int], None]


class CardWriter(Protocol):
    def write_memory_card(self, request: SummaryRequest, approach: str, report_markdown: str) -> str: ...


@dataclass(frozen=True)
class MemoryBlock:
    text: str = ""
    items: int = 0
    omitted: int = 0


def collect_memory(
    store: ReportStore, meta: ReportMeta, writer: CardWriter, on_progress: ProgressCallback | None = None
) -> MemoryBlock:
    """Bloco de memória para `meta`: fichas dos itens anteriores da mesma pasta."""
    siblings = _earlier_siblings(store, meta)
    if not siblings:
        return MemoryBlock()

    total = len(siblings)
    chosen: list[tuple[int, ReportMeta, str]] = []
    used = 0
    omitted = 0
    # Do mais novo ao mais antigo: o orçamento protege os itens recentes e só gera fichas que cabem.
    for done, (position, sibling) in enumerate(reversed(list(enumerate(siblings, start=1))), start=1):
        if on_progress:
            on_progress(done, total)
        if used >= MEMORY_MAX_CHARS:
            omitted += 1
            continue
        card = _card_for(store, sibling, writer)
        if card is None:
            continue
        rendered = _render_card(position, sibling, card)
        if chosen and used + len(rendered) > MEMORY_MAX_CHARS:
            omitted += 1
            used = MEMORY_MAX_CHARS
            continue
        chosen.append((position, sibling, rendered))
        used += len(rendered)

    if not chosen:
        return MemoryBlock()
    parts = [MEMORY_HEADER]
    if omitted:
        older = "item mais antigo omitido" if omitted == 1 else "itens mais antigos omitidos"
        parts.append(f"({omitted} {older} por limite de tamanho.)")
    parts.extend(rendered for _, _, rendered in reversed(chosen))
    return MemoryBlock(text="\n\n".join(parts), items=len(chosen), omitted=omitted)


def _earlier_siblings(store: ReportStore, meta: ReportMeta) -> list[ReportMeta]:
    if not meta.folder:
        return []
    key = (meta.created_at, meta.id)
    siblings = [
        other
        for other in store.list()
        if other.id != meta.id
        and other.report_ready
        and other.category == meta.category
        and other.folder == meta.folder
        and (other.created_at, other.id) < key
    ]
    return sorted(siblings, key=lambda other: (other.created_at, other.id))


def _card_for(store: ReportStore, sibling: ReportMeta, writer: CardWriter) -> str | None:
    """A ficha salva ou, se faltar, gera e salva. None = pular o irmão (nunca derruba o relatório)."""
    try:
        saved = store.file_path(sibling.id, ReportFile.MEMORY).read_text(encoding="utf-8").strip()
        if saved:
            return saved
    except (ReportNotFound, OSError, ValueError):
        pass
    try:
        markdown = store.file_path(sibling.id, ReportFile.MARKDOWN).read_text(encoding="utf-8")
        request = SummaryRequest(subject=sibling.subject, category=Category(sibling.category))
        card = writer.write_memory_card(request, sibling.approach, markdown).strip()
        store.write_text(sibling.id, ReportFile.MEMORY, card)
    except (SummaryError, ReportNotFound, OSError, ValueError) as error:
        logger.warning("Ficha de %s não gerada: %s", sibling.id, error)
        return None
    return card or None


def _render_card(position: int, sibling: ReportMeta, card: str) -> str:
    tag = f" {PARTIAL_TAG}" if not sibling.complete else ""
    return f"### Item {position} — {sibling.subject} ({sibling.created_at[:10]}){tag}\n{card[:MAX_CARD_CHARS]}"
