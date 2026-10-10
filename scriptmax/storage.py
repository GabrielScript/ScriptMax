"""Relatórios salvos em data/reports/<id>/ — sem índice global (sem corrida de escrita).

O id é um uuid4 hex validado por regex; o assunto digitado pelo usuário nunca
vira caminho de arquivo, o que elimina path traversal.
"""
from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

REPORT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
META_FILE = "meta.json"


class ReportFile(str, Enum):
    HTML = "report.html"
    PDF = "report.pdf"
    MARKDOWN = "report.md"
    TRANSCRIPT_TEXT = "transcript.txt"
    TRANSCRIPT_JSON = "transcript.json"
    MEMORY = "memory.md"


class ReportNotFound(LookupError):
    """Id inválido ou relatório inexistente."""


@dataclass
class ReportMeta:
    id: str
    subject: str
    source_name: str
    created_at: str
    category: str
    folder: str = ""
    library_pdf: str = ""
    approach: str = ""
    duration_seconds: float = 0.0
    report_ready: bool = False
    complete: bool = True
    math_rendered: bool = True
    memory_items: int = 0
    usage: dict[str, int] = field(default_factory=dict)

    @classmethod
    def new(cls, subject: str, source_name: str, category: str, folder: str) -> ReportMeta:
        return cls(
            id=uuid.uuid4().hex,
            subject=subject,
            source_name=source_name,
            created_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            category=category,
            folder=folder,
        )

    @property
    def library_title(self) -> str:
        return f"{self.created_at[:10]} - {self.subject}"

    @property
    def library_suffix(self) -> str:
        return f"[{self.id[:6]}]"


class ReportStore:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def directory(self, report_id: str) -> Path:
        if not REPORT_ID_PATTERN.fullmatch(report_id):
            raise ReportNotFound(report_id)
        return self._root / report_id

    def file_path(self, report_id: str, report_file: ReportFile) -> Path:
        path = self.directory(report_id) / report_file.value
        if not path.is_file():
            raise ReportNotFound(f"{report_id}/{report_file.value}")
        return path

    def write_text(self, report_id: str, report_file: ReportFile, content: str) -> Path:
        return self._write(report_id, report_file.value, content)

    def remove_file(self, report_id: str, report_file: ReportFile) -> None:
        (self.directory(report_id) / report_file.value).unlink(missing_ok=True)

    def save_meta(self, meta: ReportMeta) -> None:
        self._write(meta.id, META_FILE, json.dumps(asdict(meta), ensure_ascii=False, indent=2))

    def _write(self, report_id: str, name: str, content: str) -> Path:
        directory = self.directory(report_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        _atomic_write(path, content)
        return path

    def get(self, report_id: str) -> ReportMeta:
        meta_path = self.directory(report_id) / META_FILE
        if not meta_path.is_file():
            raise ReportNotFound(report_id)
        return ReportMeta(**json.loads(meta_path.read_text(encoding="utf-8")))

    def list(self) -> list[ReportMeta]:
        reports = []
        for meta_path in self._root.glob(f"*/{META_FILE}"):
            if REPORT_ID_PATTERN.fullmatch(meta_path.parent.name):
                reports.append(ReportMeta(**json.loads(meta_path.read_text(encoding="utf-8"))))
        return sorted(reports, key=lambda meta: meta.created_at, reverse=True)

    def delete(self, report_id: str) -> None:
        directory = self.directory(report_id)
        if not directory.is_dir():
            raise ReportNotFound(report_id)
        shutil.rmtree(directory)


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
