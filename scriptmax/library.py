"""Pastas dos relatórios e o espelho dos PDFs em diretórios reais.

Estrutura no disco: <biblioteca>/<Categoria>/<Pasta>/<Subpasta>/<data - título [id]>.pdf

Nomes digitados pelo usuário viram caminho, então cada trecho é saneado
(sem separadores, sem "..", sem nomes reservados do Windows) e o caminho final
é conferido para garantir que continua dentro da biblioteca.
"""
from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

from scriptmax.categories import Category, profile_for

logger = logging.getLogger(__name__)

MAX_FOLDER_DEPTH = 3
MAX_SEGMENT_CHARS = 60
# MAX_PATH do Windows é 260; margem para o arquivo temporário da cópia.
MAX_PATH_CHARS = 240
MIN_TITLE_CHARS = 16
FORBIDDEN_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", *(f"COM{n}" for n in range(1, 10)), *(f"LPT{n}" for n in range(1, 10))}
)


class InvalidFolder(ValueError):
    """Nome de pasta inválido."""


def safe_segment(raw: str) -> str:
    """Um nome de pasta/arquivo seguro em Windows, macOS e Linux ('' se nada sobrar)."""
    cleaned = FORBIDDEN_CHARACTERS.sub(" ", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")[:MAX_SEGMENT_CHARS].strip(" .")
    if cleaned.split(".")[0].upper() in WINDOWS_RESERVED_NAMES:
        cleaned = f"{cleaned}_"
    return cleaned


def normalize_folder(raw: str) -> str:
    """'Álgebra / Prova 1' -> 'Álgebra/Prova 1'. Vazio = raiz da categoria."""
    segments = [safe_segment(part) for part in raw.replace("\\", "/").split("/")]
    segments = [segment for segment in segments if segment]
    if len(segments) > MAX_FOLDER_DEPTH:
        raise InvalidFolder(f"Use no máximo {MAX_FOLDER_DEPTH} níveis de pasta.")
    return "/".join(segments)


class PdfLibrary:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def target_path(self, category: Category, folder: str, title: str, unique_suffix: str) -> Path:
        """<raiz>/<Categoria>/<pastas>/<título encurtado se preciso> <sufixo>.pdf"""
        parts = [profile_for(category).library_folder, *[p for p in normalize_folder(folder).split("/") if p]]
        directory = self._root.joinpath(*parts).resolve()
        if not directory.is_relative_to(self._root):
            raise InvalidFolder("Caminho fora da biblioteca.")

        fixed_chars = len(str(directory)) + len(" ") + len(unique_suffix) + len(".pdf") + 1
        title_budget = MAX_PATH_CHARS - fixed_chars
        if title_budget < MIN_TITLE_CHARS:
            raise InvalidFolder("Pastas aninhadas demais para o limite de caminho do Windows; use nomes mais curtos.")
        clean_title = safe_segment(title) or "relatorio"
        short_title = clean_title[:title_budget].rstrip(" .")
        return directory / f"{short_title} {unique_suffix}.pdf"

    def publish(self, source_pdf: Path, target: Path) -> str:
        """Copia o PDF para a biblioteca; retorna o caminho relativo (guardado no meta)."""
        target.parent.mkdir(parents=True, exist_ok=True)
        # copyfile (não copy2): o Cloud Storage FUSE recusa utime/chmod com EPERM.
        shutil.copyfile(source_pdf, target)
        return target.relative_to(self._root).as_posix()

    def remove(self, relative_path: str) -> None:
        if not relative_path:
            return
        target = (self._root / relative_path).resolve()
        if not target.is_relative_to(self._root):
            logger.warning("Ignorando caminho de biblioteca suspeito: %s", relative_path)
            return
        target.unlink(missing_ok=True)
        self._prune_empty_parents(target.parent)

    def _prune_empty_parents(self, directory: Path) -> None:
        # Remove pastas que ficaram vazias, sem nunca apagar a raiz nem as pastas de categoria.
        category_roots = {self._root / profile_for(category).library_folder for category in Category}
        while directory != self._root and directory not in category_roots and directory.is_dir():
            if any(directory.iterdir()):
                return
            directory.rmdir()
            directory = directory.parent
