from __future__ import annotations

from pathlib import Path

import pytest

from scriptmax.categories import Category
from scriptmax.emailer import safe_header_text
from scriptmax.library import InvalidFolder, PdfLibrary, normalize_folder, safe_segment
from scriptmax.storage import ReportMeta, ReportNotFound, ReportStore


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Álgebra / Prova 1", "Álgebra/Prova 1"),
        ("../../Windows", "Windows"),
        ("..\\..\\etc", "etc"),
        ("  ", ""),
        ("a//b", "a/b"),
        ("CON", "CON_"),
        ("nome:com*proibidos?", "nome com proibidos"),
    ],
)
def test_normalize_folder(raw: str, expected: str) -> None:
    assert normalize_folder(raw) == expected


def test_folder_depth_is_limited() -> None:
    with pytest.raises(InvalidFolder):
        normalize_folder("a/b/c/d")


def test_safe_segment_strips_trailing_dots() -> None:
    assert safe_segment("relatório...") == "relatório"


def test_library_paths_stay_inside_root(tmp_path: Path) -> None:
    library = PdfLibrary(tmp_path / "bib")
    target = library.target_path(Category.ACADEMIC, "../../fora", "../../x", "[abc123]")
    assert target.resolve().is_relative_to(library.root)
    assert target.parent.name == "fora"
    assert target.parent.parent.name == "Acadêmico e Conhecimento"


def test_library_publish_and_prune(tmp_path: Path) -> None:
    library = PdfLibrary(tmp_path / "bib")
    source = tmp_path / "r.pdf"
    source.write_bytes(b"%PDF")
    relative = library.publish(source, library.target_path(Category.WORK, "Projeto X/Sprint 1", "ata", "[abc123]"))
    assert (library.root / relative).read_bytes() == b"%PDF"
    library.remove(relative)
    assert not (library.root / "Trabalho" / "Projeto X").exists()
    assert (library.root / "Trabalho").exists()  # pasta da categoria fica


def test_library_publish_does_not_touch_file_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Cloud Storage FUSE (Cloud Run) recusa utime/chmod com EPERM, e no Linux o shutil.copy2 os chama
    # (no Windows o copy2 é nativo e não reproduz). Por isso o publish só pode copiar conteúdo.
    def deny(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr("shutil.copy2", deny)
    monkeypatch.setattr("shutil.copystat", deny)
    library = PdfLibrary(tmp_path / "bib")
    source = tmp_path / "r.pdf"
    source.write_bytes(b"%PDF")
    relative = library.publish(source, library.target_path(Category.WORK, "", "ata", "[abc123]"))
    assert (library.root / relative).read_bytes() == b"%PDF"


def test_long_titles_are_shortened_to_fit_windows_max_path(tmp_path: Path) -> None:
    library = PdfLibrary(tmp_path / "bib")
    folder = "/".join(["Pasta com nome bem comprido " + str(n) for n in range(3)])
    target = library.target_path(Category.ACADEMIC, folder, "Título " * 60, "[abc123]")
    assert len(str(target)) <= 240
    assert target.name.endswith(" [abc123].pdf")


def test_folder_too_deep_for_max_path_is_rejected(tmp_path: Path) -> None:
    deep_root = tmp_path / ("x" * 150)
    library = PdfLibrary(deep_root)
    with pytest.raises(InvalidFolder):
        library.target_path(Category.ACADEMIC, "a" * 60 + "/" + "b" * 60, "título", "[abc123]")


def test_library_remove_ignores_escape_attempt(tmp_path: Path) -> None:
    library = PdfLibrary(tmp_path / "bib")
    outside = tmp_path / "importante.pdf"
    outside.write_bytes(b"x")
    library.remove("../importante.pdf")
    assert outside.exists()


def test_store_rejects_invalid_ids(tmp_path: Path) -> None:
    store = ReportStore(tmp_path)
    for bad_id in ["..", "../x", "abc", "A" * 32]:
        with pytest.raises(ReportNotFound):
            store.directory(bad_id)


def test_store_roundtrip_and_ordering(tmp_path: Path) -> None:
    store = ReportStore(tmp_path)
    older = ReportMeta.new("Velho", "a.mp3", Category.WORK.value, "")
    older.created_at = "2026-01-01T00:00:00+00:00"
    newer = ReportMeta.new("Novo", "b.mp3", Category.MEDIA.value, "Séries")
    store.save_meta(older)
    store.save_meta(newer)
    assert [meta.subject for meta in store.list()] == ["Novo", "Velho"]
    store.delete(older.id)
    assert [meta.id for meta in store.list()] == [newer.id]


def test_email_subject_cannot_inject_headers() -> None:
    assert "\n" not in safe_header_text("Aula\r\nBcc: vitima@exemplo.com")
