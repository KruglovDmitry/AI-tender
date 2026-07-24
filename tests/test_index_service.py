from pathlib import Path
import os

from docx import Document

from ai_tender.services.index_service import cache_key, folder_fingerprint


def _write_docx(path: Path, text: str) -> None:
    document = Document()
    document.add_paragraph(text)
    document.save(path)


def test_folder_fingerprint_changes_when_file_changes(tmp_path: Path) -> None:
    path = tmp_path / "эталон.docx"
    _write_docx(path, "Исходный текст эталона с параметром IP54.")
    first = folder_fingerprint(tmp_path)
    _write_docx(path, "Обновлённый текст эталона с параметром IP54 и током.")
    second = folder_fingerprint(tmp_path)
    assert first != second


def test_folder_fingerprint_ignores_mtime(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("эталон", encoding="utf-8")
    first = folder_fingerprint(tmp_path)
    os.utime(path, (1_700_000_000, 1_800_000_000))
    second = folder_fingerprint(tmp_path)
    assert first == second


def test_cache_key_includes_chunk_settings(tmp_path: Path) -> None:
    _write_docx(tmp_path / "a.docx", "Достаточно длинный текст эталонного документа.")
    key_a = cache_key(tmp_path, "BAAI/bge-m3", 1024, 128)
    key_b = cache_key(tmp_path, "BAAI/bge-m3", 512, 128)
    assert key_a != key_b


def test_cache_key_includes_ocr_settings(tmp_path: Path) -> None:
    _write_docx(tmp_path / "a.docx", "Достаточно длинный текст эталонного документа.")
    key_on = cache_key(tmp_path, "BAAI/bge-m3", 1024, 128, ocr_enabled=True)
    key_off = cache_key(tmp_path, "BAAI/bge-m3", 1024, 128, ocr_enabled=False)
    key_lang = cache_key(
        tmp_path, "BAAI/bge-m3", 1024, 128, ocr_enabled=True, ocr_languages="eng"
    )
    assert key_on != key_off
    assert key_on != key_lang


def test_cache_key_portable_across_folder_paths(tmp_path: Path) -> None:
    left = tmp_path / "machine_a" / "assets"
    right = tmp_path / "machine_b" / "assets"
    left.mkdir(parents=True)
    right.mkdir(parents=True)
    (left / "spec.txt").write_text("одна и та же спецификация", encoding="utf-8")
    (right / "spec.txt").write_text("одна и та же спецификация", encoding="utf-8")
    assert cache_key(left, "BAAI/bge-m3", 1024, 128) == cache_key(
        right, "BAAI/bge-m3", 1024, 128
    )
