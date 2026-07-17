from pathlib import Path

from docx import Document

from ai_tender.index import cache_key, folder_fingerprint, select_query_nodes
from llama_index.core.schema import TextNode


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


def test_cache_key_includes_chunk_settings(tmp_path: Path) -> None:
    _write_docx(tmp_path / "a.docx", "Достаточно длинный текст эталонного документа.")
    key_a = cache_key(tmp_path, "BAAI/bge-m3", 1024, 128)
    key_b = cache_key(tmp_path, "BAAI/bge-m3", 512, 128)
    assert key_a != key_b


def test_select_query_nodes_limits() -> None:
    nodes = [TextNode(text=f"chunk with enough words for filter {index} " * 8) for index in range(100)]
    selected = select_query_nodes(nodes, limit=10)
    assert len(selected) == 10
