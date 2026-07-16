"""Загрузка документов: архивы + LlamaIndex SimpleDirectoryReader."""

from __future__ import annotations

import tempfile
from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from .utils import ARCHIVES, expand_archives

# Форматы, которые SimpleDirectoryReader читает через llama-index-readers-file.
SUPPORTED_SUFFIXES = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".xlsx",
    ".xls",
}


def load_documents(
    folder: Path,
    corpus: str,
    technical_only: bool = False,
) -> tuple[list[Document], list[str]]:
    if not folder.is_dir():
        raise ValueError(f"Папка не найдена: {folder}")

    warnings: list[str] = []
    temp_dirs: list[tempfile.TemporaryDirectory[str]] = []
    documents: list[Document] = []

    try:
        root_files = [
            (path, path.relative_to(folder).as_posix())
            for path in folder.rglob("*")
            if path.is_file()
        ]
        work_items = expand_archives(root_files, temp_dirs, warnings)

        if technical_only:
            preferred = [
                item
                for item in work_items
                if any(m in item[1].lower() for m in ("тз", "техническ", "извещение"))
            ]
            if preferred:
                work_items = preferred

        reader_files = [
            (path, label)
            for path, label in work_items
            if path.suffix.lower() in SUPPORTED_SUFFIXES
        ]
        for path, label in work_items:
            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES and suffix not in ARCHIVES:
                warnings.append(f"Формат пропущен: {label}")

        if reader_files:
            documents.extend(_load_with_llamaindex(reader_files, corpus, warnings))
    finally:
        for temp in temp_dirs:
            temp.cleanup()

    return documents, warnings


def _load_with_llamaindex(
    files: list[tuple[Path, str]],
    corpus: str,
    warnings: list[str],
) -> list[Document]:
    documents: list[Document] = []
    label_by_name: dict[str, list[str]] = {}
    for path, label in files:
        label_by_name.setdefault(path.name, []).append(label)

    try:
        reader = SimpleDirectoryReader(
            input_files=[str(path) for path, _ in files],
            filename_as_id=True,
            raise_on_error=False,
        )
        loaded = reader.load_data()
    except Exception as exc:
        warnings.append(f"SimpleDirectoryReader: {exc}")
        return documents

    used_labels: set[str] = set()
    for doc in loaded:
        file_name = doc.metadata.get("file_name") or Path(
            str(doc.metadata.get("file_path", "unknown"))
        ).name
        candidates = label_by_name.get(file_name, [file_name])
        label = next((item for item in candidates if item not in used_labels), candidates[0])
        used_labels.add(label)
        location = (
            doc.metadata.get("page_label")
            or doc.metadata.get("page_number")
            or doc.metadata.get("sheet_name")
            or "документ"
        )
        location_text = (
            f"стр. {location}" if doc.metadata.get("page_number") is not None else str(location)
        )
        doc.metadata.update(
            {
                "file_name": Path(label).name,
                "file_path": label,
                "corpus": corpus,
                "location": location_text,
            }
        )
        if doc.text and len(doc.text.strip()) >= 20:
            documents.append(doc)
        else:
            warnings.append(f"Пустой или слишком короткий документ: {label}")
    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
):
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.get_nodes_from_documents(documents)
