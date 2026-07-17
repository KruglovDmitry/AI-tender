"""Загрузка документов: архивы + LlamaIndex SimpleDirectoryReader + OCR."""

from __future__ import annotations

import tempfile
from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from .ocr import extract_pdf_with_ocr, ocr_status
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
    ocr_enabled: bool = True,
    ocr_languages: str = "rus+eng",
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
            documents.extend(
                _load_with_llamaindex(
                    reader_files,
                    corpus,
                    warnings,
                    ocr_enabled=ocr_enabled,
                    ocr_languages=ocr_languages,
                )
            )
    finally:
        for temp in temp_dirs:
            temp.cleanup()

    return documents, warnings


def _load_with_llamaindex(
    files: list[tuple[Path, str]],
    corpus: str,
    warnings: list[str],
    ocr_enabled: bool = True,
    ocr_languages: str = "rus+eng",
) -> list[Document]:
    documents: list[Document] = []
    label_by_name: dict[str, list[str]] = {}
    path_by_label: dict[str, Path] = {}
    for path, label in files:
        label_by_name.setdefault(path.name, []).append(label)
        path_by_label[label] = path

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
    kept_labels: set[str] = set()
    empty_counts: dict[str, int] = {}

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
            kept_labels.add(label)
        else:
            empty_counts[label] = empty_counts.get(label, 0) + 1

    ocr_available, ocr_hint = ocr_status()
    ocr_hint_shown = False

    for label, count in list(empty_counts.items()):
        if label in kept_labels and count == 0:
            continue

        path = path_by_label.get(label)
        if ocr_enabled and path and path.suffix.lower() == ".pdf":
            ocr_docs, note = extract_pdf_with_ocr(
                path,
                label,
                corpus=corpus,
                languages=ocr_languages,
            )
            if ocr_docs:
                documents = [doc for doc in documents if doc.metadata.get("file_path") != label]
                documents.extend(ocr_docs)
                kept_labels.add(label)
                if note:
                    warnings.append(note)
                continue
            if note and not ocr_hint_shown:
                warnings.append(f"OCR недоступен: {note}")
                ocr_hint_shown = True

        if ocr_enabled and not ocr_available and not ocr_hint_shown:
            warnings.append(f"OCR недоступен: {ocr_hint}")
            ocr_hint_shown = True

        warnings.append(
            f"Нет текстового слоя (нужен OCR): {label} "
            f"({count} стр./фрагментов без текста)"
        )

    expected = {label for _, label in files}
    missing = expected - kept_labels - set(empty_counts)
    for label in sorted(missing):
        warnings.append(f"Файл не попал в индекс (не прочитан): {label}")

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
):
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.get_nodes_from_documents(documents)
