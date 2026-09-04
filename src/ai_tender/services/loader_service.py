"""Загрузка документов: архивы + LlamaIndex SimpleDirectoryReader + .doc."""

from __future__ import annotations

import base64
import io
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader

from .archive_service import ARCHIVES, expand_archives

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
LEGACY_WORD_SUFFIXES = {".doc", ".dot"}
READABLE_SUFFIXES = SUPPORTED_SUFFIXES | LEGACY_WORD_SUFFIXES

IGNORED_BASENAMES = frozenset({"thumbs.db", "desktop.ini", ".ds_store"})
IGNORED_NAME_PREFIXES = ("~$",)

TECHNICAL_MARKERS = ("тз", "техническ", "извещение")

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"})


def is_ignored_file(label: str) -> bool:
    name = Path(label).name.lower()
    if name in IGNORED_BASENAMES:
        return True
    return any(name.startswith(prefix.lower()) for prefix in IGNORED_NAME_PREFIXES)


@dataclass
class TenderInventory:
    """Распакованные пути тендерных файлов; temp_dirs нужно освободить через cleanup()."""

    work_items: list[tuple[Path, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    _temp_dirs: list[tempfile.TemporaryDirectory[str]] = field(default_factory=list)

    def cleanup(self) -> None:
        for temp in self._temp_dirs:
            temp.cleanup()
        self._temp_dirs.clear()

    def __enter__(self) -> TenderInventory:
        return self

    def __exit__(self, *args: object) -> None:
        self.cleanup()


def inventory_tender_folder(
    folder: Path,
    *,
    technical_only: bool = False,
    only_labels: set[str] | None = None,
) -> TenderInventory:
    if not folder.is_dir():
        raise ValueError(f"Папка не найдена: {folder}")

    warnings: list[str] = []
    temp_dirs: list[tempfile.TemporaryDirectory[str]] = []
    root_files = [
        (path, path.relative_to(folder).as_posix())
        for path in folder.rglob("*")
        if path.is_file()
    ]
    work_items = expand_archives(root_files, temp_dirs, warnings)
    work_items = [item for item in work_items if not is_ignored_file(item[1])]

    if technical_only:
        preferred = [
            item
            for item in work_items
            if any(marker in item[1].lower() for marker in TECHNICAL_MARKERS)
        ]
        if preferred:
            work_items = preferred

    if only_labels is not None:
        work_items = [item for item in work_items if item[1] in only_labels]

    return TenderInventory(work_items=work_items, warnings=warnings, _temp_dirs=temp_dirs)


def load_documents(
    folder: Path,
    corpus: str,
    technical_only: bool = False,
    only_labels: set[str] | None = None,
    inventory: TenderInventory | None = None,
) -> tuple[list[Document], list[str]]:
    owns_inventory = inventory is None
    if inventory is None:
        inventory = inventory_tender_folder(
            folder,
            technical_only=technical_only,
            only_labels=only_labels,
        )

    warnings: list[str] = list(inventory.warnings)
    documents: list[Document] = []

    try:
        work_items = list(inventory.work_items)
        if only_labels is not None and owns_inventory:
            work_items = [item for item in work_items if item[1] in only_labels]

        reader_files = [
            (path, label)
            for path, label in work_items
            if path.suffix.lower() in SUPPORTED_SUFFIXES
        ]
        legacy_doc_files = [
            (path, label)
            for path, label in work_items
            if path.suffix.lower() in LEGACY_WORD_SUFFIXES
        ]
        for path, label in work_items:
            suffix = path.suffix.lower()
            if suffix in READABLE_SUFFIXES or suffix in ARCHIVES:
                continue
            if is_ignored_file(label):
                continue
            warnings.append(f"Формат пропущен: {label}")

        if reader_files:
            documents.extend(_load_with_llamaindex(reader_files, corpus, warnings))
        if legacy_doc_files:
            documents.extend(_load_legacy_doc_files(legacy_doc_files, corpus, warnings))
    finally:
        if owns_inventory:
            inventory.cleanup()

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

    expected = {label for _, label in files}
    for label in sorted(expected - kept_labels):
        count = empty_counts.get(label, 0)
        if count:
            warnings.append(
                f"Нет текстового слоя (нужен VL): {label} "
                f"({count} стр./фрагментов без текста)"
            )
        else:
            warnings.append(f"Файл не попал в индекс (не прочитан): {label}")

    return documents


def extract_doc_text(path: Path, *, timeout_sec: int = 90) -> tuple[str, str | None]:
    """Извлекает текст из Word 97–2003 (.doc). Возвращает (text, error)."""
    try:
        import sharepoint2text

        result = next(sharepoint2text.read_file(path))
        text = (result.get_full_text() or "").strip()
        if text:
            return text, None
        return "", "Пустой текст после sharepoint-to-text"
    except ImportError:
        pass
    except StopIteration:
        return "", "sharepoint-to-text: пустой результат"
    except Exception as exc:
        return "", f"sharepoint-to-text: {exc}"

    antiword = shutil.which("antiword")
    if antiword:
        try:
            proc = subprocess.run(
                [antiword, "-m", "UTF-8.txt", str(path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
                check=False,
            )
            text = (proc.stdout or "").strip()
            if proc.returncode == 0 and text:
                return text, None
            err = (proc.stderr or "").strip() or f"antiword: код {proc.returncode}"
            return text, err if not text else None
        except Exception as exc:
            return "", f"antiword: {exc}"

    return (
        "",
        "Формат .doc: установите пакет sharepoint-to-text (pip) или antiword в PATH",
    )


def read_document_text(path: Path) -> str:
    """Локальный текст документа для Qwen extract (pdf/docx/doc/txt/md/csv)."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        import fitz

        with fitz.open(path) as doc:
            parts = [(doc.load_page(i).get_text() or "") for i in range(doc.page_count)]
        return "\n\n".join(parts).strip()
    if ext in {".txt", ".md", ".csv"}:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    if ext == ".docx":
        import docx2txt

        return (docx2txt.process(str(path)) or "").strip()
    if ext == ".doc":
        text, error = extract_doc_text(path)
        if not text.strip():
            raise ValueError(error or "пустой текст .doc")
        return text.strip()
    raise ValueError(f"локальный текст не поддержан для {ext}")


def render_document_images(
    path: Path,
    *,
    dpi_scale: float = 1.5,
    max_pages: int = 80,
    jpeg_quality: int = 85,
) -> list[tuple[int, str]]:
    """Возвращает список (page_number_1based, data_url) для multimodal VL."""
    ext = path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        raw = path.read_bytes()
        mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
        }.get(ext, "image/jpeg")
        b64 = base64.b64encode(raw).decode("ascii")
        return [(1, f"data:{mime};base64,{b64}")]

    if ext != ".pdf":
        raise ValueError(f"VL-рендер не поддержан для {ext}")

    import fitz
    from PIL import Image

    out: list[tuple[int, str]] = []
    matrix = fitz.Matrix(dpi_scale, dpi_scale)
    with fitz.open(path) as doc:
        total = min(doc.page_count, max_pages)
        for i in range(total):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            out.append((i + 1, f"data:image/jpeg;base64,{b64}"))
    if not out:
        raise ValueError(f"PDF без страниц: {path.name}")
    return out


def _load_legacy_doc_files(
    files: list[tuple[Path, str]],
    corpus: str,
    warnings: list[str],
) -> list[Document]:
    documents: list[Document] = []
    for path, label in files:
        text, error = extract_doc_text(path)
        if text and len(text.strip()) >= 20:
            documents.append(
                Document(
                    text=text,
                    metadata={
                        "file_name": Path(label).name,
                        "file_path": label,
                        "corpus": corpus,
                        "location": "документ",
                    },
                )
            )
            continue
        if error:
            warnings.append(f"Не удалось прочитать {label}: {error}")
        else:
            warnings.append(f"Пустой .doc: {label}")
    return documents
