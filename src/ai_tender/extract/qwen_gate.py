"""Pre-flight gate: можно ли отправить файл в Qwen doc/long или нужен scan/legacy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

# DashScope / Qwen doc: лимит на файл.
MAX_FILE_BYTES = 150 * 1024 * 1024

# PDF без текстового слоя → отдельный scan-контракт (VL), не doc-extract.
PDF_MIN_TOTAL_CHARS = 500
PDF_MIN_CHARS_PER_PAGE = 80

# В Qwen doc (whole-file) — документы и презентации, не таблицы.
QWEN_DOC_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".txt",
    ".md",
    ".csv",
    ".ppt",
    ".pptx",
}

# Gate «знает» формат, но в Qwen не отправляет — legacy (pandas/текст LLM).
LEGACY_SPREADSHEET_EXTENSIONS = {".xls", ".xlsx"}

# Отдельный scan-контракт (qwen3-vl-plus и т.п.).
SCAN_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}

# Полностью неподдерживаемое → legacy без сети.
UNSUPPORTED_EXTENSIONS = {".rar", ".zip", ".7z", ".exe", ".bin"}


class ExtractRoute(StrEnum):
    """Маршрут извлечения. scan — другой API, не флаг doc-вызова."""

    qwen_doc = "qwen_doc"
    qwen_long = "qwen_long"
    qwen_scan = "qwen_scan"
    legacy = "legacy"


@dataclass(frozen=True)
class GateDecision:
    ok: bool
    route: ExtractRoute
    reason: str

    @property
    def sends_to_qwen_doc(self) -> bool:
        return self.ok and self.route in {ExtractRoute.qwen_doc, ExtractRoute.qwen_long}

    @property
    def uses_scan_contract(self) -> bool:
        return self.ok and self.route == ExtractRoute.qwen_scan


def _pdf_text_layer_stats(path: Path) -> tuple[int, int]:
    import fitz

    with fitz.open(path) as doc:
        pages = max(doc.page_count, 1)
        chars = 0
        for i in range(doc.page_count):
            chars += len(doc.load_page(i).get_text() or "")
    return chars, pages


def can_send_to_qwen(path: Path, *, purpose: str = "extract") -> GateDecision:
    """
    Решает маршрут ДО сети.

    purpose: «catalog» | «tender» — для логов; логика gate общая.
    """
    del purpose
    if not path.is_file():
        return GateDecision(False, ExtractRoute.legacy, f"файл не найден: {path}")

    ext = path.suffix.lower()
    size = path.stat().st_size

    if ext in UNSUPPORTED_EXTENSIONS:
        return GateDecision(
            False,
            ExtractRoute.legacy,
            f"архив/неподдерживаемый формат {ext} — распакуйте локально",
        )

    if ext in LEGACY_SPREADSHEET_EXTENSIONS:
        return GateDecision(
            False,
            ExtractRoute.legacy,
            f"таблица {ext} — legacy (локальный разбор, не Qwen doc)",
        )

    if size > MAX_FILE_BYTES:
        return GateDecision(
            False,
            ExtractRoute.legacy,
            f"файл {size} байт > лимита {MAX_FILE_BYTES} байт",
        )

    if ext in SCAN_IMAGE_EXTENSIONS:
        return GateDecision(
            True,
            ExtractRoute.qwen_scan,
            "изображение — отдельный VL/scan контракт",
        )

    if ext not in QWEN_DOC_EXTENSIONS:
        return GateDecision(
            False,
            ExtractRoute.legacy,
            f"расширение {ext} не в списке Qwen doc",
        )

    if ext == ".pdf":
        try:
            chars, pages = _pdf_text_layer_stats(path)
        except Exception as exc:
            return GateDecision(
                False,
                ExtractRoute.legacy,
                f"не удалось прочитать PDF: {exc}",
            )
        per_page = chars / max(pages, 1)
        if chars < PDF_MIN_TOTAL_CHARS or per_page < PDF_MIN_CHARS_PER_PAGE:
            return GateDecision(
                True,
                ExtractRoute.qwen_scan,
                f"PDF без текстового слоя (chars={chars}, per_page={per_page:.0f})",
            )

    # TODO: эвристика qwen_long (много файлов / оценка токенов > порога doc).
    return GateDecision(True, ExtractRoute.qwen_doc, "текстовый документ → qwen-doc-turbo")
