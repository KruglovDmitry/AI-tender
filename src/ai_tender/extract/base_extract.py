"""Базовый клиент whole-file извлечения Qwen DashScope (local-text + VL)."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Self, TypeVar

from pydantic import BaseModel

from ..models import Settings
from ..services.loader_service import read_document_text, render_document_images

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DATA_EGRESS_WARNING = (
    "Внимание: текст/изображения документа уходят на серверы DashScope (Qwen)."
)
TEXT_CHUNK_CHARS = 28_000
TEXT_CHUNK_OVERLAP = 1_200
EXTRACT_MAX_OUTPUT_TOKENS = 8_192
VL_MAX_TEXT_PAGES = 12

# Pre-flight gate
MAX_FILE_BYTES = 150 * 1024 * 1024
PDF_MIN_TOTAL_CHARS = 500
PDF_MIN_CHARS_PER_PAGE = 80
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
LEGACY_SPREADSHEET_EXTENSIONS = {".xls", ".xlsx"}
SCAN_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}
UNSUPPORTED_EXTENSIONS = {".rar", ".zip", ".7z", ".exe", ".bin"}


class ExtractRoute(StrEnum):
    """Маршрут извлечения. scan — другой API, не флаг doc-вызова."""

    qwen_doc = "qwen_doc"
    qwen_scan = "qwen_scan"
    legacy = "legacy"


@dataclass(frozen=True)
class GateDecision:
    ok: bool
    route: ExtractRoute
    reason: str

    @property
    def sends_to_qwen_doc(self) -> bool:
        return self.ok and self.route == ExtractRoute.qwen_doc


def _pdf_text_layer_stats(path: Path) -> tuple[int, int]:
    import fitz

    with fitz.open(path) as doc:
        pages = max(doc.page_count, 1)
        chars = 0
        for i in range(doc.page_count):
            chars += len(doc.load_page(i).get_text() or "")
    return chars, pages


def can_send_to_qwen(path: Path) -> GateDecision:
    """Решает маршрут ДО сети."""
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

    return GateDecision(True, ExtractRoute.qwen_doc, "текстовый документ → local-text + chat")


def dashscope_api_key() -> str:
    return (
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("QWEN_API_KEY")
        or ""
    ).strip()


class QwenExtractor:
    """Gate, local-text и VL — без доменной логики catalog/tender."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_QWEN_BASE_URL,
        doc_model: str = "qwen-plus",
        vl_model: str = "qwen-vl-plus",
        vl_enabled: bool = True,
        vl_pages_per_call: int = 2,
        vl_max_pages: int = 80,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        self.base_url = base_url
        self.doc_model = doc_model
        self.vl_model = vl_model
        self.vl_enabled = vl_enabled
        self.vl_pages_per_call = max(1, int(vl_pages_per_call))
        self.vl_max_pages = max(1, int(vl_max_pages))
        self.on_progress = on_progress

    @classmethod
    def _kwargs_from_settings(
        cls,
        settings: Settings,
        *,
        on_progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        return {
            "base_url": settings.qwen_base_url,
            "doc_model": settings.qwen_doc_model,
            "vl_model": settings.qwen_vl_model,
            "vl_enabled": settings.vl_enabled,
            "vl_pages_per_call": settings.qwen_vl_pages_per_call,
            "vl_max_pages": settings.qwen_vl_max_pages,
            "on_progress": on_progress,
        }

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        on_progress: Callable[[str], None] | None = None,
    ) -> Self:
        return cls(**cls._kwargs_from_settings(settings, on_progress=on_progress))

    def gate(self, path: Path) -> GateDecision:
        return can_send_to_qwen(path)

    def should_use_vl(self, gate: GateDecision, path: Path | None = None) -> bool:
        """VL: сканы всегда; текстовые PDF — только если включено и документ небольшой."""
        if gate.route == ExtractRoute.qwen_scan:
            return True
        if not self.vl_enabled:
            return False
        if path is not None and path.suffix.lower() == ".pdf":
            pages = self._pdf_page_count(path)
            if pages is not None and pages > VL_MAX_TEXT_PAGES:
                logger.info(
                    "VL пропущен для %s: %s стр. > %s — local-text extract",
                    path.name,
                    pages,
                    VL_MAX_TEXT_PAGES,
                )
                return False
        return True

    def _emit(self, message: str) -> None:
        logger.info("%s", message)
        if self.on_progress:
            self.on_progress(message)

    def _openai_client(self):
        from openai import OpenAI

        return OpenAI(
            base_url=self.base_url.rstrip("/"),
            api_key=dashscope_api_key() or "MISSING",
            timeout=300.0,
        )

    @staticmethod
    def _pdf_page_count(path: Path) -> int | None:
        if path.suffix.lower() != ".pdf":
            return None
        try:
            import fitz

            with fitz.open(path) as doc:
                return int(doc.page_count)
        except Exception:
            return None

    @staticmethod
    def _chunk_document_text(
        text: str,
        *,
        chunk_chars: int = TEXT_CHUNK_CHARS,
        overlap: int = TEXT_CHUNK_OVERLAP,
    ) -> list[str]:
        cleaned = text.strip()
        if not cleaned:
            return []
        if len(cleaned) <= chunk_chars:
            return [cleaned]
        chunks: list[str] = []
        step = max(1, chunk_chars - overlap)
        start = 0
        while start < len(cleaned):
            end = min(len(cleaned), start + chunk_chars)
            chunks.append(cleaned[start:end])
            if end >= len(cleaned):
                break
            start += step
        return chunks

    def _call_text_extract_model(
        self,
        *,
        model: str,
        document_text: str,
        prompt: str,
        schema_hint: str,
        fragment_note: str = "",
    ) -> str:
        client = self._openai_client()
        note = f"\n{fragment_note}\n" if fragment_note else "\n"
        user_content = f"{prompt}{note}--- ДОКУМЕНТ ---\n{document_text}"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Верни ТОЛЬКО валидный JSON без markdown. "
                        f"Структура: {schema_hint}"
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            max_tokens=EXTRACT_MAX_OUTPUT_TOKENS,
        )
        choice = response.choices[0] if response.choices else None
        finish = getattr(choice, "finish_reason", None) if choice else None
        raw = (choice.message.content or "") if choice else ""
        if finish == "length":
            logger.warning(
                "Qwen extract output truncated (finish_reason=length, chars=%s)",
                len(raw),
            )
        return raw

    def _call_vl_extract_model(
        self,
        *,
        model: str,
        prompt: str,
        schema_hint: str,
        image_data_urls: list[str],
        page_note: str,
    ) -> str:
        client = self._openai_client()
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"{prompt}\n\n{page_note}\n"
                    "Верни ТОЛЬКО валидный JSON без markdown."
                ),
            }
        ]
        for url in image_data_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Верни ТОЛЬКО валидный JSON без markdown. "
                        f"Структура: {schema_hint}"
                    ),
                },
                {"role": "user", "content": content},
            ],
            temperature=0,
            max_tokens=EXTRACT_MAX_OUTPUT_TOKENS,
        )
        choice = response.choices[0] if response.choices else None
        return (choice.message.content or "") if choice else ""

    @staticmethod
    def _parse_model_result(raw: str, model_cls: type[T]) -> T:
        from ..providers import try_parse_llm_json

        data = try_parse_llm_json(raw)
        if data is None:
            raise ValueError(f"Qwen вернул невалидный JSON: {raw[:500]!r}")
        return model_cls.model_validate(data)

    def extract_json(
        self,
        path: Path,
        *,
        gate: GateDecision,
        prompt: str,
        schema_hint: str,
        result_cls: type[T],
        merge_parts: Callable[[list[T]], T],
    ) -> T:
        if not gate.sends_to_qwen_doc:
            raise ValueError(f"extract_json вызван для маршрута {gate.route}: {gate.reason}")

        logger.warning(DATA_EGRESS_WARNING + " file=%s", path.name)
        model = self.doc_model
        local_text = read_document_text(path)
        if not local_text.strip():
            raise ValueError(f"не удалось извлечь локальный текст из {path.name}")
        chunks = self._chunk_document_text(local_text)
        logger.info(
            "Qwen local-text extract %s via %s (%s chars, %s chunk(s))",
            path.name,
            model,
            len(local_text),
            len(chunks),
        )
        if len(chunks) == 1:
            raw = self._call_text_extract_model(
                model=model,
                document_text=chunks[0],
                prompt=prompt,
                schema_hint=schema_hint,
            )
            return self._parse_model_result(raw, result_cls)

        parts: list[T] = []
        for idx, chunk in enumerate(chunks, start=1):
            self._emit(
                f"Qwen {path.name}: фрагмент {idx}/{len(chunks)} "
                f"({len(chunk)} символов)"
            )
            raw = self._call_text_extract_model(
                model=model,
                document_text=chunk,
                prompt=prompt,
                schema_hint=schema_hint,
                fragment_note=(
                    f"Это фрагмент {idx} из {len(chunks)} большого документа. "
                    "Извлеки ВСЕ позиции только из этого фрагмента "
                    "(без лимита 50/100); не выдумывай данные из других частей."
                ),
            )
            parts.append(self._parse_model_result(raw, result_cls))
        return merge_parts(parts)

    def extract_vl_json(
        self,
        path: Path,
        *,
        prompt: str,
        schema_hint: str,
        result_cls: type[T],
        merge_parts: Callable[[list[T]], T],
    ) -> T:
        if not dashscope_api_key():
            raise ValueError("Для VL задайте QWEN_API_KEY или DASHSCOPE_API_KEY")

        logger.warning(DATA_EGRESS_WARNING + " VL file=%s", path.name)
        pages = render_document_images(path, max_pages=self.vl_max_pages)
        self._emit(f"VL {path.name}: {len(pages)} стр., модель {self.vl_model}")
        parts: list[T] = []
        step = self.vl_pages_per_call
        for start in range(0, len(pages), step):
            batch = pages[start : start + step]
            page_nos = [str(p[0]) for p in batch]
            page_note = f"Страницы документа: {', '.join(page_nos)} (всего {len(pages)})."
            self._emit(f"VL {path.name}: стр. {', '.join(page_nos)}/{len(pages)}")
            raw = self._call_vl_extract_model(
                model=self.vl_model,
                prompt=prompt,
                schema_hint=schema_hint,
                image_data_urls=[p[1] for p in batch],
                page_note=page_note,
            )
            parts.append(self._parse_model_result(raw, result_cls))
        return merge_parts(parts)
