"""Классификация типа эталонного документа по первым страницам."""

from __future__ import annotations

from pathlib import Path

from llama_index.core.llms import LLM

from ...models import DocumentKind
from ...providers import complete_llm_json

# Больше страниц — стабильнее отличить буклет линейки от паспорта одной модели.
DEFAULT_PREVIEW_PAGES = 6
DEFAULT_PREVIEW_CHARS = 16_000
# Обложки почти без текста не считаем в лимит содержательных страниц.
_MIN_PAGE_CHARS = 120

CLASSIFY_HINT = '{"doc_type":"catalog|product|other","confidence":0.0,"reason":"..."}'

CLASSIFY_PROMPT = """\
Определи тип технического документа по фрагменту (первые страницы).

Типы (ровно один):
- catalog — каталог / буклет продуктовой линейки: несколько моделей, мощностей или
  артикулов; таблицы сравнения; модельный ряд / серия с вариантами SKU.
  Если в документе перечислены разные позиции одной серии (разные мощности, коды) —
  это catalog, даже если текст начинается с описания «серии».
- product — паспорт или подробное описание ОДНОГО конкретного изделия
  (одна модель/артикул) с глубокими характеристиками именно этой позиции.
- other — всё остальное (общие инструкции, письма, материалы без моделей).

Верни ТОЛЬКО JSON:
{{
  "doc_type": "catalog" | "product" | "other",
  "confidence": 0.0,
  "reason": "1 короткое предложение"
}}

ФАЙЛ: {filename}

ТЕКСТ:
{text}
"""


def preview_document_text(
    assets_path: Path,
    relative_path: str,
    *,
    max_pages: int = DEFAULT_PREVIEW_PAGES,
    max_chars: int = DEFAULT_PREVIEW_CHARS,
    ocr_enabled: bool = True,
    ocr_languages: str = "rus+eng",
) -> tuple[str, list[str]]:
    """Текст первых содержательных страниц файла для LLM-классификации."""
    del ocr_enabled, ocr_languages
    assets_path = assets_path.expanduser().resolve()
    rel = relative_path.replace("\\", "/").lstrip("/")
    path = assets_path / rel
    if not path.is_file():
        return "", [f"Файл не найден для превью: {rel}"]

    if path.suffix.lower() != ".pdf":
        return "", [
            f"Классификация эталонов только для PDF "
            f"(получен {path.suffix or 'без расширения'}): {rel}"
        ]
    return _preview_pdf_pages(path, max_pages=max_pages, max_chars=max_chars)


def _preview_pdf_pages(
    path: Path,
    *,
    max_pages: int,
    max_chars: int,
) -> tuple[str, list[str]]:
    import fitz

    warnings: list[str] = []
    try:
        doc = fitz.open(path)
    except Exception as exc:
        return "", [f"Не удалось открыть PDF {path.name}: {exc}"]
    try:
        parts: list[str] = []
        taken = 0
        for i in range(doc.page_count):
            raw = (doc.load_page(i).get_text() or "").strip()
            if len(raw) < _MIN_PAGE_CHARS:
                continue
            parts.append(f"--- стр. {i + 1} ---\n{raw}")
            taken += 1
            if taken >= max_pages:
                break
    finally:
        doc.close()
    text = "\n\n".join(parts).strip()
    if len(text) > max_chars:
        text = text[:max_chars]
        warnings.append(f"Превью PDF обрезано до {max_chars} символов: {path.name}")
    if not text:
        warnings.append(f"Пустой текстовый слой в первых страницах: {path.name}")
    return text, warnings


def classify_document_kind(
    llm: LLM,
    assets_path: Path,
    relative_path: str,
    *,
    max_pages: int = DEFAULT_PREVIEW_PAGES,
    ocr_enabled: bool = True,
    ocr_languages: str = "rus+eng",
) -> tuple[DocumentKind, list[str], dict]:
    """→ (kind, warnings, meta: confidence/reason/preview_len)."""
    text, warnings = preview_document_text(
        assets_path,
        relative_path,
        max_pages=max_pages,
        ocr_enabled=ocr_enabled,
        ocr_languages=ocr_languages,
    )
    meta: dict = {"preview_chars": len(text), "confidence": 0.0, "reason": ""}
    if not text.strip():
        warnings.append(
            f"Не удалось получить текст для классификации «{relative_path}» — тип other"
        )
        return DocumentKind.other, warnings, meta

    prompt = CLASSIFY_PROMPT.format(
        filename=Path(relative_path).name,
        text=text,
    )
    data, _ = complete_llm_json(
        llm,
        prompt,
        structure_hint=CLASSIFY_HINT,
        trace_name=None,
    )
    if not data:
        warnings.append(
            f"LLM не вернул тип для «{relative_path}» — считаем other"
        )
        return DocumentKind.other, warnings, meta

    raw = str(data.get("doc_type") or "other").strip().lower()
    aliases = {
        "catalog": DocumentKind.catalog,
        "catalogue": DocumentKind.catalog,
        "каталог": DocumentKind.catalog,
        "product": DocumentKind.product,
        "passport": DocumentKind.product,
        "datasheet": DocumentKind.product,
        "паспорт": DocumentKind.product,
        "описание": DocumentKind.product,
        "other": DocumentKind.other,
        "прочее": DocumentKind.other,
    }
    kind = aliases.get(raw, DocumentKind.other)
    try:
        meta["confidence"] = float(data.get("confidence") or 0.0)
    except (TypeError, ValueError):
        meta["confidence"] = 0.0
    meta["reason"] = str(data.get("reason") or "").strip()
    return kind, warnings, meta
