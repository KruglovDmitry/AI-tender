"""Схемы VL-анализа страницы тендера."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TenderDocumentLink(BaseModel):
    url: str
    title: str = ""
    reason: str = ""


class TenderPageAnalysis(BaseModel):
    tender_title: str = ""
    notice_number: str = ""
    document_links: list[TenderDocumentLink] = Field(default_factory=list)
    page_summary: str = ""
    confidence: float = 0.0


TENDER_PAGE_VL_SCHEMA = (
    '{"tender_title": str, "notice_number": str, '
    '"document_links": [{"url": str, "title": str, "reason": str}], '
    '"page_summary": str, "confidence": float}'
)

TENDER_PAGE_VL_PROMPT = """Ты анализируешь страницу тендера или закупки (zakupki.gov.ru, РТС, Сбер-АСТ и др.).

По скриншотам найди документы тендера для скачивания:
- извещение, техническое задание, проект контракта, спецификация, документация;
- файлы PDF, DOC, DOCX, XLS, XLSX, ZIP, RAR и аналогичные вложения.

Также дан список ссылок со страницы (candidate_links). Сопоставь видимые на скриншоте документы с href из списка.

Правила:
- В document_links.url указывай ТОЛЬКО URL из candidate_links или явно видимые полные ссылки на скриншоте.
- Не выдумывай URL.
- Включай все релевантные документы тендера, не навигацию и не посторонние ссылки.
- title — название документа как на странице.
- reason — кратко, почему это документ тендера."""
