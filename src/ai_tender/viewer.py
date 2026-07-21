"""Просмотр фрагмента документа с подсветкой цитаты (тендер / эталон)."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

from .models import Evidence


@dataclass
class DocumentView:
    title: str
    path: Path | None
    location: str
    body_html: str
    note: str | None = None


def resolve_evidence_path(root: Path | str | None, file_label: str) -> Path | None:
    from .anchors import resolve_document_path

    return resolve_document_path(root, file_label)


def highlight_quote(text: str, quote: str) -> str:
    """HTML с <mark> вокруг найденной цитаты; иначе цитата сверху + полный текст."""
    safe_text = text or ""
    q = " ".join((quote or "").replace("…", " ").replace("...", " ").split())
    if not q:
        return _pre(safe_text)

    parts = [re.escape(word) for word in q.split() if word]
    # Обрезанная цитата: не требуем совпадения самого хвоста целиком.
    if len(parts) > 8:
        parts = parts[:8]
    if not parts:
        return _pre(safe_text)

    pattern = r"\s+".join(parts)
    match = re.search(pattern, safe_text, flags=re.IGNORECASE)
    if not match:
        return (
            f"<p><strong>Цитата:</strong> <mark>{html.escape(q)}</mark></p>"
            f"<hr/>{_pre(safe_text)}"
        )

    start, end = match.span()
    return (
        "<pre style='white-space:pre-wrap;font-family:inherit;line-height:1.45'>"
        f"{html.escape(safe_text[:start])}"
        f"<mark style='background:#ffe566;padding:0 2px'>{html.escape(safe_text[start:end])}</mark>"
        f"{html.escape(safe_text[end:])}"
        "</pre>"
    )


def build_document_view(
    evidence: Evidence,
    root: Path | str | None,
    *,
    role: str = "документ",
) -> DocumentView:
    from .anchors import format_location, slice_by_lines

    path = resolve_evidence_path(root, evidence.file)
    title = f"{role}: {Path(evidence.file).name}"
    location = format_location(
        evidence.location,
        page=evidence.page,
        line_start=evidence.line_start,
        line_end=evidence.line_end,
    )

    if path is None:
        return DocumentView(
            title=title,
            path=None,
            location=location,
            body_html=(
                f"<p>Файл не найден: <code>{html.escape(evidence.file)}</code></p>"
                f"<p><strong>Цитата:</strong></p>{highlight_quote(evidence.quote, evidence.quote)}"
            ),
            note="Проверьте путь к папке корпуса.",
        )

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        page_text, note = _pdf_page_text(path, evidence.page, evidence.quote)
        body_source = page_text or evidence.quote
        if evidence.line_start is not None and page_text:
            body_source = slice_by_lines(
                page_text,
                evidence.line_start,
                evidence.line_end,
                context=3,
            )
        return DocumentView(
            title=title,
            path=path,
            location=location,
            body_html=highlight_quote(body_source, evidence.quote),
            note=note,
        )

    text, note = _plain_file_text(path)
    if text:
        if evidence.line_start is not None:
            window = slice_by_lines(
                text,
                evidence.line_start,
                evidence.line_end,
                context=4,
            )
        else:
            window = _window_around_quote(text, evidence.quote, radius=1200)
        return DocumentView(
            title=title,
            path=path,
            location=location,
            body_html=highlight_quote(window, evidence.quote),
            note=note,
        )

    return DocumentView(
        title=title,
        path=path,
        location=location,
        body_html=(
            f"<p>Предпросмотр для <code>{html.escape(suffix or 'без расширения')}</code> "
            "ограничен. Показана сохранённая цитата:</p>"
            f"{highlight_quote(evidence.quote, evidence.quote)}"
        ),
        note=note or "Откройте исходный файл вручную при необходимости.",
    )


def _pre(text: str) -> str:
    return (
        "<pre style='white-space:pre-wrap;font-family:inherit;line-height:1.45'>"
        f"{html.escape(text)}"
        "</pre>"
    )


def _pdf_page_text(
    path: Path,
    page: int | None,
    quote: str,
) -> tuple[str, str | None]:
    try:
        import fitz
    except ImportError:
        return "", "pymupdf не установлен"

    with fitz.open(path) as pdf:
        if page is not None and 1 <= page <= pdf.page_count:
            text = pdf.load_page(page - 1).get_text() or ""
            return text, None

        # Страница неизвестна — ищем цитату по документу.
        q_parts = [re.escape(word) for word in quote.split() if word][:12]
        pattern = r"\s+".join(q_parts) if q_parts else None
        if pattern:
            regex = re.compile(pattern, re.IGNORECASE)
            for index in range(pdf.page_count):
                text = pdf.load_page(index).get_text() or ""
                if regex.search(text):
                    return text, f"Найдено на стр. {index + 1}"

        if pdf.page_count:
            return pdf.load_page(0).get_text() or "", "Страница не определена — показана стр. 1"
    return "", "PDF пуст"


def _plain_file_text(path: Path) -> tuple[str, str | None]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".csv"}:
        try:
            return path.read_text(encoding="utf-8", errors="replace"), None
        except OSError as exc:
            return "", str(exc)

    if suffix == ".docx":
        try:
            import docx2txt

            text = docx2txt.process(str(path)) or ""
            return text, None
        except Exception as exc:
            return "", f"Не удалось прочитать DOCX: {exc}"

    return "", None


def _window_around_quote(text: str, quote: str, radius: int = 1200) -> str:
    parts = [re.escape(word) for word in quote.split() if word]
    if not parts:
        return text[: radius * 2]
    pattern = r"\s+".join(parts)
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return text[: radius * 2]
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    chunk = text[start:end]
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{chunk}{suffix}"
