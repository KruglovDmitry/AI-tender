"""Якоря цитат в тексте документа: точные строки для ссылок."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TextAnchor:
    quote: str
    line_start: int | None = None
    line_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None


def locate_quote(source: str, quote: str) -> TextAnchor | None:
    """Находит цитату в источнике и возвращает точный фрагмент + номера строк (1-based)."""
    if not source or not (quote or "").strip():
        return None

    # 1) Точное вхождение
    idx = source.find(quote)
    if idx >= 0:
        return _anchor_from_span(source, idx, idx + len(quote))

    # 2) Без учёта регистра
    lower_src = source.lower()
    lower_q = quote.lower()
    idx = lower_src.find(lower_q)
    if idx >= 0:
        return _anchor_from_span(source, idx, idx + len(quote))

    # 3) Гибкий поиск по словам (пробелы/переносы могут отличаться)
    words = [re.escape(word) for word in quote.split() if word]
    if not words:
        return None
    if len(words) > 40:
        words = words[:40]
    pattern = r"\s+".join(words)
    match = re.search(pattern, source, flags=re.IGNORECASE)
    if not match:
        return None
    return _anchor_from_span(source, match.start(), match.end())


def _anchor_from_span(source: str, start: int, end: int) -> TextAnchor:
    exact = source[start:end]
    line_start = source.count("\n", 0, start) + 1
    line_end = source.count("\n", 0, max(end - 1, start)) + 1
    return TextAnchor(
        quote=exact,
        line_start=line_start,
        line_end=line_end,
        char_start=start,
        char_end=end,
    )


def format_location(
    base: str | None,
    *,
    page: int | None = None,
    line_start: int | None = None,
    line_end: int | None = None,
) -> str:
    parts: list[str] = []
    if page is not None:
        parts.append(f"стр. {page}")
    elif base and base.strip() and not base.strip().startswith("стр."):
        parts.append(base.strip())
    elif base and base.strip().startswith("стр.") and page is None:
        parts.append(base.strip())

    if line_start is not None:
        if line_end is not None and line_end != line_start:
            parts.append(f"строки {line_start}–{line_end}")
        else:
            parts.append(f"строка {line_start}")

    if parts:
        return " · ".join(parts)
    return (base or "фрагмент").strip() or "фрагмент"


def numbered_excerpt(source: str, max_chars: int = 120_000) -> str:
    """Текст с префиксами номеров строк для LLM."""
    if not source:
        return ""
    lines = source.splitlines() or [source]
    numbered = [f"{index}|{line}" for index, line in enumerate(lines, start=1)]
    text = "\n".join(numbered)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def slice_by_lines(
    source: str,
    line_start: int | None,
    line_end: int | None,
    *,
    context: int = 2,
) -> str:
    """Вырезает окно строк вокруг якоря (с небольшим контекстом)."""
    lines = source.splitlines() or [source]
    if not lines or line_start is None:
        return source
    start = max(1, line_start - context)
    end = min(len(lines), (line_end or line_start) + context)
    chunk = lines[start - 1 : end]
    return "\n".join(f"{start + offset}|{line}" for offset, line in enumerate(chunk))


def refine_requirement_anchors(
    requirements: list,
    root: Path | str | None,
) -> list:
    """
    Перепривязывает quote к полному файлу (номера строк документа, не чанка).
    """
    from .models import ExtractedRequirement

    if root is None:
        return requirements

    refined: list[ExtractedRequirement] = []
    for req in requirements:
        if not isinstance(req, ExtractedRequirement):
            refined.append(req)
            continue
        path = resolve_document_path(root, req.file)
        if path is None:
            refined.append(req)
            continue

        full_text = _read_text_for_anchor(path)
        if not full_text:
            refined.append(req)
            continue

        anchor = locate_quote(full_text, req.quote) or locate_quote(full_text, req.text)
        if anchor is None:
            refined.append(req)
            continue

        refined.append(
            req.model_copy(
                update={
                    "quote": anchor.quote.strip()[:800] or req.quote,
                    "line_start": anchor.line_start,
                    "line_end": anchor.line_end,
                    "location": format_location(
                        req.location,
                        page=req.page,
                        line_start=anchor.line_start,
                        line_end=anchor.line_end,
                    ),
                }
            )
        )
    return refined


def resolve_document_path(root: Path | str | None, file_label: str) -> Path | None:
    label = (file_label or "").strip()
    if not label or label == "unknown":
        return None

    direct = Path(label)
    if direct.is_file():
        return direct.resolve()

    if root is None:
        return None

    base = Path(root).expanduser()
    candidate = (base / label).resolve()
    if candidate.is_file():
        return candidate

    by_name = list(base.rglob(Path(label).name))
    files = [path for path in by_name if path.is_file()]
    if len(files) == 1:
        return files[0].resolve()
    return None


def _read_text_for_anchor(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".csv"}:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
    if suffix == ".docx":
        try:
            import docx2txt

            return docx2txt.process(str(path)) or ""
        except Exception:
            return ""
    if suffix in {".doc", ".dot"}:
        from .loaders import extract_doc_text

        text, _ = extract_doc_text(path)
        return text
    if suffix == ".pdf":
        try:
            import fitz

            parts: list[str] = []
            with fitz.open(path) as pdf:
                for page in pdf:
                    parts.append(page.get_text() or "")
            return "\n".join(parts)
        except Exception:
            return ""
    return ""
