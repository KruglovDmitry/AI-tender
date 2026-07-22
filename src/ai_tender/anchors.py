"""Якоря цитат в тексте документа: точные строки для ссылок."""

from __future__ import annotations

import re
from dataclasses import dataclass


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
