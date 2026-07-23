"""Текст документов: якоря цитат, склейка файлов, сборка ExtractedRequirement."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from llama_index.core import Document
from llama_index.core.schema import BaseNode, TextNode

from ..models import ExtractedRequirement


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

    idx = source.find(quote)
    if idx >= 0:
        return _anchor_from_span(source, idx, idx + len(quote))

    lower_src = source.lower()
    lower_q = quote.lower()
    idx = lower_src.find(lower_q)
    if idx >= 0:
        return _anchor_from_span(source, idx, idx + len(quote))

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


def node_raw_text(node: BaseNode) -> str:
    return node.get_content(metadata_mode="none") or ""


def source_node_from_file(file_label: str, text: str, page: int | None = None) -> TextNode:
    location = f"стр. {page}" if page is not None else "документ"
    return TextNode(
        text=text,
        metadata={
            "file_path": file_label,
            "file_name": file_label,
            "location": location,
            "page_number": page,
            "corpus": "tender",
        },
    )


def _strip_line_prefixes(quote: str) -> str:
    lines = []
    for line in (quote or "").splitlines() or [quote]:
        if "|" in line[:6]:
            prefix, _, rest = line.partition("|")
            if prefix.strip().isdigit():
                lines.append(rest)
                continue
        lines.append(line)
    return "\n".join(lines).strip()


def attach_anchor(
    *,
    text: str,
    quote: str,
    source_node: BaseNode,
    priority: int,
    confidence: float,
    kind: str = "other",
) -> ExtractedRequirement:
    raw = node_raw_text(source_node)
    cleaned_quote = _strip_line_prefixes(quote) or text
    anchor = locate_quote(raw, cleaned_quote) or locate_quote(raw, text)
    meta = source_node.metadata or {}
    page = None
    try:
        if meta.get("page_number") is not None:
            page = int(meta["page_number"])
    except (TypeError, ValueError):
        page = None

    if anchor is not None:
        exact_quote = anchor.quote.strip() or cleaned_quote
        line_start, line_end = anchor.line_start, anchor.line_end
    else:
        exact_quote = " ".join(cleaned_quote.split())[:500]
        line_start = line_end = None

    location = format_location(
        str(meta.get("location") or "документ"),
        page=page,
        line_start=line_start,
        line_end=line_end,
    )
    kind_norm = (kind or "other").strip().lower()
    if kind_norm not in {"product", "specs", "other"}:
        kind_norm = "other"
    return ExtractedRequirement(
        text=text[:500],
        quote=exact_quote[:800],
        file=str(meta.get("file_path") or meta.get("file_name") or "unknown"),
        location=location,
        page=page,
        line_start=line_start,
        line_end=line_end,
        kind=kind_norm,
        priority=priority,
        confidence=confidence,
    )


def _normalize_key(text: str) -> str:
    return " ".join(text.lower().split())


def dedupe_requirements(
    items: list[ExtractedRequirement],
    limit: int,
) -> list[ExtractedRequirement]:
    """Сначала product, затем specs/other; внутри — по priority/confidence."""
    kind_rank = {"product": 0, "specs": 1, "other": 2}
    ranked = sorted(
        items,
        key=lambda item: (
            kind_rank.get(item.kind, 9),
            -item.priority,
            -item.confidence,
        ),
    )
    seen: set[str] = set()
    products: list[ExtractedRequirement] = []
    rest: list[ExtractedRequirement] = []
    for item in ranked:
        key = _normalize_key(item.text)
        if not key or key in seen:
            continue
        seen.add(key)
        if item.kind == "product":
            products.append(item)
        else:
            rest.append(item)

    output = list(products)
    room = max(0, limit - len(output))
    output.extend(rest[:room])
    return output[: max(len(products), limit)]


def merge_documents_by_file(documents: list[Document]) -> list[tuple[str, str, int | None]]:
    """Склеивает страницы/фрагменты одного файла в цельный текст."""
    buckets: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for index, doc in enumerate(documents):
        meta = doc.metadata or {}
        label = str(meta.get("file_path") or meta.get("file_name") or f"doc-{index}")
        page_raw = meta.get("page_number", meta.get("page_label"))
        try:
            page = int(page_raw) if page_raw is not None else 10**9
        except (TypeError, ValueError):
            page = 10**9
        text = (doc.text or "").strip()
        if text:
            buckets[label].append((page, index, text))

    merged: list[tuple[str, str, int | None]] = []
    for label, parts in sorted(buckets.items(), key=lambda item: item[0]):
        parts.sort(key=lambda item: (item[0], item[1]))
        text = "\n".join(part[2] for part in parts)
        pages = [part[0] for part in parts if part[0] < 10**9]
        page = pages[0] if len(pages) == 1 else None
        if text.strip():
            merged.append((label, text, page))
    return merged
