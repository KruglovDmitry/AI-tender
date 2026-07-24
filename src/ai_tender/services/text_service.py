"""Текст документов: нумерация строк, склейка файлов, сборка ExtractedRequirement."""

from __future__ import annotations

from collections import defaultdict

from llama_index.core import Document

from ..models import ExtractedRequirement


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


def _strip_line_prefixes(quote: str) -> str:
    """Убрать префиксы `N|` из цитаты модели (если скопировала нумерацию)."""
    lines = []
    for line in (quote or "").splitlines() or [quote]:
        if "|" in line[:6]:
            prefix, _, rest = line.partition("|")
            if prefix.strip().isdigit():
                lines.append(rest)
                continue
        lines.append(line)
    return "\n".join(lines).strip()


def make_requirement(
    *,
    text: str,
    quote: str,
    file: str,
    page: int | None = None,
    priority: int = 2,
    confidence: float = 0.7,
    kind: str = "other",
) -> ExtractedRequirement:
    """Собрать требование без поиска якоря в исходном тексте."""
    cleaned_quote = _strip_line_prefixes(quote) or text
    cleaned_quote = " ".join(cleaned_quote.split())[:800]
    kind_norm = (kind or "other").strip().lower()
    if kind_norm not in {"product", "specs", "other"}:
        kind_norm = "other"
    location = f"стр. {page}" if page is not None else "документ"
    return ExtractedRequirement(
        text=text[:500],
        quote=cleaned_quote,
        file=file,
        location=location,
        page=page,
        line_start=None,
        line_end=None,
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
