"""Общие хелперы текста документов и требований (scope + requirements)."""

from __future__ import annotations

from collections import defaultdict
from llama_index.core import Document
from llama_index.core.schema import BaseNode, TextNode

from .anchors import format_location, locate_quote
from .models import ExtractedRequirement


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
