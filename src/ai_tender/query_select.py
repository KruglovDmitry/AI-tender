"""Извлечение требований из тендера через LLM (документ целиком) + RAG."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from llama_index.core import Document
from llama_index.core.llms import LLM
from llama_index.core.schema import BaseNode, TextNode

from .anchors import format_location, locate_quote, numbered_excerpt
from .index import select_query_nodes
from .models import Evidence, ExtractedRequirement

EXTRACT_SCHEMA_HINT = """
Верни ТОЛЬКО JSON-объект:
{
  "requirements": [
    {
      "text": "краткая формулировка одного основного требования",
      "quote": "ДОСЛОВНАЯ непрерывная цитата из документа (место упоминания)",
      "kind": "product|specs|other",
      "priority": 3,
      "confidence": 0.9
    }
  ]
}

Приоритет извлечения:
1) Сначала product — явные артикулы / полные обозначения / названия приборов
   и позиций поставки (например «МИР С-05.10-230-5(80)-…»).
2) Затем specs — технические требования. Если для позиции уже есть product,
   НЕ дроби каждый параметр отдельно: объедини связанные ТТХ одной позиции
   в 1–3 блока specs (не 10+ пунктов про ток/напряжение/МПИ по отдельности).
3) other — только если иначе нельзя.

Не извлекай:
- реквизиты заказчика, контакты, procedural текст закупки, обеспечение заявок.

Правила:
- text — сжатая формулировка для поиска в эталоне.
- quote — дословный фрагмент из документа (без пересказа). В тексте строки "N|...".
- kind: product = артикул/название; specs = техтребования; other = иное.
- priority: 3=ключевые позиции/жёсткие ТТХ, 2=важные условия, 1=второстепенное.
- Если требований нет — {"requirements": []}.
- Не выдумывай факты вне текста документа.
""".strip()


def node_raw_text(node: BaseNode) -> str:
    return node.get_content(metadata_mode="none") or ""


def node_text(node: BaseNode, limit: int = 900) -> str:
    text = " ".join(node_raw_text(node).split())
    return text[:limit]


def _parse_json(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = (
            text.removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Ответ LLM должен быть JSON-объектом")
    return data


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


def _source_node_from_file(file_label: str, text: str, page: int | None = None) -> TextNode:
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


def requirement_to_evidence(req: ExtractedRequirement) -> Evidence:
    return Evidence(
        file=req.file,
        location=req.location,
        quote=req.quote or req.text,
        page=req.page,
        line_start=req.line_start,
        line_end=req.line_end,
    )


def requirement_to_query_node(req: ExtractedRequirement) -> TextNode:
    body = req.text.strip()
    if req.quote and req.quote.strip() and req.quote.strip() not in body:
        body = f"{body}\n\n{req.quote.strip()}"
    return TextNode(
        text=body[:1200],
        metadata={
            "file_path": req.file,
            "file_name": req.file,
            "location": req.location,
            "page_number": req.page,
            "line_start": req.line_start,
            "line_end": req.line_end,
            "requirement_text": req.text,
            "corpus": "tender",
        },
    )


def split_products_and_specs(
    items: list[ExtractedRequirement],
) -> tuple[list[ExtractedRequirement], list[ExtractedRequirement]]:
    products = [item for item in items if item.kind == "product"]
    specs = [item for item in items if item.kind != "product"]
    return products, specs


def product_match_succeeded(
    findings: list,
    *,
    min_confidence: float = 0.55,
) -> bool:
    from .models import Status

    return any(
        item.status in (Status.found, Status.partial)
        and item.confidence >= min_confidence
        for item in findings
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

    # Продукты не вытесняем лимитом: всегда оставляем их, остаток — specs.
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


def extract_requirements_from_document(
    llm: LLM,
    *,
    file_label: str,
    text: str,
    page: int | None = None,
    max_chars: int = 120_000,
) -> tuple[list[ExtractedRequirement], bool]:
    """Один цельный документ → список требований. truncated=True если текст обрезан."""
    numbered = numbered_excerpt(text, max_chars=max_chars)
    truncated = len(numbered) >= max_chars or numbered.endswith("…")
    source_node = _source_node_from_file(file_label, text, page=page)

    prompt = (
        "Ты аналитик закупок. Ниже — тендерный документ ЦЕЛИКОМ (строки вида N|текст). "
        "Выдели основные требования, если они есть, и укажи места упоминания "
        "дословными цитатами.\n\n"
        f"{EXTRACT_SCHEMA_HINT}\n\n"
        f"ФАЙЛ: {file_label}\n"
        f"{'(документ обрезан по лимиту длины) ' if truncated else ''}"
        f"ДОКУМЕНТ:\n{numbered}"
    )
    response = llm.complete(prompt)
    data = _parse_json(str(response))
    extracted: list[ExtractedRequirement] = []
    for item in data.get("requirements", []):
        req_text = str(item.get("text", "")).strip()
        quote = str(item.get("quote", "")).strip() or req_text
        if not req_text:
            continue
        try:
            priority = int(item.get("priority", 2))
        except (TypeError, ValueError):
            priority = 2
        try:
            confidence = float(item.get("confidence", 0.7))
        except (TypeError, ValueError):
            confidence = 0.7
        kind = str(item.get("kind", "other")).strip().lower()
        extracted.append(
            attach_anchor(
                text=req_text,
                quote=quote,
                source_node=source_node,
                priority=min(max(priority, 0), 3),
                confidence=min(max(confidence, 0.0), 1.0),
                kind=kind,
            )
        )
    return extracted, truncated


def fallback_requirements_from_nodes(
    nodes: list[BaseNode],
    limit: int,
) -> list[ExtractedRequirement]:
    selected = select_query_nodes(nodes, limit)
    output: list[ExtractedRequirement] = []
    for node in selected:
        raw = node_raw_text(node)
        quote = raw.strip()[:500] or node_text(node, limit=500)
        output.append(
            attach_anchor(
                text=quote[:300],
                quote=quote,
                source_node=node,
                priority=1,
                confidence=0.4,
            )
        )
    return output


def extract_tender_requirements_from_documents(
    documents: list[Document],
    limit: int,
    llm: LLM | None = None,
    *,
    use_llm: bool = True,
    max_chars_per_doc: int = 120_000,
    file_order: list[str] | None = None,
    early_stop: bool = False,
    early_stop_min_specs: int = 2,
    early_stop_min_confidence: float = 0.55,
    early_stop_min_files: int = 2,
    max_files_to_process: int | None = None,
) -> tuple[list[ExtractedRequirement], dict[str, Any]]:
    """
    Основной режим: каждый файл тендера целиком → LLM extract → дедуп → top-N.
    При file_order — обработка в заданном порядке; early_stop — остановка при достаточном наборе.
    """
    files = merge_documents_by_file(documents)
    if file_order:
        order_index = {label: index for index, label in enumerate(file_order)}
        files = sorted(
            files,
            key=lambda item: (order_index.get(item[0], len(order_index)), item[0].lower()),
        )

    stats: dict[str, Any] = {
        "mode": "uniform_fallback",
        "input_documents": len(documents),
        "files": len(files),
        "extracted_raw": 0,
        "requirements": 0,
        "selected": 0,
        "anchored": 0,
        "truncated_files": [],
        "files_processed": [],
        "early_stopped": False,
    }
    if not files:
        return [], stats

    if max_files_to_process is not None:
        files = files[: max(0, max_files_to_process)]

    if not use_llm or llm is None:
        # Без LLM: берём равномерные куски склеенных файлов как псевдо-требования.
        nodes = [
            _source_node_from_file(label, text, page=page)
            for label, text, page in files
        ]
        # Нарежем крупные файлы через select_query_nodes по «виртуальным» чанкам текста.
        pseudo_nodes: list[BaseNode] = []
        for node in nodes:
            raw = node_raw_text(node)
            step = 1200
            for start in range(0, max(len(raw), 1), step):
                chunk = raw[start : start + step]
                if chunk.strip():
                    pseudo_nodes.append(
                        TextNode(text=chunk, metadata=dict(node.metadata or {}))
                    )
        reqs = fallback_requirements_from_nodes(pseudo_nodes or nodes, limit)
        stats["mode"] = "uniform"
        stats["requirements"] = len(reqs)
        stats["selected"] = len(reqs)
        stats["anchored"] = sum(1 for item in reqs if item.line_start is not None)
        return reqs, stats

    raw_all: list[ExtractedRequirement] = []
    try:
        from .doc_select import extraction_is_sufficient

        for label, text, page in files:
            items, truncated = extract_requirements_from_document(
                llm,
                file_label=label,
                text=text,
                page=page,
                max_chars=max_chars_per_doc,
            )
            raw_all.extend(items)
            stats["files_processed"].append(label)
            if truncated:
                stats["truncated_files"].append(label)
            if early_stop and extraction_is_sufficient(
                raw_all,
                min_specs=early_stop_min_specs,
                min_confidence=early_stop_min_confidence,
            ) and len(stats["files_processed"]) >= max(1, early_stop_min_files):
                stats["early_stopped"] = True
                break
    except Exception as exc:
        nodes = [_source_node_from_file(label, text, page=page) for label, text, page in files]
        reqs = fallback_requirements_from_nodes(nodes, limit)
        stats["mode"] = "fallback_uniform"
        stats["error"] = str(exc)
        stats["requirements"] = len(reqs)
        stats["selected"] = len(reqs)
        stats["anchored"] = sum(1 for item in reqs if item.line_start is not None)
        return reqs, stats

    stats["extracted_raw"] = len(raw_all)
    if not raw_all:
        nodes = [_source_node_from_file(label, text, page=page) for label, text, page in files]
        reqs = fallback_requirements_from_nodes(nodes, limit)
        stats["mode"] = "llm_empty_fallback_uniform"
        stats["requirements"] = len(reqs)
        stats["selected"] = len(reqs)
        stats["anchored"] = sum(1 for item in reqs if item.line_start is not None)
        return reqs, stats

    reqs = dedupe_requirements(raw_all, limit)
    stats["mode"] = "llm_extract_whole_doc"
    if early_stop:
        stats["mode"] = "llm_extract_sequential"
    stats["requirements"] = len(raw_all)
    stats["selected"] = len(reqs)
    stats["anchored"] = sum(1 for item in reqs if item.line_start is not None)
    stats["top_requirements"] = [
        {
            "priority": item.priority,
            "confidence": item.confidence,
            "text": item.text[:160],
            "location": item.location,
            "line_start": item.line_start,
            "line_end": item.line_end,
        }
        for item in reqs[:5]
    ]
    return reqs, stats


# Совместимость: старый API по чанкам больше не основной путь.
def extract_tender_requirements(
    nodes: list[BaseNode],
    limit: int,
    llm: LLM | None = None,
    *,
    use_llm: bool = True,
    max_extract_candidates: int = 40,
    extract_batch_size: int = 8,
) -> tuple[list[ExtractedRequirement], dict[str, Any]]:
    del max_extract_candidates, extract_batch_size
    documents = [
        Document(text=node_raw_text(node), metadata=dict(node.metadata or {}))
        for node in nodes
        if node_raw_text(node).strip()
    ]
    return extract_tender_requirements_from_documents(
        documents,
        limit,
        llm=llm,
        use_llm=use_llm,
    )


def select_requirement_query_nodes(
    nodes: list[BaseNode],
    limit: int,
    llm: LLM | None = None,
    *,
    use_llm: bool = True,
    max_classify_candidates: int = 40,
    classify_batch_size: int = 12,
) -> tuple[list[BaseNode], dict[str, Any]]:
    del max_classify_candidates, classify_batch_size
    reqs, stats = extract_tender_requirements(nodes, limit, llm=llm, use_llm=use_llm)
    return [requirement_to_query_node(item) for item in reqs], stats
