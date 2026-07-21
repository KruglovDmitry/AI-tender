"""LLM-извлечение предмета закупки и требований из документов тендера."""

from __future__ import annotations

from typing import Any

from llama_index.core import Document
from llama_index.core.llms import LLM

from .anchors import numbered_excerpt
from .models import ExtractedRequirement
from .providers import parse_llm_json
from .query_select import (
    attach_anchor,
    dedupe_requirements,
    merge_documents_by_file,
    source_node_from_file,
)


SCOPE_SCHEMA_HINT = """
Верни ТОЛЬКО JSON-объект:
{
  "scope_summary": "коротко: что закупают (1-2 предложения)",
  "scope_items": [
    {
      "name": "название/обозначение позиции (или единый предмет, если только он один)",
      "confidence": 0.0..1.0,
      "quote": "дословная цитата из документа (место, где это описано)"
    }
  ],
  "overall_confidence": 0.0..1.0,
  "needs_more_docs": true|false,
  "missing_signals": "если needs_more_docs=true — что именно не удалось вытащить"
}

Правила:
- Используй ТОЛЬКО текст тендера, который ниже.
- Если предмет закупки не удаётся выделить: scope_items = [] и needs_more_docs = true.
- Если предмет закупки состоит из одного пункта — верни scope_items с единственным элементом.
""".strip()


SCOPE_REQUIREMENTS_SCHEMA_HINT = """
Верни ТОЛЬКО JSON-объект вида:
{
  "requirements": [
    {
      "text": "краткая формулировка одного требования",
      "quote": "дословная непрерывная цитата из документа",
      "kind": "product|specs|other",
      "priority": 1..3,
      "confidence": 0.0..1.0,
      "scope_index": 0..(len(scope_items)-1) | -1
    }
  ]
}

Правила:
- Извлекай ТОЛЬКО требования, относящиеся к любому из scope_items.
- scope_index: укажи индекс scope_items, которому это требование ближе всего.
  Если нельзя привязать — scope_index = -1.
- quote — дословный фрагмент из документа. Не пересказывай.
""".strip()


def extract_procurement_scope_from_documents(
    documents: list[Document],
    llm: LLM,
    *,
    max_chars_per_doc: int,
    scope_max_items: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files = merge_documents_by_file(documents)
    if not files:
        return [], {
            "scope_summary": "",
            "overall_confidence": 0.0,
            "needs_more_docs": True,
            "missing_signals": "нет документов для извлечения scope",
        }

    scope_docs: list[str] = []
    for label, text, _page in files:
        numbered = numbered_excerpt(text, max_chars=max_chars_per_doc)
        scope_docs.append(f"ФАЙЛ: {label}\n{numbered}")

    joined_docs = "\n\n".join(scope_docs)
    prompt = (
        "Ты аналитик закупок. ВЫДЕЛИ ПРЕДМЕТ ЗАКУПКИ и/или ПЕРЕЧЕНЬ ПОЗИЦИЙ, "
        "которые нужно поставить/выполнить.\n\n"
        f"{SCOPE_SCHEMA_HINT}\n\n"
        f"ДОКУМЕНТЫ ТЕНДЕРА:\n{joined_docs}"
    )
    response = llm.complete(prompt)
    data = parse_llm_json(str(response))

    scope_items: list[dict[str, Any]] = []
    for item in data.get("scope_items", [])[:scope_max_items]:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        try:
            confidence = float(item.get("confidence", 0.5) or 0.5)
        except (TypeError, ValueError):
            confidence = 0.5
        scope_items.append(
            {
                "name": name,
                "confidence": confidence,
                "quote": str(item.get("quote", "")).strip(),
            }
        )
    try:
        overall = float(data.get("overall_confidence", 0.5) or 0.5)
    except (TypeError, ValueError):
        overall = 0.5
    meta = {
        "scope_summary": str(data.get("scope_summary", "")).strip(),
        "overall_confidence": overall,
        "needs_more_docs": bool(data.get("needs_more_docs", False)),
        "missing_signals": str(data.get("missing_signals", "")).strip(),
    }
    return scope_items, meta


def extract_requirements_for_scope_from_documents(
    documents: list[Document],
    *,
    scope_items: list[dict[str, Any]],
    llm: LLM,
    limit: int,
    max_chars_per_doc: int,
    file_order: list[str] | None = None,
) -> tuple[list[ExtractedRequirement], dict[str, Any]]:
    files = merge_documents_by_file(documents)
    if file_order:
        order_index = {label: index for index, label in enumerate(file_order)}
        files = sorted(
            files,
            key=lambda item: (order_index.get(item[0], len(order_index)), item[0].lower()),
        )

    raw_all: list[ExtractedRequirement] = []
    truncated_files: list[str] = []
    stats: dict[str, Any] = {
        "mode": "scope_requirements",
        "files": len(files),
        "extracted_raw": 0,
        "requirements": 0,
        "selected": 0,
        "anchored": 0,
        "truncated_files": truncated_files,
        "scope_items_count": len(scope_items),
    }

    scope_list = "\n".join(
        f"{idx}) {item.get('name', '').strip()}" for idx, item in enumerate(scope_items)
    ).strip()
    scope_names = [str(item.get("name", "")).strip() for item in scope_items]

    for label, text, page in files:
        numbered = numbered_excerpt(text, max_chars=max_chars_per_doc)
        truncated = len(numbered) >= max_chars_per_doc or numbered.endswith("…")
        if truncated:
            truncated_files.append(label)
        source_node = source_node_from_file(label, text, page=page)

        prompt = (
            "Ты аналитик закупок. ТЕНДЕРНЫЙ ДОКУМЕНТ НИЖЕ.\n"
            "У нас уже выделен ПРЕДМЕТ ЗАКУПКИ как scope_items. "
            "Извлеки только требования, относящиеся к одному или нескольким scope_items. "
            "Если в документе нет требований под scope_items — верни пустой список.\n\n"
            f"scope_items:\n{scope_list}\n\n"
            f"{SCOPE_REQUIREMENTS_SCHEMA_HINT}\n\n"
            f"ФАЙЛ: {label}\n"
            f"{'(документ обрезан по лимиту длины) ' if truncated else ''}"
            f"ДОКУМЕНТ:\n{numbered}"
        )
        response = llm.complete(prompt)
        data = parse_llm_json(str(response))

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
            if kind not in {"product", "specs", "other"}:
                kind = "other"

            try:
                scope_index_i = int(item.get("scope_index", -1))
            except (TypeError, ValueError):
                scope_index_i = -1
            scope_item_name = (
                scope_names[scope_index_i] if 0 <= scope_index_i < len(scope_names) else None
            )

            req = attach_anchor(
                text=req_text,
                quote=quote,
                source_node=source_node,
                priority=min(max(priority, 0), 3),
                confidence=min(max(confidence, 0.0), 1.0),
                kind=kind,
            )
            req.scope_item = scope_item_name
            raw_all.append(req)

    stats["extracted_raw"] = len(raw_all)
    if not raw_all:
        return [], stats

    reqs = dedupe_requirements(raw_all, limit)
    stats["requirements"] = len(raw_all)
    stats["selected"] = len(reqs)
    stats["anchored"] = sum(1 for item in reqs if item.line_start is not None)
    return reqs, stats
