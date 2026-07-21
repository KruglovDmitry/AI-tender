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
  "scope_summary": "общий титул/название закупки (1-2 предложения, без списка позиций)",
  "scope_items": [
    {
      "name": "формулировка одной позиции перечня работ/оборудования (без количества)",
      "qty": 24,
      "unit": "шт.",
      "confidence": 0.0..1.0,
      "quote": "дословная цитата строки из документа"
    }
  ],
  "overall_confidence": 0.0..1.0,
  "needs_more_docs": true|false,
  "missing_signals": "если needs_more_docs=true — чего не хватает"
}

Правила:
- Используй ТОЛЬКО текст тендера ниже.
- scope_summary = официальное общее название закупки (титул), НЕ подменяй им перечень.
- scope_items = детальный ПЕРЕЧЕНЬ позиций (работы/оборудование), обычно в блоках
  «перечень», «максимальное количество», маркированных списках вида «– … – N шт.».
- Каждая строка перечня = отдельный scope_item. Не схлопывай список в один item.
- qty/unit: извлекай число и единицу (шт., компл. и т.п.), если есть; иначе qty=null, unit="".
- Если есть только общий титул без количественного перечня позиций:
  scope_items=[] (или один item без qty) и needs_more_docs=true.
- Если перечень найден (даже 1 позиция с qty) — needs_more_docs=false.
- quote — дословный фрагмент. Не пересказывай.
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


def _parse_optional_float(value: Any, default: float) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _parse_optional_qty(value: Any) -> float | int | None:
    if value is None or value == "":
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num != num:  # NaN
        return None
    if abs(num - int(num)) < 1e-9:
        return int(num)
    return num


def scope_has_detailed_list(scope_items: list[dict[str, Any]]) -> bool:
    """Достаточный детальный перечень: ≥2 позиций или хотя бы одна с qty."""
    if len(scope_items) >= 2:
        return True
    if len(scope_items) == 1 and scope_items[0].get("qty") is not None:
        return True
    return False


def extract_procurement_scope_from_documents(
    documents: list[Document],
    llm: LLM,
    *,
    max_chars_per_doc: int,
    scope_max_items: int = 40,
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
        "Ты аналитик закупок. Нужны ДВА слоя:\n"
        "1) scope_summary — общий титул закупки;\n"
        "2) scope_items — детальный перечень позиций работ/оборудования "
        "(как в ТЗ: «замена ПКУ … – 24 шт.»), а НЕ одно общее название.\n"
        "Ищи блоки «максимальное количество», «перечень», списки с «шт.».\n\n"
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
        qty = _parse_optional_qty(item.get("qty"))
        unit = str(item.get("unit", "") or "").strip()
        if qty is not None and not unit:
            unit = "шт."
        scope_items.append(
            {
                "name": name,
                "qty": qty,
                "unit": unit,
                "confidence": min(
                    max(_parse_optional_float(item.get("confidence"), 0.5), 0.0),
                    1.0,
                ),
                "quote": str(item.get("quote", "")).strip(),
            }
        )

    overall = min(max(_parse_optional_float(data.get("overall_confidence"), 0.5), 0.0), 1.0)
    needs_more = bool(data.get("needs_more_docs", False))
    missing_signals = str(data.get("missing_signals", "")).strip()

    if not scope_has_detailed_list(scope_items):
        needs_more = True
        if not missing_signals:
            missing_signals = (
                "нет детального перечня позиций с количествами "
                "(есть только общий титул или пустой список)"
            )

    meta = {
        "scope_summary": str(data.get("scope_summary", "")).strip(),
        "overall_confidence": overall,
        "needs_more_docs": needs_more,
        "missing_signals": missing_signals,
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
