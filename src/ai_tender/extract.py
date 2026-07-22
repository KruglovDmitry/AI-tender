"""LLM-извлечение предмета закупки и требований из документов тендера."""

from __future__ import annotations

from typing import Any

from llama_index.core import Document
from llama_index.core.llms import LLM

from .anchors import numbered_excerpt
from .models import ExtractedRequirement
from .providers import parse_llm_json, try_parse_llm_json
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


PER_SCOPE_REQUIREMENTS_SCHEMA_HINT = """
Верни ТОЛЬКО валидный JSON-объект (без markdown и комментариев):
{
  "items": [
    {
      "scope_index": 0,
      "requirements": [
        {
          "text": "краткая формулировка одного требования",
          "quote": "короткая дословная цитата до 160 символов",
          "kind": "product|specs|other",
          "priority": 1..3,
          "confidence": 0.0..1.0
        }
      ]
    }
  ]
}

Правила:
- Строго валидный JSON: экранируй кавычки в строках как \\", без висячих запятых.
- Для каждого scope_index из запроса — отдельный блок (можно с пустым requirements).
- Не более max_per_item требований на позицию.
- quote короткий (до 160 символов), без переносов строк внутри строки JSON.
- Не пересказывай документ целиком.
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


def parse_requirements_per_scope_payload(
    data: dict[str, Any],
    *,
    scope_items: list[dict[str, Any]],
    source_node,
    max_per_item: int,
) -> list[list[ExtractedRequirement]]:
    """Разбор JSON LLM → списки требований, выровненные по scope_items."""
    scope_names = [str(item.get("name", "")).strip() for item in scope_items]
    buckets: list[list[ExtractedRequirement]] = [[] for _ in scope_items]

    for block in data.get("items", []) or []:
        try:
            scope_index = int(block.get("scope_index", -1))
        except (TypeError, ValueError):
            continue
        if not (0 <= scope_index < len(scope_items)):
            continue
        scope_name = scope_names[scope_index]
        raw: list[ExtractedRequirement] = []
        for item in block.get("requirements", []) or []:
            req_text = str(item.get("text", "")).strip()
            quote = str(item.get("quote", "")).strip() or req_text
            if not req_text:
                continue
            try:
                priority = int(item.get("priority", 2))
            except (TypeError, ValueError):
                priority = 2
            confidence = min(
                max(_parse_optional_float(item.get("confidence"), 0.7), 0.0),
                1.0,
            )
            kind = str(item.get("kind", "other")).strip().lower()
            if kind not in {"product", "specs", "other"}:
                kind = "other"
            req = attach_anchor(
                text=req_text,
                quote=quote,
                source_node=source_node,
                priority=min(max(priority, 0), 3),
                confidence=confidence,
                kind=kind,
            )
            req.scope_item = scope_name
            raw.append(req)
        if raw:
            buckets[scope_index].extend(raw)

    capped: list[list[ExtractedRequirement]] = []
    for bucket in buckets:
        capped.append(dedupe_requirements(bucket, max_per_item) if bucket else [])
    return capped


def _complete_requirements_json(llm: LLM, prompt: str) -> dict[str, Any] | None:
    """Запрос к LLM + мягкий retry при битом JSON."""
    response = llm.complete(prompt)
    data = try_parse_llm_json(str(response))
    if data is not None:
        return data

    repair_prompt = (
        "Предыдущий ответ был НЕВАЛИДНЫМ JSON. "
        "Верни ТОЛЬКО исправленный валидный JSON-объект той же структуры "
        "(поле items с scope_index и requirements). "
        "Без markdown, без комментариев. Экранируй кавычки в строках.\n\n"
        f"ИСХОДНЫЙ ОТВЕТ:\n{str(response)[:12000]}"
    )
    repaired = llm.complete(repair_prompt)
    return try_parse_llm_json(str(repaired))


def extract_requirements_per_scope_items(
    documents: list[Document],
    *,
    scope_items: list[dict[str, Any]],
    llm: LLM,
    max_per_item: int,
    max_chars_per_doc: int,
    file_order: list[str] | None = None,
    batch_size: int = 3,
) -> tuple[list[list[ExtractedRequirement]], dict[str, Any]]:
    """Для каждой позиции перечня — свой список требований (лимит max_per_item)."""
    files = merge_documents_by_file(documents)
    if file_order:
        order_index = {label: index for index, label in enumerate(file_order)}
        files = sorted(
            files,
            key=lambda item: (order_index.get(item[0], len(order_index)), item[0].lower()),
        )

    n = len(scope_items)
    buckets: list[list[ExtractedRequirement]] = [[] for _ in range(n)]
    truncated_files: list[str] = []
    parse_errors: list[str] = []
    stats: dict[str, Any] = {
        "mode": "per_scope_requirements",
        "files": len(files),
        "scope_items_count": n,
        "max_per_item": max_per_item,
        "batch_size": batch_size,
        "truncated_files": truncated_files,
        "parse_errors": parse_errors,
        "extracted_raw": 0,
        "selected": 0,
    }

    if not files or not scope_items:
        return buckets, stats

    for label, text, page in files:
        numbered = numbered_excerpt(text, max_chars=max_chars_per_doc)
        truncated = len(numbered) >= max_chars_per_doc or numbered.endswith("…")
        if truncated:
            truncated_files.append(label)
        source_node = source_node_from_file(label, text, page=page)

        # Батчи по позициям — меньше шанс получить огромный/битый JSON.
        for batch_start in range(0, n, max(1, batch_size)):
            batch_indexes = list(range(batch_start, min(batch_start + batch_size, n)))
            scope_list = "\n".join(
                f"{idx}) {scope_items[idx].get('name', '').strip()}"
                + (
                    f" — {scope_items[idx].get('qty')} {scope_items[idx].get('unit', '')}".rstrip()
                    if scope_items[idx].get("qty") is not None
                    else ""
                )
                for idx in batch_indexes
            ).strip()
            prompt = (
                "Ты аналитик закупок. Ниже тендерный документ и ЧАСТЬ перечня позиций.\n"
                f"Для КАЖДОЙ позиции из списка извлеки до {max_per_item} требований. "
                "Если для позиции в документе нет требований — пустой requirements.\n"
                f"В ответе используй ТОЛЬКО эти scope_index: {batch_indexes}.\n\n"
                f"scope_items:\n{scope_list}\n\n"
                f"{PER_SCOPE_REQUIREMENTS_SCHEMA_HINT}\n\n"
                f"ФАЙЛ: {label}\n"
                f"{'(документ обрезан по лимиту длины) ' if truncated else ''}"
                f"ДОКУМЕНТ:\n{numbered}"
            )
            data = _complete_requirements_json(llm, prompt)
            if data is None:
                parse_errors.append(f"{label} [batch {batch_indexes[0]}-{batch_indexes[-1]}]")
                continue
            parsed = parse_requirements_per_scope_payload(
                data,
                scope_items=scope_items,
                source_node=source_node,
                max_per_item=max_per_item,
            )
            for index in batch_indexes:
                buckets[index].extend(parsed[index])

    selected = 0
    raw_total = 0
    result: list[list[ExtractedRequirement]] = []
    for bucket in buckets:
        raw_total += len(bucket)
        capped = dedupe_requirements(bucket, max_per_item) if bucket else []
        selected += len(capped)
        result.append(capped)

    stats["extracted_raw"] = raw_total
    stats["selected"] = selected
    return result, stats
