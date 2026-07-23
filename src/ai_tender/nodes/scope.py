"""LLM-извлечение предмета закупки (scope) из документов тендера."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from llama_index.core import Document
from llama_index.core.llms import LLM

from ..services.text_service import merge_documents_by_file, numbered_excerpt
from ..models import PipelineState, Settings
from ..providers import parse_llm_json


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




def node_load_next_scope_file(state: PipelineState) -> dict[str, Any]:
    from .common import load_labels, next_unloaded, progress

    label = next_unloaded(state)
    if not label:
        return {}
    progress(state, f"Загрузка файла для scope: {Path(label).name}", 0.28)
    docs, warns = load_labels(state, [label])
    loaded = list(state.get("loaded_labels") or [])
    if label not in loaded:
        loaded.append(label)
    return {
        "documents": docs,
        "loaded_labels": loaded,
        "scope_files_used": [label],
        "warnings": warns,
    }


def node_extract_scope(state: PipelineState) -> dict[str, Any]:
    from .common import progress

    settings: Settings = state["settings"]
    progress(state, "LangGraph: предмет закупки (перечень позиций)", 0.32)
    docs = list(state.get("documents") or [])
    scope_items, scope_meta = extract_procurement_scope_from_documents(
        docs,
        state["llm"],
        max_chars_per_doc=settings.max_extract_chars_per_doc,
    )
    return {"scope_items": scope_items, "scope_meta": scope_meta}


def route_after_scope(
    state: PipelineState,
) -> Literal["load_next_scope_file", "load_next_requirement_file"]:
    from .common import next_unloaded

    scope_items = state.get("scope_items") or []
    scope_meta = state.get("scope_meta") or {}
    needs_more = bool(scope_meta.get("needs_more_docs", False)) or not scope_has_detailed_list(
        scope_items
    )
    if needs_more and next_unloaded(state) is not None:
        return "load_next_scope_file"
    return "load_next_requirement_file"
