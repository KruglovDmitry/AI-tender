"""Адаптер TenderExtractResult → state графа (scope + requirements)."""

from __future__ import annotations

from typing import Any

from ..models import ExtractedRequirement
from ..services.text_service import dedupe_requirements, make_requirement
from .schemas import ScopeItemExtract, TenderExtractResult


def _parse_optional_qty(value: Any) -> float | int | None:
    if value is None or value == "":
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num != num:
        return None
    if abs(num - int(num)) < 1e-9:
        return int(num)
    return num


def _scope_has_detailed_list(scope_items: list[dict[str, Any]]) -> bool:
    if len(scope_items) >= 2:
        return True
    if len(scope_items) == 1 and scope_items[0].get("qty") is not None:
        return True
    return False


TENDER_QWEN_PROMPT = """
Ты аналитик закупок. По ПОЛНОМУ файлу тендерной документации извлеки:

1) scope_summary — общий титул/предмет закупки (1–2 предложения), НЕ технические требования.
2) scope_items — перечень позиций закупки (товары/работы/оборудование):
   - маркированные списки «– … – N шт.»;
   - таблицы с колонками «Наименование», «Ед. изм.», «Кол-во» (извещения 223-ФЗ);
   - даже одна позиция с qty — valid.
3) requirements — для КАЖДОЙ позиции scope_items список требований из этого же файла
   (характеристики, ГОСТ, параметры установки и т.п.), только относящиеся к этой позиции.

Верни JSON:
{
  "scope_summary": "...",
  "scope_items": [
    {
      "name": "краткое наименование без qty в name",
      "qty": 1,
      "unit": "шт.",
      "confidence": 0.9,
      "quote": "цитата строки",
      "requirements": [
        {"text": "...", "quote": "...", "kind": "specs|product|other", "priority": 2, "confidence": 0.9}
      ]
    }
  ],
  "overall_confidence": 0.9,
  "needs_more_docs": false,
  "missing_signals": ""
}

Правила:
- Не выдумывай позиции и требования — только из файла.
- «или эквивалент» включай в name как в документе.
- Если перечень с qty найден — needs_more_docs=false.
- Если только титул без позиций с qty — needs_more_docs=true, scope_items=[].
""".strip()


def _norm_name(name: str) -> str:
    return " ".join(name.lower().split())


def scope_item_to_dict(item: ScopeItemExtract, *, source_file: str) -> dict[str, Any]:
    qty = _parse_optional_qty(item.qty)
    unit = (item.unit or "").strip()
    if qty is not None and not unit:
        unit = "шт."
    return {
        "name": item.name.strip(),
        "qty": qty,
        "unit": unit,
        "confidence": min(max(float(item.confidence), 0.0), 1.0),
        "quote": (item.quote or "").strip(),
        "source_file": source_file,
    }


def merge_scope_item_lists(
    existing: list[dict[str, Any]],
    new_items: list[dict[str, Any]],
    *,
    scope_max_items: int = 40,
) -> list[dict[str, Any]]:
    merged = [dict(i) for i in existing]
    by_name = {_norm_name(str(i.get("name") or "")): i for i in merged if i.get("name")}
    for raw in new_items:
        if len(merged) >= scope_max_items:
            break
        name = str(raw.get("name") or "").strip()
        if not name:
            continue
        key = _norm_name(name)
        if key in by_name:
            prev = by_name[key]
            for field in ("qty", "unit", "quote", "confidence", "source_file"):
                val = raw.get(field)
                if val is not None and val != "" and not prev.get(field):
                    prev[field] = val
        else:
            merged.append(dict(raw))
            by_name[key] = merged[-1]
    return merged


def tender_result_to_scope(
    result: TenderExtractResult,
    *,
    source_file: str,
    scope_max_items: int = 40,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scope_items = [
        scope_item_to_dict(item, source_file=source_file)
        for item in result.scope_items[:scope_max_items]
        if item.name.strip()
    ]
    needs_more = bool(result.needs_more_docs)
    missing = (result.missing_signals or "").strip()
    if not _scope_has_detailed_list(scope_items):
        needs_more = True
        if not missing:
            missing = (
                "нет детального перечня позиций с количествами "
                "(есть только общий титул или пустой список)"
            )
    meta = {
        "scope_summary": result.scope_summary.strip(),
        "overall_confidence": min(max(float(result.overall_confidence), 0.0), 1.0),
        "needs_more_docs": needs_more,
        "missing_signals": missing,
        "extraction_mode": "qwen_whole_file",
    }
    return scope_items, meta


def tender_result_to_requirements(
    result: TenderExtractResult,
    *,
    source_file: str,
    scope_items: list[dict[str, Any]],
    max_per_item: int,
) -> list[list[ExtractedRequirement]]:
    """Строит buckets по порядку scope_items (после merge)."""
    by_name: dict[str, list] = {}
    for item in result.scope_items:
        name = item.name.strip()
        if name:
            by_name[_norm_name(name)] = item.requirements

    buckets: list[list[ExtractedRequirement]] = [[] for _ in scope_items]
    for index, scope_item in enumerate(scope_items):
        name = str(scope_item.get("name") or "").strip()
        raw_reqs = by_name.get(_norm_name(name), [])
        parsed: list[ExtractedRequirement] = []
        for raw in raw_reqs:
            text = str(raw.text or "").strip()
            if not text:
                continue
            kind = str(raw.kind or "other").strip().lower()
            if kind not in {"product", "specs", "other"}:
                kind = "other"
            req = make_requirement(
                text=text,
                quote=str(raw.quote or text)[:160],
                file=source_file,
                page=None,
                priority=min(max(int(raw.priority), 0), 3),
                confidence=min(max(float(raw.confidence), 0.0), 1.0),
                kind=kind,
            )
            req.scope_item = name
            parsed.append(req)
        buckets[index] = dedupe_requirements(parsed, max_per_item)
    return buckets


def merge_requirements_buckets(
    existing: list[list[ExtractedRequirement]],
    new_buckets: list[list[ExtractedRequirement]],
    *,
    max_per_item: int,
) -> list[list[ExtractedRequirement]]:
    if not existing:
        return new_buckets
    if len(existing) != len(new_buckets):
        return new_buckets
    merged: list[list[ExtractedRequirement]] = []
    for old, new in zip(existing, new_buckets):
        combined = dedupe_requirements(list(old) + list(new), max_per_item)
        merged.append(combined)
    return merged


def merge_scope_meta(existing: dict[str, Any], new_meta: dict[str, Any]) -> dict[str, Any]:
    summary = str(new_meta.get("scope_summary") or existing.get("scope_summary") or "").strip()
    if not summary:
        summary = str(existing.get("scope_summary") or "").strip()
    return {
        "scope_summary": summary,
        "overall_confidence": max(
            float(existing.get("overall_confidence") or 0),
            float(new_meta.get("overall_confidence") or 0),
        ),
        "needs_more_docs": bool(new_meta.get("needs_more_docs", existing.get("needs_more_docs"))),
        "missing_signals": str(
            new_meta.get("missing_signals") or existing.get("missing_signals") or ""
        ).strip(),
        "extraction_mode": new_meta.get("extraction_mode") or existing.get("extraction_mode"),
    }
