"""TenderExtractor: extract тендера + адаптер → state графа."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models import ExtractedRequirement, ScopeItemExtract, TenderExtractResult
from ..services.text_service import dedupe_requirements, make_requirement
from .base_extract import QwenExtractor


class TenderExtractor(QwenExtractor):
    """Whole-file extract тендерной документации + маппинг в scope/requirements."""

    PROMPT = """
Ты аналитик закупок. По файлу тендерной документации извлеки ВСЕ позиции закупки.

1) scope_summary — общий титул/предмет закупки (1–2 предложения).
2) scope_items — ПОЛНЫЙ перечень позиций (товары/работы/оборудование):
   - все строки таблиц «Наименование / Ед. изм. / Кол-во»;
   - маркированные списки «– … – N шт.»;
   - НЕ ограничивай число позиций (не останавливайся на 50 или 100) — нужны ВСЕ строки.
3) requirements — только если в той же строке/ячейке есть явные характеристики;
   иначе оставь requirements: [].

Верни JSON:
{
  "scope_summary": "...",
  "scope_items": [
    {
      "name": "краткое наименование без qty в name",
      "qty": 1,
      "unit": "шт.",
      "confidence": 0.9,
      "quote": "короткая цитата",
      "requirements": []
    }
  ],
  "overall_confidence": 0.9,
  "needs_more_docs": false,
  "missing_signals": ""
}

Правила:
- Не выдумывай позиции — только из файла.
- «или эквивалент» включай в name как в документе.
- Если перечень с qty найден — needs_more_docs=false.
- Если только титул без позиций с qty — needs_more_docs=true, scope_items=[].
- Приоритет: полнота списка позиций. Компактность важнее подробных требований.
- quote ≤ 80 символов; requirements ≤ 2 на позицию; text ≤ 120 символов.
- Не оборачивай ответ в markdown; один JSON-объект.
""".strip()

    SCHEMA_HINT = (
        '{"scope_summary":"","scope_items":[{"name":"","qty":1,"unit":"шт.",'
        '"requirements":[{"text":"","kind":"specs"}]}]}'
    )

    def extract(self, path: Path, *, prompt: str | None = None) -> TenderExtractResult:
        prompt = self.PROMPT if prompt is None else prompt
        gate = self.gate(path)
        if self.should_use_vl(gate, path):
            return self.extract_vl_json(
                path,
                prompt=prompt,
                schema_hint=self.SCHEMA_HINT,
                result_cls=TenderExtractResult,
                merge_parts=self.merge_results,
            )
        if not gate.sends_to_qwen_doc:
            raise ValueError(gate.reason)
        return self.extract_json(
            path,
            gate=gate,
            prompt=prompt,
            schema_hint=self.SCHEMA_HINT,
            result_cls=TenderExtractResult,
            merge_parts=self.merge_results,
        )

    @classmethod
    def merge_results(cls, parts: list[TenderExtractResult]) -> TenderExtractResult:
        if not parts:
            return TenderExtractResult()

        by_name: dict[str, ScopeItemExtract] = {}
        summaries: list[str] = []
        missing: list[str] = []
        needs_more = False
        confidences: list[float] = []
        for part in parts:
            if part.scope_summary.strip():
                summaries.append(part.scope_summary.strip())
            if part.missing_signals.strip():
                missing.append(part.missing_signals.strip())
            needs_more = needs_more or part.needs_more_docs
            confidences.append(float(part.overall_confidence or 0.0))
            for item in part.scope_items:
                key = (item.name or "").strip().casefold()
                if not key:
                    continue
                prev = by_name.get(key)
                if prev is None or len(item.requirements) > len(prev.requirements):
                    by_name[key] = item
        return TenderExtractResult(
            scope_summary=" ".join(summaries).strip(),
            scope_items=list(by_name.values()),
            overall_confidence=(sum(confidences) / len(confidences)) if confidences else 0.0,
            needs_more_docs=needs_more,
            missing_signals="; ".join(missing),
        )

    @staticmethod
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

    @staticmethod
    def _scope_has_detailed_list(scope_items: list[dict[str, Any]]) -> bool:
        if len(scope_items) >= 2:
            return True
        if len(scope_items) == 1 and scope_items[0].get("qty") is not None:
            return True
        return False

    @staticmethod
    def _norm_name(name: str) -> str:
        return " ".join(name.lower().split())

    @classmethod
    def scope_item_to_dict(cls, item: ScopeItemExtract, *, source_file: str) -> dict[str, Any]:
        qty = cls._parse_optional_qty(item.qty)
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

    @classmethod
    def merge_scope_items(
        cls,
        existing: list[dict[str, Any]],
        new_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged = [dict(i) for i in existing]
        by_name = {cls._norm_name(str(i.get("name") or "")): i for i in merged if i.get("name")}
        for raw in new_items:
            name = str(raw.get("name") or "").strip()
            if not name:
                continue
            key = cls._norm_name(name)
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

    @classmethod
    def result_to_scope(
        cls,
        result: TenderExtractResult,
        *,
        source_file: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        scope_items = [
            cls.scope_item_to_dict(item, source_file=source_file)
            for item in result.scope_items
            if item.name.strip()
        ]
        needs_more = bool(result.needs_more_docs)
        missing = (result.missing_signals or "").strip()
        if not cls._scope_has_detailed_list(scope_items):
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

    @classmethod
    def result_to_requirements(
        cls,
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
                by_name[cls._norm_name(name)] = item.requirements

        buckets: list[list[ExtractedRequirement]] = [[] for _ in scope_items]
        for index, scope_item in enumerate(scope_items):
            name = str(scope_item.get("name") or "").strip()
            raw_reqs = by_name.get(cls._norm_name(name), [])
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

    @staticmethod
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

    @staticmethod
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
