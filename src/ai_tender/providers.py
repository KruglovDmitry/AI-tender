"""LLM-провайдеры и оценка найденных вхождений."""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any

from llama_index.core.llms import LLM

from .models import (
    DEFAULT_USER_INSTRUCTION,
    STATUS_PRIORITY,
    Finding,
    Settings,
    Status,
)

ASSESS_SCHEMA_HINT = """
Верни ТОЛЬКО JSON-объект вида:
{
  "items": [
    {
      "index": 0,
      "status": "found|partial|not_found|uncertain",
      "explanation": "1-2 коротких предложения: требование тендера ↔ что подтверждает эталон",
      "confidence": 0.0
    }
  ]
}

status (для требования тендера):
- found — цитаты эталона подтверждают применимость к этому требованию
- partial — подтверждена только часть требования
- not_found — по цитатам применимость не подтверждается
- uncertain — данных недостаточно

Строгие правила:
- Опирайся ТОЛЬКО на цитаты tender и asset_hits. Не используй внешние знания.
- Не называй модели/ТС/параметры, которых нет в этих цитатах.
- Если хиты эталона слабо связаны с требованием (реквизиты, общие фразы) —
  ставь not_found или uncertain, не found.
- Не пиши длинных резюме.
""".strip()

PRODUCT_ASSESS_HINT = """
Режим PRODUCT: оцени, подтверждается ли в цитатах эталона наличие/применимость
указанного артикула, обозначения или названия прибора (или явного эквивалента
в тексте эталона).

- found — модель/артикул (или явный эквивалент) есть в цитатах эталона
- partial — близкое семейство/частичное совпадение обозначения
- not_found — в цитатах эталона этой позиции нет
- uncertain — данных недостаточно

Не подтверждай product только по общим ТТХ без упоминания модели/серии в эталоне.
""".strip()


def build_llm(settings: Settings) -> LLM:
    provider = settings.llm_provider.lower().strip()
    if provider == "openai":
        from llama_index.llms.openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Не указан OPENAI_API_KEY")
        return OpenAI(
            model=settings.llm_model,
            api_key=api_key,
            api_base=settings.openai_base_url,
            temperature=0,
        )

    from llama_index.llms.openai_like import OpenAILike

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Не указан DEEPSEEK_API_KEY")

    return OpenAILike(
        model=settings.llm_model,
        api_base=settings.deepseek_base_url,
        api_key=api_key,
        is_chat_model=True,
        is_function_calling_model=False,
        temperature=0,
    )


def assess_findings(
    llm: LLM,
    candidates: list[tuple[Any, list[Any]]],
    node_to_evidence,
    user_instruction: str | None = None,
    max_findings: int = 12,
    *,
    mode: str = "specs",
    match_mode: str = "",
    select_important: bool = True,
) -> tuple[str, list[Finding]]:
    if not candidates:
        return "Кандидаты для сопоставления не найдены.", []

    from .models import Evidence, ExtractedRequirement
    from .query_select import requirement_to_evidence

    instruction = (user_instruction or DEFAULT_USER_INSTRUCTION).strip()
    payload_items: list[dict[str, Any]] = []
    prepared: list[Finding] = []

    for index, (tender_part, hits) in enumerate(candidates):
        requirement_text = ""
        kind = "other"
        if isinstance(tender_part, ExtractedRequirement):
            tender = requirement_to_evidence(tender_part)
            requirement_text = tender_part.text
            kind = tender_part.kind
        elif isinstance(tender_part, Evidence):
            tender = tender_part
            requirement_text = tender.quote
        else:
            tender = node_to_evidence(tender_part)
            meta = getattr(tender_part, "metadata", None) or {}
            requirement_text = str(meta.get("requirement_text") or tender.quote)
            kind = str(meta.get("kind") or "other")

        asset_hits = [node_to_evidence(hit.node, hit.score) for hit in hits]
        query_text = (requirement_text or tender.quote)[:300]

        prepared.append(
            Finding(
                query_text=query_text,
                tender=tender,
                asset_hits=asset_hits,
                kind=kind,
                match_mode=match_mode,
            )
        )

        payload_items.append(
            {
                "index": index,
                "kind": kind,
                "requirement": requirement_text,
                "tender": tender.model_dump(),
                "asset_hits": [item.model_dump() for item in asset_hits],
            }
        )

    mode_hint = PRODUCT_ASSESS_HINT if mode == "product" else ASSESS_SCHEMA_HINT
    role = (
        "Для каждой позиции (артикул/название) оцени наличие в цитатах эталона."
        if mode == "product"
        else "Для каждого технического требования оцени подтверждение в цитатах эталона."
    )

    batch_size = 8
    for start in range(0, len(payload_items), batch_size):
        batch = payload_items[start : start + batch_size]
        prompt = (
            f"Ты аналитик закупок. {role} "
            "Поле requirement — сформулированное требование; tender.quote — якорь; "
            "asset_hits — фрагменты эталона (RAG).\n"
            f"ЗАДАЧА ОТ ПОЛЬЗОВАТЕЛЯ:\n{instruction}\n\n"
            f"{mode_hint}\n\n"
            f"{ASSESS_SCHEMA_HINT}\n\n"
            f"ДАННЫЕ:\n{json.dumps(batch, ensure_ascii=False)}"
        )

        response = llm.complete(prompt)
        data = _parse_json(str(response))

        for item in data.get("items", []):
            local = int(item.get("index", -1))
            if local < 0 or start + local >= len(prepared):
                continue

            target = prepared[start + local]
            status_raw = str(item.get("status", "uncertain"))

            try:
                target.status = Status(status_raw)
            except ValueError:
                target.status = Status.uncertain

            target.explanation = str(item.get("explanation", "")).strip()
            try:
                target.confidence = float(item.get("confidence", 0))
            except (TypeError, ValueError):
                target.confidence = 0.0

            target.confidence = min(max(target.confidence, 0.0), 1.0)

    findings = (
        select_important_findings(prepared, max_findings=max_findings)
        if select_important
        else prepared
    )
    summary = build_compact_summary(prepared, findings, match_mode=match_mode)
    return summary, findings


def select_important_findings(
    findings: list[Finding],
    max_findings: int = 12,
) -> list[Finding]:
    """Оставляем в основном подтверждённые/частичные совпадения; product в приоритете."""
    preferred = [
        item
        for item in findings
        if item.status in (Status.found, Status.partial)
        or (item.status == Status.uncertain and item.confidence >= 0.6)
    ]
    pool = preferred or [
        item for item in findings if item.status != Status.not_found
    ] or list(findings)

    kind_rank = {"product": 0, "specs": 1, "other": 2}
    pool.sort(
        key=lambda item: (
            kind_rank.get(item.kind, 9),
            STATUS_PRIORITY.get(item.status, 9),
            -item.confidence,
        )
    )
    return pool[: max(1, max_findings)]


def build_compact_summary(
    all_findings: list[Finding],
    shown: list[Finding],
    *,
    match_mode: str = "",
    strategy: str = "",
) -> str:
    counts = Counter(item.status for item in all_findings)
    products = [item for item in all_findings if item.kind == "product"]
    lines: list[str] = []

    mode_labels = {
        "product_first": "гибрид: артикул подтверждён, ТТХ выборочно",
        "specs_fallback": "гибрид: артикул не подтверждён — разбор ТТХ",
        "specs_only": "только техсоответствие (ТТХ)",
        "product_only": "только поиск по продукту/артикулу",
    }
    strategy_labels = {
        "hybrid": "Гибридный",
        "product": "По продукту",
        "specs": "По техсоответствию",
    }
    if strategy:
        lines.append(f"Стратегия: {strategy_labels.get(strategy, strategy)}.")
    if match_mode:
        lines.append(f"Ход: {mode_labels.get(match_mode, match_mode)}.")

    if products:
        ok = sum(
            1
            for item in products
            if item.status in (Status.found, Status.partial)
        )
        lines.append(f"Позиции (product): {len(products)}, подтверждено: {ok}.")

    lines.append(f"Проверено пунктов: {len(all_findings)}.")
    lines.append(
        f"Сводка: применимо {counts[Status.found]}, "
        f"частично {counts[Status.partial]}, "
        f"не подтверждено {counts[Status.not_found]}, "
        f"неясно {counts[Status.uncertain]}."
    )
    lines.append(f"В таблице — {len(shown)} наиболее значимых.")

    highlights = [
        item.explanation
        for item in shown
        if item.explanation and item.status in (Status.found, Status.partial)
    ][:2]
    if highlights:
        lines.append("Ключевые:")
        lines.extend(f"— {text}" for text in highlights)

    lines.append(f"Итог: {_brief_conclusion(all_findings, match_mode=match_mode)}")
    return "\n".join(lines)


def _brief_conclusion(findings: list[Finding], *, match_mode: str = "") -> str:
    if not findings:
        return "Недостаточно данных для заключения."

    products = [item for item in findings if item.kind == "product"]
    product_ok = [
        item
        for item in products
        if item.status in (Status.found, Status.partial) and item.confidence >= 0.55
    ]
    found = sum(1 for item in findings if item.status == Status.found)
    partial = sum(1 for item in findings if item.status == Status.partial)
    bad = sum(
        1 for item in findings if item.status in (Status.not_found, Status.uncertain)
    )

    if match_mode == "product_only":
        if product_ok:
            return "Позиция по артикулу/названию в эталоне подтверждена."
        if products:
            return "Явная позиция в эталоне не подтверждена."
        return "Артикул/название для проверки не выделены."

    if product_ok and match_mode == "product_first":
        if partial and not bad:
            return "Продукт найден в эталоне; отдельные ТТХ подтверждены частично."
        if bad:
            return "Продукт найден в эталоне; по части ТТХ есть расхождения или пробелы."
        return "Продукт найден в эталоне; выборочные ТТХ согласуются."

    if found and not bad and not partial:
        return "По проверенным пунктам эталон подтверждает требования."
    if found or partial:
        if bad:
            return "Есть подтверждения, но часть пунктов без надёжного покрытия эталоном."
        return "Требования в целом подтверждаются эталоном с оговорками."
    return "Надёжного подтверждения по эталону не получено."


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

    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("Ответ LLM должен быть JSON-объектом")
    return result
