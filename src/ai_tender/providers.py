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

Не опирайся на знания вне цитат. Не пиши длинных резюме.
""".strip()


def assess_findings(
    llm: LLM,
    candidates: list[tuple[Any, list[Any]]],
    node_to_evidence,
    user_instruction: str | None = None,
    max_findings: int = 12,
) -> tuple[str, list[Finding]]:
    if not candidates:
        return "Кандидаты для сопоставления не найдены.", []

    instruction = (user_instruction or DEFAULT_USER_INSTRUCTION).strip()
    payload_items: list[dict[str, Any]] = []
    prepared: list[Finding] = []

    for index, (tender_node, hits) in enumerate(candidates):
        tender = node_to_evidence(tender_node)
        asset_hits = [node_to_evidence(hit.node, hit.score) for hit in hits]
        query_text = tender.quote[:300]

        prepared.append(
            Finding(
                query_text=query_text,
                tender=tender,
                asset_hits=asset_hits,
            )
        )

        payload_items.append(
            {
                "index": index,
                "tender": tender.model_dump(),
                "asset_hits": [item.model_dump() for item in asset_hits],
            }
        )

    batch_size = 8
    for start in range(0, len(payload_items), batch_size):
        batch = payload_items[start : start + batch_size]
        prompt = (
            "Ты аналитик закупок. Для каждого требования тендера оцени "
            "подтверждение в цитатах эталона.\n"
            f"ЗАДАЧА ОТ ПОЛЬЗОВАТЕЛЯ:\n{instruction}\n\n"
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

    findings = select_important_findings(prepared, max_findings=max_findings)
    summary = build_compact_summary(prepared, findings)
    return summary, findings


def select_important_findings(
    findings: list[Finding],
    max_findings: int = 12,
) -> list[Finding]:
    """Оставляем в основном подтверждённые/частичные совпадения, без шума."""
    preferred = [
        item
        for item in findings
        if item.status in (Status.found, Status.partial)
        or (item.status == Status.uncertain and item.confidence >= 0.6)
    ]
    pool = preferred or [
        item for item in findings if item.status != Status.not_found
    ] or list(findings)

    pool.sort(
        key=lambda item: (
            STATUS_PRIORITY.get(item.status, 9),
            -item.confidence,
        )
    )
    return pool[: max(1, max_findings)]


def build_compact_summary(
    all_findings: list[Finding],
    shown: list[Finding],
) -> str:
    counts = Counter(item.status for item in all_findings)
    parts = [
        f"Проверено требований тендера: {len(all_findings)}.",
        f"Применимо: {counts[Status.found]}, "
        f"частично: {counts[Status.partial]}, "
        f"не подтверждено: {counts[Status.not_found]}, "
        f"неясно: {counts[Status.uncertain]}.",
        f"В таблице — {len(shown)} наиболее значимых.",
    ]
    highlights = [
        item.explanation
        for item in shown
        if item.explanation and item.status in (Status.found, Status.partial)
    ][:2]
    if highlights:
        parts.append("Ключевые: " + " | ".join(highlights))
    return " ".join(parts)


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
