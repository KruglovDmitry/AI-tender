"""LLM-провайдеры и оценка найденных вхождений."""

from __future__ import annotations

import json
import os
from typing import Any

from llama_index.core.llms import LLM

from .models import Finding, Settings, Status


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
  "summary": "краткое резюме на русском",
  "items": [
    {
      "index": 0,
      "status": "found|partial|not_found|uncertain",
      "explanation": "почему такой статус, только по цитатам",
      "confidence": 0.0
    }
  ]
}

status:
- found — в тендере явно есть то же ТС/параметр из эталона (по цитатам)
- partial — частичное пересечение (часть параметра подтверждена)
- not_found — по переданным цитатам вхождения нет
- uncertain — данных недостаточно

Не опирайся на знания вне цитат.
""".strip()


def assess_findings(
    llm: LLM,
    candidates: list[tuple[Any, list[Any]]],
    node_to_evidence,
) -> tuple[str, list[Finding]]:
    if not candidates:
        return "Кандидаты для сопоставления не найдены.", []

    payload_items: list[dict[str, Any]] = []
    prepared: list[Finding] = []

    for index, (asset_node, hits) in enumerate(candidates):
        asset = node_to_evidence(asset_node)
        tender_hits = [node_to_evidence(hit.node, hit.score) for hit in hits]
        query_text = asset.quote[:300]

        prepared.append(
            Finding(
                query_text=query_text,
                asset=asset,
                tender_hits=tender_hits,
            )
        )

        payload_items.append(
            {
                "index": index,
                "asset": asset.model_dump(),
                "tender_hits": [item.model_dump() for item in tender_hits],
            }
        )

    # Батчами, чтобы не раздувать контекст.
    batch_size = 8
    summary_parts: list[str] = []

    for start in range(0, len(payload_items), batch_size):
        batch = payload_items[start : start + batch_size]
        prompt = (
            "Ты аналитик закупок. По каждой записи сравни эталонный фрагмент "
            "ТС/параметра с найденными фрагментами тендера.\n"
            f"{ASSESS_SCHEMA_HINT}\n\nДАННЫЕ:\n{json.dumps(batch, ensure_ascii=False)}"
        )

        response = llm.complete(prompt)
        data = _parse_json(str(response))

        if data.get("summary"):
            summary_parts.append(str(data["summary"]))

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

            target.explanation = str(item.get("explanation", ""))
            try:
                target.confidence = float(item.get("confidence", 0))
            except (TypeError, ValueError):
                target.confidence = 0.0

            target.confidence = min(max(target.confidence, 0.0), 1.0)

    summary = (
        " ".join(summary_parts).strip()
        or "Оценка выполнена по найденным кандидатам."
    )
    return summary, prepared


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

