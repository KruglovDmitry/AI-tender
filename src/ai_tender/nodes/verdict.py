"""Нода: итоговый вердикт по тендеру."""

from __future__ import annotations

import json
from typing import Any

from llama_index.core.llms import LLM

from ..models import PositionMatchStatus, ScopePositionMatch
from ..providers import parse_llm_json
from ..state import PipelineState
from .common import progress


VERDICT_SCHEMA_HINT = """
Верни ТОЛЬКО JSON-объект:
{
  "suitable": true|false|null,
  "label": "подходит|с оговорками|не подходит",
  "verdict": "2-4 предложения итогового вывода по тендеру"
}

Правила:
- suitable=true если закрыта большая часть позиций (matched/partial).
- suitable=false если подходящих вариантов мало или нет.
- suitable=null только если данных совсем недостаточно.
- partial считай закрытием позиции (с оговорками).
- Пиши кратко, по делу, на русском.
""".strip()


def build_tender_verdict(
    llm: LLM,
    matches: list[ScopePositionMatch],
    *,
    scope_summary: str = "",
) -> str:
    """Итоговый вывод по тендеру отдельным запросом к LLM."""
    if not matches:
        return "Перечень позиций пуст — вывод о пригодности тендера сформировать нельзя."

    covered = sum(
        1
        for item in matches
        if item.status in (PositionMatchStatus.matched, PositionMatchStatus.partial)
    )
    payload = {
        "scope_summary": scope_summary,
        "total_positions": len(matches),
        "covered_positions": covered,
        "positions": [
            {
                "name": item.scope_name,
                "qty": item.qty,
                "unit": item.unit,
                "status": item.status.value,
                "product_name": item.product_name,
                "requirements_count": len(item.requirements),
                "explanation": item.explanation,
            }
            for item in matches
        ],
    }
    prompt = (
        "Ты аналитик закупок. По результатам подбора эталонных вариантов к позициям "
        "перечня сделай итоговый вывод: подходит ли тендер (можно ли закрыть "
        "большую часть позиций нашей продукцией).\n\n"
        f"{VERDICT_SCHEMA_HINT}\n\n"
        f"ДАННЫЕ:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    response = llm.complete(prompt)
    data = parse_llm_json(str(response))
    verdict = str(data.get("verdict") or "").strip()
    label = str(data.get("label") or "").strip()
    if label and verdict:
        return f"{label.capitalize()}. {verdict}"
    if verdict:
        return verdict
    ratio = covered / max(len(matches), 1)
    if ratio >= 0.7:
        return (
            f"Тендер в целом подходит: закрыто {covered} из {len(matches)} позиций."
        )
    if ratio >= 0.4:
        return (
            f"Тендер подходит с оговорками: закрыто {covered} из {len(matches)} позиций."
        )
    return (
        f"Тендер скорее не подходит: закрыто лишь {covered} из {len(matches)} позиций."
    )


def node_build_verdict(state: PipelineState) -> dict[str, Any]:
    progress(state, "Итоговый вывод по тендеру", 0.92)
    scope_meta = state.get("scope_meta") or {}
    verdict = build_tender_verdict(
        state["llm"],
        list(state.get("position_matches") or []),
        scope_summary=str(scope_meta.get("scope_summary") or ""),
    )
    return {"verdict": verdict}
