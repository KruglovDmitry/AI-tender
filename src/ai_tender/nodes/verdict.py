from __future__ import annotations

import json
from typing import Any

from ..models import PipelineState, PositionMatchStatus
from ..providers import complete_llm_json


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


def node_build_verdict(state: PipelineState) -> dict[str, Any]:
    callback = state.get("progress")
    if callable(callback):
        callback("Итоговый вывод по тендеру", 0.92)

    matches = list(state.get("position_matches") or [])
    if not matches:
        return {
            "verdict": "Перечень позиций пуст — вывод о пригодности тендера сформировать нельзя."
        }

    covered = sum(
        1
        for item in matches
        if item.status in (PositionMatchStatus.matched, PositionMatchStatus.partial)
    )
    fallback = f"Закрыто {covered} из {len(matches)} позиций."
    scope_meta = state.get("scope_meta") or {}
    payload = {
        "scope_summary": str(scope_meta.get("scope_summary") or ""),
        "total_positions": len(matches),
        "covered_positions": covered,
        "positions": [
            {
                "name": item.scope_name,
                "qty": item.qty,
                "unit": item.unit,
                "status": item.status.value,
                "required_product": item.required_product,
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
    try:
        data, _n_calls = complete_llm_json(
            state["llm"],
            prompt,
            structure_hint=VERDICT_SCHEMA_HINT,
            trace_name="tender_verdict",
        )
    except Exception:
        return {"verdict": fallback}

    if not data:
        return {"verdict": fallback}

    verdict = str(data.get("verdict") or "").strip()
    label = str(data.get("label") or "").strip()
    if label and verdict:
        return {"verdict": f"{label.capitalize()}. {verdict}"}
    if verdict:
        return {"verdict": verdict}
    return {"verdict": fallback}
