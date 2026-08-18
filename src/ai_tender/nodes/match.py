"""Нода match: retrieval по эталону + LLM-подбор варианта на позицию."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from llama_index.core.llms import LLM

from ..services.index_service import node_to_evidence, retrieve_for_queries
from ..services.logging_service import trace_note, trace_retrieval
from ..models import (
    DEFAULT_USER_INSTRUCTION,
    Evidence,
    ExtractedRequirement,
    PipelineState,
    PositionMatchStatus,
    ScopePositionMatch,
    Settings,
)
from ..providers import complete_llm_json
from .common import cap_evidence_per_file, cap_hits_per_file, progress


POSITION_MATCH_SCHEMA_HINT = """
Верни ТОЛЬКО JSON-объект:
{
  "matched": true|false,
  "status": "matched|partial|none",
  "required_product": "конкретная модель/тип из тендера (position/requirements) или пустая строка",
  "product_name": "подобранная модель/серия из цитат эталона или пустая строка",
  "explanation": "1-2 предложения: почему подходит или почему нет",
  "confidence": 0.0..1.0
}

Правила:
- Опирайся ТОЛЬКО на position, requirements и asset_hits.
- required_product и product_name — РАЗНЫЕ поля, не смешивай источники.
- required_product: заполняй ТОЛЬКО если в названии позиции или в требованиях явно
  указано конкретное обозначение/тип/модель того, что нужно купить
  (в т.ч. формулировки «тип X или аналог»). Пиши кратко обозначение, без «или аналог».
  Если в тендере только обобщённое описание без конкретной модели — оставь "".
- product_name: заполняй ТОЛЬКО из цитат эталона (asset_hits); пиши как в цитате.
  Предпочтительно полный артикул/модель из таблицы каталога, а не только имя серии.
  Пример: «SR33020-6x9», а не «SR33».
  Серию ставь, только если в цитатах нет конкретной строки модели.
  Не общие слова категории («ИБП», «зарядное устройство», «светильник») без модели.
- Выбор строки таблицы: если в цитатах есть несколько моделей одной серии,
  выбери ту, чьи характеристики ближе к требованиям именно этой позиции.
  Не бери заведомо несовместимый ряд, если есть ближе.
  Если часть requirements явно про другое изделие/другую строку перечня —
  игнорируй их при подборе, не отвергай из‑за них подходящий аналог.
- Аналог разрешён только того же класса изделия (источник питания к источнику питания,
  зарядное к зарядному, розетка к розетке). Чужой тип в тендере («тип X или аналог»)
  не обязан встречаться в эталоне — подбирай нашу модель того же класса из цитат.
- Запрещено подменять класс: внутренний узел другого изделия (например зарядка
  в составе ИБП) — это не позиция «зарядное устройство».
- product_name обязан буквально встречаться в одной из цитат asset_hits
  (артикул/модель как в тексте). Если в цитатах этого нет — product_name=""
  и status=none. Не копируй required_product из тендера.
- matched=true и status=matched|partial допустимы ТОЛЬКО если product_name непустой.
  Иначе matched=false и status=none.
- required_product сам по себе НИКОГДА не даёт matched/partial.
- Одно лишь упоминание категории без конкретной модели/серии в цитатах → status=none.
- status=matched — конкретное изделие из эталона закрывает позицию (как аналог или
  как то же обозначение) и ключевые требования подтверждены цитатами.
- status=partial — конкретное изделие из эталона есть, но покрытие неполное
  (часть требований/комплектующих не подтверждена).
- Если позиция — комплект, а в эталоне подтверждено только основное изделие —
  status=partial, matched=true при непустом product_name.
- status=none — в цитатах нет конкретного изделия нашего каталога по этой позиции.

Явное обозначение в предмете/позиции закупки:
- Совпадение кода тендера с линейкой в эталоне усиливает уверенность (можно matched).
- Если кода тендера в эталоне нет, но есть конкретный аналог (серия/модель в цитате)
  того же типа изделия — это тоже подбор, обычно partial, не none.
- Без непустого product_name из эталона код в тендере НЕ даёт matched/partial.
- Опции в коде изделия не переводи в partial только из‑за отсутствия отдельной
  цитаты — если само изделие уже подобрано из эталона.
""".strip()


def _normalize_ground_text(text: str) -> str:
    return "".join(ch.casefold() if ch.isalnum() else " " for ch in text)


def product_name_in_hits(product_name: str, hits: list[Evidence]) -> bool:
    """True, если обозначение изделия есть в цитатах эталона (не только в тендере)."""
    name = " ".join((product_name or "").split())
    if len(name) < 3:
        return False
    quotes = " ".join(hit.quote or "" for hit in hits)
    spaced = " ".join(_normalize_ground_text(quotes).split())
    needle = " ".join(_normalize_ground_text(name).split())
    if needle and needle in spaced:
        return True
    compact_name = "".join(ch for ch in name.casefold() if ch.isalnum())
    compact_quotes = "".join(ch for ch in quotes.casefold() if ch.isalnum())
    return len(compact_name) >= 5 and compact_name in compact_quotes


def _stable_requirements(requirements: list[ExtractedRequirement],) -> list[ExtractedRequirement]:
    kind_rank = {"product": 0, "specs": 1, "other": 2}
    return sorted(
        requirements,
        key=lambda req: (
            kind_rank.get(req.kind, 9),
            -req.priority,
            -req.confidence,
            req.text.casefold(),
        ),
    )


def match_scope_position(llm: LLM, *, scope_item: dict[str, Any], requirements: list[ExtractedRequirement], asset_hits: list[Evidence], user_instruction: str | None = None,) -> ScopePositionMatch:
    """Подбор варианта из эталона для одной позиции перечня."""
    scope_name = str(scope_item.get("name") or "").strip()
    qty = scope_item.get("qty")
    unit = str(scope_item.get("unit") or "").strip()
    requirements = _stable_requirements(requirements)
    asset_hits = cap_evidence_per_file(asset_hits, per_file=5, limit=12)
    base = ScopePositionMatch(
        scope_name=scope_name,
        qty=qty if isinstance(qty, (int, float)) or qty is None else None,
        unit=unit,
        requirements=list(requirements),
        asset_hits=list(asset_hits),
    )
    if not asset_hits:
        base.status = PositionMatchStatus.none
        base.explanation = "Подходящего варианта в эталоне нет (нет релевантных фрагментов)."
        trace_note(
            "match_skip_no_hits",
            base.explanation,
            meta={
                "scope_name": scope_name,
                "requirements_count": len(requirements),
            },
        )
        return base

    instruction = (user_instruction or DEFAULT_USER_INSTRUCTION).strip()
    payload = {
        "position": {
            "name": scope_name,
            "qty": qty,
            "unit": unit,
        },
        "requirements": [
            {
                "text": req.text,
                "quote": req.quote,
                "kind": req.kind,
                "priority": req.priority,
            }
            for req in requirements
        ],
        "asset_hits": [hit.model_dump() for hit in asset_hits],
    }
    prompt = (
        "Ты аналитик закупок. Подбери конкретный вариант продукции из цитат эталона "
        "для позиции перечня с учётом её требований.\n"
        f"ЗАДАЧА ОТ ПОЛЬЗОВАТЕЛЯ:\n{instruction}\n\n"
        f"{POSITION_MATCH_SCHEMA_HINT}\n\n"
        f"ДАННЫЕ:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    try:
        data, _n_calls = complete_llm_json(
            llm,
            prompt,
            structure_hint=POSITION_MATCH_SCHEMA_HINT,
            trace_name="match_position",
        )
    except Exception as exc:
        trace_note(
            "match_llm_error",
            f"Ошибка LLM при подборе эталона: {exc}",
            meta={"scope_name": scope_name},
        )
        base.status = PositionMatchStatus.none
        base.explanation = f"Не удалось подобрать эталон (ошибка модели): {exc}"
        return base

    if data is None:
        base.status = PositionMatchStatus.none
        base.explanation = "Не удалось разобрать ответ модели при подборе эталона."
        trace_note(
            "match_json_failed",
            base.explanation,
            meta={"scope_name": scope_name},
        )
        return base

    status_raw = str(data.get("status") or "").strip().lower()
    matched = bool(data.get("matched", False))
    try:
        status = PositionMatchStatus(status_raw)
    except ValueError:
        status = PositionMatchStatus.matched if matched else PositionMatchStatus.none
    if not matched and status == PositionMatchStatus.matched:
        status = PositionMatchStatus.none
    if matched and status == PositionMatchStatus.none:
        status = PositionMatchStatus.partial

    required_product = " ".join(str(data.get("required_product") or "").split())
    product_name = " ".join(str(data.get("product_name") or "").split())
    if product_name and not product_name_in_hits(product_name, asset_hits):
        product_name = ""
    # Без конкретного изделия из эталона matched/partial недопустимы.
    if not product_name:
        status = PositionMatchStatus.none
    elif status == PositionMatchStatus.none:
        product_name = ""

    try:
        confidence = float(data.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    base.status = status
    base.required_product = required_product
    base.product_name = product_name
    base.explanation = str(data.get("explanation") or "").strip()
    if not base.explanation and status == PositionMatchStatus.none:
        base.explanation = "Подходящего варианта в эталоне нет."
    base.confidence = min(max(confidence, 0.0), 1.0)
    return base


def position_to_query_text(scope_name: str, requirements: list[ExtractedRequirement], *, max_reqs: int = 8,) -> str:
    """Текстовый запрос в индекс эталонов по позиции + её требованиям."""
    kind_rank = {"product": 0, "specs": 1, "other": 2}
    ranked = sorted(
        requirements,
        key=lambda req: (
            kind_rank.get(req.kind, 9),
            -req.priority,
            -req.confidence,
            req.text.casefold(),
        ),
    )
    lines = [f"Позиция закупки: {scope_name.strip()}"]
    for req in ranked[:max_reqs]:
        lines.append(f"- {req.text.strip()}")
        if req.quote and req.quote.strip() and req.quote.strip() != req.text.strip():
            lines.append(f"  цитата: {req.quote.strip()[:240]}")
    return "\n".join(lines).strip()


def retrieve_hits_for_position(scope_name: str, requirements: list[ExtractedRequirement], assets_index, top_k: int,) -> list:
    query = position_to_query_text(scope_name, requirements)
    if not query:
        trace_note(
            "retrieve_empty_query",
            "Пустой query для позиции — retrieval пропущен",
            meta={"scope_name": scope_name, "requirements_count": len(requirements)},
        )
        return []
    fetch_k = max(top_k * 4, 16)
    hit_lists = retrieve_for_queries(assets_index, [query], top_k=fetch_k)
    raw_hits = list(hit_lists[0]) if hit_lists else []
    hits = cap_hits_per_file(raw_hits, per_file=5, limit=max(top_k * 3, 12))
    hit_payload = []
    for hit in hits:
        node = getattr(hit, "node", None)
        meta = (getattr(node, "metadata", None) or {}) if node is not None else {}
        text = ""
        if node is not None and hasattr(node, "get_content"):
            text = node.get_content(metadata_mode="none") or ""
        hit_payload.append(
            {
                "score": getattr(hit, "score", None),
                "file": meta.get("file_path") or meta.get("file_name"),
                "location": meta.get("location"),
                "text_preview": text[:800],
            }
        )
    trace_retrieval(
        "retrieve_position",
        query=query,
        hits=hit_payload,
        meta={
            "scope_name": scope_name,
            "requirements_count": len(requirements),
            "top_k": top_k,
            "raw_hits": len(raw_hits),
            "deduped_hits": len(hits),
        },
    )
    return hits


def _failed_position_match(
    scope_item: dict[str, Any],
    requirements: list[ExtractedRequirement],
    *,
    explanation: str,
) -> ScopePositionMatch:
    qty = scope_item.get("qty")
    return ScopePositionMatch(
        scope_name=str(scope_item.get("name") or "").strip() or "позиция",
        qty=qty if isinstance(qty, (int, float)) or qty is None else None,
        unit=str(scope_item.get("unit") or "").strip(),
        requirements=list(requirements),
        status=PositionMatchStatus.none,
        explanation=explanation,
    )


def _match_one_position(*, llm: LLM, scope_item: dict[str, Any], requirements: list[ExtractedRequirement], assets_index, top_k: int, user_instruction: str | None,) -> ScopePositionMatch:
    name = str(scope_item.get("name") or "").strip() or "позиция"
    hits = retrieve_hits_for_position(name, requirements, assets_index, top_k=top_k)
    asset_evidence = [node_to_evidence(hit.node, hit.score) for hit in hits]
    return match_scope_position(
        llm,
        scope_item=scope_item,
        requirements=requirements,
        asset_hits=asset_evidence,
        user_instruction=user_instruction,
    )


def node_match_positions(state: PipelineState) -> dict[str, Any]:
    settings: Settings = state["settings"]
    scope_items = list(state.get("scope_items") or [])
    reqs_by_item = list(state.get("requirements_by_item") or [])
    assets_index = state.get("assets_index")
    top_k = max(settings.top_k, 5)
    total = max(len(scope_items), 1)
    workers = max(1, int(settings.match_parallelism or 1))
    llm = state["llm"]
    instruction = settings.user_instruction

    if not scope_items:
        return {"position_matches": []}

    progress(
        state,
        f"Подбор эталона: {len(scope_items)} позиций (parallelism={workers})",
        0.72,
    )

    def _one(index: int) -> tuple[int, ScopePositionMatch, str | None]:
        scope_item = scope_items[index]
        requirements = reqs_by_item[index] if index < len(reqs_by_item) else []
        name = str(scope_item.get("name") or "").strip() or f"позиция {index + 1}"
        try:
            match = _match_one_position(
                llm=llm,
                scope_item=scope_item,
                requirements=requirements,
                assets_index=assets_index,
                top_k=top_k,
                user_instruction=instruction,
            )
            return index, match, None
        except Exception as exc:
            trace_note(
                "match_position_failed",
                f"Ошибка подбора эталона для «{name}»: {exc}",
                meta={"scope_name": name, "index": index},
            )
            fallback = _failed_position_match(
                scope_item,
                requirements,
                explanation=f"Не удалось подобрать эталон: {exc}",
            )
            return index, fallback, f"Match «{name}»: {exc}"

    matches: list[ScopePositionMatch | None] = [None] * len(scope_items)
    warnings: list[str] = []
    done = 0
    done_lock = threading.Lock()

    def _on_done(index: int, match: ScopePositionMatch, warning: str | None) -> None:
        nonlocal done
        matches[index] = match
        if warning:
            warnings.append(warning)
        with done_lock:
            done += 1
            current = done
        name = str(scope_items[index].get("name") or "").strip() or f"позиция {index + 1}"
        progress(
            state,
            f"Подбор эталона: {current}/{len(scope_items)} — {name[:60]}",
            0.7 + 0.2 * (current / total),
        )

    if workers <= 1 or len(scope_items) == 1:
        for index in range(len(scope_items)):
            idx, match, warning = _one(index)
            _on_done(idx, match, warning)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(scope_items))) as pool:
            futures = {pool.submit(_one, index): index for index in range(len(scope_items))}
            for future in as_completed(futures):
                try:
                    idx, match, warning = future.result()
                except Exception as exc:
                    idx = futures[future]
                    scope_item = scope_items[idx]
                    requirements = reqs_by_item[idx] if idx < len(reqs_by_item) else []
                    name = str(scope_item.get("name") or "").strip() or f"позиция {idx + 1}"
                    match = _failed_position_match(
                        scope_item,
                        requirements,
                        explanation=f"Не удалось подобрать эталон: {exc}",
                    )
                    warning = f"Match «{name}»: {exc}"
                _on_done(idx, match, warning)

    return {
        "position_matches": [item for item in matches if item is not None],
        "warnings": warnings,
    }