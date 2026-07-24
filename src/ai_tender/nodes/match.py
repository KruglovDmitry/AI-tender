"""Нода match: retrieval по эталону + LLM-подбор варианта на позицию."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from llama_index.core.llms import LLM

from ..services.index_service import node_to_evidence, retrieve_for_queries
from ..services.logging_service import trace_llm, trace_note, trace_retrieval
from ..models import (
    DEFAULT_USER_INSTRUCTION,
    Evidence,
    ExtractedRequirement,
    PipelineState,
    PositionMatchStatus,
    ScopePositionMatch,
    Settings,
)
from ..providers import parse_llm_json
from .common import dedupe_evidence_by_file, dedupe_hits_by_file, progress


POSITION_MATCH_SCHEMA_HINT = """
Верни ТОЛЬКО JSON-объект:
{
  "matched": true|false,
  "status": "matched|partial|none",
  "product_name": "модель/серия/обозначение из цитат эталона или пустая строка",
  "explanation": "1-2 предложения: почему подходит или почему нет",
  "confidence": 0.0..1.0
}

Правила:
- Опирайся ТОЛЬКО на position, requirements и asset_hits.
- product_name указывай ТОЛЬКО если он явно есть в цитатах эталона; пиши как в цитате.
- matched=true если есть конкретный подходящий вариант основного изделия позиции.
- status=matched — основное изделие закрыто и ключевые требования подтверждены цитатами.
- status=partial — есть подходящий вариант основного изделия, но покрытие неполное
  (часть требований или комплектующих не подтверждена цитатами эталона).
- Если позиция — комплект из нескольких типов изделий, а в эталоне подтверждено
  только основное изделие (без части комплектующих) — status=partial, matched=true.
  Не ставь none только из-за отсутствия комплектующих в цитатах.
- status=none и matched=false — только если нет подходящего основного изделия.
""".strip()


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
    asset_hits = dedupe_evidence_by_file(asset_hits)
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
    response = llm.complete(prompt)
    raw = str(response)
    trace_llm(
        "match_position",
        prompt=prompt,
        response=raw,
        meta={
            "scope_name": scope_name,
            "requirements_count": len(requirements),
            "asset_hits_count": len(asset_hits),
        },
    )
    data = parse_llm_json(raw)

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

    product_name = " ".join(str(data.get("product_name") or "").split())
    if status == PositionMatchStatus.none:
        product_name = ""

    try:
        confidence = float(data.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    base.status = status
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
    fetch_k = max(top_k * 2, top_k)
    hit_lists = retrieve_for_queries(assets_index, [query], top_k=fetch_k)
    raw_hits = list(hit_lists[0]) if hit_lists else []
    hits = dedupe_hits_by_file(raw_hits, limit=top_k)
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

    def _one(index: int) -> tuple[int, ScopePositionMatch]:
        scope_item = scope_items[index]
        requirements = reqs_by_item[index] if index < len(reqs_by_item) else []
        match = _match_one_position(
            llm=llm,
            scope_item=scope_item,
            requirements=requirements,
            assets_index=assets_index,
            top_k=top_k,
            user_instruction=instruction,
        )
        return index, match

    matches: list[ScopePositionMatch | None] = [None] * len(scope_items)
    done = 0
    done_lock = threading.Lock()

    def _on_done(index: int, match: ScopePositionMatch) -> None:
        nonlocal done
        matches[index] = match
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
            idx, match = _one(index)
            _on_done(idx, match)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(scope_items))) as pool:
            futures = {pool.submit(_one, index): index for index in range(len(scope_items))}
            for future in as_completed(futures):
                idx, match = future.result()
                _on_done(idx, match)

    return {"position_matches": [item for item in matches if item is not None]}