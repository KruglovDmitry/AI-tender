"""Match: retrieval по каталогу + Qwen JSON-вердикт + параллельная нода."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from llama_index.core.llms import LLM

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
from ..services.catalog_retrieval import (
    CatalogProductHit,
    ProductCatalog,
    catalog_hit_to_evidence,
    format_product_quote,
    search_catalog,
)
from ..services.logging_service import trace_note, trace_retrieval
from .common import progress

MAX_HITS_FOR_LLM = 12
MAX_REQS_IN_QUERY = 8
_REQ_KIND_RANK = {"product": 0, "specs": 1, "other": 2}

MATCH_SCHEMA_HINT = """
Верни ТОЛЬКО JSON:
{
  "matched": true|false,
  "status": "matched|partial|none",
  "required_product": "модель/тип из тендера или \"\"",
  "product_name": "модель из asset_hits или \"\"",
  "explanation": "1-2 предложения",
  "confidence": 0.0..1.0
}

Правила:
- Только position, requirements и asset_hits (каталог эталонов).
- required_product — из тендера; product_name — только из asset_hits (дословно в цитате).
- matched/partial только при непустом product_name из эталона.
- status=matched — позиция закрыта; partial — неполное покрытие; none — нет изделия в цитатах.
- Аналог того же класса изделия разрешён; не подменяй класс (ИБП ≠ зарядка).
- Если в характеристиках asset_hits прямо указано «аналог …» / «полный аналог …» для модели из позиции — это matched или partial.
- Игнорируй requirements, явно относящиеся к другой позиции перечня.
""".strip()


def _rank_requirements(requirements: list[ExtractedRequirement]) -> list[ExtractedRequirement]:
    return sorted(
        requirements,
        key=lambda req: (
            _REQ_KIND_RANK.get(req.kind, 9),
            -req.priority,
            -req.confidence,
            req.text.casefold(),
        ),
    )


def _scope_qty(qty: Any) -> float | int | None:
    return qty if isinstance(qty, (int, float)) or qty is None else None


def _normalize_ground(text: str) -> str:
    return "".join(ch.casefold() if ch.isalnum() else " " for ch in text)


def product_name_in_hits(product_name: str, hits: list[Evidence]) -> bool:
    name = " ".join((product_name or "").split())
    if len(name) < 3:
        return False
    quotes = " ".join(hit.quote or "" for hit in hits)
    spaced = " ".join(_normalize_ground(quotes).split())
    needle = " ".join(_normalize_ground(name).split())
    if needle and needle in spaced:
        return True
    compact_name = "".join(ch for ch in name.casefold() if ch.isalnum())
    compact_quotes = "".join(ch for ch in quotes.casefold() if ch.isalnum())
    return len(compact_name) >= 5 and compact_name in compact_quotes


def position_to_query_text(
    scope_name: str,
    requirements: list[ExtractedRequirement],
    *,
    max_reqs: int = MAX_REQS_IN_QUERY,
) -> str:
    lines = [f"Позиция закупки: {scope_name.strip()}"]
    for req in _rank_requirements(requirements)[:max_reqs]:
        lines.append(f"- {req.text.strip()}")
        if req.quote and req.quote.strip() and req.quote.strip() != req.text.strip():
            lines.append(f"  цитата: {req.quote.strip()[:240]}")
    return "\n".join(lines).strip()


def retrieve_hits_for_position(
    scope_name: str,
    requirements: list[ExtractedRequirement],
    catalog: ProductCatalog,
    *,
    top_k: int,
    embedding_model: str,
    device: str | None,
) -> list[CatalogProductHit]:
    query = position_to_query_text(scope_name, requirements)
    if not query:
        trace_note(
            "retrieve_empty_query",
            "Пустой query — retrieval пропущен",
            meta={"scope_name": scope_name},
        )
        return []

    fetch_k = max(top_k * 4, 12)
    hits = search_catalog(
        catalog,
        query,
        top_k=fetch_k,
        embedding_model=embedding_model,
        device=device,
    )
    trace_retrieval(
        "retrieve_position",
        query=query,
        hits=[
            {
                "score": h.score,
                "file": h.source_file,
                "model": h.product.model,
                "text_preview": format_product_quote(
                    h.product,
                    catalog_name=h.catalog_name,
                    source_file=h.source_file,
                )[:800],
            }
            for h in hits
        ],
        meta={
            "scope_name": scope_name,
            "top_k": top_k,
            "fetch_k": fetch_k,
            "catalog_products": catalog.size,
            "retrieval": "qwen_catalog",
        },
    )
    return hits[: max(top_k * 3, MAX_HITS_FOR_LLM)]


def _base_match(
    scope_item: dict[str, Any],
    requirements: list[ExtractedRequirement],
    asset_hits: list[Evidence],
) -> ScopePositionMatch:
    return ScopePositionMatch(
        scope_name=str(scope_item.get("name") or "").strip(),
        qty=_scope_qty(scope_item.get("qty")),
        unit=str(scope_item.get("unit") or "").strip(),
        requirements=list(requirements),
        asset_hits=list(asset_hits[:MAX_HITS_FOR_LLM]),
    )


def _parse_llm_match(
    data: dict[str, Any],
    asset_hits: list[Evidence],
) -> tuple[PositionMatchStatus, str, str, str, float]:
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
    if not product_name:
        status = PositionMatchStatus.none
    elif status == PositionMatchStatus.none:
        product_name = ""

    try:
        confidence = float(data.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    explanation = str(data.get("explanation") or "").strip()
    if not explanation and status == PositionMatchStatus.none:
        explanation = "Подходящего варианта в эталоне нет."
    return status, required_product, product_name, explanation, min(max(confidence, 0.0), 1.0)


def match_scope_position(
    llm: LLM,
    *,
    scope_item: dict[str, Any],
    requirements: list[ExtractedRequirement],
    asset_hits: list[Evidence],
    user_instruction: str | None = None,
) -> ScopePositionMatch:
    """Подбор варианта из эталона для одной позиции (Qwen chat JSON)."""
    scope_name = str(scope_item.get("name") or "").strip()
    reqs = _rank_requirements(requirements)
    hits = asset_hits[:MAX_HITS_FOR_LLM]
    result = _base_match(scope_item, reqs, hits)

    if not hits:
        result.status = PositionMatchStatus.none
        result.explanation = "Подходящего варианта в каталоге эталонов нет (нет кандидатов)."
        trace_note("match_skip_no_hits", result.explanation, meta={"scope_name": scope_name})
        return result

    payload = {
        "position": {
            "name": scope_name,
            "qty": scope_item.get("qty"),
            "unit": str(scope_item.get("unit") or "").strip(),
        },
        "requirements": [
            {"text": r.text, "quote": r.quote, "kind": r.kind, "priority": r.priority}
            for r in reqs
        ],
        "asset_hits": [h.model_dump() for h in hits],
    }
    instruction = (user_instruction or DEFAULT_USER_INSTRUCTION).strip()
    prompt = (
        "Ты аналитик закупок. Выбери вариант из каталога эталонов (asset_hits) "
        "для позиции перечня.\n"
        f"ЗАДАЧА:\n{instruction}\n\n{MATCH_SCHEMA_HINT}\n\n"
        f"ДАННЫЕ:\n{json.dumps(payload, ensure_ascii=False)}"
    )

    try:
        data, _ = complete_llm_json(
            llm,
            prompt,
            structure_hint=MATCH_SCHEMA_HINT,
            trace_name="match_position",
        )
    except Exception as exc:
        trace_note("match_llm_error", str(exc), meta={"scope_name": scope_name})
        result.status = PositionMatchStatus.none
        result.explanation = f"Не удалось подобрать эталон (ошибка модели): {exc}"
        return result

    if data is None:
        result.status = PositionMatchStatus.none
        result.explanation = "Не удалось разобрать ответ модели при подборе эталона."
        trace_note("match_json_failed", result.explanation, meta={"scope_name": scope_name})
        return result

    status, req_prod, prod_name, explanation, confidence = _parse_llm_match(data, hits)
    result.status = status
    result.required_product = req_prod
    result.product_name = prod_name
    result.explanation = explanation
    result.confidence = confidence
    return result


def match_one_position(
    *,
    llm: LLM,
    scope_item: dict[str, Any],
    requirements: list[ExtractedRequirement],
    catalog: ProductCatalog,
    top_k: int,
    user_instruction: str | None,
    embedding_model: str,
    embedding_device: str | None,
) -> ScopePositionMatch:
    name = str(scope_item.get("name") or "").strip() or "позиция"
    hits = retrieve_hits_for_position(
        name,
        requirements,
        catalog,
        top_k=top_k,
        embedding_model=embedding_model,
        device=embedding_device,
    )
    evidence = [catalog_hit_to_evidence(h) for h in hits]
    return match_scope_position(
        llm,
        scope_item=scope_item,
        requirements=requirements,
        asset_hits=evidence,
        user_instruction=user_instruction,
    )


def failed_position_match(
    scope_item: dict[str, Any],
    requirements: list[ExtractedRequirement],
    *,
    explanation: str,
) -> ScopePositionMatch:
    return ScopePositionMatch(
        scope_name=str(scope_item.get("name") or "").strip() or "позиция",
        qty=_scope_qty(scope_item.get("qty")),
        unit=str(scope_item.get("unit") or "").strip(),
        requirements=list(requirements),
        status=PositionMatchStatus.none,
        explanation=explanation,
    )


def node_match_positions(state: PipelineState) -> dict[str, Any]:
    settings: Settings = state["settings"]
    scope_items = list(state.get("scope_items") or [])
    reqs_by_item = list(state.get("requirements_by_item") or [])
    catalog = state.get("product_catalog")

    if not scope_items:
        return {"position_matches": []}

    if catalog is None or catalog.size == 0:
        return {
            "position_matches": [
                failed_position_match(
                    scope_items[i],
                    reqs_by_item[i] if i < len(reqs_by_item) else [],
                    explanation="Каталог эталонов не содержит продуктов для подбора.",
                )
                for i in range(len(scope_items))
            ],
            "warnings": [
                "Каталог эталонов пуст — подбор по позициям невозможен. Переиндексируйте эталоны."
            ],
        }

    top_k = max(settings.top_k, 5)
    workers = max(1, min(int(settings.match_parallelism or 1), len(scope_items)))
    llm = state["llm"]
    instruction = settings.user_instruction
    embed_model = settings.embedding_model
    embed_device = settings.embedding_device
    total = len(scope_items)

    progress(state, f"Подбор эталона: {total} позиций (workers={workers})", 0.72)

    def _run(index: int) -> tuple[int, ScopePositionMatch, str | None]:
        item = scope_items[index]
        reqs = reqs_by_item[index] if index < len(reqs_by_item) else []
        name = str(item.get("name") or "").strip() or f"позиция {index + 1}"
        try:
            match = match_one_position(
                llm=llm,
                scope_item=item,
                requirements=reqs,
                catalog=catalog,
                top_k=top_k,
                user_instruction=instruction,
                embedding_model=embed_model,
                embedding_device=embed_device,
            )
            return index, match, None
        except Exception as exc:
            trace_note(
                "match_position_failed",
                f"Ошибка подбора «{name}»: {exc}",
                meta={"scope_name": name, "index": index},
            )
            return (
                index,
                failed_position_match(item, reqs, explanation=f"Не удалось подобрать эталон: {exc}"),
                f"Match «{name}»: {exc}",
            )

    matches: list[ScopePositionMatch | None] = [None] * total
    warnings: list[str] = []
    done = 0

    def _store(index: int, match: ScopePositionMatch, warning: str | None) -> None:
        nonlocal done
        matches[index] = match
        if warning:
            warnings.append(warning)
        done += 1
        label = str(scope_items[index].get("name") or "").strip() or f"позиция {index + 1}"
        progress(state, f"Подбор эталона: {done}/{total} — {label[:60]}", 0.7 + 0.2 * done / total)

    if workers == 1:
        for i in range(total):
            _store(*_run(i))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run, i): i for i in range(total)}
            for future in as_completed(futures):
                _store(*future.result())

    return {
        "position_matches": [m for m in matches if m is not None],
        "warnings": warnings,
    }
