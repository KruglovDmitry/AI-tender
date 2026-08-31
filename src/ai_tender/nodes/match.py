"""Нода match: retrieval по эталону + LLM-подбор варианта на позицию."""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from llama_index.core.llms import LLM

from ..services.catalog_retrieval import (
    VlCatalog,
    catalog_hit_to_evidence,
    search_catalog,
)
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
from .common import progress


POSITION_MATCH_FEW_SHOT = """
Пример:
Вход:
{"position": {"name": "BRD-X200", "qty": 10, "unit": "шт."}, "requirements": [],
 "catalog_product": {"file": "catalog.pdf", "location": "стр. 2", "model": "Преобразователь интерфейса P-100",
 "quote": "Модель: Преобразователь интерфейса P-100\\nОписание: шлюз RS-485 - Ethernet.\\nХарактеристики: Полный аналог BRD-X200/210 (по потребности); монтаж на DIN-рейку"}}
Выход:
{"fits": true, "status": "matched", "product_name": "Преобразователь интерфейса P-100",
 "explanation": "В характеристиках указан полный аналог BRD-X200 — изделие подходит.", "confidence": 0.9}
""".strip()

POSITION_MATCH_PROMPT_HINT = f"""
Верни ТОЛЬКО JSON:
{{"fits": true|false, "status": "matched|partial|none", "product_name": "...",
 "explanation": "...", "confidence": 0.0..1.0}}

Правила:
- Оцени ТОЛЬКО один catalog_product в ЗАДАНИИ относительно position/requirements.
- product_name — модель/артикул из catalog_product (поле model или цитата), не обозначение из тендера.
- fits=true и status matched|partial только если product_name непустой и встречается в catalog_product.
- Изделие должно быть того же класса/назначения, что и позиция перечня.

ВАЖНО: В характеристиках catalog_product может быть явное указание
«Полный аналог XXX» (или аналогичная формулировка) — это означает, что продукт ПОДХОДИТ.
Если XXX совпадает с требуемым обозначением из position/requirements — приоритет: следовать
этому указанию; ставь fits=true, status=matched (или partial при неполном покрытии требований).

{POSITION_MATCH_FEW_SHOT}

- Если не подходит — fits=false, status=none, product_name="".
""".strip()

# Сколько кандидатов из retrieval оценивать отдельными LLM-запросами.
MATCH_LLM_EVAL_MAX = 8
# Полная цитата одного продукта в промпте (vLLM 8192 хватает на 1 продукт).
MATCH_LLM_PRODUCT_QUOTE_CHARS = 1600


@dataclass
class _CandidateEval:
    hit: Evidence
    fits: bool
    status: PositionMatchStatus
    product_name: str
    explanation: str
    confidence: float
    llm_calls: int = 0


def _compact_hits_for_llm(
    asset_hits: list[Evidence],
    *,
    max_hits: int = MATCH_LLM_EVAL_MAX,
    max_quote: int = MATCH_LLM_PRODUCT_QUOTE_CHARS,
) -> list[dict[str, Any]]:
    """Укороченные хиты (legacy helper для тестов)."""
    compact: list[dict[str, Any]] = []
    for hit in asset_hits[:max_hits]:
        quote = (hit.quote or "").strip()
        if len(quote) > max_quote:
            quote = quote[: max_quote - 1] + "…"
        compact.append(
            {
                "file": hit.file,
                "location": hit.location,
                "score": hit.score,
                "quote": quote,
            }
        )
    return compact


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


def _looks_like_product_designation(text: str) -> bool:
    t = text.strip()
    if len(t) < 4:
        return False
    if re.search(r"\b[A-Z0-9]{2,}[A-Z0-9/-]*\b", t) and re.search(r"\d", t):
        return True
    return False


def _extract_designation_from_scope_name(name: str) -> str:
    name = name.strip()
    if not name:
        return ""
    words = name.split()
    for i, word in enumerate(words):
        if re.search(r"[A-Z]", word) or (
            re.search(r"\d", word) and re.search(r"[A-Za-z]", word)
        ):
            candidate = " ".join(words[i:])
            if _looks_like_product_designation(candidate):
                return candidate
    if _looks_like_product_designation(name):
        return name
    return ""


def _infer_required_product(
    scope_name: str,
    requirements: list[ExtractedRequirement],
) -> str:
    for req in requirements:
        if req.kind == "product" and (req.text or "").strip():
            text = req.text.strip()
            lowered = text.lower()
            if "или аналог" in lowered:
                return text[: lowered.index("или аналог")].strip(" ,—–-")
            return text
    return _extract_designation_from_scope_name(scope_name)


def _product_quote_for_llm(hit: Evidence, *, max_chars: int = MATCH_LLM_PRODUCT_QUOTE_CHARS) -> str:
    quote = (hit.quote or "").strip()
    if len(quote) <= max_chars:
        return quote
    return quote[: max_chars - 1] + "…"


def _status_rank(status: PositionMatchStatus) -> int:
    return {
        PositionMatchStatus.matched: 2,
        PositionMatchStatus.partial: 1,
        PositionMatchStatus.none: 0,
    }.get(status, 0)


def _parse_candidate_evaluation(
    data: dict[str, Any] | None,
    hit: Evidence,
) -> _CandidateEval | None:
    if not data:
        return None
    status_raw = str(data.get("status") or "").strip().lower()
    fits = bool(data.get("fits", data.get("matched", False)))
    try:
        status = PositionMatchStatus(status_raw)
    except ValueError:
        status = PositionMatchStatus.matched if fits else PositionMatchStatus.none
    if fits and status == PositionMatchStatus.none:
        status = PositionMatchStatus.partial
    if not fits and status != PositionMatchStatus.none:
        status = PositionMatchStatus.none

    product_name = " ".join(str(data.get("product_name") or "").split())
    if product_name and not product_name_in_hits(product_name, [hit]):
        product_name = ""
    if not product_name:
        status = PositionMatchStatus.none
        fits = False
    elif status == PositionMatchStatus.none:
        product_name = ""
        fits = False

    try:
        confidence = float(data.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    explanation = str(data.get("explanation") or "").strip()
    return _CandidateEval(
        hit=hit,
        fits=fits,
        status=status,
        product_name=product_name,
        explanation=explanation,
        confidence=min(max(confidence, 0.0), 1.0),
    )


def build_match_candidate_prompt(
    *,
    scope_item: dict[str, Any],
    requirements: list[ExtractedRequirement],
    hit: Evidence,
    instruction: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Собирает промпт и payload для оценки одного кандидата (как в match)."""
    scope_name = str(scope_item.get("name") or "").strip()
    qty = scope_item.get("qty")
    unit = str(scope_item.get("unit") or "").strip()
    quote = _product_quote_for_llm(hit)
    model_line = ""
    for line in quote.splitlines():
        if line.lower().startswith("модель:"):
            model_line = line.split(":", 1)[-1].strip()
            break

    payload = {
        "position": {"name": scope_name, "qty": qty, "unit": unit},
        "requirements": [
            {
                "text": req.text,
                "quote": (req.quote or "")[:240] if req.quote else req.quote,
                "kind": req.kind,
                "priority": req.priority,
            }
            for req in requirements
        ],
        "catalog_product": {
            "file": hit.file,
            "location": hit.location,
            "model": model_line,
            "quote": quote,
        },
    }
    instruction_text = (instruction or "").strip()
    extra_instruction = ""
    if instruction_text and instruction_text != DEFAULT_USER_INSTRUCTION.strip():
        extra_instruction = f"Дополнительно: {instruction_text}\n\n"
    payload_json = json.dumps(payload, ensure_ascii=False)
    prompt = (
        "Ты аналитик закупок. Оцени, насколько ОДИН продукт из каталога "
        "подходит к позиции перечня.\n\n"
        f"{extra_instruction}"
        f"{POSITION_MATCH_PROMPT_HINT}\n\n"
        "Формат ответа: один JSON-объект, начинается с {{ и заканчивается }}. "
        "Без markdown, без блоков ```, без текста до или после JSON.\n\n"
        f"ЗАДАНИЕ:\n{payload_json}\n\n"
        "JSON:"
    )
    return prompt, payload


def _evaluate_product_candidate(
    llm: LLM,
    *,
    scope_item: dict[str, Any],
    requirements: list[ExtractedRequirement],
    hit: Evidence,
    instruction: str,
    candidate_index: int,
) -> tuple[_CandidateEval | None, int]:
    scope_name = str(scope_item.get("name") or "").strip()
    prompt, _payload = build_match_candidate_prompt(
        scope_item=scope_item,
        requirements=requirements,
        hit=hit,
        instruction=instruction,
    )
    try:
        data, n_calls = complete_llm_json(
            llm,
            prompt,
            structure_hint=POSITION_MATCH_PROMPT_HINT,
            trace_name=f"match_candidate_{candidate_index}",
            repair_context=prompt,
        )
    except Exception as exc:
        trace_note(
            "match_candidate_error",
            f"Ошибка LLM при оценке кандидата: {exc}",
            meta={"scope_name": scope_name, "file": hit.file, "index": candidate_index},
        )
        return None, 0
    parsed = _parse_candidate_evaluation(data, hit)
    if parsed is None:
        return None, n_calls
    parsed.llm_calls = n_calls
    return parsed, n_calls


def _pick_best_candidate(candidates: list[_CandidateEval]) -> _CandidateEval | None:
    if not candidates:
        return None
    viable = [c for c in candidates if c.product_name and c.status != PositionMatchStatus.none]
    if not viable:
        return None
    return max(
        viable,
        key=lambda c: (
            _status_rank(c.status),
            c.confidence,
            c.hit.score or 0.0,
        ),
    )


def match_scope_position(llm: LLM, *, scope_item: dict[str, Any], requirements: list[ExtractedRequirement], asset_hits: list[Evidence], user_instruction: str | None = None,) -> ScopePositionMatch:
    """Подбор варианта из эталона для одной позиции перечня."""
    scope_name = str(scope_item.get("name") or "").strip()
    qty = scope_item.get("qty")
    unit = str(scope_item.get("unit") or "").strip()
    requirements = _stable_requirements(requirements)
    asset_hits = asset_hits[:12]
    base = ScopePositionMatch(
        scope_name=scope_name,
        qty=qty if isinstance(qty, (int, float)) or qty is None else None,
        unit=unit,
        requirements=list(requirements),
        asset_hits=list(asset_hits),
    )
    if not asset_hits:
        base.status = PositionMatchStatus.none
        base.explanation = "Подходящего варианта в VL-каталоге нет (нет кандидатов)."
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
    required_product = _infer_required_product(scope_name, requirements)
    candidates: list[_CandidateEval] = []
    total_calls = 0
    for index, hit in enumerate(asset_hits[:MATCH_LLM_EVAL_MAX]):
        evaluated, n_calls = _evaluate_product_candidate(
            llm,
            scope_item=scope_item,
            requirements=requirements,
            hit=hit,
            instruction=instruction,
            candidate_index=index,
        )
        total_calls += n_calls
        if evaluated is not None:
            candidates.append(evaluated)

    best = _pick_best_candidate(candidates)
    if best is None:
        base.status = PositionMatchStatus.none
        base.required_product = required_product
        if candidates:
            base.explanation = "Подходящего варианта в эталоне нет."
        else:
            base.explanation = "Не удалось разобрать ответ модели при подборе эталона."
        trace_note(
            "match_json_failed" if not candidates else "match_no_viable",
            base.explanation,
            meta={"scope_name": scope_name, "llm_calls": total_calls},
        )
        return base

    base.status = best.status
    base.required_product = required_product
    base.product_name = best.product_name
    base.explanation = best.explanation
    if not base.explanation and best.status == PositionMatchStatus.none:
        base.explanation = "Подходящего варианта в эталоне нет."
    base.confidence = best.confidence
    trace_note(
        "match_position_best",
        base.explanation or best.status.value,
        meta={
            "scope_name": scope_name,
            "product_name": best.product_name,
            "status": best.status.value,
            "confidence": best.confidence,
            "candidates_evaluated": len(candidates),
            "llm_calls": total_calls,
            "retrieval_score": best.hit.score,
        },
    )
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


def retrieve_hits_for_position(
    scope_name: str,
    requirements: list[ExtractedRequirement],
    catalog: VlCatalog,
    *,
    top_k: int,
    embedding_model: str,
    device: str | None,
) -> list:
    query = position_to_query_text(scope_name, requirements)
    if not query:
        trace_note(
            "retrieve_empty_query",
            "Пустой query для позиции — retrieval пропущен",
            meta={"scope_name": scope_name, "requirements_count": len(requirements)},
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
    hit_payload = [
        {
            "score": hit.score,
            "file": hit.source_file,
            "model": hit.product.model,
            "location": f"стр. {hit.product.source.page}" if hit.product.source.page else "VL-каталог",
            "text_preview": format_product_preview(hit),
        }
        for hit in hits
    ]
    trace_retrieval(
        "retrieve_position",
        query=query,
        hits=hit_payload,
        meta={
            "scope_name": scope_name,
            "requirements_count": len(requirements),
            "top_k": top_k,
            "fetch_k": fetch_k,
            "catalog_products": catalog.size,
            "retrieval": "vl_catalog",
        },
    )
    return hits[: max(top_k * 3, 12)]


def format_product_preview(hit) -> str:
    from ..services.catalog_retrieval import format_product_quote

    return format_product_quote(
        hit.product,
        catalog_name=hit.catalog_name,
        source_file=hit.source_file,
    )[:800]


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


def _match_one_position(
    *,
    llm: LLM,
    scope_item: dict[str, Any],
    requirements: list[ExtractedRequirement],
    product_catalog: VlCatalog,
    top_k: int,
    user_instruction: str | None,
    embedding_model: str,
    embedding_device: str | None,
) -> ScopePositionMatch:
    name = str(scope_item.get("name") or "").strip() or "позиция"
    hits = retrieve_hits_for_position(
        name,
        requirements,
        product_catalog,
        top_k=top_k,
        embedding_model=embedding_model,
        device=embedding_device,
    )
    asset_evidence = [catalog_hit_to_evidence(hit) for hit in hits]
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
    product_catalog = state.get("product_catalog")
    if product_catalog is None or product_catalog.size == 0:
        warnings = [
            "VL-каталог пуст — подбор по позициям невозможен. Переиндексируйте эталоны."
        ]
        matches = [
            _failed_position_match(
                scope_items[i],
                reqs_by_item[i] if i < len(reqs_by_item) else [],
                explanation="VL-каталог не содержит продуктов для подбора.",
            )
            for i in range(len(scope_items))
        ]
        return {"position_matches": matches, "warnings": warnings}

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
                product_catalog=product_catalog,
                top_k=top_k,
                user_instruction=instruction,
                embedding_model=settings.embedding_model,
                embedding_device=settings.embedding_device,
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