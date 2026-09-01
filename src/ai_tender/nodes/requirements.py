from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

from llama_index.core.llms import LLM

from ..services.text_service import (
    dedupe_requirements,
    make_requirement,
    merge_documents_by_file,
    numbered_excerpt,
)
from ..services.logging_service import trace_note
from ..models import ExtractedRequirement, PipelineState, Settings
from ..providers import complete_llm_json
from .common import (
    EXT_PREF,
    heuristic_role_level,
    meta_for_label,
    normalized_stem,
    parse_optional_float,
)


PER_ITEM_REQUIREMENTS_SCHEMA_HINT = """
Верни ТОЛЬКО валидный JSON-объект (без markdown и комментариев):
{
  "found": true|false,
  "requirements": [
    {
      "text": "краткая формулировка одного требования",
      "quote": "короткая дословная цитата до 160 символов",
      "kind": "product|specs|other",
      "priority": 1..3,
      "confidence": 0.0..1.0
    }
  ]
}

Правила:
- Строго валидный JSON: экранируй кавычки в строках как \\", без висячих запятых.
- found=true только если в документе есть требования именно к указанной позиции.
- Если к позиции нет своих требований — found=false и requirements=[].
  Не заполняй список чужими пунктами, даже если соседние позиции в том же документе подробнее.
- Верни только факты об ЭТОЙ позиции. Лучше короткий список, чем лишние пункты.
  Не добивай список до max_per_item: это потолок, не норма.
- quote короткий (до 160 символов), без переносов строк внутри строки JSON.
- Не пересказывай документ целиком.
- Не бери характеристики соседних пунктов перечня и не читай соседние строки таблицы
  как требования этой позиции. Пример: у позиции свой тип/номинал — не копируй
  номиналы с других строк ведомости.
""".strip()

_REQ_ROLE_RANK = {
    "specs": 0,
    "tz_main": 1,
    "other": 2,
    "notice": 3,
    "contract": 4,
    "nmck": 5,
}

def _level_rank(level: int) -> int:
    if level == 2:
        return 0
    if level == 1:
        return 1
    return 2


def order_requirement_files(labels: list[str], *, doc_selection: dict[str, Any] | None = None,) -> list[str]:
    """specs / level=2 раньше; docx раньше pdf; stem-дедуп."""
    order_index = {
        label.replace("\\", "/"): index for index, label in enumerate(labels)
    }

    def sort_key(label: str) -> tuple:
        norm = label.replace("\\", "/")
        meta = meta_for_label(label, doc_selection)
        if meta:
            role = str(meta.get("role") or "other").strip().lower() or "other"
            try:
                level = int(meta.get("scope_level", 3))
            except (TypeError, ValueError):
                level = 3
            try:
                priority = int(meta.get("priority", 9))
            except (TypeError, ValueError):
                priority = 9
        else:
            role, level = heuristic_role_level(label)
            priority = 9
        return (
            _REQ_ROLE_RANK.get(role, _REQ_ROLE_RANK["other"]),
            _level_rank(level),
            priority,
            EXT_PREF.get(Path(label).suffix.lower(), 9),
            order_index.get(norm, order_index.get(label, 10_000)),
            norm.lower(),
        )

    ordered = sorted(labels, key=sort_key)
    seen: set[str] = set()
    result: list[str] = []
    for label in ordered:
        stem = normalized_stem(label)
        if stem and stem in seen:
            continue
        if stem:
            seen.add(stem)
        result.append(label)
    return result


def parse_single_item_requirements(data: dict[str, Any], *, scope_item: dict[str, Any], file_label: str, page: int | None = None, max_per_item: int,) -> list[ExtractedRequirement]:
    scope_name = str(scope_item.get("name", "")).strip()
    found = data.get("found")
    raw_reqs = list(data.get("requirements") or [])
    if found is False and not raw_reqs:
        return []

    parsed: list[ExtractedRequirement] = []
    for item in raw_reqs:
        req_text = str(item.get("text", "")).strip()
        quote = str(item.get("quote", "")).strip() or req_text
        if not req_text:
            continue
        try:
            priority = int(item.get("priority", 2))
        except (TypeError, ValueError):
            priority = 2
        confidence = min(
            max(parse_optional_float(item.get("confidence"), 0.7), 0.0),
            1.0,
        )
        kind = str(item.get("kind", "other")).strip().lower()
        if kind not in {"product", "specs", "other"}:
            kind = "other"
        req = make_requirement(
            text=req_text,
            quote=quote,
            file=file_label,
            page=page,
            priority=min(max(priority, 0), 3),
            confidence=confidence,
            kind=kind,
        )
        req.scope_item = scope_name
        parsed.append(req)
    return dedupe_requirements(parsed, max_per_item) if parsed else []


def _scope_item_line(scope_item: dict[str, Any]) -> str:
    name = str(scope_item.get("name", "")).strip()
    qty = scope_item.get("qty")
    if qty is None:
        return name
    unit = str(scope_item.get("unit", "") or "").strip()
    return f"{name} — {qty} {unit}".rstrip()


def _single_item_prompt(*, scope_item: dict[str, Any], max_per_item: int, label: str, numbered: str, truncated: bool, retry: bool = False,) -> str:
    retry_note = ""
    if retry:
        retry_note = (
            "\nПОВТОРНЫЙ ЗАПРОС: для других позиций этого документа требования уже найдены. "
            "Ищи только то, что относится к указанной позиции. "
            "Не копируй пункты соседей. Если своих требований нет — found=false, "
            "даже если название позиции встречается в тексте.\n"
        )
    return (
        "Ты аналитик закупок. Ниже тендерный документ и ОДНА позиция перечня.\n"
        f"Извлеки только требования к этой позиции (потолок {max_per_item}, не цель).\n"
        "Не дополняй список, если фактов меньше. Не бери чужие строки перечня.\n"
        "Если своих требований нет — found=false и пустой requirements.\n"
        f"{retry_note}\n"
        f"ПОЗИЦИЯ: {_scope_item_line(scope_item)}\n\n"
        f"{PER_ITEM_REQUIREMENTS_SCHEMA_HINT.replace('max_per_item', str(max_per_item))}\n\n"
        f"ФАЙЛ: {label}\n"
        f"{'(документ обрезан по лимиту длины) ' if truncated else ''}"
        f"ДОКУМЕНТ:\n{numbered}"
    )

def _extract_one_position(
    *,
    llm: LLM,
    scope_item: dict[str, Any],
    max_per_item: int,
    label: str,
    numbered: str,
    truncated: bool,
    page: int | None,
    index: int,
    retry: bool,
) -> tuple[int, list[ExtractedRequirement], int, str | None]:
    prompt = _single_item_prompt(
        scope_item=scope_item,
        max_per_item=max_per_item,
        label=label,
        numbered=numbered,
        truncated=truncated,
        retry=retry,
    )
    data, n_calls = complete_llm_json(
        llm,
        prompt,
        structure_hint="той же структуры (поля found и requirements)",
        trace_name="requirements",
    )
    if data is None:
        return index, [], n_calls, f"{label} [scope {index}]"
    parsed = parse_single_item_requirements(
        data,
        scope_item=scope_item,
        file_label=label,
        page=page,
        max_per_item=max_per_item,
    )
    return index, parsed, n_calls, None


def _run_positions(
    *,
    llm: LLM,
    scope_items: list[dict[str, Any]],
    indexes: list[int],
    max_per_item: int,
    label: str,
    numbered: str,
    truncated: bool,
    page: int | None,
    retry: bool,
    workers: int,
) -> list[tuple[int, list[ExtractedRequirement], int, str | None]]:
    if not indexes:
        return []

    def run(index: int) -> tuple[int, list[ExtractedRequirement], int, str | None]:
        return _extract_one_position(
            llm=llm,
            scope_item=scope_items[index],
            max_per_item=max_per_item,
            label=label,
            numbered=numbered,
            truncated=truncated,
            page=page,
            index=index,
            retry=retry,
        )

    if workers <= 1 or len(indexes) == 1:
        return [run(index) for index in indexes]

    out: list[tuple[int, list[ExtractedRequirement], int, str | None]] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(indexes))) as pool:
        futures = [pool.submit(run, index) for index in indexes]
        for future in as_completed(futures):
            out.append(future.result())
    out.sort(key=lambda item: item[0])
    return out


def extract_requirements_from_file(
    *,
    label: str,
    text: str,
    page: Any = None,
    scope_items: list[dict[str, Any]],
    llm: LLM,
    max_per_item: int,
    max_chars_per_doc: int,
    existing_buckets: list[list[ExtractedRequirement]] | None = None,
    parallelism: int = 6,
    retry_if_sibling_found: bool = True,
) -> tuple[list[list[ExtractedRequirement]], dict[str, Any]]:
    """Один файл → LLM по пустым позициям (+ retry, если соседи нашлись)."""
    workers = max(1, int(parallelism or 1))
    n = len(scope_items)
    if existing_buckets and len(existing_buckets) == n:
        buckets = [list(bucket) for bucket in existing_buckets]
    else:
        buckets = [[] for _ in range(n)]

    numbered = numbered_excerpt(text, max_chars=max_chars_per_doc)
    truncated = len(numbered) >= max_chars_per_doc or numbered.endswith("…")
    try:
        page_num = int(page) if page is not None else None
    except (TypeError, ValueError):
        page_num = None
    source_by_item: list[str | None] = [
        (bucket[0].file if bucket else None) for bucket in buckets
    ]
    stats: dict[str, Any] = {
        "mode": "per_item_requirements",
        "file": label,
        "scope_items_count": n,
        "max_per_item": max_per_item,
        "parallelism": workers,
        "truncated": truncated,
        "parse_errors": [],
        "retries": 0,
        "llm_calls": 0,
        "selected": 0,
        "source_by_item": source_by_item,
    }
    if not scope_items or not (text or "").strip():
        return buckets, stats

    pending = [index for index in range(n) if not buckets[index]]
    if not pending:
        stats["selected"] = sum(len(b) for b in buckets)
        return buckets, stats

    llm_calls = 0
    parse_errors: list[str] = []

    for index, parsed, n_calls, err in _run_positions(
        llm=llm,
        scope_items=scope_items,
        indexes=pending,
        max_per_item=max_per_item,
        label=label,
        numbered=numbered,
        truncated=truncated,
        page=page_num,
        retry=False,
        workers=workers,
    ):
        llm_calls += n_calls
        if err:
            parse_errors.append(err)
        if parsed:
            buckets[index] = parsed
            source_by_item[index] = label

    still_empty = [index for index in pending if not buckets[index]]
    retries = 0
    sibling_in_file = any(source_by_item[i] == label for i in range(n))
    if retry_if_sibling_found and still_empty and sibling_in_file:
        retries = len(still_empty)
        for index, parsed, n_calls, err in _run_positions(
            llm=llm,
            scope_items=scope_items,
            indexes=still_empty,
            max_per_item=max_per_item,
            label=label,
            numbered=numbered,
            truncated=truncated,
            page=page_num,
            retry=True,
            workers=workers,
        ):
            llm_calls += n_calls
            if err:
                parse_errors.append(err)
            if parsed:
                buckets[index] = parsed
                source_by_item[index] = label

    stats.update(
        {
            "llm_calls": llm_calls,
            "retries": retries,
            "parse_errors": parse_errors,
            "selected": sum(len(b) for b in buckets),
            "source_by_item": source_by_item,
        }
    )
    return buckets, stats


def _ensure_requirement_queue(state: PipelineState) -> list[str]:
    existing = list(state.get("requirement_queue") or [])
    if existing:
        return existing
    return order_requirement_files(
        list(state.get("ranked_paths") or []),
        doc_selection=state.get("doc_selection") or None,
    )


def node_load_next_requirement_file(state: PipelineState) -> dict[str, Any]:
    from .common import load_label_updates, progress

    settings: Settings = state["settings"]
    queue = _ensure_requirement_queue(state)
    tried = set(state.get("requirement_files_tried") or [])
    max_files = max(1, settings.max_requirement_files)
    if len(tried) >= max_files:
        return {"requirement_queue": queue, "current_requirement_file": ""}

    next_label = next((path for path in queue if path not in tried), None)
    if not next_label:
        return {"requirement_queue": queue, "current_requirement_file": ""}

    progress(state, f"Файл для требований: {Path(next_label).name}", 0.42)
    updates: dict[str, Any] = {
        "requirement_queue": queue,
        "current_requirement_file": next_label,
    }
    updates.update(load_label_updates(state, next_label))
    return updates


def node_extract_requirements(state: PipelineState) -> dict[str, Any]:
    from .common import ensure_docs_for_label, progress

    settings: Settings = state["settings"]
    scope_items = list(state.get("scope_items") or [])
    current = str(state.get("current_requirement_file") or "").strip()
    if not current:
        return {}

    qwen_files = {
        label.replace("\\", "/")
        for label in (state.get("qwen_extracted_files") or [])
    }
    current_norm = current.replace("\\", "/")
    if current_norm in qwen_files:
        tried = list(state.get("requirement_files_tried") or [])
        if current not in tried:
            tried.append(current)
        return {
            "requirement_files_tried": tried,
        }

    progress(
        state,
        f"Требования из {Path(current).name} (макс. {settings.max_reqs_per_scope_item}/позиция)",
        0.5,
    )

    existing = list(state.get("requirements_by_item") or [])
    if not existing or len(existing) != len(scope_items):
        existing = [[] for _ in scope_items]

    matched, updates = ensure_docs_for_label(state, current)
    files = merge_documents_by_file(matched)
    missing_text = not files
    if missing_text:
        label, text, page = current, "", None
    else:
        label, text, page = files[0]

    reqs_by_item, req_stats = extract_requirements_from_file(
        label=label,
        text=text,
        page=page,
        scope_items=scope_items,
        llm=state["llm"],
        max_per_item=settings.max_reqs_per_scope_item,
        max_chars_per_doc=settings.max_extract_chars_per_doc,
        existing_buckets=existing,
        parallelism=settings.requirements_parallelism,
    )

    tried = list(state.get("requirement_files_tried") or [])
    if current not in tried:
        tried.append(current)

    prev = dict(state.get("requirements_stats") or {})
    merged = {
        **prev,
        **req_stats,
        "llm_calls": int(prev.get("llm_calls", 0)) + int(req_stats.get("llm_calls", 0)),
        "retries": int(prev.get("retries", 0)) + int(req_stats.get("retries", 0)),
        "files_used": list(dict.fromkeys(list(prev.get("files_used") or []) + [current])),
        "requirement_files_tried": tried,
        "passes": int(prev.get("passes", 0)) + 1,
    }
    warnings = list(updates.get("warnings") or [])
    warnings.extend(
        f"Не разобран JSON требований: {err}"
        for err in (req_stats.get("parse_errors") or [])
    )
    if missing_text:
        warnings.append(f"Нет текста для файла требований: {current}")
    trace_note(
        "requirements_stats",
        f"Проход требований: {Path(current).name}",
        meta={
            "selected": merged.get("selected"),
            "llm_calls": merged.get("llm_calls"),
            "retries": merged.get("retries"),
            "per_item_counts": [len(bucket) for bucket in reqs_by_item],
            "current_file": current,
            "tried": tried,
            "missing_text": missing_text,
        },
    )
    updates.update(
        {
            "requirements_by_item": reqs_by_item,
            "requirements_stats": merged,
            "requirement_files_tried": tried,
            "warnings": warnings,
        }
    )
    return updates


def route_after_requirements(state: PipelineState,) -> Literal["load_next_requirement_file", "build_assets_index"]:
    settings: Settings = state["settings"]
    scope_items = state.get("scope_items") or []
    buckets = state.get("requirements_by_item") or []
    empty = any(
        index >= len(buckets) or not buckets[index] for index in range(len(scope_items))
    )
    queue = _ensure_requirement_queue(state)
    tried = set(state.get("requirement_files_tried") or [])
    has_more = any(path not in tried for path in queue)
    if empty and has_more and len(tried) < max(1, settings.max_requirement_files):
        return "load_next_requirement_file"
    return "build_assets_index"