"""LLM-извлечение требований по позициям scope из документов тендера."""

from __future__ import annotations

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

from llama_index.core import Document
from llama_index.core.llms import LLM

from ..services.text_service import (
    attach_anchor,
    dedupe_requirements,
    merge_documents_by_file,
    numbered_excerpt,
    source_node_from_file,
)
from ..services.logging_service import trace_llm, trace_note
from ..models import ExtractedRequirement, PipelineState, Settings
from ..providers import try_parse_llm_json


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
- Если требований к позиции нет — found=false и requirements=[].
- Не более max_per_item требований.
- quote короткий (до 160 символов), без переносов строк внутри строки JSON.
- Не пересказывай документ целиком и не подменяй требования других позиций.
""".strip()


def _parse_optional_float(value: Any, default: float) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _complete_requirements_json(llm: LLM, prompt: str) -> tuple[dict[str, Any] | None, int]:
    """Запрос к LLM + мягкий retry при битом JSON. Возвращает (data, число complete)."""
    response = llm.complete(prompt)
    raw = str(response)
    trace_llm("requirements", prompt=prompt, response=raw, meta={"phase": "extract"})
    data = try_parse_llm_json(raw)
    if data is not None:
        return data, 1

    repair_prompt = (
        "Предыдущий ответ был НЕВАЛИДНЫМ JSON. "
        "Верни ТОЛЬКО исправленный валидный JSON-объект той же структуры "
        "(поля found и requirements). "
        "Без markdown, без комментариев. Экранируй кавычки в строках.\n\n"
        f"ИСХОДНЫЙ ОТВЕТ:\n{raw[:12000]}"
    )
    repaired = llm.complete(repair_prompt)
    repaired_raw = str(repaired)
    trace_llm(
        "requirements_repair",
        prompt=repair_prompt,
        response=repaired_raw,
        meta={"phase": "repair"},
    )
    return try_parse_llm_json(repaired_raw), 2


def parse_single_item_requirements(
    data: dict[str, Any],
    *,
    scope_item: dict[str, Any],
    source_node,
    max_per_item: int,
) -> list[ExtractedRequirement]:
    """Разбор JSON одной позиции → список требований."""
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
            max(_parse_optional_float(item.get("confidence"), 0.7), 0.0),
            1.0,
        )
        kind = str(item.get("kind", "other")).strip().lower()
        if kind not in {"product", "specs", "other"}:
            kind = "other"
        req = attach_anchor(
            text=req_text,
            quote=quote,
            source_node=source_node,
            priority=min(max(priority, 0), 3),
            confidence=confidence,
            kind=kind,
        )
        req.scope_item = scope_name
        parsed.append(req)
    return dedupe_requirements(parsed, max_per_item) if parsed else []


def _scope_item_line(scope_items: list[dict[str, Any]], idx: int) -> str:
    item = scope_items[idx]
    name = str(item.get("name", "")).strip()
    qty = item.get("qty")
    if qty is None:
        return f"{idx}) {name}"
    unit = str(item.get("unit", "") or "").strip()
    return f"{idx}) {name} — {qty} {unit}".rstrip()


_REQ_ROLE_RANK = {
    "specs": 0,
    "tz_main": 1,
    "other": 2,
    "notice": 3,
    "contract": 4,
    "nmck": 5,
}

_EXT_PREF = {
    ".docx": 0,
    ".doc": 1,
    ".odt": 2,
    ".rtf": 3,
    ".pdf": 4,
    ".txt": 5,
    ".md": 6,
}


def _requirement_scope_level_rank(level: int) -> int:
    """Для требований детальное ТЗ (2) важнее титула (1)."""
    if level == 2:
        return 0
    if level == 1:
        return 1
    return 2


def _heuristic_requirement_file_rank(label: str) -> tuple[int, int]:
    """Fallback, если файла нет в doc_selection: роль/уровень по имени."""
    lower = label.lower().replace("\\", "/")
    name = Path(label).name.lower()
    in_tz_folder = "техническ" in lower
    is_detailed = ("тз" in name or name.startswith("тз")) and (
        "фз" in name or "522" in name or in_tz_folder
    )
    is_title = "провед" in lower and "закуп" in lower
    is_notice = "извещ" in lower
    is_admin = any(
        token in lower
        for token in ("регламент", "допуск", "аттестац", "обеспечен", "договор", "нмц")
    )
    if is_detailed:
        return _REQ_ROLE_RANK["specs"], 2
    if is_title:
        return _REQ_ROLE_RANK["tz_main"], 1
    if is_notice:
        return _REQ_ROLE_RANK["notice"], 1
    if is_admin:
        return _REQ_ROLE_RANK["nmck"], 3
    return _REQ_ROLE_RANK["other"], 3


def _selection_meta_by_path(doc_selection: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for item in (doc_selection or {}).get("files") or []:
        path = str(item.get("path") or "").strip().replace("\\", "/")
        if path:
            meta[path] = item
    return meta


def _order_requirement_files(
    files: list[tuple[str, str, Any]],
    *,
    file_order: list[str] | None,
    prefer_labels: list[str] | None,
    doc_selection: dict[str, Any] | None = None,
) -> list[tuple[str, str, Any]]:
    """Порядок по LLM-разметке scope: specs / scope_level=2 раньше титула и извещений."""
    prefer = {label.replace("\\", "/") for label in (prefer_labels or [])}
    order_index = {
        label.replace("\\", "/"): index for index, label in enumerate(file_order or [])
    }
    selected = _selection_meta_by_path(doc_selection)

    def sort_key(item: tuple[str, str, Any]) -> tuple:
        label, _text, _page = item
        norm = label.replace("\\", "/")
        meta = selected.get(norm) or selected.get(label) or {}
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
            role_rank = _REQ_ROLE_RANK.get(role, _REQ_ROLE_RANK["other"])
        else:
            role_rank, level = _heuristic_requirement_file_rank(norm)
            priority = 9
        return (
            role_rank,
            _requirement_scope_level_rank(level),
            0 if norm in prefer or label in prefer else 1,
            priority,
            order_index.get(norm, order_index.get(label, len(order_index))),
            norm.lower(),
        )

    return sorted(files, key=sort_key)


def _normalized_stem(label: str) -> str:
    stem = Path(label).stem.lower()
    stem = re.sub(r"[\s_\-]+", "", stem)
    stem = re.sub(r"[^\wа-яё]+", "", stem, flags=re.IGNORECASE)
    return stem


def _content_fingerprint(text: str) -> str:
    norm = re.sub(r"\s+", " ", (text or "").lower()).strip()
    if len(norm) < 200:
        sample = norm
    else:
        mid = len(norm) // 2
        sample = norm[:3000] + norm[mid : mid + 1500] + norm[-1500:]
    return hashlib.sha1(sample.encode("utf-8")).hexdigest()


def _dedupe_equivalent_files(
    files: list[tuple[str, str, Any]],
) -> tuple[list[tuple[str, str, Any]], list[str]]:
    """Убрать docx/pdf-дубликаты одного ТЗ; предпочитаем docx над pdf."""
    kept: list[tuple[str, str, Any]] = []
    skipped: list[str] = []
    seen_stem: dict[str, int] = {}
    seen_fp: dict[str, int] = {}

    def better(new: tuple[str, str, Any], old: tuple[str, str, Any]) -> bool:
        new_ext = Path(new[0]).suffix.lower()
        old_ext = Path(old[0]).suffix.lower()
        return (_EXT_PREF.get(new_ext, 9), -len(new[1])) < (
            _EXT_PREF.get(old_ext, 9),
            -len(old[1]),
        )

    for item in files:
        label, text, _page = item
        stem = _normalized_stem(label)
        fp = _content_fingerprint(text)

        replace_at: int | None = None
        if stem and stem in seen_stem:
            replace_at = seen_stem[stem]
        elif fp in seen_fp:
            replace_at = seen_fp[fp]

        if replace_at is None:
            idx = len(kept)
            kept.append(item)
            if stem:
                seen_stem[stem] = idx
            seen_fp[fp] = idx
            continue

        old = kept[replace_at]
        if better(item, old):
            skipped.append(f"{old[0]} → заменён на {label}")
            old_stem = _normalized_stem(old[0])
            old_fp = _content_fingerprint(old[1])
            kept[replace_at] = item
            if old_stem and seen_stem.get(old_stem) == replace_at:
                del seen_stem[old_stem]
            if seen_fp.get(old_fp) == replace_at:
                del seen_fp[old_fp]
            if stem:
                seen_stem[stem] = replace_at
            seen_fp[_content_fingerprint(text)] = replace_at
        else:
            skipped.append(f"{label} → дубликат {old[0]}")

    return kept, skipped


def _single_item_prompt(
    *,
    scope_item: dict[str, Any],
    max_per_item: int,
    label: str,
    numbered: str,
    truncated: bool,
    retry: bool = False,
) -> str:
    line = _scope_item_line([scope_item], 0).removeprefix("0) ").strip()
    retry_note = ""
    if retry:
        retry_note = (
            "\nПОВТОРНЫЙ ЗАПРОС: для соседних позиций в ЭТОМ же документе требования уже найдены. "
            "Ещё раз внимательно найди требования именно к указанной позиции "
            "(блоки ПКУ/ПУ/ТТ, технические характеристики, СТО/ГОСТ, условия поставки). "
            "found=false только если позиции реально нет в тексте документа.\n"
        )
    return (
        "Ты аналитик закупок. Ниже тендерный документ и ОДНА позиция перечня.\n"
        f"Извлеки до {max_per_item} требований, относящихся именно к этой позиции.\n"
        "Если в документе нет требований к позиции — found=false и пустой requirements.\n"
        f"{retry_note}\n"
        f"ПОЗИЦИЯ: {line}\n\n"
        f"{PER_ITEM_REQUIREMENTS_SCHEMA_HINT.replace('max_per_item', str(max_per_item))}\n\n"
        f"ФАЙЛ: {label}\n"
        f"{'(документ обрезан по лимиту длины) ' if truncated else ''}"
        f"ДОКУМЕНТ:\n{numbered}"
    )


def build_requirement_file_queue(
    labels: list[str],
    *,
    prefer_labels: list[str] | None = None,
    doc_selection: dict[str, Any] | None = None,
) -> list[str]:
    """Упорядоченный список уникальных файлов-кандидатов для требований (без pdf/docx-дублей)."""
    fake = [(label, "", None) for label in labels]
    ordered = _order_requirement_files(
        fake,
        file_order=labels,
        prefer_labels=prefer_labels,
        doc_selection=doc_selection,
    )
    seen_stem: set[str] = set()
    result: list[str] = []
    for label, _text, _page in ordered:
        stem = _normalized_stem(label)
        if stem and stem in seen_stem:
            continue
        if stem:
            seen_stem.add(stem)
        result.append(label)
    return result


def extract_requirements_per_scope_items(
    documents: list[Document],
    *,
    scope_items: list[dict[str, Any]],
    llm: LLM,
    max_per_item: int,
    max_chars_per_doc: int,
    file_order: list[str] | None = None,
    prefer_labels: list[str] | None = None,
    doc_selection: dict[str, Any] | None = None,
    max_files: int = 3,
    existing_buckets: list[list[ExtractedRequirement]] | None = None,
    only_labels: set[str] | None = None,
    retry_if_sibling_found: bool = True,
    parallelism: int = 6,
) -> tuple[list[list[ExtractedRequirement]], dict[str, Any]]:
    """
    Для каждой (ещё пустой) позиции — LLM-запрос к документу.

    Внутри одного файла запросы по позициям идут параллельно; если часть
    нашлась, а часть нет — параллельный retry для пустых.
    """
    workers = max(1, int(parallelism or 1))
    files = merge_documents_by_file(documents)
    files_available = len(files)
    if only_labels:
        wanted = {label.replace("\\", "/") for label in only_labels}
        files = [
            item
            for item in files
            if item[0].replace("\\", "/") in wanted or item[0] in only_labels
        ]
    files = _order_requirement_files(
        files,
        file_order=file_order,
        prefer_labels=prefer_labels,
        doc_selection=doc_selection,
    )
    files, deduped = _dedupe_equivalent_files(files)
    if max_files > 0:
        files = files[:max_files]

    n = len(scope_items)
    if existing_buckets and len(existing_buckets) == n:
        buckets: list[list[ExtractedRequirement]] = [
            list(bucket) for bucket in existing_buckets
        ]
    else:
        buckets = [[] for _ in range(n)]

    truncated_files: list[str] = []
    parse_errors: list[str] = []
    files_used: list[str] = []
    files_tried_by_item: list[list[str]] = [[] for _ in range(n)]
    source_by_item: list[str | None] = [
        (bucket[0].file if bucket else None) for bucket in buckets
    ]
    retries = 0
    llm_calls = 0
    prepared: list[tuple[str, str, bool, Any]] = []
    for label, text, page in files:
        numbered = numbered_excerpt(text, max_chars=max_chars_per_doc)
        truncated = len(numbered) >= max_chars_per_doc or numbered.endswith("…")
        if truncated:
            truncated_files.append(label)
        prepared.append(
            (label, numbered, truncated, source_node_from_file(label, text, page=page))
        )

    stats: dict[str, Any] = {
        "mode": "per_item_requirements",
        "files": len(prepared),
        "files_available": files_available,
        "files_after_dedupe": len(prepared),
        "deduped_files": deduped,
        "scope_items_count": n,
        "max_per_item": max_per_item,
        "max_files": max_files,
        "parallelism": workers,
        "truncated_files": truncated_files,
        "parse_errors": parse_errors,
        "files_used": files_used,
        "files_tried_by_item": files_tried_by_item,
        "source_by_item": source_by_item,
        "retries": 0,
        "llm_calls": 0,
        "early_stop": False,
        "extracted_raw": 0,
        "selected": 0,
    }

    if not prepared or not scope_items:
        return buckets, stats

    for label, numbered, truncated, source_node in prepared:
        if label not in files_used:
            files_used.append(label)

        pending = [index for index in range(n) if not buckets[index]]
        if not pending:
            break

        for index in pending:
            files_tried_by_item[index].append(label)

        results = _extract_positions_parallel(
            llm=llm,
            scope_items=scope_items,
            indexes=pending,
            max_per_item=max_per_item,
            label=label,
            numbered=numbered,
            truncated=truncated,
            source_node=source_node,
            retry=False,
            workers=workers,
        )
        for index, parsed, n_calls, err in results:
            llm_calls += n_calls
            if err:
                parse_errors.append(err)
            if parsed:
                buckets[index] = parsed
                source_by_item[index] = label

        still_empty = [index for index in pending if not buckets[index]]
        sibling_found = any(
            source_by_item[index] == label for index in range(n) if buckets[index]
        )
        if retry_if_sibling_found and still_empty and sibling_found:
            retries += len(still_empty)
            retry_results = _extract_positions_parallel(
                llm=llm,
                scope_items=scope_items,
                indexes=still_empty,
                max_per_item=max_per_item,
                label=label,
                numbered=numbered,
                truncated=truncated,
                source_node=source_node,
                retry=True,
                workers=workers,
            )
            for index, parsed, n_calls, err in retry_results:
                llm_calls += n_calls
                if err:
                    parse_errors.append(err)
                if parsed:
                    buckets[index] = parsed
                    source_by_item[index] = label

    selected = sum(len(bucket) for bucket in buckets)
    stats["extracted_raw"] = selected
    stats["selected"] = selected
    stats["files"] = len(files_used)
    stats["retries"] = retries
    stats["llm_calls"] = llm_calls
    stats["source_by_item"] = source_by_item
    return buckets, stats


def _extract_positions_parallel(
    *,
    llm: LLM,
    scope_items: list[dict[str, Any]],
    indexes: list[int],
    max_per_item: int,
    label: str,
    numbered: str,
    truncated: bool,
    source_node,
    retry: bool,
    workers: int,
) -> list[tuple[int, list[ExtractedRequirement], int, str | None]]:
    if not indexes:
        return []
    if workers <= 1 or len(indexes) == 1:
        return [
            _extract_one_position_result(
                llm=llm,
                scope_item=scope_items[index],
                max_per_item=max_per_item,
                label=label,
                numbered=numbered,
                truncated=truncated,
                source_node=source_node,
                index=index,
                retry=retry,
            )
            for index in indexes
        ]

    out: list[tuple[int, list[ExtractedRequirement], int, str | None]] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(indexes))) as pool:
        futures = {
            pool.submit(
                _extract_one_position_result,
                llm=llm,
                scope_item=scope_items[index],
                max_per_item=max_per_item,
                label=label,
                numbered=numbered,
                truncated=truncated,
                source_node=source_node,
                index=index,
                retry=retry,
            ): index
            for index in indexes
        }
        for future in as_completed(futures):
            out.append(future.result())
    out.sort(key=lambda item: item[0])
    return out


def _extract_one_position_result(
    *,
    llm: LLM,
    scope_item: dict[str, Any],
    max_per_item: int,
    label: str,
    numbered: str,
    truncated: bool,
    source_node,
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
    data, n_calls = _complete_requirements_json(llm, prompt)
    if data is None:
        return (
            index,
            [],
            n_calls,
            f"{label} [scope {index}{' retry' if retry else ''}]",
        )
    parsed = parse_single_item_requirements(
        data,
        scope_item=scope_item,
        source_node=source_node,
        max_per_item=max_per_item,
    )
    return index, parsed, n_calls, None



def _ensure_requirement_queue(state: PipelineState) -> list[str]:
    existing = list(state.get("requirement_queue") or [])
    if existing:
        return existing
    return build_requirement_file_queue(
        list(state.get("ranked_paths") or []),
        prefer_labels=list(state.get("scope_files_used") or []),
        doc_selection=state.get("doc_selection") or None,
    )


def node_load_next_requirement_file(state: PipelineState) -> dict[str, Any]:
    from .common import load_labels, progress

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
    loaded = set(state.get("loaded_labels") or [])
    if next_label not in loaded:
        docs, warns = load_labels(state, [next_label])
        loaded_list = list(state.get("loaded_labels") or [])
        loaded_list.append(next_label)
        updates.update(
            {
                "documents": docs,
                "loaded_labels": loaded_list,
                "warnings": warns,
            }
        )
    return updates


def node_extract_requirements(state: PipelineState) -> dict[str, Any]:
    from .common import progress

    settings: Settings = state["settings"]
    scope_items = list(state.get("scope_items") or [])
    current = str(state.get("current_requirement_file") or "").strip()
    if not current:
        return {}

    progress(
        state,
        (
            f"Требования из {Path(current).name} "
            f"(макс. {settings.max_reqs_per_scope_item}/позиция, retry при пустом)"
        ),
        0.5,
    )
    existing = list(state.get("requirements_by_item") or [])
    if existing and len(existing) != len(scope_items):
        existing = [[] for _ in scope_items]
    elif not existing:
        existing = [[] for _ in scope_items]

    docs = [
        doc
        for doc in (state.get("documents") or [])
        if str((doc.metadata or {}).get("file_path") or (doc.metadata or {}).get("file_name") or "")
        .replace("\\", "/")
        == current.replace("\\", "/")
        or Path(
            str((doc.metadata or {}).get("file_path") or (doc.metadata or {}).get("file_name") or "")
        ).name
        == Path(current).name
    ]
    if not docs:
        docs = list(state.get("documents") or [])

    reqs_by_item, req_stats = extract_requirements_per_scope_items(
        docs,
        scope_items=scope_items,
        llm=state["llm"],
        max_per_item=settings.max_reqs_per_scope_item,
        max_chars_per_doc=settings.max_extract_chars_per_doc,
        file_order=state.get("ranked_paths") or None,
        prefer_labels=list(state.get("scope_files_used") or []),
        doc_selection=state.get("doc_selection") or None,
        max_files=1,
        existing_buckets=existing,
        only_labels={current},
        retry_if_sibling_found=True,
        parallelism=settings.requirements_parallelism,
    )

    tried = list(state.get("requirement_files_tried") or [])
    if current not in tried:
        tried.append(current)

    prev_stats = dict(state.get("requirements_stats") or {})
    merged_stats = {
        **prev_stats,
        **req_stats,
        "llm_calls": int(prev_stats.get("llm_calls", 0)) + int(req_stats.get("llm_calls", 0)),
        "retries": int(prev_stats.get("retries", 0)) + int(req_stats.get("retries", 0)),
        "files_used": list(
            dict.fromkeys(
                list(prev_stats.get("files_used") or []) + list(req_stats.get("files_used") or [])
            )
        ),
        "requirement_files_tried": tried,
        "passes": int(prev_stats.get("passes", 0)) + 1,
    }
    warnings = [
        f"Не разобран JSON требований: {err}"
        for err in (req_stats.get("parse_errors") or [])
    ]
    trace_note(
        "requirements_stats",
        f"Проход требований: {Path(current).name}",
        meta={
            "selected": merged_stats.get("selected"),
            "llm_calls": merged_stats.get("llm_calls"),
            "retries": merged_stats.get("retries"),
            "files_used": merged_stats.get("files_used"),
            "source_by_item": req_stats.get("source_by_item"),
            "per_item_counts": [len(bucket) for bucket in reqs_by_item],
            "current_file": current,
            "tried": tried,
        },
    )
    return {
        "requirements_by_item": reqs_by_item,
        "requirements_stats": merged_stats,
        "requirement_files_tried": tried,
        "warnings": warnings,
    }


def route_after_requirements(
    state: PipelineState,
) -> Literal["load_next_requirement_file", "build_assets_index"]:
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
