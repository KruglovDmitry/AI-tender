from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from ..extract.qwen_settings import build_qwen_extractor
from ..extract.tender_adapter import (
    TENDER_QWEN_PROMPT,
    merge_requirements_buckets,
    merge_scope_item_lists,
    merge_scope_meta,
    tender_result_to_requirements,
    tender_result_to_scope,
)
from ..models import ExtractedRequirement, PipelineState, Settings
from ..services.logging_service import trace_note
from .load_next import next_unloaded

logger = logging.getLogger(__name__)


def scope_has_detailed_list(scope_items: list[dict[str, Any]]) -> bool:
    """Достаточный детальный перечень: ≥2 позиций или хотя бы одна с qty."""
    if len(scope_items) >= 2:
        return True
    if len(scope_items) == 1 and scope_items[0].get("qty") is not None:
        return True
    return False


def extract_scope_from_file(
    path: Path,
    *,
    relative_label: str,
    settings: Settings,
    existing_items: list[dict[str, Any]],
    existing_meta: dict[str, Any],
    existing_reqs: list[list[ExtractedRequirement]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[list[ExtractedRequirement]], list[str]]:
    """Один файл → Qwen whole-file (scope + requirements)."""
    warnings: list[str] = []
    extractor = build_qwen_extractor(settings)
    gate = extractor.gate(path, purpose="tender")
    result = extractor.extract_tender(path, prompt=TENDER_QWEN_PROMPT)
    route = "qwen_scan" if extractor.should_use_vl(gate, path) else str(gate.route)
    trace_note(
        "extract_scope_qwen",
        f"Qwen tender: {Path(relative_label).name}",
        meta={
            "route": route,
            "items": len(result.scope_items),
            "file": relative_label,
        },
    )

    new_items, new_meta = tender_result_to_scope(
        result,
        source_file=relative_label,
    )
    scope_items = merge_scope_item_lists(existing_items, new_items)
    scope_meta = merge_scope_meta(existing_meta, new_meta)

    new_buckets = tender_result_to_requirements(
        result,
        source_file=relative_label,
        scope_items=scope_items,
        max_per_item=settings.max_reqs_per_scope_item,
    )
    if existing_reqs and len(existing_reqs) == len(scope_items):
        reqs = merge_requirements_buckets(
            existing_reqs,
            new_buckets,
            max_per_item=settings.max_reqs_per_scope_item,
        )
    elif existing_reqs and len(existing_reqs) == len(new_items):
        reqs = new_buckets
        if len(reqs) < len(scope_items):
            reqs.extend([[] for _ in range(len(scope_items) - len(reqs))])
    else:
        reqs = new_buckets
        if len(reqs) < len(scope_items):
            reqs.extend([[] for _ in range(len(scope_items) - len(reqs))])

    scope_meta["requirements_mode"] = "qwen_whole_file"
    return scope_items, scope_meta, reqs, warnings


def node_extract_scope(state: PipelineState) -> dict[str, Any]:
    settings: Settings = state["settings"]
    callback = state.get("progress")
    if callable(callback):
        callback("LangGraph: предмет закупки (перечень позиций)", 0.32)

    existing_items = list(state.get("scope_items") or [])
    existing_meta = dict(state.get("scope_meta") or {})
    existing_reqs = list(state.get("requirements_by_item") or [])
    files_used = list(state.get("scope_files_used") or [])
    current_label = files_used[-1] if files_used else ""

    if not current_label:
        return {}

    path = Path(state["tender_path"]) / current_label
    if callable(callback):
        callback(f"Qwen whole-file: {path.name}", 0.34)

    try:
        scope_items, scope_meta, reqs, warnings = extract_scope_from_file(
            path,
            relative_label=current_label,
            settings=settings,
            existing_items=existing_items,
            existing_meta=existing_meta,
            existing_reqs=existing_reqs,
        )
    except Exception as exc:
        logger.exception("Qwen scope extract failed for %s", current_label)
        return {"warnings": [f"Qwen ошибка {path.name}: {exc}"]}

    prev_stats = dict(state.get("requirements_stats") or {})
    return {
        "scope_items": scope_items,
        "scope_meta": scope_meta,
        "requirements_by_item": reqs,
        "qwen_extracted_files": [current_label.replace("\\", "/")],
        "requirements_stats": {
            **prev_stats,
            "mode": "qwen_whole_file",
            "selected": sum(len(b) for b in reqs),
            "files_used": list(
                dict.fromkeys(list(prev_stats.get("files_used") or []) + [current_label])
            ),
        },
        "warnings": warnings,
    }


def route_after_scope(
    state: PipelineState,
) -> Literal["load_next_scope_file", "load_catalog"]:
    scope_items = state.get("scope_items") or []
    scope_meta = state.get("scope_meta") or {}
    needs_more = bool(scope_meta.get("needs_more_docs", False)) or not scope_has_detailed_list(
        scope_items
    )
    if needs_more and next_unloaded(state) is not None:
        return "load_next_scope_file"
    return "load_catalog"
