"""Извлечение предмета закупки (scope + requirements) через Qwen whole-file."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from ..extract.qwen_settings import uses_qwen_extract
from ..models import PipelineState, Settings
from .scope_qwen import extract_scope_qwen_from_file

logger = logging.getLogger(__name__)


def scope_has_detailed_list(scope_items: list[dict[str, Any]]) -> bool:
    """Достаточный детальный перечень: ≥2 позиций или хотя бы одна с qty."""
    if len(scope_items) >= 2:
        return True
    if len(scope_items) == 1 and scope_items[0].get("qty") is not None:
        return True
    return False


def node_load_next_scope_file(state: PipelineState) -> dict[str, Any]:
    from .common import load_label_updates, next_unloaded, progress

    label = next_unloaded(state)
    if not label:
        return {}
    progress(state, f"Загрузка файла для scope: {Path(label).name}", 0.28)
    updates: dict[str, Any] = {"scope_files_used": [label]}
    updates.update(load_label_updates(state, label))
    return updates


def node_extract_scope(state: PipelineState) -> dict[str, Any]:
    from .common import progress

    settings: Settings = state["settings"]
    progress(state, "LangGraph: предмет закупки (перечень позиций)", 0.32)

    existing_items = list(state.get("scope_items") or [])
    existing_meta = dict(state.get("scope_meta") or {})
    existing_reqs = list(state.get("requirements_by_item") or [])
    files_used = list(state.get("scope_files_used") or [])
    current_label = files_used[-1] if files_used else ""

    if not current_label:
        return {}

    if not uses_qwen_extract(settings):
        return {
            "warnings": [
                "Qwen extract не настроен: задайте AI_TENDER_EXTRACT_BACKEND=qwen "
                "и QWEN_API_KEY или DASHSCOPE_API_KEY"
            ],
        }

    path = Path(state["tender_path"]) / current_label
    progress(state, f"Qwen whole-file: {path.name}", 0.34)

    try:
        scope_items, scope_meta, reqs, warnings = extract_scope_qwen_from_file(
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
) -> Literal["load_next_scope_file", "build_assets_index"]:
    from .common import next_unloaded

    scope_items = state.get("scope_items") or []
    scope_meta = state.get("scope_meta") or {}
    needs_more = bool(scope_meta.get("needs_more_docs", False)) or not scope_has_detailed_list(
        scope_items
    )
    if needs_more and next_unloaded(state) is not None:
        return "load_next_scope_file"
    return "build_assets_index"
