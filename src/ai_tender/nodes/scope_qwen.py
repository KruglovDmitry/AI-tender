"""Qwen whole-file extract: scope + requirements за один вызов."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..extract.qwen_settings import build_qwen_extractor
from ..extract.tender_adapter import (
    TENDER_QWEN_PROMPT,
    merge_requirements_buckets,
    merge_scope_item_lists,
    merge_scope_meta,
    tender_result_to_requirements,
    tender_result_to_scope,
)
from ..models import ExtractedRequirement, Settings
from ..services.logging_service import trace_note

logger = logging.getLogger(__name__)


def extract_scope_qwen_from_file(
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
