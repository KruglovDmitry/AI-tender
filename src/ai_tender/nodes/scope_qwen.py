"""Подключение Qwen whole-file extract к узлу scope."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from llama_index.core import Document
from llama_index.core.llms import LLM

from ..extract.qwen_gate import ExtractRoute
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
from .scope import extract_procurement_scope_from_documents

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

    if gate.route == ExtractRoute.qwen_scan:
        warnings.append(
            f"{relative_label}: скан — Qwen scan-контракт ещё не реализован, нужен legacy"
        )
        raise ValueError(gate.reason)

    if not gate.sends_to_qwen_doc:
        raise ValueError(gate.reason)

    result = extractor.extract_tender(path, prompt=TENDER_QWEN_PROMPT)
    trace_note(
        "extract_scope_qwen",
        f"Qwen tender: {Path(relative_label).name}",
        meta={
            "route": str(gate.route),
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
        # scope_items выросли — пересобрать buckets по новому списку
        reqs = new_buckets
        if len(reqs) < len(scope_items):
            reqs.extend([[] for _ in range(len(scope_items) - len(reqs))])
    else:
        reqs = new_buckets
        if len(reqs) < len(scope_items):
            reqs.extend([[] for _ in range(len(scope_items) - len(reqs))])

    scope_meta["requirements_mode"] = "qwen_whole_file"
    return scope_items, scope_meta, reqs, warnings


def extract_scope_qwen_with_legacy_fallback(
    *,
    path: Path,
    relative_label: str,
    documents: list[Document],
    llm: LLM,
    settings: Settings,
    existing_items: list[dict[str, Any]],
    existing_meta: dict[str, Any],
    existing_reqs: list[list[ExtractedRequirement]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    list[list[ExtractedRequirement]] | None,
    list[str],
    bool,
]:
    """
    Qwen whole-file; при ошибке/gate legacy → text LLM (только scope).
    → (scope_items, scope_meta, requirements_by_item|None, warnings, qwen_ok).
    """
    warnings: list[str] = []
    try:
        items, meta, reqs, w = extract_scope_qwen_from_file(
            path,
            relative_label=relative_label,
            settings=settings,
            existing_items=existing_items,
            existing_meta=existing_meta,
            existing_reqs=existing_reqs,
        )
        warnings.extend(w)
        return items, meta, reqs, warnings, True
    except NotImplementedError as exc:
        warnings.append(f"Qwen: {exc}")
    except ValueError as exc:
        warnings.append(f"Qwen gate/маршрут {relative_label}: {exc}")
    except Exception as exc:
        logger.exception("Qwen extract failed for %s", relative_label)
        warnings.append(f"Qwen ошибка {Path(relative_label).name}: {exc}")

    warnings.append(f"Legacy fallback (text LLM) для scope: {Path(relative_label).name}")
    from .common import docs_for_label

    matched = docs_for_label(documents, relative_label) or documents
    items, meta = extract_procurement_scope_from_documents(
        matched,
        llm,
        max_chars_per_doc=settings.max_extract_chars_per_doc,
    )
    meta["extraction_mode"] = "legacy_fallback"
    merged_items = merge_scope_item_lists(existing_items, items)
    merged_meta = merge_scope_meta(existing_meta, meta)
    return merged_items, merged_meta, None, warnings, False
