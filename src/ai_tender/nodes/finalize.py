"""Нода: сборка метаданных результата."""

from __future__ import annotations

from typing import Any

from ..models import PipelineState
from .common import progress


def node_finalize(state: PipelineState) -> dict[str, Any]:
    scope_items = state.get("scope_items") or []
    scope_meta = state.get("scope_meta") or {}
    loaded_labels = list(state.get("loaded_labels") or [])
    doc_selection = dict(state.get("doc_selection") or {})
    position_matches = list(state.get("position_matches") or [])
    reqs_by_item = list(state.get("requirements_by_item") or [])
    all_requirements = [req for bucket in reqs_by_item for req in bucket]
    covered = sum(1 for item in position_matches if item.status.value != "none")
    verdict = str(state.get("verdict") or "")

    query_selection = {
        "scope_first": True,
        "graph": "langgraph",
        "scope": {
            "items": scope_items,
            "summary": scope_meta.get("scope_summary"),
            "overall_confidence": scope_meta.get("overall_confidence"),
            "needs_more_docs": scope_meta.get("needs_more_docs"),
            "missing_signals": scope_meta.get("missing_signals"),
            "files_used": list(state.get("scope_files_used") or []),
        },
        "requirements_stats": state.get("requirements_stats") or {},
        "loaded_labels": loaded_labels,
        "loaded_labels_count": len(loaded_labels),
        "doc_selection": {
            "mode": doc_selection.get("mode"),
            "catalog_count": doc_selection.get("catalog_count"),
            "selected": doc_selection.get("files"),
            "skipped": doc_selection.get("skip"),
            "loaded": loaded_labels,
            **(
                {"error": doc_selection["error"]}
                if doc_selection.get("error")
                else {}
            ),
        },
    }

    progress(
        state,
        (
            f"Готово: позиций={len(position_matches)}, закрыто={covered}, "
            f"требований={len(all_requirements)}"
        ),
        0.97,
    )
    return {
        "query_selection": query_selection,
        "verdict": verdict,
    }
