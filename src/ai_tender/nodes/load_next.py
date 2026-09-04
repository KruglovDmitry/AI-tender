from __future__ import annotations

from pathlib import Path
from typing import Any

from llama_index.core import Document

from ..models import PipelineState
from ..services.loader_service import load_documents


def next_unloaded(state: PipelineState) -> str | None:
    loaded = set(state.get("loaded_labels") or [])
    for path in state.get("scope_queue") or []:
        if path not in loaded:
            return path
    for path in state.get("ranked_paths") or []:
        if path not in loaded:
            return path
    return None


def _load_labels(state: PipelineState, labels: list[str]) -> tuple[list[Document], list[str]]:
    if not labels:
        return [], []
    return load_documents(
        Path(state["tender_path"]),
        corpus="tender",
        inventory=state.get("inventory"),
        only_labels=set(labels),
    )


def _load_label_updates(state: PipelineState, label: str) -> dict[str, Any]:
    """Патч state: загрузить label, если его ещё нет в loaded_labels."""
    if not label:
        return {}
    loaded = set(state.get("loaded_labels") or [])
    if label in loaded:
        return {}
    docs, warns = _load_labels(state, [label])
    loaded_list = list(state.get("loaded_labels") or [])
    loaded_list.append(label)
    return {
        "documents": docs,
        "loaded_labels": loaded_list,
        "warnings": warns,
    }


def node_load_next_scope_file(state: PipelineState) -> dict[str, Any]:
    label = next_unloaded(state)
    if not label:
        return {}
    callback = state.get("progress")
    if callable(callback):
        callback(f"Загрузка файла для scope: {Path(label).name}", 0.28)
    updates: dict[str, Any] = {"scope_files_used": [label]}
    updates.update(_load_label_updates(state, label))
    return updates
