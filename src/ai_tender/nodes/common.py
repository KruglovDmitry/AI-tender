"""Общие хелперы нод: прогресс и загрузка файлов тендера."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llama_index.core import Document

from ..loaders import load_documents
from ..models import Settings
from ..state import PipelineState


def progress(state: PipelineState, message: str, value: float) -> None:
    callback = state.get("progress")
    if callable(callback):
        callback(message, value)


def next_unloaded(state: PipelineState) -> str | None:
    loaded = set(state.get("loaded_labels") or [])
    for path in state.get("scope_queue") or []:
        if path not in loaded:
            return path
    for path in state.get("ranked_paths") or []:
        if path not in loaded:
            return path
    return None


def load_labels(state: PipelineState, labels: list[str]) -> tuple[list[Document], list[str]]:
    if not labels:
        return [], []
    settings: Settings = state["settings"]
    docs, warns = load_documents(
        Path(state["tender_path"]),
        corpus="tender",
        inventory=state.get("inventory"),
        only_labels=set(labels),
        ocr_enabled=settings.ocr_enabled,
        ocr_languages=settings.ocr_languages,
    )
    return docs, warns
