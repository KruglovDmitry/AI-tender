"""Общие хелперы нод: прогресс, загрузка файлов, дедуп по файлу."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from llama_index.core import Document

from ..services.loader_service import load_documents
from ..models import Evidence, PipelineState, Settings

EXT_PREF = {
    ".docx": 0,
    ".doc": 1,
    ".odt": 2,
    ".rtf": 3,
    ".pdf": 4,
    ".txt": 5,
    ".md": 6,
}


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


def normalized_stem(label: str) -> str:
    """Имя файла без расширения/пробелов — ключ для дедупа docx/pdf."""
    stem = Path(label).stem.lower()
    stem = re.sub(r"[\s_\-]+", "", stem)
    stem = re.sub(r"[^\wа-яё]+", "", stem, flags=re.IGNORECASE)
    return stem


def meta_for_label(label: str, doc_selection: dict[str, Any] | None) -> dict[str, Any]:
    """Метаданные файла из результата select_files (role, scope_level, …)."""
    norm = label.replace("\\", "/")
    for item in (doc_selection or {}).get("files") or []:
        path = str(item.get("path") or "").strip().replace("\\", "/")
        if path == norm or path == label:
            return item
    return {}


def docs_for_label(documents: list[Document], label: str) -> list[Document]:
    """Документы из state для одного файла: сначала точный path, иначе имя."""
    want = label.replace("\\", "/")
    want_name = Path(label).name
    exact: list[Document] = []
    by_name: list[Document] = []
    for doc in documents:
        meta = doc.metadata or {}
        path = str(meta.get("file_path") or meta.get("file_name") or "").replace("\\", "/")
        if path == want:
            exact.append(doc)
        elif Path(path).name == want_name:
            by_name.append(doc)
    return exact or by_name


def heuristic_role_level(label: str) -> tuple[str, int]:
    """Fallback role/scope_level по имени файла, если нет doc_selection."""
    lower = label.lower().replace("\\", "/")
    name = Path(label).name.lower()
    if ("тз" in name or name.startswith("тз")) and (
        "фз" in name or "522" in name or "техническ" in lower
    ):
        return "specs", 2
    if "извещ" in lower:
        return "notice", 1
    return "other", 3


def parse_optional_float(value: Any, default: float) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def load_label_updates(state: PipelineState, label: str) -> dict[str, Any]:
    """Патч state: загрузить label, если его ещё нет в loaded_labels."""
    if not label:
        return {}
    loaded = set(state.get("loaded_labels") or [])
    if label in loaded:
        return {}
    docs, warns = load_labels(state, [label])
    loaded_list = list(state.get("loaded_labels") or [])
    loaded_list.append(label)
    return {
        "documents": docs,
        "loaded_labels": loaded_list,
        "warnings": warns,
    }


def ensure_docs_for_label(
    state: PipelineState,
    label: str,
) -> tuple[list[Document], dict[str, Any]]:
    """Документы label из state; при пустом match — догрузка, если ещё не loaded."""
    docs = list(state.get("documents") or [])
    matched = docs_for_label(docs, label)
    if matched:
        return matched, {}
    updates = load_label_updates(state, label)
    extra = list(updates.get("documents") or [])
    if extra:
        matched = docs_for_label(docs + extra, label)
    return matched, updates


def _file_key(raw: str, fallback: object) -> str:
    key = (raw or "").replace("\\", "/").casefold()
    return key or f"id:{id(fallback)}"


def _dedupe_by_file(
    items: list,
    *,
    file_of,
    score_of,
    limit: int | None = None,
) -> list:
    """Лучший элемент на файл (по score), порядок первого появления."""
    best: dict[str, Any] = {}
    order: list[str] = []
    for item in items:
        key = _file_key(str(file_of(item) or ""), item)
        if key not in best:
            order.append(key)
            best[key] = item
            continue
        prev = best[key]
        if (score_of(item) or 0.0) > (score_of(prev) or 0.0):
            best[key] = item
    output = [best[key] for key in order]
    if limit is not None:
        return output[: max(0, limit)]
    return output


def dedupe_evidence_by_file(
    hits: list[Evidence],
    *,
    limit: int | None = None,
) -> list[Evidence]:
    """Один лучший Evidence на файл (по score)."""
    return _dedupe_by_file(
        hits,
        file_of=lambda hit: hit.file,
        score_of=lambda hit: hit.score,
        limit=limit,
    )


def _hit_file_path(hit) -> str:
    node = getattr(hit, "node", None)
    meta = (getattr(node, "metadata", None) or {}) if node is not None else {}
    return str(meta.get("file_path") or meta.get("file_name") or "")


def dedupe_hits_by_file(hits: list, *, limit: int | None = None) -> list:
    """Один лучший retrieval-hit на файл (по score)."""
    return _dedupe_by_file(
        hits,
        file_of=_hit_file_path,
        score_of=lambda hit: getattr(hit, "score", None),
        limit=limit,
    )
