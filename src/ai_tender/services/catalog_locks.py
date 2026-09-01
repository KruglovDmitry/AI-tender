"""Блокировки на время индексации эталона."""

from __future__ import annotations

import threading

_lock = threading.Lock()
_active: set[str] = set()


class AssetFileLockedError(PermissionError):
    """Файл занят — обычно идёт индексация."""


def _normalize_rel(relative_path: str) -> str:
    return relative_path.replace("\\", "/").lstrip("/")


def mark_indexing(relative_path: str, *, active: bool) -> None:
    rel = _normalize_rel(relative_path)
    if not rel:
        return
    with _lock:
        if active:
            _active.add(rel)
        else:
            _active.discard(rel)


def is_indexing(relative_path: str) -> bool:
    rel = _normalize_rel(relative_path)
    with _lock:
        return rel in _active
