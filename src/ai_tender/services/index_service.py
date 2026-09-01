"""Embeddings и утилиты каталога эталонов (product_json, без chunk-индекса)."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from .catalog_locks import AssetFileLockedError, is_indexing

_INDEXABLE_SUFFIXES = {".pdf"}


def _unlink_file(path: Path, *, retries: int = 8, delay_sec: float = 0.25) -> None:
    """Удаляет файл; на Windows повторяет при WinError 32 (файл занят)."""
    last_err: OSError | None = None
    for attempt in range(retries):
        try:
            path.unlink()
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            winerr = getattr(exc, "winerror", None)
            if winerr == 32 or exc.errno in (13, 16, 26):
                last_err = exc
                if attempt + 1 < retries:
                    time.sleep(delay_sec)
                    continue
            raise
    raise AssetFileLockedError(
        f"Файл занят другим процессом (возможно, идёт индексация): {path.name}"
    ) from last_err


def _resolve_device(device: str | None) -> str | None:
    if device:
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return None


def configure_embeddings(model_name: str, device: str | None = None) -> HuggingFaceEmbedding:
    kwargs: dict = {"model_name": model_name, "embed_batch_size": 32}
    resolved = _resolve_device(device)
    if resolved:
        kwargs["device"] = resolved
    embed_model = HuggingFaceEmbedding(**kwargs)
    Settings.embed_model = embed_model
    return embed_model


def _iter_asset_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in _INDEXABLE_SUFFIXES
    )


def file_fingerprint(path: Path, *, relative: str | None = None) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "rel_path": relative if relative is not None else path.name,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest()[:16],
    }


def scan_assets_files(folder: Path) -> dict[str, dict[str, Any]]:
    """Словарь rel_path → fingerprint для PDF эталонов на диске."""
    folder = folder.expanduser().resolve()
    result: dict[str, dict[str, Any]] = {}
    for path in _iter_asset_files(folder):
        rel = path.relative_to(folder).as_posix()
        result[rel] = file_fingerprint(path, relative=rel)
    return result


def delete_asset_file(
    assets_path: Path,
    cache_dir: Path,
    relative_path: str,
    *,
    delete_file: bool = True,
) -> list[str]:
    """Удаляет product_json/embeddings и (по умолчанию) файл эталона с диска."""
    from .catalog_persistence import delete_product_artifacts

    assets_path = assets_path.expanduser().resolve()
    rel = relative_path.replace("\\", "/").lstrip("/")
    if not rel or ".." in Path(rel).parts:
        raise ValueError(f"Некорректный путь эталона: {relative_path!r}")

    warnings: list[str] = []
    if delete_file and is_indexing(rel):
        raise AssetFileLockedError(
            f"Идёт индексация «{Path(rel).name}» — дождитесь завершения и повторите удаление"
        )

    delete_product_artifacts(cache_dir, rel)

    if delete_file:
        target = (assets_path / rel).resolve()
        if assets_path.resolve() not in target.parents and target != assets_path.resolve():
            raise ValueError(f"Путь вне каталога эталонов: {rel}")
        if target.is_file():
            _unlink_file(target)
        else:
            warnings.append(f"Файл на диске не найден: {rel}")

    return warnings
