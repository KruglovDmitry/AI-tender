"""Индексация эталонов: Qwen extract → product_json + embeddings, scan/delete."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from ..extract.catalog_extractor import CatalogExtractor
from ..models import (
    DOCUMENT_KIND_LABELS,
    CatalogExtractResult,
    DocumentKind,
    IndexingContext,
    IndexingResult,
    IndexingStatus,
    ProductDocumentIndex,
    Settings as AppSettings,
)
from .catalog_service import (
    delete_product_artifacts,
    save_product_embeddings,
    save_product_index,
)

logger = logging.getLogger(__name__)

QwenCatalogFn = Callable[[Path], CatalogExtractResult]

_INDEXABLE_SUFFIXES = {".pdf"}
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


def _index_log(rel: str, message: str) -> None:
    print(f"[assets index] {rel}: {message}", flush=True)


def _emit_progress(
    context: IndexingContext,
    *,
    rel: str,
    phase: str,
    detail: str,
) -> None:
    _index_log(rel, f"{phase} — {detail}")
    callback = context.extra.get("on_index_progress")
    if callable(callback):
        callback(phase, 0, 0, detail)


def _embed_texts(texts: list[str], context: IndexingContext) -> np.ndarray:
    embedder = context.extra.get("embed_model")
    if embedder is None:
        embedder = configure_embeddings(
            context.embedding_model, context.embedding_device
        )
    if hasattr(embedder, "get_text_embedding_batch"):
        vectors = embedder.get_text_embedding_batch(texts)
    else:
        vectors = [embedder.get_text_embedding(t) for t in texts]
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _extract_file(
    path: Path,
    *,
    relative_path: str,
    context: IndexingContext,
) -> tuple[ProductDocumentIndex, list[str]]:
    rel = relative_path.replace("\\", "/").lstrip("/")
    if not path.is_file():
        return (
            ProductDocumentIndex(source_file=rel, doc_kind=DocumentKind.asset, products=[]),
            [f"Файл не найден: {rel}"],
        )

    warnings: list[str] = []
    custom: QwenCatalogFn | None = context.extra.get("qwen_catalog_extract")
    extractor: CatalogExtractor | None = context.extra.get("qwen_extractor")
    settings: AppSettings | None = context.extra.get("settings")

    if custom is not None:
        result = custom(path)
    else:
        if settings is None:
            raise ValueError("IndexingContext.extra['settings'] обязателен")
        if extractor is None:
            progress_cb = context.extra.get("on_index_progress")

            def _vl_progress(message: str) -> None:
                _index_log(rel, message)
                if callable(progress_cb):
                    progress_cb("vl", 0, 0, message)

            extractor = CatalogExtractor.from_settings(settings, on_progress=_vl_progress)
        gate = extractor.gate(path)
        result = extractor.extract(path)
        warnings.append(
            f"Qwen catalog route={'qwen_scan' if extractor.should_use_vl(gate) else gate.route}"
        )

    index = CatalogExtractor.result_to_index(
        result,
        source_file=rel,
        doc_kind=DocumentKind.asset,
    )
    logger.info("Qwen catalog %s → %d products", path.name, len(index.products))
    return index, warnings


def _persist_index(
    index: ProductDocumentIndex,
    context: IndexingContext,
) -> list[str]:
    warnings: list[str] = []
    if context.cache_dir is None:
        raise ValueError("IndexingContext.cache_dir обязателен")

    index.embedding_model = context.embedding_model
    rel = index.source_file
    _index_log(rel, f"persist: JSON + embeddings для {len(index.products)} продуктов")
    save_product_index(context.cache_dir, index)
    if not index.products:
        return warnings

    from .catalog_service import embedding_text

    texts = [embedding_text(p) for p in index.products]
    ids = [p.id for p in index.products]
    vectors = _embed_texts(texts, context)
    save_product_embeddings(
        context.cache_dir,
        index.source_file,
        ids,
        vectors,
        embedding_model=context.embedding_model,
    )
    return warnings


def index_catalog_file(
    path: Path,
    *,
    relative_path: str,
    context: IndexingContext,
) -> IndexingResult:
    kind = DocumentKind.asset
    label = DOCUMENT_KIND_LABELS[kind]
    name = Path(relative_path).name
    rel = relative_path.replace("\\", "/").lstrip("/")

    if context.cache_dir is None:
        return IndexingResult(
            relative_path=rel,
            doc_kind=kind,
            status=IndexingStatus.failed,
            message=f"«{name}»: не задан cache_dir",
        )

    try:
        _emit_progress(context, rel=rel, phase="extract", detail=f"Qwen whole-file {name}")
        doc_index, warnings = _extract_file(path, relative_path=rel, context=context)
        _emit_progress(context, rel=rel, phase="persist", detail="сохранение JSON и эмбеддингов")
        warnings.extend(_persist_index(doc_index, context))
    except Exception as exc:
        logger.exception("Catalog index failed for %s", rel)
        return IndexingResult(
            relative_path=rel,
            doc_kind=kind,
            status=IndexingStatus.failed,
            message=f"«{label}»: ошибка индексации «{name}»: {exc}",
        )

    n = len(doc_index.products)
    msg = f"«{name}» — {n} продукт(ов) (Qwen whole-file)"
    if doc_index.catalog_name:
        msg += f", документ «{doc_index.catalog_name}»"
    return IndexingResult(
        relative_path=rel,
        doc_kind=kind,
        status=IndexingStatus.indexed,
        message=msg,
        details={
            "product_count": n,
            "catalog_name": doc_index.catalog_name,
            "warnings": warnings,
        },
    )


def index_asset_files(
    assets_path: Path,
    relative_paths: list[str],
    *,
    cache_dir: Path,
    settings: AppSettings | None = None,
    extra: dict | None = None,
) -> tuple[list[IndexingResult], list[str]]:
    """Индексирует эталоны через Qwen whole-file. → (results, warnings)."""
    assets_path = assets_path.expanduser().resolve()
    cache_dir = cache_dir.expanduser().resolve()
    extra_dict = dict(extra or {})
    if settings is not None:
        extra_dict.setdefault("settings", settings)

    context = IndexingContext(
        assets_path=assets_path,
        cache_dir=cache_dir,
        embedding_model=settings.embedding_model if settings else "BAAI/bge-m3",
        embedding_device=settings.embedding_device if settings else None,
        extra=extra_dict,
    )

    results: list[IndexingResult] = []
    warnings: list[str] = []

    for raw in relative_paths:
        rel = raw.replace("\\", "/").lstrip("/")
        if not rel:
            continue
        path = assets_path / rel
        mark_indexing(rel, active=True)
        try:
            result = index_catalog_file(path, relative_path=rel, context=context)
        finally:
            mark_indexing(rel, active=False)
        results.append(result)
        warnings.extend(result.details.get("warnings") or [])

    return results, warnings
