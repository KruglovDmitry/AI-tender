"""Qwen whole-file индексация эталонов → product_json + embeddings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np

from ..extract.catalog_adapter import CATALOG_QWEN_PROMPT, catalog_result_to_index
from ..extract.qwen_extract import QwenExtractor
from ..extract.qwen_gate import ExtractRoute
from ..extract.qwen_settings import build_qwen_extractor, uses_qwen_extract
from ..extract.schemas import CatalogExtractResult
from ..models import (
    DOCUMENT_KIND_LABELS,
    DocumentKind,
    IndexingContext,
    IndexingResult,
    IndexingStatus,
    ProductDocumentIndex,
    Settings,
)
from .catalog_locks import mark_indexing
from .catalog_persistence import (
    MIN_PRODUCTS_PER_CATALOG,
    delete_product_artifacts,
    save_product_embeddings,
    save_product_index,
)

logger = logging.getLogger(__name__)

QwenCatalogFn = Callable[[Path], CatalogExtractResult]


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
        from .index_service import configure_embeddings

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


def extract_catalog_qwen(
    path: Path,
    *,
    relative_path: str,
    context: IndexingContext,
    settings: Settings,
) -> tuple[ProductDocumentIndex, list[str]]:
    rel = relative_path.replace("\\", "/").lstrip("/")
    warnings: list[str] = []
    custom: QwenCatalogFn | None = context.extra.get("qwen_catalog_extract")
    extractor: QwenExtractor | None = context.extra.get("qwen_extractor")

    if custom is not None:
        result = custom(path)
    else:
        if extractor is None:
            extractor = build_qwen_extractor(settings)
        gate = extractor.gate(path, purpose="catalog")
        if gate.route == ExtractRoute.qwen_scan:
            raise NotImplementedError(
                f"Скан «{rel}» — Qwen scan-контракт ещё не реализован ({gate.reason})"
            )
        if not gate.sends_to_qwen_doc:
            raise ValueError(gate.reason)
        result = extractor.extract_catalog(path, prompt=CATALOG_QWEN_PROMPT)
        warnings.append(f"Qwen catalog route={gate.route}")

    index = catalog_result_to_index(
        result,
        source_file=rel,
        doc_kind=DocumentKind.asset,
    )
    logger.info("Qwen catalog %s → %d products", path.name, len(index.products))
    return index, warnings


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

    settings: Settings | None = context.extra.get("settings")
    if not uses_qwen_extract(settings):
        return (
            ProductDocumentIndex(source_file=rel, doc_kind=DocumentKind.asset, products=[]),
            [
                "Qwen extract не настроен: задайте AI_TENDER_EXTRACT_BACKEND=qwen "
                "и QWEN_API_KEY или DASHSCOPE_API_KEY"
            ],
        )

    assert settings is not None
    return extract_catalog_qwen(
        path,
        relative_path=rel,
        context=context,
        settings=settings,
    )


def _persist_index(
    index: ProductDocumentIndex,
    context: IndexingContext,
) -> list[str]:
    warnings: list[str] = []
    if context.cache_dir is None:
        raise ValueError("IndexingContext.cache_dir обязателен")

    index.embedding_model = context.embedding_model
    rel = index.source_file
    if len(index.products) < MIN_PRODUCTS_PER_CATALOG:
        delete_product_artifacts(context.cache_dir, rel)
        warnings.append(
            f"Каталог не сохранён: требуется минимум {MIN_PRODUCTS_PER_CATALOG} продукт(ов)"
        )
        return warnings

    _index_log(rel, f"persist: JSON + embeddings для {len(index.products)} продуктов")
    save_product_index(context.cache_dir, index)

    texts = [
        (p.canonical_desc or p.model or p.raw_chunk or p.id).strip() or p.id
        for p in index.products
    ]
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
    if n < MIN_PRODUCTS_PER_CATALOG:
        removed = delete_product_artifacts(context.cache_dir, rel)
        if removed:
            _index_log(rel, f"удалены пустые артефакты: {len(removed)} файл(ов)")
        fail_msg = (
            f"«{name}» — требуется минимум {MIN_PRODUCTS_PER_CATALOG} продукт(ов), "
            f"извлечено {n}. Переиндексируйте после исправления Qwen extract."
        )
        warnings.append(fail_msg)
        return IndexingResult(
            relative_path=rel,
            doc_kind=kind,
            status=IndexingStatus.failed,
            message=fail_msg,
            details={
                "product_count": n,
                "catalog_name": doc_index.catalog_name,
                "warnings": warnings,
                "min_products_required": MIN_PRODUCTS_PER_CATALOG,
            },
        )

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
    settings: Settings | None = None,
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
