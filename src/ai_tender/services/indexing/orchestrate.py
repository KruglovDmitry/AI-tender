"""Оркестрация: classify → индексатор по типу (без legacy chunk-индекса)."""

from __future__ import annotations

from pathlib import Path

from llama_index.core.llms import LLM

from ...models import DocumentKind, IndexingContext, IndexingResult
from .catalog import CatalogDocumentIndexer
from .classify import classify_document_kind
from .other import OtherDocumentIndexer
from .product import ProductDocumentIndexer

_INDEXERS = {
    DocumentKind.catalog: CatalogDocumentIndexer(),
    DocumentKind.product: ProductDocumentIndexer(),
    DocumentKind.other: OtherDocumentIndexer(),
}


def index_asset_files(
    assets_path: Path,
    relative_paths: list[str],
    llm: LLM,
    *,
    cache_dir: Path,
    embedding_model: str = "BAAI/bge-m3",
    embedding_device: str | None = None,
    ocr_enabled: bool = False,
    ocr_languages: str = "rus+eng",
) -> tuple[list[IndexingResult], list[str]]:
    """Классифицирует и индексирует каждый файл своим обработчиком.

    → (results, warnings).
    """
    assets_path = assets_path.expanduser().resolve()
    cache_dir = cache_dir.expanduser().resolve()
    context = IndexingContext(
        assets_path=assets_path,
        cache_dir=cache_dir,
        llm=llm,
        embedding_model=embedding_model,
        embedding_device=embedding_device,
        ocr_enabled=ocr_enabled,
        ocr_languages=ocr_languages,
    )
    results: list[IndexingResult] = []
    warnings: list[str] = []

    for raw in relative_paths:
        rel = raw.replace("\\", "/").lstrip("/")
        if not rel:
            continue
        kind, classify_warnings, meta = classify_document_kind(
            llm,
            assets_path,
            rel,
            ocr_enabled=ocr_enabled,
            ocr_languages=ocr_languages,
        )
        warnings.extend(classify_warnings)
        indexer = _INDEXERS[kind]
        path = assets_path / rel
        result = indexer.index(path, relative_path=rel, context=context)
        result.details.update(
            {
                "confidence": meta.get("confidence"),
                "reason": meta.get("reason"),
                "preview_chars": meta.get("preview_chars"),
            }
        )
        results.append(result)

    return results, warnings
