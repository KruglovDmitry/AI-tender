"""Whole-file извлечение продуктов каталога через Qwen."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from ...models import DocumentKind, IndexingContext, ProductDocumentIndex, Settings
from ...extract.catalog_adapter import CATALOG_QWEN_PROMPT, catalog_result_to_index
from ...extract.qwen_extract import QwenExtractor
from ...extract.qwen_gate import ExtractRoute
from ...extract.qwen_settings import build_qwen_extractor, uses_qwen_extract
from ...extract.schemas import CatalogExtractResult

logger = logging.getLogger(__name__)

QwenCatalogFn = Callable[[Path], CatalogExtractResult]


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
                "Скан каталога — отдельный scan-контракт; используйте VL fallback"
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
    logger.info(
        "Qwen catalog %s → %d products",
        path.name,
        len(index.products),
    )
    return index, warnings


def try_extract_catalog_qwen(
    path: Path,
    *,
    relative_path: str,
    context: IndexingContext,
) -> tuple[ProductDocumentIndex | None, list[str]]:
    """None → вызывающий код использует VL legacy."""
    settings: Settings | None = context.extra.get("settings")
    if not uses_qwen_extract(settings):
        return None, []

    assert settings is not None
    try:
        return extract_catalog_qwen(
            path,
            relative_path=relative_path,
            context=context,
            settings=settings,
        )
    except NotImplementedError as exc:
        return None, [f"Qwen catalog: {exc}"]
    except ValueError as exc:
        return None, [f"Qwen catalog gate: {exc}"]
    except Exception as exc:
        logger.exception("Qwen catalog extract failed for %s", relative_path)
        return None, [f"Qwen catalog ошибка: {exc}"]
