"""Нода: загрузка уже проиндексированного каталога эталонов для retrieval."""

from __future__ import annotations

from typing import Any

from ..models import PipelineState, Settings
from ..services.catalog_retrieval import load_product_catalog
from .common import progress


def node_load_catalog(state: PipelineState) -> dict[str, Any]:
    settings: Settings = state["settings"]
    progress(
        state,
        "Каталог эталонов: загрузка product_json и эмбеддингов…",
        0.6,
    )
    catalog, catalog_warnings = load_product_catalog(
        settings.cache_dir,
        embedding_model=settings.embedding_model,
        device=settings.embedding_device,
    )
    progress(
        state,
        (
            f"Каталог готов ({catalog.size} продуктов из "
            f"{len(catalog.indexed_files)} файлов)"
        ),
        0.7,
    )
    return {
        "product_catalog": catalog,
        "indexed_files": catalog.indexed_files,
        "index_reused": True,
        "warnings": catalog_warnings,
    }
