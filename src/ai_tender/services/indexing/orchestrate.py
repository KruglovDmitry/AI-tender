"""Оркестрация: PDF эталонов → постраничный VL extract (без classify)."""

from __future__ import annotations

from pathlib import Path

from ...models import IndexingContext, IndexingResult, Settings
from .base import AssetVlIndexer
from .locks import mark_indexing


def index_asset_files(
    assets_path: Path,
    relative_paths: list[str],
    *,
    cache_dir: Path,
    settings: Settings | None = None,
    extra: dict | None = None,
) -> tuple[list[IndexingResult], list[str]]:
    """Индексирует каждый PDF постранично через VL. → (results, warnings)."""
    assets_path = assets_path.expanduser().resolve()
    cache_dir = cache_dir.expanduser().resolve()

    if settings is None:
        context = IndexingContext(
            assets_path=assets_path,
            cache_dir=cache_dir,
            extra=dict(extra or {}),
        )
    else:
        context = IndexingContext(
            assets_path=assets_path,
            cache_dir=cache_dir,
            embedding_model=settings.embedding_model,
            embedding_device=settings.embedding_device,
            vl_base_url=settings.vl_base_url,
            vl_model=settings.vl_model,
            vl_api_key=settings.vl_api_key,
            vl_max_pages=settings.vl_max_pages,
            vl_image_scale=settings.vl_image_scale,
            vl_timeout_sec=settings.vl_timeout_sec,
            vl_max_output_tokens=settings.vl_max_output_tokens,
            extra=dict(extra or {}),
        )

    indexer = AssetVlIndexer()
    results: list[IndexingResult] = []
    warnings: list[str] = []

    for raw in relative_paths:
        rel = raw.replace("\\", "/").lstrip("/")
        if not rel:
            continue
        path = assets_path / rel
        mark_indexing(rel, active=True)
        try:
            result = indexer.index(path, relative_path=rel, context=context)
        finally:
            mark_indexing(rel, active=False)
        results.append(result)
        warnings.extend(result.details.get("warnings") or [])

    return results, warnings
