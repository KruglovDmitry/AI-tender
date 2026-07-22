"""Нода: индекс эталонов."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..index import indexed_file_paths, load_or_build_assets_index
from ..models import Settings
from ..state import PipelineState
from .common import progress


def node_build_assets_index(state: PipelineState) -> dict[str, Any]:
    settings: Settings = state["settings"]
    progress(
        state,
        "Индекс эталонов: чтение PDF и эмбеддинги (после смены файлов — полная пересборка)",
        0.6,
    )
    assets_index, asset_nodes, asset_warnings, index_reused = load_or_build_assets_index(
        Path(state["assets_path"]),
        settings.cache_dir,
        settings.embedding_model,
        settings.chunk_size,
        settings.chunk_overlap,
        settings.embedding_device,
        ocr_enabled=settings.ocr_enabled,
        ocr_languages=settings.ocr_languages,
    )
    files = indexed_file_paths(asset_nodes)
    progress(
        state,
        (
            f"Индекс эталонов готов ({len(asset_nodes)} чанков, "
            f"{'из кэша' if index_reused else 'построен заново'})"
        ),
        0.7,
    )
    return {
        "assets_index": assets_index,
        "indexed_files": files,
        "index_reused": index_reused,
        "warnings": asset_warnings,
    }
