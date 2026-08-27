"""Постраничная VL-индексация эталонов (без классификации типа)."""

from ...models import (
    DOCUMENT_KIND_LABELS,
    DocumentKind,
    IndexingContext,
    IndexingResult,
    IndexingStatus,
)
from .base import AssetVlIndexer
from .orchestrate import index_asset_files
from .persistance import delete_product_artifacts

__all__ = [
    "DOCUMENT_KIND_LABELS",
    "AssetVlIndexer",
    "DocumentKind",
    "IndexingContext",
    "IndexingResult",
    "IndexingStatus",
    "delete_product_artifacts",
    "index_asset_files",
]
