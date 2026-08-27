"""Индексация эталонов по типу документа."""

from ...models import (
    DOCUMENT_KIND_LABELS,
    DocumentKind,
    IndexingContext,
    IndexingResult,
    IndexingStatus,
)
from .classify import classify_document_kind
from .orchestrate import index_asset_files
from .persistance import delete_product_artifacts

__all__ = [
    "DOCUMENT_KIND_LABELS",
    "DocumentKind",
    "IndexingContext",
    "IndexingResult",
    "IndexingStatus",
    "classify_document_kind",
    "delete_product_artifacts",
    "index_asset_files",
]
