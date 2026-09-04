"""Whole-file извлечение через Qwen DashScope (doc / long / scan)."""

from ..models import (
    CatalogExtractResult,
    ProductRecord,
    ScopeItemExtract,
    TenderExtractResult,
)
from .base_extract import ExtractRoute, GateDecision, can_send_to_qwen

__all__ = [
    "CatalogExtractResult",
    "ExtractRoute",
    "GateDecision",
    "ProductRecord",
    "ScopeItemExtract",
    "TenderExtractResult",
    "can_send_to_qwen",
]
