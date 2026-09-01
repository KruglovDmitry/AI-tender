"""Whole-file извлечение через Qwen DashScope (doc / long / scan)."""

from .qwen_cache import QwenExtractCache, content_sha256
from .qwen_gate import ExtractRoute, GateDecision, can_send_to_qwen
from .schemas import (
    EXTRACT_SCHEMA_VERSION,
    CatalogExtractResult,
    ProductRecord,
    ScopeItemExtract,
    TenderExtractResult,
)

__all__ = [
    "EXTRACT_SCHEMA_VERSION",
    "CatalogExtractResult",
    "ExtractRoute",
    "GateDecision",
    "ProductRecord",
    "QwenExtractCache",
    "ScopeItemExtract",
    "TenderExtractResult",
    "can_send_to_qwen",
    "content_sha256",
]
