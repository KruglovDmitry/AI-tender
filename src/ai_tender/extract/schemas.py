"""Pydantic-схемы JSON-извлечения Qwen (каталог и тендер)."""

from __future__ import annotations

from pydantic import BaseModel, Field

# Инкремент при изменении промпта/полей — инвалидирует кэш.
EXTRACT_SCHEMA_VERSION = "1"


class ProductRecord(BaseModel):
    model: str = ""
    manufacturer: str = ""
    category: str = ""
    canonical_desc: str = ""
    raw_chunk: str = ""
    characteristics: list[str] = Field(default_factory=list)
    standards: list[str] = Field(default_factory=list)


class CatalogExtractResult(BaseModel):
    catalog_name: str = ""
    products: list[ProductRecord] = Field(default_factory=list)


class RequirementRecord(BaseModel):
    text: str
    quote: str = ""
    kind: str = "other"
    priority: int = Field(default=2, ge=0, le=3)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class ScopeItemExtract(BaseModel):
    name: str
    qty: float | int | None = None
    unit: str = ""
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    quote: str = ""
    requirements: list[RequirementRecord] = Field(default_factory=list)


class TenderExtractResult(BaseModel):
    scope_summary: str = ""
    scope_items: list[ScopeItemExtract] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_more_docs: bool = False
    missing_signals: str = ""
