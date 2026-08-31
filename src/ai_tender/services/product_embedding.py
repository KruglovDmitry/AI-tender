"""Текст для эмбеддинга продуктов VL-каталога."""

from __future__ import annotations

from ..models import Product


def embedding_text(product: Product) -> str:
    """Модель, описание и характеристики продукта."""
    parts: list[str] = []
    if product.model:
        parts.append(product.model.strip())
    if product.manufacturer:
        parts.append(product.manufacturer.strip())
    if product.category:
        parts.append(product.category.strip())
    desc = (product.canonical_desc or "").strip()
    if desc:
        parts.append(desc)
    chunk = (product.raw_chunk or "").strip()
    if chunk and chunk != desc:
        parts.append(chunk)
    if product.characteristics:
        parts.append("; ".join(c.strip() for c in product.characteristics if c.strip()))
    if product.standards:
        parts.append("; ".join(s.strip() for s in product.standards if s.strip()))
    text = "\n".join(p for p in parts if p).strip()
    return text or product.id
