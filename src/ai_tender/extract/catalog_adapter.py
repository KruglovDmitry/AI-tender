"""Адаптер CatalogExtractResult → ProductDocumentIndex."""

from __future__ import annotations

import hashlib

from ..models import DocumentKind, Product, ProductDocumentIndex, ProductSource
from .schemas import CatalogExtractResult, ProductRecord


CATALOG_QWEN_PROMPT = """
Ты индексатор эталонного каталога продукции. По ПОЛНОМУ файлу извлеки все позиции продукции.

Верни JSON:
{
  "catalog_name": "название документа/каталога если видно",
  "products": [
    {
      "model": "артикул/модель/тип изделия",
      "manufacturer": "производитель",
      "category": "категория",
      "canonical_desc": "краткое каноническое описание",
      "raw_chunk": "фрагмент исходного текста",
      "characteristics": ["характеристика как в документе"],
      "standards": ["ГОСТ …"]
    }
  ]
}

Правила:
- Каждый отдельный тип изделия/модуль/артикул = отдельный product.
- characteristics — фразы КАК В ТЕКСТЕ, без выдумок.
- Если на странице/в документе есть таблица моделей — извлеки все строки.
- Не заполняй id — его проставит система.
""".strip()


def _stable_id(source_file: str, page: int | None, model: str, idx: int) -> str:
    raw = f"{source_file}|{page or 0}|{model.strip().lower()}|{idx}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _record_to_product(
    record: ProductRecord,
    *,
    source_file: str,
    idx: int,
) -> Product | None:
    model = record.model.strip()
    canonical = record.canonical_desc.strip()
    if not model and not canonical:
        return None
    standards = [str(s).strip() for s in record.standards if str(s).strip()]
    return Product(
        id=_stable_id(source_file, None, model or canonical, idx),
        model=model,
        manufacturer=record.manufacturer.strip(),
        category=record.category.strip(),
        canonical_desc=canonical or model,
        raw_chunk=record.raw_chunk.strip(),
        source=ProductSource(catalog_id=source_file, page=None),
        characteristics=[c.strip() for c in record.characteristics if c.strip()],
        standards=standards,
    )


def catalog_result_to_index(
    result: CatalogExtractResult,
    *,
    source_file: str,
    doc_kind: DocumentKind = DocumentKind.asset,
) -> ProductDocumentIndex:
    products: list[Product] = []
    for idx, record in enumerate(result.products):
        product = _record_to_product(record, source_file=source_file, idx=idx)
        if product is not None:
            products.append(product)
    return ProductDocumentIndex(
        source_file=source_file,
        doc_kind=doc_kind,
        catalog_name=result.catalog_name.strip(),
        products=products,
        product_pages=[],
        warnings=["extraction_mode=qwen_whole_file"],
    )
