"""CatalogExtractor: extract каталога + адаптер → ProductDocumentIndex."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..models import (
    CatalogExtractResult,
    DocumentKind,
    Product,
    ProductDocumentIndex,
    ProductRecord,
    ProductSource,
)
from .base_extract import QwenExtractor


class CatalogExtractor(QwenExtractor):
    """Whole-file extract эталонного каталога + маппинг в ProductDocumentIndex."""

    PROMPT = """
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
- Каждая строка таблицы заказа / конкретный артикул / модификация = отдельный product с полным обозначением в model.
- Не дублируй базовый тип (напр. «МИР С-04») отдельным product, если ниже уже извлечены модификации, исполнения или артикулы с тем же базовым типом.
- Общее описание серии из начала раздела — не отдельный product, только конкретные позиции.
- characteristics — фразы КАК В ТЕКСТЕ, без выдумок.
- Если на странице/в документе есть таблица моделей — извлеки все строки с полными обозначениями.
- Не заполняй id — его проставит система.
""".strip()

    SCHEMA_HINT = '{"catalog_name":"","products":[{"model":"","characteristics":[]}]}'

    def extract(self, path: Path, *, prompt: str | None = None) -> CatalogExtractResult:
        prompt = self.PROMPT if prompt is None else prompt
        gate = self.gate(path)
        if self.should_use_vl(gate, path):
            return self.extract_vl_json(
                path,
                prompt=prompt,
                schema_hint=self.SCHEMA_HINT,
                result_cls=CatalogExtractResult,
                merge_parts=self.merge_results,
            )
        if not gate.sends_to_qwen_doc:
            raise ValueError(gate.reason)
        return self.extract_json(
            path,
            gate=gate,
            prompt=prompt,
            schema_hint=self.SCHEMA_HINT,
            result_cls=CatalogExtractResult,
            merge_parts=self.merge_results,
        )

    @staticmethod
    def _product_key(product: ProductRecord) -> str:
        model = (product.model or "").strip().casefold()
        if model:
            return f"m:{model}"
        desc = (product.canonical_desc or product.raw_chunk or "").strip().casefold()
        return f"d:{desc[:120]}"

    @classmethod
    def merge_results(cls, parts: list[CatalogExtractResult]) -> CatalogExtractResult:
        catalog_name = next((p.catalog_name.strip() for p in parts if p.catalog_name.strip()), "")
        seen: set[str] = set()
        products: list[ProductRecord] = []
        for part in parts:
            for product in part.products:
                key = cls._product_key(product)
                if not key or key in {"m:", "d:"}:
                    continue
                if key in seen:
                    continue
                seen.add(key)
                products.append(product)
        return CatalogExtractResult(catalog_name=catalog_name, products=products)

    @staticmethod
    def _stable_id(source_file: str, page: int | None, model: str, idx: int) -> str:
        raw = f"{source_file}|{page or 0}|{model.strip().lower()}|{idx}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _is_more_specific_variant(base: str, candidate: str) -> bool:
        base = base.strip()
        candidate = candidate.strip()
        if not base or candidate.casefold() == base.casefold():
            return False
        if not candidate.casefold().startswith(base.casefold()):
            return False
        if len(candidate) <= len(base):
            return False
        suffix = candidate[len(base) : len(base) + 1]
        return suffix in {" ", ".", "-", "/", "("}

    @staticmethod
    def _merge_characteristics(existing: Product, incoming: Product) -> None:
        seen = {c.casefold() for c in existing.characteristics}
        for item in incoming.characteristics:
            text = item.strip()
            if text and text.casefold() not in seen:
                existing.characteristics.append(text)
                seen.add(text.casefold())
        seen_std = {s.casefold() for s in existing.standards}
        for item in incoming.standards:
            text = item.strip()
            if text and text.casefold() not in seen_std:
                existing.standards.append(text)
                seen_std.add(text.casefold())
        if len(incoming.canonical_desc) > len(existing.canonical_desc):
            existing.canonical_desc = incoming.canonical_desc
        if len(incoming.raw_chunk) > len(existing.raw_chunk):
            existing.raw_chunk = incoming.raw_chunk
        if not existing.category and incoming.category:
            existing.category = incoming.category

    @classmethod
    def _merge_duplicate_models(cls, products: list[Product]) -> list[Product]:
        merged: dict[tuple[str, str], Product] = {}
        for product in products:
            key = (product.model.strip().casefold(), product.manufacturer.strip().casefold())
            if key not in merged:
                merged[key] = product
                continue
            cls._merge_characteristics(merged[key], product)
        return list(merged.values())

    @classmethod
    def _drop_redundant_base_models(cls, products: list[Product]) -> list[Product]:
        redundant_ids: set[str] = set()
        for product in products:
            base = product.model.strip()
            if not base:
                continue
            base_cat = product.category.strip().casefold()
            has_specific = False
            for other in products:
                if other.id == product.id:
                    continue
                other_model = other.model.strip()
                if other.category.strip().casefold() != base_cat:
                    continue
                if cls._is_more_specific_variant(base, other_model):
                    has_specific = True
                    break
            if has_specific:
                redundant_ids.add(product.id)
        return [p for p in products if p.id not in redundant_ids]

    @classmethod
    def dedupe_products(cls, products: list[Product]) -> list[Product]:
        """Слияние одинаковых model и удаление базовых типов при наличии уточнений."""
        if not products:
            return []
        merged = cls._merge_duplicate_models(products)
        return cls._drop_redundant_base_models(merged)

    @classmethod
    def _record_to_product(
        cls,
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
            id=cls._stable_id(source_file, None, model or canonical, idx),
            model=model,
            manufacturer=record.manufacturer.strip(),
            category=record.category.strip(),
            canonical_desc=canonical or model,
            raw_chunk=record.raw_chunk.strip(),
            source=ProductSource(catalog_id=source_file, page=None),
            characteristics=[c.strip() for c in record.characteristics if c.strip()],
            standards=standards,
        )

    @classmethod
    def result_to_index(
        cls,
        result: CatalogExtractResult,
        *,
        source_file: str,
        doc_kind: DocumentKind = DocumentKind.asset,
    ) -> ProductDocumentIndex:
        products: list[Product] = []
        for idx, record in enumerate(result.products):
            product = cls._record_to_product(record, source_file=source_file, idx=idx)
            if product is not None:
                products.append(product)
        before = len(products)
        products = cls.dedupe_products(products)
        warnings = ["extraction_mode=qwen_whole_file"]
        if before > len(products):
            warnings.append(f"deduped_products={before}->{len(products)}")
        return ProductDocumentIndex(
            source_file=source_file,
            doc_kind=doc_kind,
            catalog_name=result.catalog_name.strip(),
            products=products,
            product_pages=[],
            warnings=warnings,
        )
