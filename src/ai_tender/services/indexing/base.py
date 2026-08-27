"""Базовый индексатор документов эталонов."""

from __future__ import annotations
import fitz
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from ...models import (
    CANONICAL_ATTRIBUTE_KEYS,
    DOCUMENT_KIND_LABELS,
    AttributeType,
    AttributeValueNorm,
    DocumentKind,
    IndexingContext,
    IndexingResult,
    IndexingStatus,
    Product,
    ProductAttribute,
    ProductDocumentIndex,
    ProductSource,
)
from .persistance import (
    delete_product_artifacts,
    save_product_embeddings,
    save_product_index,
)

_KEYS_LIST = ", ".join(CANONICAL_ATTRIBUTE_KEYS)

PRODUCT_SCHEMA_HINT = (
    '{"products":[{"id":"","model":"","manufacturer":"","category":"",'
    '"canonical_desc":"","raw_chunk":"","attributes":[{"key_canonical":"voltage",'
    '"key_raw":"","value_norm":{"num":220,"unit":"V","tol":null},"value_raw":"",'
    '"type":"numeric_range|categorical|bool|standard_ref|text"}],'
    '"standards":["ГОСТ …"]}],"catalog_name":""}'
)

ATTR_RULES = f"""\
Атрибуты:
- key_canonical — ТОЛЬКО из списка: {_KEYS_LIST}
- key_raw — как в документе
- value_raw — исходная строка
- value_norm: num/num_max/unit/tol для чисел; text для категорий; bool_value для bool
- type: numeric_range | categorical | bool | standard_ref | text

canonical_desc — 1–3 предложения на русском: модель, назначение, ключевые параметры
(для семантического поиска). Без воды.
raw_chunk — короткий исходный фрагмент (строка таблицы / абзац), не весь документ.
"""


class StructuredDocumentIndexer(ABC):
    """Каталог/паспорт: extract → JSON + embeddings → IndexingResult."""

    kind: DocumentKind
    max_pages: int = 10
    page_max_chars: int = 8_000

    @staticmethod
    def stable_id(source_file: str, page: int | None, model: str, idx: int) -> str:
        raw = f"{source_file}|{page or 0}|{model.strip().lower()}|{idx}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def as_float(value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def as_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def as_bbox(value: Any) -> list[float] | None:
        if not isinstance(value, (list, tuple)) or len(value) < 4:
            return None
        try:
            return [float(x) for x in value[:4]]
        except (TypeError, ValueError):
            return None

    @abstractmethod
    def extract(self, relative_path: str, context: IndexingContext,) -> tuple[ProductDocumentIndex, list[str]]:
        """Извлечь продукты из файла (специфика типа документа)."""

    @abstractmethod
    def success_message(self, name: str, label: str, doc_index: ProductDocumentIndex, product_count: int,) -> str:
        """Сообщение для UI после extract/persist."""

    def index(self, path: Path, *, relative_path: str, context: IndexingContext,) -> IndexingResult:
        del path
        label = DOCUMENT_KIND_LABELS[self.kind]
        name = Path(relative_path).name
        if context.llm is None:
            return IndexingResult(
                relative_path=relative_path,
                doc_kind=self.kind,
                status=IndexingStatus.failed,
                message=f"«{name}»: нет LLM для индексации",
            )
        if context.cache_dir is None:
            return IndexingResult(
                relative_path=relative_path,
                doc_kind=self.kind,
                status=IndexingStatus.failed,
                message=f"«{name}»: не задан cache_dir",
            )

        try:
            doc_index, warnings = self.extract(relative_path, context)
            persist_warnings = self.persist(doc_index, context)
            warnings.extend(persist_warnings)
        except Exception as exc:
            return IndexingResult(
                relative_path=relative_path,
                doc_kind=self.kind,
                status=IndexingStatus.failed,
                message=f"Тип «{label}»: ошибка индексации «{name}»: {exc}",
            )

        return self.build_result(relative_path, doc_index, warnings)

    def read_document_pages(self, relative_path: str, context: IndexingContext, *, max_pages: int | None = None, max_chars_per_page: int | None = None,) -> tuple[list[tuple[int, str]], list[str]]:      
        if max_pages is None:
            max_pages = self.max_pages
        if max_chars_per_page is None:
            max_chars_per_page = self.page_max_chars
        assets_path = context.assets_path.expanduser().resolve()
        rel = relative_path.replace("\\", "/").lstrip("/")
        path = assets_path / rel
        if not path.is_file():
            return [], [f"Файл не найден: {rel}"]
        if path.suffix.lower() != ".pdf":
            return [], [
                f"Эталоны индексируются только из PDF "
                f"(получен {path.suffix or 'без расширения'}): {rel}"
            ]

        warnings: list[str] = []
        try:
            doc = fitz.open(path)
        except Exception as exc:
            return [], [f"Не удалось открыть PDF {path.name}: {exc}"]
        try:
            n = min(max_pages, doc.page_count)
            pages: list[tuple[int, str]] = []
            for i in range(n):
                raw = doc.load_page(i).get_text() or ""
                text = raw.strip()
                if len(text) > max_chars_per_page:
                    text = text[:max_chars_per_page]
                    warnings.append(
                        f"Стр. {i + 1} обрезана до {max_chars_per_page} символов: "
                        f"{path.name}"
                    )
                pages.append((i + 1, text))
        finally:
            doc.close()
        return pages, warnings

    def build_result(self, relative_path: str, doc_index: ProductDocumentIndex, warnings: list[str],) -> IndexingResult:
        label = DOCUMENT_KIND_LABELS[self.kind]
        name = Path(relative_path).name
        n = len(doc_index.products)
        status = IndexingStatus.indexed if n else IndexingStatus.failed
        message = self.success_message(name, label, doc_index, n)
        details: dict[str, Any] = {"product_count": n, "warnings": warnings}
        if doc_index.catalog_name:
            details["catalog_name"] = doc_index.catalog_name
        return IndexingResult(
            relative_path=relative_path,
            doc_kind=self.kind,
            status=status,
            message=message,
            details=details,
        )

    def persist(self, index: ProductDocumentIndex, context: IndexingContext,) -> list[str]:
        warnings: list[str] = []
        if context.cache_dir is None:
            raise ValueError("IndexingContext.cache_dir обязателен для сохранения индекса")

        delete_product_artifacts(context.cache_dir, index.source_file)
        index.embedding_model = context.embedding_model
        save_product_index(context.cache_dir, index)
        if not index.products:
            return warnings

        texts = [
            (p.canonical_desc or p.model or p.raw_chunk or p.id).strip() or p.id
            for p in index.products
        ]
        ids = [p.id for p in index.products]
        vectors = self.embed_texts(texts, context)
        save_product_embeddings(
            context.cache_dir,
            index.source_file,
            ids,
            vectors,
            embedding_model=context.embedding_model,
        )
        return warnings

    def embed_texts(self, texts: list[str], context: IndexingContext) -> np.ndarray:     
        embedder = context.extra.get("embed_model")
        if embedder is None:
            from ..index_service import configure_embeddings
            embedder = configure_embeddings(context.embedding_model, context.embedding_device)

        if hasattr(embedder, "get_text_embedding_batch"):
            vectors = embedder.get_text_embedding_batch(texts)
        else:
            vectors = [embedder.get_text_embedding(t) for t in texts]
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def parse_products_payload(
        self,
        data: dict[str, Any] | None,
        *,
        source_file: str,
        page: int | None,
    ) -> tuple[str, list[Product]]:
        if not data:
            return "", []
        catalog_name = str(data.get("catalog_name") or "").strip()
        raw_list = data.get("products") or []
        if not isinstance(raw_list, list):
            return catalog_name, []
        products: list[Product] = []
        for idx, item in enumerate(raw_list):
            if not isinstance(item, dict):
                continue
            product = self.coerce_product(
                item, source_file=source_file, page=page, idx=idx
            )
            if product:
                products.append(product)
        return catalog_name, products

    def coerce_product(
        self,
        raw: dict[str, Any],
        *,
        source_file: str,
        page: int | None,
        idx: int,
    ) -> Product | None:
        model = str(raw.get("model") or "").strip()
        canonical = str(raw.get("canonical_desc") or "").strip()
        if not model and not canonical:
            return None
        attrs_raw = raw.get("attributes") or []
        attributes: list[ProductAttribute] = []
        if isinstance(attrs_raw, list):
            for item in attrs_raw:
                if isinstance(item, dict):
                    attr = self.coerce_attr(item)
                    if attr:
                        attributes.append(attr)
        standards_raw = raw.get("standards") or []
        standards = (
            [str(s).strip() for s in standards_raw if str(s).strip()]
            if isinstance(standards_raw, list)
            else []
        )
        src = raw.get("source") if isinstance(raw.get("source"), dict) else {}
        product_id = str(raw.get("id") or "").strip() or self.stable_id(
            source_file, page, model or canonical, idx
        )
        return Product(
            id=product_id,
            model=model,
            manufacturer=str(raw.get("manufacturer") or "").strip(),
            category=str(raw.get("category") or "").strip(),
            canonical_desc=canonical or model,
            raw_chunk=str(raw.get("raw_chunk") or "").strip(),
            source=ProductSource(
                catalog_id=str(src.get("catalog_id") or source_file).strip()
                or source_file,
                version=str(src.get("version") or "").strip(),
                page=page if page is not None else self.as_int(src.get("page")),
                bbox=self.as_bbox(src.get("bbox")),
            ),
            attributes=attributes,
            standards=standards,
        )

    def coerce_attr(self, raw: dict[str, Any]) -> ProductAttribute | None:
        key = str(raw.get("key_canonical") or "other").strip().lower()
        if key not in CANONICAL_ATTRIBUTE_KEYS:
            key = "other"
        type_raw = str(raw.get("type") or "text").strip().lower()
        try:
            attr_type = AttributeType(type_raw)
        except ValueError:
            attr_type = AttributeType.text
        vn = raw.get("value_norm") or {}
        if not isinstance(vn, dict):
            vn = {}
        return ProductAttribute(
            key_canonical=key,
            key_raw=str(raw.get("key_raw") or "").strip(),
            value_raw=str(raw.get("value_raw") or "").strip(),
            type=attr_type,
            value_norm=AttributeValueNorm(
                num=self.as_float(vn.get("num")),
                num_max=self.as_float(vn.get("num_max")),
                unit=(
                    str(vn["unit"]).strip() if vn.get("unit") is not None else None
                ),
                tol=self.as_float(vn.get("tol")),
                text=(
                    str(vn["text"]).strip() if vn.get("text") is not None else None
                ),
                bool_value=(
                    vn.get("bool_value")
                    if isinstance(vn.get("bool_value"), bool)
                    else None
                ),
            ),
        )