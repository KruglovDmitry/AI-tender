"""Постраничная VL-индексация эталонов (без классификации типа)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import fitz
import numpy as np

from ...models import (
    DOCUMENT_KIND_LABELS,
    DocumentKind,
    IndexingContext,
    IndexingResult,
    IndexingStatus,
    Product,
    ProductDocumentIndex,
    ProductSource,
)
from .persistance import (
    delete_product_artifacts,
    save_product_embeddings,
    save_product_index,
)
from .vl_client import complete_vl_json

PRODUCT_SCHEMA_HINT = (
    '{"catalog_name":"","products":[{"model":"","manufacturer":"","category":"",'
    '"canonical_desc":"","raw_chunk":"","characteristics":[""],"standards":[]}]}'
)

PAGE_EXTRACT_PROMPT = """\
Извлеки продукты и их характеристики со страницы технического документа (изображение).
Если на странице нет продуктов/моделей/артикулов — верни пустой массив products.

Верни ТОЛЬКО JSON:
{{
  "catalog_name": "название каталога/документа, если видно, иначе пустая строка",
  "products": []
}}

Поля Product: model, manufacturer, category, canonical_desc, raw_chunk,
characteristics[], standards[]. Поле id не заполняй.

Правила границ продукта (важно):
- Каждый отдельный тип изделия, модуль, артикул или строка таблицы = отдельный product.
- Если на странице есть и базовая станция/платформа/серия, и её модули/варианты —
  заведи ОТДЕЛЬНЫЙ product на саму станцию/платформу/серию И отдельные products
  на каждую входящую позицию. Не оставляй только модули и не сливай всё в один.
- Не дублируй один и тот же product. Различающиеся варианты — отдельные model.
- model должен однозначно отличать позицию на странице.

characteristics — массив строк: основные характеристики КАК В ТЕКСТЕ документа
(короткие фразы/значения со страницы). Без ключей, без нормализации, без выдумок.
Пустой массив, если характеристик нет.

canonical_desc — 1–2 предложения: что это. Без воды.
raw_chunk — короткий исходный фрагмент (заголовок блока / строка таблицы).

ИСТОЧНИК: {filename}
СТРАНИЦА: {page}
НАЗВАНИЕ ДОКУМЕНТА (если уже известно): {catalog_name}
"""


class AssetVlIndexer:
    """PDF → страница за страницей в VL → JSON продуктов + embeddings."""

    kind = DocumentKind.asset

    def index(
        self,
        path: Path,
        *,
        relative_path: str,
        context: IndexingContext,
    ) -> IndexingResult:
        del path
        label = DOCUMENT_KIND_LABELS[self.kind]
        name = Path(relative_path).name
        if context.cache_dir is None:
            return IndexingResult(
                relative_path=relative_path,
                doc_kind=self.kind,
                status=IndexingStatus.failed,
                message=f"«{name}»: не задан cache_dir",
            )

        try:
            doc_index, warnings = self.extract(relative_path, context)
            warnings.extend(self.persist(doc_index, context))
        except Exception as exc:
            return IndexingResult(
                relative_path=relative_path,
                doc_kind=self.kind,
                status=IndexingStatus.failed,
                message=f"«{label}»: ошибка индексации «{name}»: {exc}",
            )

        n = len(doc_index.products)
        status = IndexingStatus.indexed if n else IndexingStatus.failed
        msg = (
            f"«{name}» — {n} продукт(ов) (VL, постранично)"
            if n
            else f"«{name}» — продукты не извлечены"
        )
        if doc_index.catalog_name:
            msg += f", документ «{doc_index.catalog_name}»"
        return IndexingResult(
            relative_path=relative_path,
            doc_kind=self.kind,
            status=status,
            message=msg,
            details={
                "product_count": n,
                "catalog_name": doc_index.catalog_name,
                "warnings": warnings,
            },
        )

    def extract(
        self,
        relative_path: str,
        context: IndexingContext,
    ) -> tuple[ProductDocumentIndex, list[str]]:
        rel = relative_path.replace("\\", "/").lstrip("/")
        path = context.assets_path.expanduser().resolve() / rel
        warnings: list[str] = []
        if not path.is_file():
            return (
                ProductDocumentIndex(
                    source_file=rel, doc_kind=self.kind, products=[]
                ),
                [f"Файл не найден: {rel}"],
            )
        if path.suffix.lower() != ".pdf":
            return (
                ProductDocumentIndex(
                    source_file=rel, doc_kind=self.kind, products=[]
                ),
                [f"Ожидается PDF: {rel}"],
            )

        filename = Path(rel).name
        catalog_name = ""
        products: list[Product] = []
        empty_pages = 0

        try:
            doc = fitz.open(path)
        except Exception as exc:
            return (
                ProductDocumentIndex(
                    source_file=rel, doc_kind=self.kind, products=[]
                ),
                [f"Не удалось открыть PDF {filename}: {exc}"],
            )

        try:
            n_pages = min(context.vl_max_pages, doc.page_count)
            scale = max(0.5, float(context.vl_image_scale))
            matrix = fitz.Matrix(scale, scale)
            vl_call = context.extra.get("vl_complete") or complete_vl_json

            for i in range(n_pages):
                page_num = i + 1
                try:
                    pix = doc.load_page(i).get_pixmap(matrix=matrix, alpha=False)
                    image_bytes = pix.tobytes("jpeg")
                    prompt = PAGE_EXTRACT_PROMPT.format(
                        filename=filename,
                        page=page_num,
                        catalog_name=catalog_name or "(ещё не известно)",
                    )
                    data, _ = vl_call(
                        image_bytes=image_bytes,
                        prompt=prompt,
                        base_url=context.vl_base_url,
                        model=context.vl_model,
                        api_key=context.vl_api_key,
                        image_mime="image/jpeg",
                        max_tokens=context.vl_max_output_tokens,
                        timeout_sec=context.vl_timeout_sec,
                        structure_hint=PRODUCT_SCHEMA_HINT,
                    )
                except Exception as exc:
                    warnings.append(f"стр. {page_num}: {exc}")
                    empty_pages += 1
                    continue
                if data is None:
                    warnings.append(f"стр. {page_num}: VL JSON не разобран")
                    empty_pages += 1
                    continue
                page_name, page_products = self.parse_products_payload(
                    data, source_file=rel, page=page_num
                )
                if page_name and not catalog_name:
                    catalog_name = page_name
                if not page_products:
                    empty_pages += 1
                    continue
                products.extend(page_products)
        finally:
            doc.close()

        if empty_pages and not products:
            warnings.append(
                f"На обработанных страницах «{rel}» продукты не найдены"
            )
        if not catalog_name:
            catalog_name = filename

        return (
            ProductDocumentIndex(
                source_file=rel,
                doc_kind=self.kind,
                catalog_name=catalog_name,
                products=products,
                warnings=list(warnings),
            ),
            warnings,
        )

    def persist(
        self,
        index: ProductDocumentIndex,
        context: IndexingContext,
    ) -> list[str]:
        warnings: list[str] = []
        if context.cache_dir is None:
            raise ValueError("IndexingContext.cache_dir обязателен")

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

            embedder = configure_embeddings(
                context.embedding_model, context.embedding_device
            )
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
        characteristics = self.coerce_characteristics(
            raw.get("characteristics") or raw.get("attributes")
        )
        standards_raw = raw.get("standards") or []
        standards = (
            [str(s).strip() for s in standards_raw if str(s).strip()]
            if isinstance(standards_raw, list)
            else []
        )
        src = raw.get("source") if isinstance(raw.get("source"), dict) else {}
        # id всегда стабильный у нас: LLM часто ставит один id на семейство (дубли).
        product_id = self.stable_id(
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
            characteristics=characteristics,
            standards=standards,
        )

    def coerce_characteristics(self, raw: Any) -> list[str]:
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        for item in raw:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                # совместимость со старым форматом attributes
                parts = [
                    str(item.get("key") or item.get("key_canonical") or "").strip(),
                    str(item.get("value") or item.get("value_raw") or "").strip(),
                    str(item.get("unit") or "").strip(),
                ]
                text = " ".join(p for p in parts if p).strip()
            else:
                text = str(item).strip() if item is not None else ""
            if text:
                out.append(text)
        return out

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
