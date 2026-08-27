"""Индексатор каталогов: постраничное LLM-извлечение продуктов."""

from __future__ import annotations

from pathlib import Path

from ...models import DocumentKind, IndexingContext, Product, ProductDocumentIndex
from ...providers import complete_llm_json
from .base import ATTR_RULES, PRODUCT_SCHEMA_HINT, StructuredDocumentIndexer

CATALOG_PAGE_PROMPT = """\
Извлеки продукты со страницы каталога. Если на странице нет описаний/строк продуктов —
верни пустой массив products.

Верни ТОЛЬКО JSON:
{{
  "catalog_name": "название каталога, если видно на странице, иначе пустая строка",
  "products": [ /* 0..N объектов Product */ ]
}}

Поля Product: id (можно пустой), model, manufacturer, category,
canonical_desc, raw_chunk, attributes[], standards[].

{attr_rules}

ИСТОЧНИК: {filename}
СТРАНИЦА: {page}
НАЗВАНИЕ КАТАЛОГА (если уже известно): {catalog_name}

ТЕКСТ СТРАНИЦЫ:
{text}
"""


class CatalogDocumentIndexer(StructuredDocumentIndexer):
    kind = DocumentKind.catalog
    max_pages = 10

    def extract(self, relative_path: str, context: IndexingContext,) -> tuple[ProductDocumentIndex, list[str]]:
        rel = relative_path.replace("\\", "/").lstrip("/")
        pages, warnings = self.read_document_pages(rel, context)
        filename = Path(rel).name
        catalog_name = ""
        products: list[Product] = []
        empty_pages = 0

        for page_num, text in pages:
            if not text.strip():
                empty_pages += 1
                continue
            prompt = CATALOG_PAGE_PROMPT.format(
                attr_rules=ATTR_RULES,
                filename=filename,
                page=page_num,
                catalog_name=catalog_name or "(ещё не известно)",
                text=text,
            )
            data, _ = complete_llm_json(context.llm, prompt, structure_hint=PRODUCT_SCHEMA_HINT,)
            page_name, page_products = self.parse_products_payload(data, source_file=rel, page=page_num)
            if page_name and not catalog_name:
                catalog_name = page_name
            if not page_products:
                empty_pages += 1
                continue
            products.extend(page_products)

        if empty_pages and not products:
            warnings.append(f"В первых {len(pages)} стр. каталога «{rel}» продукты не найдены")
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

    def success_message(self, name: str, label: str, doc_index: ProductDocumentIndex, product_count: int,) -> str:
        if product_count:
            return (
                f"Тип «{label}»: «{name}» — {product_count} продукт(ов), "
                f"каталог «{doc_index.catalog_name}»"
            )
        return f"Тип «{label}»: «{name}» — продукты не извлечены"