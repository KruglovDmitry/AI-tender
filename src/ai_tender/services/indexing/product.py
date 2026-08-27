"""Индексатор описания/паспорта продукта."""

from __future__ import annotations

from pathlib import Path

from ...models import DocumentKind, IndexingContext, Product, ProductDocumentIndex
from ...providers import complete_llm_json
from .base import ATTR_RULES, PRODUCT_SCHEMA_HINT, StructuredDocumentIndexer

PRODUCT_FILE_PROMPT = """\
Извлеки описание изделия/изделий из паспорта или описания типа продукта.
Обычно 1 продукт; если явно несколько моделей одной серии — несколько.

Верни ТОЛЬКО JSON:
{{
  "catalog_name": "",
  "products": [ /* 1..N объектов Product */ ]
}}

Поля Product: id (можно пустой), model, manufacturer, category,
canonical_desc, raw_chunk, attributes[], standards[].

{attr_rules}

ИСТОЧНИК: {filename}

ТЕКСТ ДОКУМЕНТА (ограничен первыми страницами):
{text}
"""


class ProductDocumentIndexer(StructuredDocumentIndexer):
    kind = DocumentKind.product
    max_pages = 25
    file_max_chars = 40_000

    def join_pages_text(self, pages: list[tuple[int, str]], *, max_chars: int | None = None,) -> str:
        if max_chars is None:
            max_chars = self.file_max_chars
        parts = [
            f"--- стр. {num} ---\n{text}" for num, text in pages if text.strip()
        ]
        text = "\n\n".join(parts).strip()
        return text[:max_chars] if len(text) > max_chars else text

    def extract(self, relative_path: str, context: IndexingContext,) -> tuple[ProductDocumentIndex, list[str]]:
        rel = relative_path.replace("\\", "/").lstrip("/")
        pages, warnings = self.read_document_pages(rel, context)
        text = self.join_pages_text(pages)
        filename = Path(rel).name
        products: list[Product] = []
        if text.strip():
            prompt = PRODUCT_FILE_PROMPT.format(
                attr_rules=ATTR_RULES,
                filename=filename,
                text=text,
            )
            data, _ = complete_llm_json(context.llm, prompt, structure_hint=PRODUCT_SCHEMA_HINT,)
            _, products = self.parse_products_payload(data, source_file=rel, page=None)
            for product in products:
                if product.source.page is None and pages:
                    product.source.page = pages[0][0]
        else:
            warnings.append(f"Пустой текст для извлечения продукта: {rel}")

        if not products:
            warnings.append(f"LLM не извлёк продукты из «{rel}»")

        return (
            ProductDocumentIndex(
                source_file=rel,
                doc_kind=self.kind,
                catalog_name="",
                products=products,
                warnings=list(warnings),
            ),
            warnings,
        )

    def success_message(self, name: str, label: str, doc_index: ProductDocumentIndex, product_count: int,) -> str:
        del doc_index
        if product_count:
            return (
                f"Тип «{label}»: «{name}» — {product_count} продукт(ов) проиндексировано"
            )
        return f"Тип «{label}»: «{name}» — продукт не извлечён"