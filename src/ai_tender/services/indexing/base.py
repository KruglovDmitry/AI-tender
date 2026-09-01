"""Постраничная VL-индексация эталонов (скан → слияние с контекстом)."""

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
    MIN_PRODUCTS_PER_CATALOG,
    delete_product_artifacts,
    embeddings_paths,
    save_product_embeddings,
    save_product_index,
)
from .qwen_catalog import try_extract_catalog_qwen
from .vl_client import complete_vl_json

PAGE_SCAN_SCHEMA_HINT = '{"summary":"","has_products":true}'


def _index_log(rel: str, message: str) -> None:
    print(f"[assets index] {rel}: {message}", flush=True)


def _emit_progress(
    context: IndexingContext,
    *,
    rel: str,
    phase: str,
    page: int,
    total: int,
    detail: str,
) -> None:
    _index_log(rel, f"{phase} {page}/{total} — {detail}")
    callback = context.extra.get("on_index_progress")
    if callable(callback):
        callback(phase, page, total, detail)

PAGE_MERGE_SCHEMA_HINT = (
    '{"catalog_name":"","update":false,'
    '"products_add":[{"model":"","manufacturer":"","category":"",'
    '"canonical_desc":"","raw_chunk":"","characteristics":[""],"standards":[]}],'
    '"products_patch":[{"match_model":"","canonical_desc":"",'
    '"characteristics_add":[""],"standards_add":[]}]}'
)

PAGE_SCAN_PROMPT = """\
ПРОХОД_СКАНИРОВАНИЯ страницы PDF.
Кратко опиши содержание (1–2 предложения) и реши, похоже ли, что на странице
есть данные по продукции.

Верни ТОЛЬКО JSON:
{{
  "summary": "краткое описание страницы",
  "has_products": true
}}

has_products — ориентир для лога (true/false). На втором проходе все страницы
всё равно будут просмотрены с накопленным контекстом.

ИСТОЧНИК: {filename}
СТРАНИЦА: {page}
"""

PAGE_MERGE_PROMPT = """\
ПРОХОД_СЛИЯНИЯ: проанализируй ТЕКУЩУЮ страницу с учётом контекста каталога.
Если на странице нет полезных данных по продуктам — ничего не меняй.

Верни ТОЛЬКО JSON:
{{
  "catalog_name": "название документа если видно, иначе пустая строка",
  "update": false,
  "products_add": [],
  "products_patch": []
}}

Правила:
- update=false и пустые массивы, если страница не добавляет важных данных.
- update=true, если нужно добавить новые продукты или дополнить уже известные.
- products_add — только НОВЫЕ позиции, которых ещё нет в KNOWN PRODUCTS.
- products_patch — дополнение уже известных (match_model = точное model из списка).
  В patch можно передать: canonical_desc (если лучше/подробнее), manufacturer,
  category, raw_chunk, characteristics_add[], standards_add[].
  Не повторяй уже известные характеристики.
- Каждый отдельный тип изделия/модуль/артикул = отдельный product.
- Серия/платформа и её модули — разные products.
- Поле id не заполняй.
- ОБЯЗАТЕЛЬНО: если на странице есть таблица моделей, типоразмеры, артикулы или
  технические характеристики изделий — добавь хотя бы один product в products_add
  (или дополни через products_patch). Каталог не может остаться пустым, если
  документ описывает продукцию. Серию/линейку тоже указывай как model, если
  конкретного артикула на странице нет.

Поля нового product: model, manufacturer, category, canonical_desc, raw_chunk,
characteristics[], standards[].
characteristics — фразы КАК В ТЕКСТЕ, без выдумок.

ИСТОЧНИК: {filename}
ТЕКУЩАЯ СТРАНИЦА: {page}
НАЗВАНИЕ ДОКУМЕНТА: {catalog_name}

ОПИСАНИЯ СТРАНИЦ (скан):
{page_summaries}

KNOWN PRODUCTS:
{known_products}
"""


class AssetVlIndexer:
    """PDF → скан страниц → постраничное слияние продуктов с контекстом."""

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
            _index_log(relative_path, "сохранение JSON и эмбеддингов…")
            _emit_progress(
                context,
                rel=relative_path,
                phase="persist",
                page=0,
                total=0,
                detail="сохранение JSON и эмбеддингов",
            )
            warnings.extend(self.persist(doc_index, context))
        except Exception as exc:
            return IndexingResult(
                relative_path=relative_path,
                doc_kind=self.kind,
                status=IndexingStatus.failed,
                message=f"«{label}»: ошибка индексации «{name}»: {exc}",
            )

        n = len(doc_index.products)
        if n < MIN_PRODUCTS_PER_CATALOG:
            removed = delete_product_artifacts(context.cache_dir, relative_path)
            if removed:
                _index_log(relative_path, f"удалены пустые артефакты: {len(removed)} файл(ов)")
            fail_msg = (
                f"«{name}» — требуется минимум {MIN_PRODUCTS_PER_CATALOG} продукт(ов), "
                f"извлечено {n}. Переиндексируйте после исправления VL."
            )
            warnings.append(fail_msg)
            return IndexingResult(
                relative_path=relative_path,
                doc_kind=self.kind,
                status=IndexingStatus.failed,
                message=fail_msg,
                details={
                    "product_count": n,
                    "catalog_name": doc_index.catalog_name,
                    "product_pages": list(doc_index.product_pages),
                    "warnings": warnings,
                    "min_products_required": MIN_PRODUCTS_PER_CATALOG,
                },
            )

        status = IndexingStatus.indexed
        qwen_mode = any(
            "extraction_mode=qwen_whole_file" in w for w in doc_index.warnings
        ) or any(w.startswith("Qwen catalog route=") for w in warnings)
        if qwen_mode:
            msg = f"«{name}» — {n} продукт(ов) (Qwen whole-file)"
        else:
            msg = f"«{name}» — {n} продукт(ов) (VL, слияние по страницам)"
        if doc_index.catalog_name:
            msg += f", документ «{doc_index.catalog_name}»"
        if doc_index.product_pages:
            msg += f", страницы с изменениями: {doc_index.product_pages}"
        return IndexingResult(
            relative_path=relative_path,
            doc_kind=self.kind,
            status=status,
            message=msg,
            details={
                "product_count": n,
                "catalog_name": doc_index.catalog_name,
                "product_pages": list(doc_index.product_pages),
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

        qwen_index, qwen_warnings = try_extract_catalog_qwen(
            path,
            relative_path=rel,
            context=context,
        )
        warnings.extend(qwen_warnings)
        if qwen_index is not None:
            return qwen_index, warnings

        if path.suffix.lower() != ".pdf":
            return (
                ProductDocumentIndex(
                    source_file=rel, doc_kind=self.kind, products=[]
                ),
                warnings + [f"Ожидается PDF для VL: {rel}"],
            )

        filename = Path(rel).name
        catalog_name = ""
        products: list[Product] = []
        touched_pages: list[int] = []

        try:
            doc = fitz.open(path)
        except Exception as exc:
            return (
                ProductDocumentIndex(
                    source_file=rel, doc_kind=self.kind, products=[]
                ),
                [f"Не удалось открыть PDF {filename}: {exc}"],
            )

        with doc:
            n_pages = min(context.vl_max_pages, doc.page_count)
            scale = max(0.5, float(context.vl_image_scale))
            matrix = fitz.Matrix(scale, scale)
            vl_call = context.extra.get("vl_complete") or complete_vl_json
            scan_max_tokens = min(256, context.vl_max_output_tokens)
            total_vl_calls = n_pages * 2

            _index_log(
                rel,
                "старт "
                f"pages={n_pages}/{doc.page_count} scale={scale} "
                f"vl={context.vl_base_url} model={context.vl_model!r} "
                f"timeout={context.vl_timeout_sec}s scan_tokens={scan_max_tokens} "
                f"merge_tokens={context.vl_max_output_tokens} "
                f"vl_вызовов≈{total_vl_calls} (2 прохода)",
            )
            _emit_progress(
                context,
                rel=rel,
                phase="start",
                page=0,
                total=n_pages,
                detail=f"индексация {filename}, {n_pages} стр.",
            )

            # --- Проход 1: краткие описания всех страниц ---
            scan_rows: list[dict[str, Any]] = []
            for i in range(n_pages):
                page_num = i + 1
                _emit_progress(
                    context,
                    rel=rel,
                    phase="scan",
                    page=page_num,
                    total=n_pages,
                    detail=f"ожидание VL (scan стр. {page_num})",
                )
                try:
                    pix = doc.load_page(i).get_pixmap(matrix=matrix, alpha=False)
                    image_bytes = pix.tobytes("jpeg")
                    scan_prompt = PAGE_SCAN_PROMPT.format(
                        filename=filename, page=page_num
                    )
                    scan_data, _ = vl_call(
                        image_bytes=image_bytes,
                        prompt=scan_prompt,
                        base_url=context.vl_base_url,
                        model=context.vl_model,
                        api_key=context.vl_api_key,
                        image_mime="image/jpeg",
                        max_tokens=scan_max_tokens,
                        timeout_sec=context.vl_timeout_sec,
                        structure_hint=PAGE_SCAN_SCHEMA_HINT,
                        log_context=f"{rel} scan p{page_num}/{n_pages}",
                    )
                except Exception as exc:
                    warnings.append(f"скан стр. {page_num}: {exc}")
                    _index_log(rel, f"scan p{page_num}/{n_pages} ERROR: {exc}")
                    scan_rows.append(
                        {"page": page_num, "summary": "", "has_products": True}
                    )
                    continue

                summary, has_products = self.parse_scan_payload(scan_data)
                if scan_data is None:
                    warnings.append(f"скан стр. {page_num}: VL JSON не разобран")
                    has_products = True
                scan_rows.append(
                    {
                        "page": page_num,
                        "summary": summary,
                        "has_products": has_products,
                    }
                )
                _index_log(
                    rel,
                    f"scan p{page_num}/{n_pages} ok "
                    f"has_products={has_products} summary={summary[:80]!r}",
                )

            scan_hint_pages = [
                int(row["page"]) for row in scan_rows if row.get("has_products")
            ]
            _index_log(rel, f"scan завершён hint_pages={scan_hint_pages}")
            warnings.append(f"scan_hint_pages={scan_hint_pages}")

            page_summaries_text = self.format_page_summaries(scan_rows)

            # --- Проход 2: все страницы + контекст, merge ---
            _index_log(rel, "merge: второй проход по всем страницам")
            for i in range(n_pages):
                page_num = i + 1
                known_n = len(products)
                _emit_progress(
                    context,
                    rel=rel,
                    phase="merge",
                    page=page_num,
                    total=n_pages,
                    detail=f"ожидание VL (merge стр. {page_num}, продуктов={known_n})",
                )
                try:
                    pix = doc.load_page(i).get_pixmap(matrix=matrix, alpha=False)
                    image_bytes = pix.tobytes("jpeg")
                    prompt = PAGE_MERGE_PROMPT.format(
                        filename=filename,
                        page=page_num,
                        catalog_name=catalog_name or "(ещё не известно)",
                        page_summaries=page_summaries_text,
                        known_products=self.format_known_products(products),
                    )
                    _index_log(
                        rel,
                        f"merge p{page_num}/{n_pages} → VL "
                        f"image={len(image_bytes)}B prompt={len(prompt)}ch "
                        f"known_products={known_n}",
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
                        structure_hint=PAGE_MERGE_SCHEMA_HINT,
                        log_context=f"{rel} merge p{page_num}/{n_pages}",
                    )
                except Exception as exc:
                    warnings.append(f"стр. {page_num}: {exc}")
                    _index_log(rel, f"merge p{page_num}/{n_pages} ERROR: {exc}")
                    continue
                if data is None:
                    warnings.append(f"стр. {page_num}: VL JSON не разобран")
                    _index_log(rel, f"merge p{page_num}/{n_pages} JSON не разобран")
                    continue

                changed, catalog_name = self.apply_page_merge(
                    data,
                    products=products,
                    catalog_name=catalog_name,
                    source_file=rel,
                    page=page_num,
                )
                if changed:
                    touched_pages.append(page_num)
                _index_log(
                    rel,
                    f"merge p{page_num}/{n_pages} "
                    f"changed={changed} products={len(products)} "
                    f"catalog={catalog_name!r}",
                )
            _index_log(rel, f"merge завершён products={len(products)}")
            _emit_progress(
                context,
                rel=rel,
                phase="done",
                page=n_pages,
                total=n_pages,
                detail=f"VL готово, продуктов={len(products)}",
            )

        if not products:
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
                product_pages=touched_pages,
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

        index.embedding_model = context.embedding_model
        rel = index.source_file
        if len(index.products) < MIN_PRODUCTS_PER_CATALOG:
            delete_product_artifacts(context.cache_dir, rel)
            warnings.append(
                f"Каталог не сохранён: требуется минимум {MIN_PRODUCTS_PER_CATALOG} продукт(ов)"
            )
            return warnings

        _index_log(rel, f"persist: JSON + embeddings для {len(index.products)} продуктов")
        save_product_index(context.cache_dir, index)

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

    def parse_scan_payload(
        self, data: dict[str, Any] | None
    ) -> tuple[str, bool]:
        """→ (summary, has_products)."""
        if not data:
            return "", False
        summary = str(data.get("summary") or "").strip()
        raw_flag = data.get("has_products")
        if isinstance(raw_flag, bool):
            has_products = raw_flag
        elif isinstance(raw_flag, (int, float)):
            has_products = bool(raw_flag)
        elif isinstance(raw_flag, str):
            has_products = raw_flag.strip().lower() in {
                "1",
                "true",
                "yes",
                "да",
            }
        else:
            has_products = False
        return summary, has_products

    def apply_page_merge(
        self,
        data: dict[str, Any],
        *,
        products: list[Product],
        catalog_name: str,
        source_file: str,
        page: int,
    ) -> tuple[bool, str]:
        """Применяет add/patch со страницы. → (changed, catalog_name)."""
        name = str(data.get("catalog_name") or "").strip()
        if name and not catalog_name:
            catalog_name = name

        update_flag = data.get("update")
        if isinstance(update_flag, str):
            do_update = update_flag.strip().lower() in {"1", "true", "yes", "да"}
        else:
            do_update = bool(update_flag)

        adds = data.get("products_add") or []
        patches = data.get("products_patch") or []
        if not isinstance(adds, list):
            adds = []
        if not isinstance(patches, list):
            patches = []

        # Совместимость: старый формат {"products": [...]} без update.
        legacy = data.get("products")
        if isinstance(legacy, list) and legacy and not adds and not patches:
            do_update = True
            adds = legacy

        if not do_update and not adds and not patches:
            return False, catalog_name

        changed = False
        by_model = {p.model.strip().lower(): p for p in products if p.model.strip()}

        for item in patches:
            if not isinstance(item, dict):
                continue
            if self.patch_product(item, products=products, by_model=by_model):
                changed = True

        for idx, item in enumerate(adds):
            if not isinstance(item, dict):
                continue
            model = str(item.get("model") or "").strip()
            key = model.lower()
            if key and key in by_model:
                # уже есть — как patch
                patch = {
                    "match_model": model,
                    "canonical_desc": item.get("canonical_desc"),
                    "manufacturer": item.get("manufacturer"),
                    "category": item.get("category"),
                    "raw_chunk": item.get("raw_chunk"),
                    "characteristics_add": item.get("characteristics")
                    or item.get("characteristics_add"),
                    "standards_add": item.get("standards")
                    or item.get("standards_add"),
                }
                if self.patch_product(patch, products=products, by_model=by_model):
                    changed = True
                continue
            product = self.coerce_product(
                item,
                source_file=source_file,
                page=page,
                idx=len(products) + idx,
            )
            if product is None:
                continue
            products.append(product)
            if product.model.strip():
                by_model[product.model.strip().lower()] = product
            changed = True

        return changed, catalog_name

    def patch_product(
        self,
        patch: dict[str, Any],
        *,
        products: list[Product],
        by_model: dict[str, Product],
    ) -> bool:
        match = str(
            patch.get("match_model") or patch.get("model") or ""
        ).strip()
        if not match:
            return False
        product = by_model.get(match.lower())
        if product is None:
            # частичное совпадение по началу model
            for key, candidate in by_model.items():
                if key.startswith(match.lower()) or match.lower().startswith(key):
                    product = candidate
                    break
        if product is None:
            return False

        changed = False
        desc = str(patch.get("canonical_desc") or "").strip()
        if desc and desc != product.canonical_desc:
            if len(desc) >= len(product.canonical_desc):
                product.canonical_desc = desc
                changed = True
        manufacturer = str(patch.get("manufacturer") or "").strip()
        if manufacturer and not product.manufacturer:
            product.manufacturer = manufacturer
            changed = True
        category = str(patch.get("category") or "").strip()
        if category and not product.category:
            product.category = category
            changed = True
        raw_chunk = str(patch.get("raw_chunk") or "").strip()
        if raw_chunk and not product.raw_chunk:
            product.raw_chunk = raw_chunk
            changed = True

        chars_add = self.coerce_characteristics(
            patch.get("characteristics_add") or patch.get("characteristics")
        )
        if chars_add:
            existing = {c.lower() for c in product.characteristics}
            for item in chars_add:
                if item.lower() not in existing:
                    product.characteristics.append(item)
                    existing.add(item.lower())
                    changed = True

        standards_raw = patch.get("standards_add") or patch.get("standards") or []
        if isinstance(standards_raw, list):
            existing_std = {s.lower() for s in product.standards}
            for item in standards_raw:
                text = str(item).strip()
                if text and text.lower() not in existing_std:
                    product.standards.append(text)
                    existing_std.add(text.lower())
                    changed = True

        del products  # list mutated in place via product refs
        return changed

    def format_page_summaries(self, scan_rows: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for row in scan_rows:
            summary = str(row.get("summary") or "").strip() or "(нет описания)"
            lines.append(f"- стр. {row.get('page')}: {summary}")
        return "\n".join(lines) if lines else "(пусто)"

    def format_known_products(self, products: list[Product]) -> str:
        if not products:
            return "(пока пусто)"
        lines: list[str] = []
        for product in products:
            chars = "; ".join(product.characteristics[:8])
            if len(product.characteristics) > 8:
                chars += "; …"
            desc = (product.canonical_desc or "")[:220]
            lines.append(
                f"- model={product.model!r} | category={product.category!r} | "
                f"desc={desc!r} | characteristics=[{chars}]"
            )
        # ограничим размер контекста
        text = "\n".join(lines)
        if len(text) > 12000:
            text = text[:12000] + "\n…"
        return text

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
