"""Каталог эталонов: хранение product_json/embeddings + поиск для match."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..models import Evidence, Product, ProductDocumentIndex

JSON_DIR_NAME = "product_json"
EMBED_DIR_NAME = "product_embeddings"

MIN_PRODUCTS_PER_CATALOG = 1


def catalog_is_indexed(index: ProductDocumentIndex | None) -> bool:
    """Каталог считается проиндексированным только если есть ≥1 продукт."""
    return index is not None and len(index.products) >= MIN_PRODUCTS_PER_CATALOG


def product_json_root(cache_dir: Path) -> Path:
    return cache_dir.expanduser().resolve() / JSON_DIR_NAME


def product_embeddings_root(cache_dir: Path) -> Path:
    return cache_dir.expanduser().resolve() / EMBED_DIR_NAME


def _mirror_stem(relative_path: str) -> Path:
    rel = relative_path.replace("\\", "/").lstrip("/")
    return Path(rel).with_suffix("")


def json_path_for(cache_dir: Path, relative_path: str) -> Path:
    return product_json_root(cache_dir) / _mirror_stem(relative_path).with_suffix(".json")


def embeddings_paths(cache_dir: Path, relative_path: str) -> tuple[Path, Path]:
    stem = product_embeddings_root(cache_dir) / _mirror_stem(relative_path)
    return stem.with_suffix(".npy"), stem.with_suffix(".ids.json")


def save_product_index(cache_dir: Path, index: ProductDocumentIndex) -> Path:
    path = json_path_for(cache_dir, index.source_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        index.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
    )
    path.write_text(payload, encoding="utf-8")
    return path


def load_product_index(
    cache_dir: Path, relative_path: str
) -> ProductDocumentIndex | None:
    path = json_path_for(cache_dir, relative_path)
    if not path.is_file():
        return None
    return ProductDocumentIndex.model_validate_json(path.read_text(encoding="utf-8"))


def save_product_embeddings(
    cache_dir: Path,
    relative_path: str,
    product_ids: list[str],
    vectors: np.ndarray,
    *,
    embedding_model: str,
) -> None:
    npy_path, ids_path = embeddings_paths(cache_dir, relative_path)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2-D (n_products, dim)")
    if len(product_ids) != vectors.shape[0]:
        raise ValueError("product_ids length must match vectors rows")
    np.save(npy_path, vectors.astype(np.float32, copy=False))
    ids_path.write_text(
        json.dumps(
            {
                "embedding_model": embedding_model,
                "ids": product_ids,
                "dim": int(vectors.shape[1]) if vectors.size else 0,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_product_embeddings(
    cache_dir: Path, relative_path: str
) -> tuple[list[str], np.ndarray, str] | None:
    npy_path, ids_path = embeddings_paths(cache_dir, relative_path)
    if not npy_path.is_file() or not ids_path.is_file():
        return None
    meta = json.loads(ids_path.read_text(encoding="utf-8"))
    ids = list(meta.get("ids") or [])
    vectors = np.load(npy_path)
    return ids, vectors, str(meta.get("embedding_model") or "")


def delete_product_artifacts(cache_dir: Path, relative_path: str) -> list[str]:
    removed: list[str] = []
    for path in (
        json_path_for(cache_dir, relative_path),
        *embeddings_paths(cache_dir, relative_path),
    ):
        if path.is_file():
            path.unlink()
            removed.append(str(path))
    return removed


@dataclass
class CatalogProductHit:
    product: Product
    source_file: str
    catalog_name: str
    score: float


@dataclass
class ProductCatalog:
    products: list[Product] = field(default_factory=list)
    vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    product_ids: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    catalog_names: list[str] = field(default_factory=list)
    embedding_model: str = ""
    indexed_files: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.products)


def embedding_text(product: Product) -> str:
    """Текст для эмбеддинга и keyword-поиска: модель + описание + характеристики."""
    parts: list[str] = []
    for value in (
        product.model,
        product.manufacturer,
        product.category,
        product.canonical_desc,
    ):
        text = str(value or "").strip()
        if text and text not in parts:
            parts.append(text)
    if product.characteristics:
        parts.extend(str(c).strip() for c in product.characteristics if str(c).strip())
    if product.standards:
        parts.extend(str(s).strip() for s in product.standards if str(s).strip())
    chunk = str(product.raw_chunk or "").strip()
    if chunk and chunk not in parts and chunk not in (product.canonical_desc or ""):
        parts.append(chunk)
    return "\n".join(parts).strip() or product.id


def _query_terms(query: str) -> list[str]:
    import re

    terms: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[\w\d]+", query.casefold()):
        if len(token) < 3 or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms


def _keyword_overlap_score(query: str, product: Product) -> float:
    terms = _query_terms(query)
    if not terms:
        return 0.0
    haystack = embedding_text(product).casefold()
    hits = sum(1 for term in terms if term in haystack)
    return hits / len(terms)


def _combine_scores(vector_score: float, keyword_score: float) -> float:
    # Keyword помогает, когда в характеристиках явно указан аналог (Moxa NPort …).
    return 0.65 * vector_score + 0.35 * keyword_score


def format_product_quote(
    product: Product,
    *,
    catalog_name: str,
    source_file: str,
) -> str:
    lines: list[str] = []
    if catalog_name:
        lines.append(f"Каталог: {catalog_name}")
    if product.model:
        lines.append(f"Модель: {product.model}")
    if product.manufacturer:
        lines.append(f"Производитель: {product.manufacturer}")
    if product.category:
        lines.append(f"Категория: {product.category}")
    if product.canonical_desc:
        lines.append(product.canonical_desc)
    if product.characteristics:
        lines.append("Характеристики: " + "; ".join(product.characteristics[:16]))
    if product.standards:
        lines.append("Стандарты: " + "; ".join(product.standards))
    if product.raw_chunk:
        chunk = product.raw_chunk.strip()
        if chunk and chunk not in (product.canonical_desc or ""):
            lines.append(chunk)
    if not lines:
        lines.append(f"Источник: {source_file}")
    return "\n".join(lines)


def catalog_hit_to_evidence(hit: CatalogProductHit) -> Evidence:
    page = hit.product.source.page
    location = f"стр. {page}" if page else "каталог эталонов"
    file_ref = hit.product.source.catalog_id or hit.source_file
    quote = format_product_quote(
        hit.product,
        catalog_name=hit.catalog_name,
        source_file=hit.source_file,
    )
    if len(quote) > 1600:
        quote = quote[:1599] + "…"
    return Evidence(
        file=file_ref,
        location=location,
        quote=quote,
        score=round(float(hit.score), 4),
        page=page,
    )


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def load_product_catalog(
    cache_dir: Path,
    *,
    embedding_model: str,
    device: str | None = None,
) -> tuple[ProductCatalog, list[str]]:
    """Собирает каталог из product_json/ и product_embeddings/ (Qwen extract)."""
    from .index_service import configure_embeddings

    configure_embeddings(embedding_model, device)
    warnings: list[str] = []
    root = product_json_root(cache_dir)
    if not root.is_dir():
        warnings.append(
            "Каталог эталонов не найден (нет product_json/). Переиндексируйте эталоны через Qwen."
        )
        return ProductCatalog(embedding_model=embedding_model), warnings

    products: list[Product] = []
    vectors_rows: list[np.ndarray] = []
    product_ids: list[str] = []
    source_files: list[str] = []
    catalog_names: list[str] = []
    indexed: set[str] = set()

    for json_path in sorted(root.rglob("*.json")):
        try:
            doc_index = ProductDocumentIndex.model_validate_json(
                json_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            warnings.append(f"Не удалось прочитать {json_path.name}: {exc}")
            continue

        if not doc_index.products:
            continue

        emb = load_product_embeddings(cache_dir, doc_index.source_file)
        if emb is None:
            warnings.append(
                f"Нет эмбеддингов для «{doc_index.source_file}» — файл пропущен"
            )
            continue
        ids, matrix, emb_model = emb
        if emb_model and emb_model != embedding_model:
            warnings.append(
                f"«{doc_index.source_file}»: модель эмбеддингов {emb_model!r} "
                f"≠ {embedding_model!r}"
            )
        id_to_row = {pid: i for i, pid in enumerate(ids)}
        for product in doc_index.products:
            row_idx = id_to_row.get(product.id)
            if row_idx is None:
                warnings.append(
                    f"Пропущен продукт {product.id!r} в «{doc_index.source_file}» "
                    "(нет в embeddings)"
                )
                continue
            products.append(product)
            vectors_rows.append(matrix[row_idx])
            product_ids.append(product.id)
            source_files.append(doc_index.source_file)
            catalog_names.append(doc_index.catalog_name or doc_index.source_file)
        indexed.add(doc_index.source_file)

    if not products:
        warnings.append(
            "Каталог пуст: ни один эталон не содержит продуктов с эмбеддингами. "
            "Переиндексируйте файлы в разделе «Эталоны» (Qwen whole-file)."
        )
        return ProductCatalog(embedding_model=embedding_model), warnings

    stacked = np.vstack(vectors_rows).astype(np.float32, copy=False)
    return (
        ProductCatalog(
            products=products,
            vectors=_normalize_rows(stacked),
            product_ids=product_ids,
            source_files=source_files,
            catalog_names=catalog_names,
            embedding_model=embedding_model,
            indexed_files=sorted(indexed),
        ),
        warnings,
    )


def embed_query(text: str, *, embedding_model: str, device: str | None) -> np.ndarray:
    from .index_service import configure_embeddings

    embedder = configure_embeddings(embedding_model, device)
    vec = np.array(embedder.get_text_embedding(text), dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def search_catalog(
    catalog: ProductCatalog,
    query: str,
    *,
    top_k: int,
    embedding_model: str,
    device: str | None = None,
) -> list[CatalogProductHit]:
    if catalog.size == 0:
        return []
    q = embed_query(query, embedding_model=embedding_model, device=device)
    vector_scores = catalog.vectors @ q
    keyword_scores = np.array(
        [_keyword_overlap_score(query, catalog.products[i]) for i in range(catalog.size)],
        dtype=np.float32,
    )
    scores = np.array(
        [_combine_scores(float(vector_scores[i]), float(keyword_scores[i])) for i in range(catalog.size)],
        dtype=np.float32,
    )
    k = min(max(top_k, 1), catalog.size)
    if k == catalog.size:
        top_idx = np.argsort(-scores)
    else:
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
    hits: list[CatalogProductHit] = []
    for idx in top_idx:
        i = int(idx)
        hits.append(
            CatalogProductHit(
                product=catalog.products[i],
                source_file=catalog.source_files[i],
                catalog_name=catalog.catalog_names[i],
                score=float(scores[i]),
            )
        )
    return hits
