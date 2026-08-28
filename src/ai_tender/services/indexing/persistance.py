"""Хранение JSON-индекса продуктов и отдельных эмбеддингов."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ...models import ProductDocumentIndex

JSON_DIR_NAME = "product_json"
EMBED_DIR_NAME = "product_embeddings"


def product_json_root(cache_dir: Path) -> Path:
    return cache_dir.expanduser().resolve() / JSON_DIR_NAME


def product_embeddings_root(cache_dir: Path) -> Path:
    return cache_dir.expanduser().resolve() / EMBED_DIR_NAME


def _mirror_stem(relative_path: str) -> Path:
    """assets/foo/bar.pdf → foo/bar (без суффикса исходника)."""
    rel = relative_path.replace("\\", "/").lstrip("/")
    return Path(rel).with_suffix("")


def json_path_for(cache_dir: Path, relative_path: str) -> Path:
    return product_json_root(cache_dir) / _mirror_stem(relative_path).with_suffix(".json")


def embeddings_paths(cache_dir: Path, relative_path: str) -> tuple[Path, Path]:
    """→ (vectors.npy, ids.json)."""
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
    """Удаляет JSON и эмбеддинги для исходного файла. → список удалённых путей."""
    removed: list[str] = []
    for path in (
        json_path_for(cache_dir, relative_path),
        *embeddings_paths(cache_dir, relative_path),
    ):
        if path.is_file():
            path.unlink()
            removed.append(str(path))
    return removed
