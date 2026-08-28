"""Тесты VL-индексации эталонов."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz

from ai_tender.models import DocumentKind, IndexingContext, IndexingStatus, Settings
from ai_tender.services.indexing import AssetVlIndexer, index_asset_files
from ai_tender.services.indexing.persistance import (
    delete_product_artifacts,
    json_path_for,
    load_product_embeddings,
    load_product_index,
)


def _write_pdf(path: Path, pages: list[str]) -> None:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(page.rect + (36, 36, -36, -36), text, fontsize=11)
    doc.save(str(path))
    doc.close()


def _page_num(prompt: str) -> int:
    for n in range(1, 20):
        if f"СТРАНИЦА: {n}" in prompt or f"ТЕКУЩАЯ СТРАНИЦА: {n}" in prompt:
            return n
    return 0


def _fake_vl(**kwargs: Any) -> tuple[dict[str, Any] | None, int]:
    prompt = str(kwargs.get("prompt") or "")
    page = _page_num(prompt)

    if "ПРОХОД_СКАНИРОВАНИЯ" in prompt:
        return (
            {
                "summary": f"страница {page}",
                "has_products": page in {1, 2},
            },
            1,
        )

    if "ПРОХОД_СЛИЯНИЯ" in prompt:
        if page == 1:
            return (
                {
                    "catalog_name": "Каталог тест",
                    "update": True,
                    "products_add": [
                        {
                            "model": "UPS-1000",
                            "manufacturer": "ACME",
                            "category": "ИБП",
                            "canonical_desc": "ИБП UPS-1000 1000ВА",
                            "raw_chunk": "UPS-1000 | 1000ВА",
                            "characteristics": ["напряжение 220 В"],
                            "standards": [],
                        }
                    ],
                    "products_patch": [],
                },
                1,
            )
        if page == 2:
            return (
                {
                    "catalog_name": "",
                    "update": True,
                    "products_add": [
                        {
                            "model": "UPS-2000",
                            "canonical_desc": "ИБП UPS-2000",
                            "raw_chunk": "UPS-2000",
                            "characteristics": [],
                            "standards": [],
                        }
                    ],
                    "products_patch": [
                        {
                            "match_model": "UPS-1000",
                            "characteristics_add": ["мощность 1000 ВА"],
                        }
                    ],
                },
                1,
            )
        return {"catalog_name": "", "update": False, "products_add": [], "products_patch": []}, 1

    return {"catalog_name": "", "update": False, "products_add": [], "products_patch": []}, 1


class _FakeEmbedder:
    def get_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0, 0.0] for t in texts]


def test_vl_indexer_page_by_page(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    cache = tmp_path / "cache"
    assets.mkdir()
    _write_pdf(
        assets / "cat.pdf",
        [
            "страница один UPS-1000 " * 20,
            "страница два UPS-2000 " * 20,
            "обложка без продукции " * 20,
        ],
    )
    ctx = IndexingContext(
        assets_path=assets,
        cache_dir=cache,
        embedding_model="fake",
        vl_max_pages=10,
        extra={"vl_complete": _fake_vl, "embed_model": _FakeEmbedder()},
    )
    result = AssetVlIndexer().index(
        assets / "cat.pdf", relative_path="cat.pdf", context=ctx
    )
    assert result.status == IndexingStatus.indexed
    assert result.doc_kind == DocumentKind.asset
    assert result.details["product_count"] == 2
    assert result.details["product_pages"] == [1, 2]

    loaded = load_product_index(cache, "cat.pdf")
    assert loaded is not None
    assert loaded.catalog_name == "Каталог тест"
    assert loaded.product_pages == [1, 2]
    assert [p.model for p in loaded.products] == ["UPS-1000", "UPS-2000"]
    assert "мощность 1000 ВА" in loaded.products[0].characteristics
    assert loaded.products[0].source.page == 1
    assert loaded.products[1].source.page == 2

    emb = load_product_embeddings(cache, "cat.pdf")
    assert emb is not None
    ids, vectors, model = emb
    assert len(ids) == 2
    assert vectors.shape == (2, 3)
    assert model == "fake"

    assert delete_product_artifacts(cache, "cat.pdf")
    assert load_product_index(cache, "cat.pdf") is None
    assert not json_path_for(cache, "cat.pdf").exists()


def test_index_asset_files_orchestrate(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    cache = tmp_path / "cache"
    assets.mkdir()
    _write_pdf(assets / "a.pdf", ["модель A " * 30])

    settings = Settings(
        cache_dir=cache,
        embedding_model="fake",
        vl_max_pages=5,
    )
    results, _ = index_asset_files(
        assets,
        ["a.pdf"],
        cache_dir=cache,
        settings=settings,
        extra={"vl_complete": _fake_vl, "embed_model": _FakeEmbedder()},
    )
    assert len(results) == 1
    assert results[0].status == IndexingStatus.indexed
    assert load_product_index(cache, "a.pdf") is not None


def test_empty_pages_fail_without_products(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    cache = tmp_path / "cache"
    assets.mkdir()
    _write_pdf(assets / "empty.pdf", ["обложка без моделей"])

    def _empty_vl(**kwargs: Any) -> tuple[dict[str, Any] | None, int]:
        prompt = str(kwargs.get("prompt") or "")
        if "ПРОХОД_СКАНИРОВАНИЯ" in prompt:
            return {"summary": "обложка", "has_products": False}, 1
        return {
            "catalog_name": "",
            "update": False,
            "products_add": [],
            "products_patch": [],
        }, 1

    ctx = IndexingContext(
        assets_path=assets,
        cache_dir=cache,
        extra={"vl_complete": _empty_vl, "embed_model": _FakeEmbedder()},
    )
    result = AssetVlIndexer().index(
        assets / "empty.pdf", relative_path="empty.pdf", context=ctx
    )
    assert result.status == IndexingStatus.failed
    assert result.details["product_count"] == 0
    assert result.details["product_pages"] == []
    assert load_product_index(cache, "empty.pdf") is None
    assert not json_path_for(cache, "empty.pdf").exists()
