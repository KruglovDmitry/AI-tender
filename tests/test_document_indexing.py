"""Тесты Qwen-индексации эталонов."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from ai_tender.models import CatalogExtractResult, ProductRecord
from ai_tender.models import DocumentKind, IndexingContext, IndexingStatus, Settings
from ai_tender.services.index_service import index_asset_files, index_catalog_file
from ai_tender.services.catalog_service import (
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


class _FakeEmbedder:
    def get_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0, 0.0] for t in texts]


def _fake_qwen_catalog(path: Path) -> CatalogExtractResult:
    del path
    return CatalogExtractResult(
        catalog_name="Каталог Qwen",
        products=[
            ProductRecord(
                model="UPS-1000",
                manufacturer="ACME",
                canonical_desc="ИБП UPS-1000 1000ВА",
                raw_chunk="UPS-1000 | 1000ВА",
                characteristics=["напряжение 220 В"],
            ),
            ProductRecord(
                model="UPS-2000",
                canonical_desc="ИБП UPS-2000",
                raw_chunk="UPS-2000",
            ),
        ],
    )


def test_qwen_catalog_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_tender.extract import base_extract

    monkeypatch.setattr(base_extract, "dashscope_api_key", lambda: "test-key")

    assets = tmp_path / "assets"
    cache = tmp_path / "cache"
    assets.mkdir()
    _write_pdf(assets / "cat.pdf", ["страница с моделями UPS"])

    settings = Settings(
        cache_dir=cache,
        embedding_model="fake",
        extract_backend="qwen",
    )
    ctx = IndexingContext(
        assets_path=assets,
        cache_dir=cache,
        embedding_model="fake",
        extra={
            "settings": settings,
            "qwen_catalog_extract": _fake_qwen_catalog,
            "embed_model": _FakeEmbedder(),
        },
    )
    result = index_catalog_file(
        assets / "cat.pdf", relative_path="cat.pdf", context=ctx
    )
    assert result.status == IndexingStatus.indexed
    assert "Qwen whole-file" in result.message
    index = load_product_index(cache, "cat.pdf")
    assert index is not None
    assert len(index.products) == 2
    assert index.catalog_name == "Каталог Qwen"
    assert load_product_embeddings(cache, "cat.pdf") is not None

    assert delete_product_artifacts(cache, "cat.pdf")
    assert load_product_index(cache, "cat.pdf") is None
    assert not json_path_for(cache, "cat.pdf").exists()


def test_index_asset_files_orchestrate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_tender.extract import base_extract

    monkeypatch.setattr(base_extract, "dashscope_api_key", lambda: "test-key")

    assets = tmp_path / "assets"
    cache = tmp_path / "cache"
    assets.mkdir()
    _write_pdf(assets / "a.pdf", ["модель A " * 30])

    settings = Settings(
        cache_dir=cache,
        embedding_model="fake",
        extract_backend="qwen",
    )
    results, _ = index_asset_files(
        assets,
        ["a.pdf"],
        cache_dir=cache,
        settings=settings,
        extra={"qwen_catalog_extract": _fake_qwen_catalog, "embed_model": _FakeEmbedder()},
    )
    assert len(results) == 1
    assert results[0].status == IndexingStatus.indexed
    assert load_product_index(cache, "a.pdf") is not None


def test_empty_qwen_extract_still_indexes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_tender.extract import base_extract

    monkeypatch.setattr(base_extract, "dashscope_api_key", lambda: "test-key")

    assets = tmp_path / "assets"
    cache = tmp_path / "cache"
    assets.mkdir()
    _write_pdf(assets / "empty.pdf", ["обложка без моделей"])

    def _empty(_path: Path) -> CatalogExtractResult:
        return CatalogExtractResult(catalog_name="", products=[])

    settings = Settings(cache_dir=cache, embedding_model="fake", extract_backend="qwen")
    ctx = IndexingContext(
        assets_path=assets,
        cache_dir=cache,
        embedding_model="fake",
        extra={
            "settings": settings,
            "qwen_catalog_extract": _empty,
            "embed_model": _FakeEmbedder(),
        },
    )
    result = index_catalog_file(
        assets / "empty.pdf", relative_path="empty.pdf", context=ctx
    )
    assert result.status == IndexingStatus.indexed
    assert result.details["product_count"] == 0
    index = load_product_index(cache, "empty.pdf")
    assert index is not None
    assert index.products == []
