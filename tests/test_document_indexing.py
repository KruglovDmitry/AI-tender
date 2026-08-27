"""Тесты индексации каталога/продукта и product store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ai_tender.models import DocumentKind, IndexingContext, IndexingStatus
from ai_tender.services.indexing import classify_document_kind, index_asset_files
from ai_tender.services.indexing.catalog import CatalogDocumentIndexer
from ai_tender.services.indexing.other import OtherDocumentIndexer
from ai_tender.services.indexing.product import ProductDocumentIndexer
from ai_tender.services.indexing.persistance import (
    delete_product_artifacts,
    json_path_for,
    load_product_embeddings,
    load_product_index,
)


class _FakeEmbedder:
    def get_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0, 0.0] for t in texts]


class _FakeLLM:
    def __init__(self, mode: str = "other"):
        self.mode = mode
        self.calls = 0

    def complete(self, prompt: str) -> Any:
        self.calls += 1

        class _R:
            def __init__(self, text: str):
                self.text = text

            def __str__(self) -> str:
                return self.text

        if self.mode == "other":
            payload = {"doc_type": "other", "confidence": 0.9, "reason": "test"}
        elif self.mode == "catalog_page":
            payload = {
                "catalog_name": "Каталог ИБП",
                "products": [
                    {
                        "model": "UPS-1000",
                        "manufacturer": "ACME",
                        "category": "ИБП",
                        "canonical_desc": "ИБП UPS-1000 1000ВА, 220В",
                        "raw_chunk": "UPS-1000 | 1000ВА | 220В",
                        "attributes": [
                            {
                                "key_canonical": "voltage",
                                "key_raw": "Напряжение",
                                "value_raw": "220В",
                                "type": "numeric_range",
                                "value_norm": {"num": 220, "unit": "V"},
                            }
                        ],
                        "standards": ["IP20"],
                    }
                ],
            }
        elif self.mode == "catalog_empty":
            payload = {"catalog_name": "", "products": []}
        elif self.mode == "product_file":
            payload = {
                "catalog_name": "",
                "products": [
                    {
                        "model": "INV-500",
                        "manufacturer": "ACME",
                        "category": "инвертор",
                        "canonical_desc": "Инвертор INV-500 500Вт",
                        "raw_chunk": "INV-500 техническое описание",
                        "attributes": [],
                        "standards": [],
                    }
                ],
            }
        else:
            payload = {"doc_type": self.mode, "confidence": 0.9, "reason": "test"}
        return _R(json.dumps(payload, ensure_ascii=False))


def test_other_indexer_skips() -> None:
    result = OtherDocumentIndexer().index(
        Path("x.pdf"),
        relative_path="docs/x.pdf",
        context=IndexingContext(assets_path=Path(".")),
    )
    assert result.status == IndexingStatus.skipped
    assert result.doc_kind == DocumentKind.other


def test_classify_other(tmp_path: Path, monkeypatch) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "note.txt").write_text("note", encoding="utf-8")
    monkeypatch.setattr(
        "ai_tender.services.indexing.classify.preview_document_text",
        lambda *a, **k: ("служебный текст", []),
    )
    llm = _FakeLLM("other")
    kind, _, meta = classify_document_kind(llm, assets, "note.txt")  # type: ignore[arg-type]
    assert kind == DocumentKind.other
    assert meta["confidence"] == 0.9


def test_preview_skips_short_cover_pages(tmp_path: Path) -> None:
    import fitz

    assets = tmp_path / "assets"
    assets.mkdir()
    path = assets / "doc.pdf"
    doc = fitz.open()
    page0 = doc.new_page()
    page0.insert_text((72, 72), "SR33")  # короткая обложка
    page1 = doc.new_page()
    page1.insert_textbox(
        page1.rect + (36, 36, -36, -36),
        ("Модель SR33010 мощность 10 кВА. " * 40),
        fontsize=11,
    )
    doc.save(str(path))
    doc.close()

    from ai_tender.services.indexing.classify import preview_document_text

    text, warnings = preview_document_text(assets, "doc.pdf", max_pages=2)
    assert "SR33010" in text
    assert "--- стр. 2 ---" in text
    assert not warnings


def test_catalog_indexer_writes_json_and_embeddings(tmp_path: Path, monkeypatch) -> None:
    assets = tmp_path / "assets"
    cache = tmp_path / "cache"
    assets.mkdir()
    (assets / "cat.txt").write_text("UPS-1000 1000ВА 220В артикул", encoding="utf-8")

    monkeypatch.setattr(
        "ai_tender.services.indexing.base.StructuredDocumentIndexer.read_document_pages",
        lambda self, *a, **k: ([(1, "UPS-1000 1000ВА 220В")], []),
    )
    llm = _FakeLLM("catalog_page")
    ctx = IndexingContext(
        assets_path=assets,
        cache_dir=cache,
        llm=llm,
        embedding_model="fake",
        extra={"embed_model": _FakeEmbedder()},
    )
    result = CatalogDocumentIndexer().index(
        assets / "cat.txt", relative_path="cat.txt", context=ctx
    )
    assert result.status == IndexingStatus.indexed
    assert result.details["product_count"] == 1
    assert json_path_for(cache, "cat.txt").is_file()

    loaded = load_product_index(cache, "cat.txt")
    assert loaded is not None
    assert loaded.catalog_name == "Каталог ИБП"
    assert loaded.products[0].model == "UPS-1000"
    assert loaded.products[0].attributes[0].key_canonical == "voltage"

    emb = load_product_embeddings(cache, "cat.txt")
    assert emb is not None
    ids, vectors, model = emb
    assert ids == [loaded.products[0].id]
    assert vectors.shape == (1, 3)
    assert model == "fake"

    removed = delete_product_artifacts(cache, "cat.txt")
    assert removed
    assert load_product_index(cache, "cat.txt") is None


def test_catalog_skips_empty_pages(tmp_path: Path, monkeypatch) -> None:
    assets = tmp_path / "assets"
    cache = tmp_path / "cache"
    assets.mkdir()
    (assets / "c.txt").write_text("x", encoding="utf-8")

    pages = [(1, ""), (2, "модель UPS"), (3, "нет продуктов")]
    monkeypatch.setattr(
        "ai_tender.services.indexing.base.StructuredDocumentIndexer.read_document_pages",
        lambda self, *a, **k: (pages, []),
    )

    class _PageLLM(_FakeLLM):
        def complete(self, prompt: str) -> Any:
            if "модель UPS" in prompt:
                self.mode = "catalog_page"
            else:
                self.mode = "catalog_empty"
            return super().complete(prompt)

    llm = _PageLLM()
    ctx = IndexingContext(
        assets_path=assets,
        cache_dir=cache,
        llm=llm,
        extra={"embed_model": _FakeEmbedder()},
    )
    result = CatalogDocumentIndexer().index(
        assets / "c.txt", relative_path="c.txt", context=ctx
    )
    assert result.status == IndexingStatus.indexed
    # пустая стр.1 пропущена без LLM; стр.2 и стр.3 — два вызова
    assert llm.calls == 2


def test_product_indexer(tmp_path: Path, monkeypatch) -> None:
    assets = tmp_path / "assets"
    cache = tmp_path / "cache"
    assets.mkdir()
    (assets / "p.txt").write_text("INV-500 паспорт", encoding="utf-8")
    monkeypatch.setattr(
        "ai_tender.services.indexing.base.StructuredDocumentIndexer.read_document_pages",
        lambda self, *a, **k: ([(1, "INV-500 паспорт")], []),
    )
    ctx = IndexingContext(
        assets_path=assets,
        cache_dir=cache,
        llm=_FakeLLM("product_file"),
        extra={"embed_model": _FakeEmbedder()},
    )
    result = ProductDocumentIndexer().index(
        assets / "p.txt", relative_path="docs/p.txt", context=ctx
    )
    assert result.status == IndexingStatus.indexed
    loaded = load_product_index(cache, "docs/p.txt")
    assert loaded is not None
    assert loaded.doc_kind == DocumentKind.product
    assert loaded.products[0].model == "INV-500"


def test_index_asset_files_orchestrate(tmp_path: Path, monkeypatch) -> None:
    assets = tmp_path / "assets"
    cache = tmp_path / "cache"
    assets.mkdir()
    (assets / "cat.pdf").write_bytes(b"%PDF-1.1")

    monkeypatch.setattr(
        "ai_tender.services.indexing.classify.preview_document_text",
        lambda *a, **k: ("артикул модель таблица каталог", []),
    )
    monkeypatch.setattr(
        "ai_tender.services.indexing.base.StructuredDocumentIndexer.read_document_pages",
        lambda self, *a, **k: ([(1, "UPS-1000 220В")], []),
    )
    monkeypatch.setattr(
        "ai_tender.services.indexing.base.StructuredDocumentIndexer.embed_texts",
        lambda self, texts, context: np.asarray(
            [[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32
        ),
    )

    class _OrchLLM(_FakeLLM):
        def complete(self, prompt: str) -> Any:
            if "Определи тип технического документа" in prompt:
                self.mode = "catalog"
            else:
                self.mode = "catalog_page"
            return super().complete(prompt)

    results, warnings = index_asset_files(
        assets,
        ["cat.pdf"],
        _OrchLLM(),  # type: ignore[arg-type]
        cache_dir=cache,
        embedding_model="fake",
    )
    assert len(results) == 1
    assert results[0].doc_kind == DocumentKind.catalog
    assert results[0].status == IndexingStatus.indexed
    assert load_product_index(cache, "cat.pdf") is not None


def test_index_asset_files_other_skips(tmp_path: Path, monkeypatch) -> None:
    assets = tmp_path / "assets"
    cache = tmp_path / "cache"
    assets.mkdir()
    (assets / "x.pdf").write_bytes(b"%PDF-1.1")
    monkeypatch.setattr(
        "ai_tender.services.indexing.classify.preview_document_text",
        lambda *a, **k: ("служебная инструкция", []),
    )
    results, _ = index_asset_files(
        assets,
        ["x.pdf"],
        _FakeLLM("other"),  # type: ignore[arg-type]
        cache_dir=cache,
    )
    assert results[0].status == IndexingStatus.skipped
    assert load_product_index(cache, "x.pdf") is None
