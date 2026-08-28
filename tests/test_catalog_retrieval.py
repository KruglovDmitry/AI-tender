import numpy as np

from ai_tender.models import Product, ProductSource
from ai_tender.services.catalog_retrieval import (
    VlCatalog,
    catalog_hit_to_evidence,
    embedding_text,
    search_catalog,
)


def test_embedding_text_prefers_canonical_desc() -> None:
    product = Product(
        id="a",
        model="SR33020",
        canonical_desc="ИБП 20 кВА",
        raw_chunk="chunk",
    )
    assert embedding_text(product) == "ИБП 20 кВА"


def test_search_catalog_returns_top_by_cosine(monkeypatch) -> None:
    p1 = Product(id="1", model="MIR S-05", canonical_desc="счетчик однофазный 230В")
    p2 = Product(id="2", model="SR33020", canonical_desc="ИБП 20 кВА")
    p3 = Product(id="3", model="MAC9", canonical_desc="модуль ввода-вывода")
    basis = np.eye(3, dtype=np.float32)
    catalog = VlCatalog(
        products=[p1, p2, p3],
        vectors=basis,
        product_ids=["1", "2", "3"],
        source_files=["a.pdf", "b.pdf", "c.pdf"],
        catalog_names=["MIR", "SR33", "Tract"],
        embedding_model="mock",
        indexed_files=["a.pdf", "b.pdf", "c.pdf"],
    )

    def fake_embed(text: str, *, embedding_model: str, device: str | None) -> np.ndarray:
        if "счетчик" in text.lower() or "мир" in text.lower():
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(
        "ai_tender.services.catalog_retrieval.embed_query",
        fake_embed,
    )

    hits = search_catalog(
        catalog,
        "Позиция закупки: счетчик МИР С-05 230В",
        top_k=2,
        embedding_model="mock",
    )
    assert len(hits) == 2
    assert hits[0].product.model == "MIR S-05"
    assert hits[0].score > hits[1].score


def test_catalog_hit_to_evidence_contains_model() -> None:
    from ai_tender.services.catalog_retrieval import CatalogProductHit

    product = Product(
        id="x",
        model="SR33020-6x9",
        manufacturer="Штиль",
        canonical_desc="ИБП 20 кВА",
        source=ProductSource(catalog_id="SR33.pdf", page=3),
    )
    hit = CatalogProductHit(
        product=product,
        source_file="SR33.pdf",
        catalog_name="ИБП SR33",
        score=0.91,
    )
    ev = catalog_hit_to_evidence(hit)
    assert "SR33020-6x9" in ev.quote
    assert ev.file == "SR33.pdf"
    assert ev.page == 3
