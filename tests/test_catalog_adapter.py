from ai_tender.extract.catalog_adapter import dedupe_catalog_products
from ai_tender.models import Product, ProductSource


def _product(model: str, *, category: str = "счетчики", manufacturer: str = "МИР") -> Product:
    return Product(
        id=model,
        model=model,
        manufacturer=manufacturer,
        category=category,
        canonical_desc=f"desc {model}",
        source=ProductSource(catalog_id="cat.pdf"),
        characteristics=[f"char {model}"],
    )


def test_dedupe_merges_same_model() -> None:
    a = _product("МИР С-04", category="счетчики")
    b = _product("МИР С-04", category="счетчики")
    b.characteristics = ["другая характеристика"]
    out = dedupe_catalog_products([a, b])
    assert len(out) == 1
    assert len(out[0].characteristics) == 2


def test_dedupe_drops_base_when_specific_exists_same_category() -> None:
    base = _product("МИР С-04", category="счетчики")
    specific = _product("МИР С-04.02-230-5(100)-R-D", category="счетчики")
    out = dedupe_catalog_products([base, specific])
    assert len(out) == 1
    assert out[0].model == "МИР С-04.02-230-5(100)-R-D"


def test_dedupe_keeps_distinct_products_in_different_categories() -> None:
    app = _product("МИР ДП", category="мобильные приложения")
    display = _product("МИР ДП-01", category="внешние устройства")
    out = dedupe_catalog_products([app, display])
    assert len(out) == 2
