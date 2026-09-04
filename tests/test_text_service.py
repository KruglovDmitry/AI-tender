from ai_tender.services.text_service import dedupe_requirements, make_requirement
from ai_tender.models import ExtractedRequirement


def test_make_requirement_strips_line_prefix_from_quote() -> None:
    req = make_requirement(
        text="Напряжение 230 В",
        quote="12|Номинальное напряжение 230 В",
        file="tender.docx",
        priority=3,
        confidence=0.9,
    )
    assert req.quote == "Номинальное напряжение 230 В"
    assert req.file == "tender.docx"
    assert req.location == "документ"
    assert req.line_start is None


def test_dedupe_requirements() -> None:
    items = [
        ExtractedRequirement(
            text="Класс точности не хуже 1.0",
            quote="класс точности 1.0",
            file="a.docx",
            location="док",
            priority=3,
            confidence=0.9,
        ),
        ExtractedRequirement(
            text="Класс точности не хуже 1.0",
            quote="повтор",
            file="a.docx",
            location="док",
            priority=2,
            confidence=0.5,
        ),
    ]
    assert len(dedupe_requirements(items, limit=5)) == 1


def test_dedupe_keeps_products_first() -> None:
    items = [
        ExtractedRequirement(
            text="Ток 5(80) А",
            quote="ток",
            file="a.docx",
            location="док",
            kind="specs",
            priority=3,
            confidence=0.9,
        ),
        ExtractedRequirement(
            text="Изделие серии X-05",
            quote="серия X-05",
            file="a.docx",
            location="док",
            kind="product",
            priority=3,
            confidence=0.8,
        ),
        ExtractedRequirement(
            text="230 В",
            quote="230",
            file="a.docx",
            location="док",
            kind="specs",
            priority=2,
            confidence=0.7,
        ),
    ]
    out = dedupe_requirements(items, limit=2)
    assert out[0].kind == "product"
    assert len(out) == 2
