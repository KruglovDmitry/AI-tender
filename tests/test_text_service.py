from llama_index.core import Document

from ai_tender.services.text_service import (
    dedupe_requirements,
    make_requirement,
    merge_documents_by_file,
    numbered_excerpt,
)
from ai_tender.models import ExtractedRequirement


def test_merge_documents_by_file_joins_pages() -> None:
    docs = [
        Document(text="стр1", metadata={"file_path": "a.pdf", "page_number": 1}),
        Document(text="стр2", metadata={"file_path": "a.pdf", "page_number": 2}),
        Document(text="другой", metadata={"file_path": "b.docx"}),
    ]
    merged = merge_documents_by_file(docs)
    by_label = {label: text for label, text, _ in merged}
    assert "стр1" in by_label["a.pdf"] and "стр2" in by_label["a.pdf"]
    assert by_label["b.docx"] == "другой"


def test_numbered_excerpt_keeps_line_prefixes() -> None:
    text = numbered_excerpt("первая\nвторая", max_chars=1000)
    assert text.startswith("1|первая")
    assert "2|вторая" in text


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
