from pathlib import Path

from llama_index.core import Document
from llama_index.core.schema import TextNode

from ai_tender.anchors import locate_quote, refine_requirement_anchors
from ai_tender.models import ExtractedRequirement
from ai_tender.query_select import (
    attach_anchor,
    dedupe_requirements,
    merge_documents_by_file,
    requirement_to_evidence,
)


def _node(text: str) -> TextNode:
    return TextNode(text=text, metadata={"file_path": "tender.docx", "location": "док"})


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


def test_locate_quote_returns_line_numbers() -> None:
    source = "шапка\nКласс точности не хуже 1.0\nхвост"
    anchor = locate_quote(source, "Класс точности не хуже 1.0")
    assert anchor is not None
    assert anchor.line_start == 2


def test_attach_anchor_remembers_lines() -> None:
    node = _node("стр1\nНоминальное напряжение 230 В\nстр3")
    req = attach_anchor(
        text="Напряжение 230 В",
        quote="Номинальное напряжение 230 В",
        source_node=node,
        priority=3,
        confidence=0.9,
    )
    assert req.line_start == 2
    evidence = requirement_to_evidence(req)
    assert evidence.line_start == 2


def test_refine_anchors_against_full_file(tmp_path: Path) -> None:
    path = tmp_path / "tz.txt"
    path.write_text("a\nb\nТребование: ток 5 (80) А\nc\n", encoding="utf-8")
    req = ExtractedRequirement(
        text="ток 5 (80) А",
        quote="ток 5 (80) А",
        file="tz.txt",
        location="фрагмент",
        line_start=1,
        line_end=1,
    )
    refined = refine_requirement_anchors([req], tmp_path)
    assert refined[0].line_start == 3


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
            text="МИР С-05.10-230-5(80)",
            quote="МИР С-05",
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
