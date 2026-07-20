from pathlib import Path

from llama_index.core.schema import TextNode

from ai_tender.index import node_to_evidence
from ai_tender.models import Evidence
from ai_tender.viewer import build_document_view, highlight_quote, resolve_evidence_path


def test_highlight_quote_marks_match() -> None:
    html = highlight_quote("Альфа требование beta конец", "требование beta")
    assert "<mark" in html
    assert "требование beta" in html


def test_resolve_evidence_path_relative(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    file = folder / "tz.pdf"
    file.write_bytes(b"%PDF-1.4")
    assert resolve_evidence_path(folder, "tz.pdf") == file.resolve()


def test_build_document_view_missing_file() -> None:
    evidence = Evidence(
        file="missing.pdf",
        location="стр. 1",
        quote="тест",
        page=1,
    )
    view = build_document_view(evidence, root=None, role="Тендер")
    assert view.path is None
    assert "не найден" in view.body_html.lower() or "не найден" in view.body_html


def test_node_to_evidence_keeps_page() -> None:
    node = TextNode(
        text="слово " * 50,
        metadata={
            "file_path": "assets/a.pdf",
            "location": "стр. 3",
            "page_number": 3,
        },
    )
    evidence = node_to_evidence(node, score=0.12)
    assert evidence.page == 3
    assert evidence.file == "assets/a.pdf"
